from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import welch
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("dataset/mendeley Simultaneous EEG-fNIRS Data on Learning Capability")
OUT_DIR = Path("neurovascnet_outputs_v2")
OUT_DIR.mkdir(exist_ok=True)

FEATURE_CACHE = OUT_DIR / "mendeley_trial_features_v2.npz"
SUMMARY_JSON = OUT_DIR / "mendeley_diagnostics_v2.json"

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EXCLUDE_EEG_LABELS = {
    "DC09",
    "EEG Mark1",
    "EEG Mark2",
    "Events/Markers",
}


@dataclass
class Segment:
    subject: str
    group_name: str
    y_group: int
    event_path: Path
    edf_path: Path
    fnirs_prefix: str | None


@dataclass
class SegmentQC:
    subject: str
    group_name: str
    event_file: str
    edf_file: str
    fnirs_prefix: str | None
    q_events: int
    fnirs_mark1: int
    used_trials: int
    eeg_channels: int
    edf_duration_s: float
    status: str


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def discover_segments() -> list[Segment]:
    segments: list[Segment] = []
    for group_name, y_group in [("Yes Implicit Learning", 1), ("No Implicit Learning", 0)]:
        group_dir = ROOT / group_name
        for subj_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            subject = subj_dir.name
            eeg_dir = subj_dir / "EEG"
            fnirs_dir = subj_dir / "fNIRS"
            if not eeg_dir.exists() or not fnirs_dir.exists():
                continue

            event_paths = sorted(eeg_dir.glob("Event*_sepQ.txt"))
            for event_path in event_paths:
                event_name = event_path.name
                suffix = ""
                if "_1-1" in event_name:
                    suffix = "1-1"
                elif "_1-2" in event_name:
                    suffix = "1-2"
                if suffix:
                    edf_candidates = sorted(eeg_dir.glob(f"Truncated*{suffix}.edf"))
                    if suffix == "1-1":
                        prefix = "Part1"
                    elif suffix == "1-2":
                        prefix = "Part2"
                    else:
                        prefix = None
                else:
                    edf_candidates = sorted(eeg_dir.glob("Truncated*.edf"))
                    edf_candidates = [p for p in edf_candidates if "_1-1" not in p.name and "_1-2" not in p.name]
                    prefix = None

                if not edf_candidates:
                    continue
                segments.append(
                    Segment(
                        subject=subject,
                        group_name=group_name,
                        y_group=y_group,
                        event_path=event_path,
                        edf_path=edf_candidates[0],
                        fnirs_prefix=prefix,
                    )
                )
    return segments


def read_q_events(path: Path) -> np.ndarray:
    last_exc: Exception | None = None
    for encoding in ["utf-8", "utf-16", "latin1"]:
        try:
            df = pd.read_csv(path, sep="\t", encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_exc = exc
    else:
        raise last_exc or UnicodeDecodeError("unknown", b"", 0, 1, "unknown encoding")
    event_type = df["type"].astype(str).str.strip()
    q_names = {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "qOne",
        "qTwo",
        "qThree",
        "qFour",
        "qO",
        "qT",
    }
    q = df[event_type.isin(q_names)]["latency"].astype(float)
    return q.to_numpy()


def read_fnirs_modality(fnirs_dir: Path, subject: str, prefix: str | None, modality: str) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    marks = None
    for probe in ["Probe1", "Probe2"]:
        if prefix:
            pattern = f"{subject}_{prefix}_HBA_{probe}_{modality}.csv"
        else:
            pattern = f"{subject}_HBA_{probe}_{modality}.csv"
        path = fnirs_dir / pattern
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=39, low_memory=False)
        ch_cols = [f"CH{i}" for i in range(1, 25)]
        data = df[ch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        parts.append(data)
        if marks is None:
            marks = pd.to_numeric(df["Mark"], errors="coerce").fillna(0).to_numpy(dtype=np.int16)
    if marks is None:
        raise ValueError("No fNIRS marks found")
    return np.concatenate(parts, axis=1), marks


def fnirs_trial_targets(fnirs_dir: Path, subject: str, prefix: str | None, n_trials: int) -> tuple[np.ndarray, int]:
    oxy, marks = read_fnirs_modality(fnirs_dir, subject, prefix, "Oxy")
    deoxy, _ = read_fnirs_modality(fnirs_dir, subject, prefix, "Deoxy")
    y_full = np.concatenate([oxy, deoxy], axis=1)
    mark_idx = np.flatnonzero(marks == 1)
    used = min(n_trials, len(mark_idx))
    targets = []
    for idx in mark_idx[:used]:
        base0 = max(0, idx - 10)
        base1 = max(base0 + 1, idx)
        resp0 = min(y_full.shape[0], idx + 40)
        resp1 = min(y_full.shape[0], idx + 120)
        if resp1 <= resp0 + 3:
            continue
        baseline = np.nanmean(y_full[base0:base1], axis=0)
        response = np.nanmean(y_full[resp0:resp1], axis=0)
        targets.append(response - baseline)
    if not targets:
        return np.empty((0, 96), dtype=np.float32), len(mark_idx)
    return np.asarray(targets, dtype=np.float32), len(mark_idx)


def eeg_bandpower_features(edf_path: Path, q_times: np.ndarray, n_trials: int) -> tuple[np.ndarray, list[str], float]:
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = reader.getSignalLabels()
        freqs = reader.getSampleFrequencies()
        duration = float(reader.file_duration)
        channel_indices = [
            i
            for i, label in enumerate(labels)
            if label not in EXCLUDE_EEG_LABELS and freqs[i] >= 200
        ]
        channel_labels = [labels[i] for i in channel_indices]
        if not channel_indices:
            return np.empty((0, 0), dtype=np.float32), [], duration

        q_times = q_times[:n_trials]
        feats = np.zeros((len(q_times), len(channel_indices) * len(EEG_BANDS)), dtype=np.float32)
        col = 0
        for ch_idx in channel_indices:
            sf = float(freqs[ch_idx])
            signal = reader.readSignal(ch_idx).astype(np.float32)
            win_len = int(round(2.0 * sf))
            epochs = np.empty((len(q_times), win_len), dtype=np.float32)
            valid = np.ones(len(q_times), dtype=bool)
            for i, t in enumerate(q_times):
                start = int(round(t * sf))
                end = start + win_len
                if start < 0 or end > signal.shape[0]:
                    valid[i] = False
                    start = max(0, min(start, signal.shape[0] - win_len))
                    end = start + win_len
                epochs[i] = signal[start:end]
            epochs -= np.nanmean(epochs, axis=1, keepdims=True)
            f, pxx = welch(epochs, fs=sf, nperseg=min(512, win_len), axis=1)
            total = np.trapz(pxx[:, (f >= 1) & (f <= 45)], f[(f >= 1) & (f <= 45)], axis=1)
            total = np.maximum(total, 1e-12)
            for low, high in EEG_BANDS.values():
                mask = (f >= low) & (f < high)
                bp = np.trapz(pxx[:, mask], f[mask], axis=1)
                feats[:, col] = np.log1p(bp / total)
                feats[~valid, col] = np.nan
                col += 1
        return feats, channel_labels, duration
    finally:
        reader.close()


def build_features(force: bool = False) -> dict:
    if FEATURE_CACHE.exists() and not force:
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        return {k: data[k] for k in data.files}

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    segment_ids: list[str] = []
    qc: list[SegmentQC] = []

    for seg in discover_segments():
        status = "ok"
        q_events = read_q_events(seg.event_path)
        try:
            fnirs_dir = seg.event_path.parents[1] / "fNIRS"
            y, n_marks = fnirs_trial_targets(fnirs_dir, seg.subject, seg.fnirs_prefix, len(q_events))
            used = min(len(q_events), y.shape[0])
            if used < 20:
                raise ValueError(f"Too few aligned trials: {used}")
            x, ch_labels, duration = eeg_bandpower_features(seg.edf_path, q_events, used)
            used = min(used, x.shape[0])
            x = x[:used]
            y = y[:used]
            finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
            x = x[finite]
            y = y[finite]
            used = x.shape[0]
            if used < 20:
                raise ValueError(f"Too few finite trials: {used}")
            xs.append(x)
            ys.append(y)
            labels.extend([seg.y_group] * used)
            subjects.extend([seg.subject] * used)
            segment_ids.extend([f"{seg.subject}:{seg.event_path.stem}"] * used)
        except Exception as exc:  # noqa: BLE001 - QC should capture all bad segments.
            n_marks = -1
            used = 0
            ch_labels = []
            duration = float("nan")
            status = f"skip: {exc}"
        qc.append(
            SegmentQC(
                subject=seg.subject,
                group_name=seg.group_name,
                event_file=seg.event_path.name,
                edf_file=seg.edf_path.name,
                fnirs_prefix=seg.fnirs_prefix,
                q_events=int(len(q_events)),
                fnirs_mark1=int(n_marks),
                used_trials=int(used),
                eeg_channels=int(len(ch_labels)),
                edf_duration_s=float(duration),
                status=status,
            )
        )
        print(f"{seg.subject} {seg.event_path.name}: {status}, used={used}")

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    y_group = np.asarray(labels, dtype=np.int64)
    subject_arr = np.asarray(subjects)
    segment_arr = np.asarray(segment_ids)
    qc_json = json.dumps([asdict(item) for item in qc])
    np.savez_compressed(
        FEATURE_CACHE,
        X_eeg=X,
        Y_fnirs=Y,
        y_group=y_group,
        subjects=subject_arr,
        segments=segment_arr,
        qc_json=np.asarray(qc_json),
    )
    return {
        "X_eeg": X,
        "Y_fnirs": Y,
        "y_group": y_group,
        "subjects": subject_arr,
        "segments": segment_arr,
        "qc_json": np.asarray(qc_json),
    }


def loso_group_classification(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> dict:
    unique_subjects = np.unique(subjects)
    subj_true = []
    subj_pred = []
    subj_score = []
    for test_subj in unique_subjects:
        train = subjects != test_subj
        test = subjects == test_subj
        if len(np.unique(y[train])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=42,
            ),
        )
        clf.fit(X[train], y[train])
        prob = clf.predict_proba(X[test])[:, 1]
        score = float(np.nanmean(prob))
        subj_score.append(score)
        subj_pred.append(int(score >= 0.5))
        subj_true.append(int(y[test][0]))
    result = {
        "subjects": int(len(subj_true)),
        "accuracy": float(accuracy_score(subj_true, subj_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(subj_true, subj_pred)),
        "positive_rate_pred": float(np.mean(subj_pred)),
    }
    if len(np.unique(subj_true)) == 2:
        result["auc"] = float(roc_auc_score(subj_true, subj_score))
    else:
        result["auc"] = float("nan")
    return result


def loso_eeg_to_fnirs(X: np.ndarray, Y: np.ndarray, subjects: np.ndarray) -> dict:
    unique_subjects = np.unique(subjects)
    y_scalar = Y[:, :48].mean(axis=1)
    pred_all = np.zeros_like(Y)
    pred_scalar = np.zeros_like(y_scalar)
    for test_subj in unique_subjects:
        train = subjects != test_subj
        test = subjects == test_subj
        reg = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        reg.fit(X[train], Y[train])
        pred_all[test] = reg.predict(X[test])
        reg_s = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        reg_s.fit(X[train], y_scalar[train])
        pred_scalar[test] = reg_s.predict(X[test])

    pls_corr = float("nan")
    try:
        pred_pls = np.zeros_like(Y[:, :48])
        for test_subj in unique_subjects:
            train = subjects != test_subj
            test = subjects == test_subj
            n_comp = min(5, X.shape[1], Y[:, :48].shape[1], max(1, train.sum() - 1))
            pls = make_pipeline(StandardScaler(), PLSRegression(n_components=n_comp))
            pls.fit(X[train], Y[train, :48])
            pred_pls[test] = pls.predict(X[test])
        pls_corr = _safe_corr(pred_pls, Y[:, :48])
    except Exception:
        pass

    return {
        "ridge_vector_oxy_deoxy_pearson": _safe_corr(pred_all, Y),
        "ridge_vector_oxy_only_pearson": _safe_corr(pred_all[:, :48], Y[:, :48]),
        "ridge_scalar_oxy_mean_pearson": _safe_corr(pred_scalar, y_scalar),
        "pls_vector_oxy_only_pearson": pls_corr,
        "target_oxy_scalar_std": float(np.nanstd(y_scalar)),
    }


def summarize_qc(qc: list[dict]) -> dict:
    ok = [row for row in qc if row["status"] == "ok"]
    skipped = [row for row in qc if row["status"] != "ok"]
    by_subject = {}
    for row in ok:
        by_subject.setdefault(row["subject"], 0)
        by_subject[row["subject"]] += row["used_trials"]
    yes_subjects = sorted({row["subject"] for row in ok if row["group_name"].startswith("Yes")})
    no_subjects = sorted({row["subject"] for row in ok if row["group_name"].startswith("No")})
    return {
        "ok_segments": len(ok),
        "skipped_segments": len(skipped),
        "usable_subjects": len(by_subject),
        "usable_yes_subjects": len(yes_subjects),
        "usable_no_subjects": len(no_subjects),
        "usable_trials_total": int(sum(by_subject.values())),
        "usable_trials_min_per_subject": int(min(by_subject.values())) if by_subject else 0,
        "usable_trials_median_per_subject": float(np.median(list(by_subject.values()))) if by_subject else 0.0,
        "usable_trials_max_per_subject": int(max(by_subject.values())) if by_subject else 0,
        "skipped": skipped,
    }


def main() -> None:
    force = "--force" in sys.argv
    data = build_features(force=force)
    X = data["X_eeg"].astype(np.float32)
    Y = data["Y_fnirs"].astype(np.float32)
    y = data["y_group"].astype(np.int64)
    subjects = data["subjects"].astype(str)
    qc = json.loads(str(data["qc_json"].item()))

    fnirs_scalar = Y[:, :48].mean(axis=1, keepdims=True)
    fnirs_peak_like = Y[:, :48]
    X_fused = np.concatenate([X, fnirs_scalar, fnirs_peak_like], axis=1)

    summary = {
        "dataset": "Mendeley simultaneous EEG-fNIRS implicit learning",
        "feature_cache": str(FEATURE_CACHE),
        "shapes": {
            "X_eeg": list(X.shape),
            "Y_fnirs_oxy_deoxy": list(Y.shape),
            "subjects": int(len(np.unique(subjects))),
            "trials": int(X.shape[0]),
            "group_counts_trial_level": {
                "no_implicit_learning": int((y == 0).sum()),
                "yes_implicit_learning": int((y == 1).sum()),
            },
            "group_counts_subject_level": {
                "no_implicit_learning": int(len({s for s in np.unique(subjects) if y[subjects == s][0] == 0})),
                "yes_implicit_learning": int(len({s for s in np.unique(subjects) if y[subjects == s][0] == 1})),
            },
        },
        "qc": summarize_qc(qc),
        "loso_subject_classification": {
            "eeg_bandpower_to_group": loso_group_classification(X, y, subjects),
            "fnirs_response_to_group": loso_group_classification(Y, y, subjects),
            "eeg_plus_fnirs_to_group": loso_group_classification(X_fused, y, subjects),
        },
        "loso_eeg_to_fnirs": loso_eeg_to_fnirs(X, Y, subjects),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
