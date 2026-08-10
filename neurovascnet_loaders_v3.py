from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import scipy.io as sio
from scipy.signal import resample


ROOT = Path("dataset")


DATASET_ROLES = {
    "train_fnirs": {
        "name": "HEFMI-ICH EEG+NIRS",
        "path": ROOT / "HEFMI-ICH EEG+FNRI",
        "role": "primary EEG-fNIRS supervision",
        "loader": "load_hefmi_fnirs",
    },
    "train_fmri": {
        "name": "OpenNeuro ds002336",
        "path": ROOT / "OpenNeuro ds002336",
        "role": "primary EEG-fMRI supervision",
        "loader": "load_ds002336_fmri",
    },
    "test_fnirs": {
        "name": "TU Berlin EEG+NIRS",
        "path": ROOT / "TU Berlin EEG + NIRS",
        "role": "held-out EEG-fNIRS external test",
        "loader": "load_tuberlin_fnirs_external",
    },
    "test_fmri": {
        "name": "OpenNeuro ds002739",
        "path": ROOT / "External fMRI validation OpenNeuro ds002739",
        "role": "held-out EEG-fMRI external test",
        "loader": "load_ds002739_fmri_external",
    },
}

CACHE_DIR = Path("neurovascnet_outputs/cache")


@dataclass(frozen=True)
class LoaderConfig:
    eeg_samples: int = 6000
    fnirs_samples: int = 300
    # Match the existing shared EEG encoder caches: [N, 29, 6000].
    eeg_channels: int = 29
    # Match the existing fNIRS head/cache shape: 20 HbO + 20 HbR = [N, 40, 300].
    fnirs_channels_per_chromophore: int = 20
    fnirs_val_subjects: int = 4


def _as_epo_struct(mat_path: Path):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "epo" not in mat:
        raise KeyError(f"{mat_path} does not contain an 'epo' variable")
    return mat["epo"]


def _resample_time_first(x: np.ndarray, n_samples: int) -> np.ndarray:
    if x.shape[0] == n_samples:
        return x.astype(np.float32, copy=False)
    return resample(x, n_samples, axis=0).astype(np.float32)


def _zscore_per_trial(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # x: (channels, time)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return ((x - mu) / np.maximum(sd, eps)).astype(np.float32)


def _baseline_correct_hemo(x: np.ndarray, baseline_samples: int = 20) -> np.ndarray:
    # x: (channels, time)
    b = np.nanmean(x[:, : min(baseline_samples, x.shape[1])], axis=1, keepdims=True)
    return (x - b).astype(np.float32)


def _group_subject_ids(root: Path, group: str) -> list[str]:
    eeg_root = root / "epo_data" / group / "EEG"
    if not eeg_root.exists():
        return []
    return sorted([p.name for p in eeg_root.iterdir() if p.is_dir()], key=_subject_sort_key)


def _subject_sort_key(name: str) -> tuple[int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 10**9, name)


def _split_subjects(subjects: list[str], split: Literal["train", "val", "all"], n_val: int) -> set[str]:
    if split == "all":
        return set(subjects)
    val = set(subjects[-n_val:])
    if split == "val":
        return val
    if split == "train":
        return set(subjects) - val
    raise ValueError(f"Unknown split: {split}")


def _session_files(root: Path, group: str, subject: str) -> list[str]:
    eeg_dir = root / "epo_data" / group / "EEG" / subject
    nirs_dir = root / "epo_data" / group / "NIRS" / subject
    eeg_files = {p.name for p in eeg_dir.glob("*_epo.mat")} if eeg_dir.exists() else set()
    nirs_files = {p.name for p in nirs_dir.glob("*_epo.mat")} if nirs_dir.exists() else set()
    return sorted(eeg_files & nirs_files)


def load_hefmi_fnirs(
    split: Literal["train", "val", "all"] = "train",
    groups: Iterable[Literal["Normals", "Patients"]] = ("Normals",),
    cfg: LoaderConfig = LoaderConfig(),
    root: Path = DATASET_ROLES["train_fnirs"]["path"],
) -> dict[str, np.ndarray]:
    """Load HEFMI-ICH paired EEG/NIRS epochs in the notebook dict schema.

    Returns:
        eeg: [N, 32, 6000] by default
        hemo: [N, 180, 300] = 90 HbO + 90 HbR channels by default
        labels: trial class labels from the source files
        subject_ids: stable integer ids, unique across group and subject
        group_labels: 0=Normals, 1=Patients
        true_delays: zeros placeholder, because the dataset does not provide per-trial delays
    """
    root = Path(root)
    eeg_out: list[np.ndarray] = []
    hemo_out: list[np.ndarray] = []
    labels_out: list[int] = []
    subject_ids_out: list[int] = []
    group_labels_out: list[int] = []
    session_ids_out: list[str] = []
    audit: list[dict[str, object]] = []

    for group in groups:
        subjects = _group_subject_ids(root, group)
        selected = _split_subjects(subjects, split, cfg.fnirs_val_subjects)
        group_label = 0 if group == "Normals" else 1
        for subject in subjects:
            if subject not in selected:
                continue
            subject_num = _subject_sort_key(subject)[0]
            stable_subject_id = group_label * 1000 + subject_num
            for fname in _session_files(root, group, subject):
                eeg_path = root / "epo_data" / group / "EEG" / subject / fname
                nirs_path = root / "epo_data" / group / "NIRS" / subject / fname
                try:
                    eeg_epo = _as_epo_struct(eeg_path)
                    nirs_epo = _as_epo_struct(nirs_path)
                    eeg_x = np.asarray(eeg_epo.X, dtype=np.float32)  # time, channels, trials
                    oxy_x = np.asarray(nirs_epo.oxy.X, dtype=np.float32)
                    doxy_x = np.asarray(nirs_epo.doxy.X, dtype=np.float32)
                    eeg_y = np.asarray(eeg_epo.y).astype(np.int64).ravel()
                    nirs_y = np.asarray(nirs_epo.oxy.y).astype(np.int64).ravel()
                    n_trials = min(eeg_x.shape[2], oxy_x.shape[2], doxy_x.shape[2], eeg_y.size, nirs_y.size)
                    if n_trials == 0:
                        raise ValueError("no overlapping trials")
                    if not np.array_equal(eeg_y[:n_trials], nirs_y[:n_trials]):
                        raise ValueError("EEG and NIRS labels do not match")

                    eeg_x = eeg_x[:, : cfg.eeg_channels, :n_trials]
                    oxy_x = oxy_x[:, : cfg.fnirs_channels_per_chromophore, :n_trials]
                    doxy_x = doxy_x[:, : cfg.fnirs_channels_per_chromophore, :n_trials]
                    eeg_x = _resample_time_first(eeg_x, cfg.eeg_samples)
                    oxy_x = _resample_time_first(oxy_x, cfg.fnirs_samples)
                    doxy_x = _resample_time_first(doxy_x, cfg.fnirs_samples)

                    for trial in range(n_trials):
                        eeg_trial = _zscore_per_trial(eeg_x[:, :, trial].T)
                        oxy_trial = _baseline_correct_hemo(oxy_x[:, :, trial].T)
                        doxy_trial = _baseline_correct_hemo(doxy_x[:, :, trial].T)
                        hemo_trial = np.concatenate([oxy_trial, doxy_trial], axis=0).astype(np.float32)
                        eeg_out.append(eeg_trial)
                        hemo_out.append(hemo_trial)
                        labels_out.append(int(eeg_y[trial]))
                        subject_ids_out.append(stable_subject_id)
                        group_labels_out.append(group_label)
                        session_ids_out.append(f"{group}/{subject}/{fname}")
                    audit.append(
                        {
                            "group": group,
                            "subject": subject,
                            "session": fname,
                            "trials": int(n_trials),
                            "status": "ok",
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - data audits should capture all failures.
                    audit.append(
                        {
                            "group": group,
                            "subject": subject,
                            "session": fname,
                            "trials": 0,
                            "status": f"skip: {exc}",
                        }
                    )

    if not eeg_out:
        raise RuntimeError(f"HEFMI loader produced no samples for split={split}, groups={tuple(groups)}")

    return {
        "eeg": np.stack(eeg_out).astype(np.float32),
        "hemo": np.stack(hemo_out).astype(np.float32),
        "labels": np.asarray(labels_out, dtype=np.int64),
        "subject_ids": np.asarray(subject_ids_out, dtype=np.int64),
        "group_labels": np.asarray(group_labels_out, dtype=np.int64),
        "session_ids": np.asarray(session_ids_out),
        "true_delays": np.zeros(len(eeg_out), dtype=np.float32),
        "audit": np.asarray(audit, dtype=object),
    }


def _load_cached_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache {path}. Run the original notebook loader once, or rebuild this cache in v3."
        )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_ds002336_fmri(split: Literal["train", "val"] = "train") -> dict[str, np.ndarray]:
    """Load cached ds002336 EEG-fMRI train/validation bundle.

    This is the current primary fMRI supervision dataset. It is already in the
    shared notebook schema:
    eeg [N, 29, 6000], hemo [N, 48, 15].
    """
    name = f"ds002336_{split}_w30_e29_r48.npz"
    return _load_cached_npz(CACHE_DIR / name)


def load_tuberlin_fnirs_external() -> dict[str, np.ndarray]:
    """Load cached TU Berlin EEG-NIRS external fNIRS test bundle."""
    return _load_cached_npz(CACHE_DIR / "tuberlin_ext_w30_e29_r48.npz")


def load_ds002739_fmri_external() -> dict[str, np.ndarray]:
    """Load cached ds002739 EEG-fMRI external fMRI test bundle."""
    return _load_cached_npz(CACHE_DIR / "ds002739_ext_w30_e29_r48.npz")


def summarize_bundle(bundle: dict[str, np.ndarray]) -> dict[str, object]:
    labels, label_counts = np.unique(bundle["labels"], return_counts=True)
    subjects = np.unique(bundle["subject_ids"])
    group_summary = {}
    if "group_labels" in bundle:
        groups, group_counts = np.unique(bundle["group_labels"], return_counts=True)
        group_summary = {str(int(k)): int(v) for k, v in zip(groups, group_counts)}
    return {
        "eeg_shape": tuple(bundle["eeg"].shape),
        "hemo_shape": tuple(bundle["hemo"].shape),
        "n_subjects": int(subjects.size),
        "n_samples": int(bundle["eeg"].shape[0]),
        "label_counts": {str(int(k)): int(v) for k, v in zip(labels, label_counts)},
        "group_counts": group_summary,
        "audit_ok": int(sum(1 for row in bundle["audit"] if row["status"] == "ok")),
        "audit_skipped": int(sum(1 for row in bundle["audit"] if row["status"] != "ok")),
    }


if __name__ == "__main__":
    for split in ["train", "val"]:
        data = load_hefmi_fnirs(split=split)
        print(split, summarize_bundle(data))
