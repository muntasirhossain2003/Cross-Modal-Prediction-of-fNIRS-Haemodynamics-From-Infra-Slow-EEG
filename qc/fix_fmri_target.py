"""Can the ds002336 fMRI target be made to reproduce? A target-side ablation.

The v4 notebook measured split-half reliability of the cached 48-ROI BOLD target at
**-0.793** — it does not reproduce against itself, so every fMRI number the project has
produced is uninterpretable.

Three things are wrong with how that target was built, and this script tests each
independently, on the target alone (no EEG, no model — reliability is a property of the
target, so it can be fixed before worrying about pairing):

    1. NO NUISANCE REGRESSION. The cached series keep head motion and scanner drift.
       The audit's GLM check already showed this matters: raw event-locked averaging gave
       ~0.001% signal change while a GLM with motion + drift regressors gave +0.087%.
    2. NOT EVENT-LOCKED. v3 used 30 s sliding windows; the fNIRS fix worked partly because
       v4 epochs on actual stimulus onsets.
    3. ALL 48 ROIs. Most of the cortex does not respond to a motor task, so 40+ unpredictable
       ROIs dilute the handful that do.

Reuses the ROI time series already extracted by the audit, so no NIfTI is re-read.

Usage:  python qc/fix_fmri_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import ARRAY_DIR, DATASET_ROOT, MANIFEST_CSV, TABLE_DIR
from qc.qc_fmri import read_head_motion

TR = 2.0
EPOCH_VOL = 15                      # 30 s at TR=2
MOTOR_KEYS = ("Precentral", "Postcentral", "Supplementary", "Juxtapositional")
TASKS = ("motorloc", "MIpre", "MIpost")
RNG = np.random.default_rng(0)


def clean_series(ts: np.ndarray, motion: np.ndarray | None) -> np.ndarray:
    """Regress motion + polynomial drift out of each ROI time series."""
    n = ts.shape[0]
    cols = [np.ones(n)]
    t = np.linspace(-1, 1, n)
    for p in (1, 2, 3):
        cols.append(t ** p)
    if motion is not None and motion.shape[0] >= 2:
        m = motion[:n] if motion.shape[0] >= n else np.vstack(
            [motion, np.repeat(motion[-1:], n - motion.shape[0], axis=0)])
        for k in range(min(6, m.shape[1])):
            v = m[:, k]
            sd = v.std()
            cols.append((v - v.mean()) / (sd if sd > 0 else 1.0))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
    return ts - X @ beta


def task_onsets(events_path: Path) -> np.ndarray:
    """Onsets (s) of non-rest blocks."""
    if not events_path.exists():
        return np.array([])
    df = pd.read_csv(events_path, sep="\t")
    if "onset" not in df.columns:
        return np.array([])
    if "trial_type" in df.columns:
        df = df[~df["trial_type"].astype(str).str.lower().str.contains("rest")]
    on = pd.to_numeric(df["onset"], errors="coerce").to_numpy(dtype=float)
    return on[np.isfinite(on)]


def epoch(ts: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """(n_vol, n_roi) -> (n_trials, n_roi, EPOCH_VOL) locked to onsets."""
    out = []
    for o in onsets:
        s = int(round(o / TR))
        if s >= 0 and s + EPOCH_VOL <= ts.shape[0]:
            seg = ts[s:s + EPOCH_VOL].T
            out.append(seg - seg[:, :1])          # baseline at onset
    return np.stack(out) if out else np.empty((0, ts.shape[1], EPOCH_VOL))


def split_half(ep: np.ndarray) -> float:
    if ep.shape[0] < 4:
        return np.nan
    p = RNG.permutation(ep.shape[0])
    a = ep[p[:len(p) // 2]].mean(0).ravel()
    b = ep[p[len(p) // 2:]].mean(0).ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    man = pd.read_csv(MANIFEST_CSV)
    man = man[(man.dataset == "ds002336") & (man.modality == "fmri") &
              (man.task.isin(TASKS))]
    root = DATASET_ROOT / "OpenNeuro ds002336"

    # `legit=False` marks variants whose reliability estimate is not trustworthy:
    # overlapping sliding windows share samples between the two split halves, so the
    # split-half correlation is inflated by leakage rather than by reproducible signal.
    variants = {
        "v3-style: sliding, no clean, 48 ROI": dict(clean=False, locked=False, motor=False,
                                                    legit=False),
        "event-locked, no clean, 48 ROI":      dict(clean=False, locked=True,  motor=False,
                                                    legit=False),
        "event-locked, cleaned, 48 ROI":       dict(clean=True,  locked=True,  motor=False,
                                                    legit=True),
        "event-locked, cleaned, MOTOR ROI":    dict(clean=True,  locked=True,  motor=True,
                                                    legit=True),
    }
    acc = {k: [] for k in variants}
    rows = []

    for r in man.to_dict("records"):
        subj, task = r["subject"], r["task"]
        npz = ARRAY_DIR / "fmri_roi" / f"ds002336_{subj}_{r['session']}_{task}_fmri.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        ts = d["roi_timeseries"].astype(float)
        labels = [str(x) for x in d["roi_labels"]]
        if ts.size == 0 or not labels:
            continue

        motion = read_head_motion(r.get("aux_path"))
        onsets = task_onsets(root / f"task-{task}_events.tsv")
        motor_idx = [i for i, nm in enumerate(labels)
                     if any(k in nm for k in MOTOR_KEYS)]

        for name, cfg in variants.items():
            x = clean_series(ts, motion) if cfg["clean"] else ts
            if cfg["motor"]:
                if not motor_idx:
                    continue
                x = x[:, motor_idx]
            if cfg["locked"]:
                if onsets.size == 0:
                    continue
                ep = epoch(x, onsets)
            else:
                # v3-style sliding windows, stride 5 volumes
                segs = [x[i:i + EPOCH_VOL].T
                        for i in range(0, x.shape[0] - EPOCH_VOL, 5)]
                ep = np.stack(segs) if segs else np.empty((0, x.shape[1], EPOCH_VOL))
            rel = split_half(ep)
            if np.isfinite(rel):
                acc[name].append(rel)
            rows.append({"subject": subj, "task": task, "variant": name,
                         "n_epochs": int(ep.shape[0]),
                         "n_rois": int(ep.shape[1]) if ep.size else 0,
                         "reliability": rel})

    df = pd.DataFrame(rows)
    out = TABLE_DIR / "fmri_target_fix.csv"
    df.to_csv(out, index=False)

    print(f"{'variant':40s}{'n':>5s}{'reliab':>9s}{'median':>9s}   note")
    print("-" * 88)
    for name, cfg in variants.items():
        v = np.asarray(acc[name], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            print(f"{name:40s}{0:>5d}{'n/a':>9s}{'n/a':>9s}")
            continue
        if not cfg["legit"]:
            note = ("NOT VALID - overlapping windows leak between halves"
                    if not cfg["locked"] else
                    "NOT VALID - drift is reproducible but not neural")
        else:
            note = ""
        print(f"{name:40s}{v.size:>5d}{v.mean():>9.3f}{np.median(v):>9.3f}   {note}")
    print("-" * 88)
    print("reference: fNIRS hBCI (v4) target reliability = +0.506")
    print("           cached v3 fMRI target, as measured in the notebook = -0.793")

    legit = {k: acc[k] for k, c in variants.items() if c["legit"] and acc[k]}
    if not legit:
        print("\nno valid variant produced a reliability estimate")
        return
    best = max(legit, key=lambda k: np.nanmean(legit[k]))
    bv = float(np.nanmean(legit[best]))
    print(f"\nBEST VALID: {best}  ->  {bv:+.3f}")
    if bv > 0.3:
        print("VERDICT: the fMRI target IS fixable -- event-locking turns -0.793 into a")
        print("         positive, reproducible target. Rebuild the cache this way, then")
        print("         re-gate and re-run the EEG->BOLD decoding.")
    elif bv > 0:
        print("VERDICT: partially fixed -- reproduces, but well below the fNIRS arm (0.506).")
    else:
        print("VERDICT: still does not reproduce. The problem is upstream of preprocessing.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
