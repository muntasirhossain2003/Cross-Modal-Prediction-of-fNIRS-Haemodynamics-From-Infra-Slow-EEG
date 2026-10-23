"""Target reliability over ALL subjects, one consistent method for every arm.

The v4 notebook reported reliability on the validation split only (6 fNIRS subjects,
3 fMRI subjects), and `qc/fix_fmri_target.py` reported a per-run figure across all
subjects. The two disagreed (0.518 vs 0.357) purely because of sample and pooling, which
made the "ceiling" ambiguous. This settles it:

  * every subject in the dataset, train and validation pooled
  * one unit of analysis: split-half within a subject, across all that subject's trials
  * N_REPEAT random splits per subject, averaged -- a single split is itself noisy
  * bootstrap CI over subjects, so the estimate comes with an error bar

The reliability ceiling on any prediction is ~sqrt(reliability), so this is the number
every decoding result should be read against.

Usage:  python qc/reliability_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V3 = Path("neurovascnet_outputs/cache_v3")
V4 = Path("neurovascnet_outputs/cache_v4")
N_REPEAT = 50
N_BOOT = 2000
MIN_TRIALS = 6

DATASETS = {
    "fNIRS HEFMI (v3)":  (V3/"fnirs_train.npz",          V3/"fnirs_val.npz"),
    "fNIRS hBCI (v4)":   (V4/"hbci_train_MI.npz",        V4/"hbci_val_MI.npz"),
    "fMRI v3 (broken)":  (V3/"ds002336_train.npz",       V3/"ds002336_val.npz"),
    "fMRI v4 (fixed)":   (V4/"ds002336_fixed_train.npz", V4/"ds002336_fixed_val.npz"),
}


def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def subject_reliability(target, n_repeat=N_REPEAT, seed=0):
    """Mean split-half correlation over n_repeat random halvings of one subject."""
    rng = np.random.default_rng(seed)
    n = target.shape[0]
    if n < MIN_TRIALS:
        return np.nan
    rs = []
    for _ in range(n_repeat):
        p = rng.permutation(n)
        r = concat_r(target[p[: n // 2]].mean(0), target[p[n // 2:]].mean(0))
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else np.nan


def main():
    rows, per_subject = [], []
    for name, (ptr, pva) in DATASETS.items():
        if not ptr.exists() or not pva.exists():
            print(f"{name:20s} cache missing, skipped")
            continue
        tr, va = np.load(ptr, allow_pickle=True), np.load(pva, allow_pickle=True)
        hemo = np.concatenate([tr["hemo"], va["hemo"]])
        # validation subject ids are offset so train/val subjects never collide
        sid = np.concatenate([tr["subject_ids"],
                              va["subject_ids"] + tr["subject_ids"].max() + 1])

        vals, counts = [], []
        for s in np.unique(sid):
            idx = np.flatnonzero(sid == s)
            r = subject_reliability(hemo[idx])
            if np.isfinite(r):
                vals.append(r)
                counts.append(idx.size)
                per_subject.append({"dataset": name, "subject": int(s),
                                    "n_trials": int(idx.size), "reliability": r})
        vals = np.asarray(vals)
        if vals.size == 0:
            print(f"{name:20s} no subject had >= {MIN_TRIALS} trials")
            continue

        rng = np.random.default_rng(0)
        boot = [np.mean(rng.choice(vals, vals.size, replace=True)) for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        mean = float(vals.mean())
        rows.append({"dataset": name, "n_subjects": int(vals.size),
                     "median_trials_per_subject": int(np.median(counts)),
                     "reliability_mean": mean,
                     "ci_low": float(lo), "ci_high": float(hi),
                     "reliability_median": float(np.median(vals)),
                     "reliability_min": float(vals.min()),
                     "reliability_max": float(vals.max()),
                     "ceiling": float(np.sqrt(max(mean, 0))),
                     "ceiling_ci_low": float(np.sqrt(max(lo, 0))),
                     "ceiling_ci_high": float(np.sqrt(max(hi, 0)))})
        print(f"{name:20s} n={vals.size:2d} subj, {int(np.median(counts)):4d} trials each"
              f"   reliability {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"   ceiling {np.sqrt(max(mean,0)):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "reliability_all_subjects.csv", index=False)
    pd.DataFrame(per_subject).to_csv(TABLE_DIR / "reliability_per_subject.csv", index=False)

    print("\n" + "=" * 92)
    print(df[["dataset", "n_subjects", "reliability_mean", "ci_low", "ci_high",
              "ceiling"]].round(3).to_string(index=False))
    print("=" * 92)
    print("\nDecoding results to read against these ceilings:")
    print("  fNIRS hBCI  ridge 0.0723 (p=0.005)   deep 0.0543 (p=0.035)")
    print("  fMRI  fixed ridge 0.0161 (p=0.473)   deep 0.0385 (p=0.403)")
    print(f"\nwrote {TABLE_DIR/'reliability_all_subjects.csv'}")
    print(f"wrote {TABLE_DIR/'reliability_per_subject.csv'}")


if __name__ == "__main__":
    main()
