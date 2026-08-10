"""Phase-0 check 3: does the ds002336 BOLD target actually contain the motor response?

The audit's event-locked ROI averages were computed on raw BOLD with no drift or
motion handling, so a real 0.5-2% response could have been buried. This runs a
proper first-level GLM instead:

    task boxcar (*) canonical HRF  +  6 motion params  +  polynomial drift

and reports the t-statistic and percent-signal-change for sensorimotor ROIs
against a non-motor control ROI. If precentral/postcentral does not come out
significantly positive here, the 48-ROI target genuinely does not carry the
motor response and no EEG encoder can be expected to predict it.

Reuses the ROI time series already extracted by the audit, so it does not
re-read any NIfTI.

Usage:  python qc/glm_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import ARRAY_DIR, MANIFEST_CSV, TABLE_DIR

TASK = "motorloc"
MOTOR_KEYS = ("Precentral", "Postcentral", "Supplementary", "Juxtapositional")
CONTROL_KEYS = ("Occipital Pole", "Intracalcarine", "Lingual")


def canonical_hrf(tr: float, length_s: float = 32.0) -> np.ndarray:
    """SPM-style double-gamma HRF sampled at the TR."""
    from scipy.stats import gamma
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / np.abs(h).sum()


def build_regressors(design: np.ndarray, cond_names: list[str], tr: float,
                     motion: np.ndarray | None, n_vol: int) -> tuple:
    """Task regressor(s) + nuisance. Returns (X, task_col_index, names)."""
    hrf = canonical_hrf(tr)
    cols, names = [], []

    # task regressors: everything that is not an explicit rest/baseline block
    task_idx = None
    for j, nm in enumerate(cond_names):
        if "rest" in nm.lower():
            continue
        conv = np.convolve(design[:, j], hrf)[:n_vol]
        cols.append(conv)
        names.append(f"task:{nm}")
        if task_idx is None:
            task_idx = len(cols) - 1

    if task_idx is None:      # no non-rest condition found
        return None, None, None

    # nuisance: motion (truncated/padded to n_vol), drift, intercept
    if motion is not None and motion.shape[0] >= 2:
        m = motion[:n_vol] if motion.shape[0] >= n_vol else np.vstack(
            [motion, np.repeat(motion[-1:], n_vol - motion.shape[0], axis=0)])
        for k in range(min(6, m.shape[1])):
            v = m[:, k]
            sd = v.std()
            cols.append((v - v.mean()) / (sd if sd > 0 else 1.0))
            names.append(f"motion{k}")

    t = np.linspace(-1, 1, n_vol)
    for p in (1, 2, 3):
        cols.append(t ** p)
        names.append(f"drift{p}")
    cols.append(np.ones(n_vol))
    names.append("intercept")

    return np.column_stack(cols), task_idx, names


def fit_roi(y: np.ndarray, X: np.ndarray, task_idx: int) -> tuple[float, float]:
    """OLS; return (percent signal change, t-statistic) for the task regressor."""
    baseline = y.mean()
    if baseline == 0 or not np.isfinite(baseline):
        return np.nan, np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return np.nan, np.nan
    s2 = (resid @ resid) / dof
    try:
        xtxi = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    se = np.sqrt(max(s2 * xtxi[task_idx, task_idx], 1e-30))
    t = beta[task_idx] / se if se > 0 else np.nan
    psc = 100.0 * beta[task_idx] / abs(baseline)
    return float(psc), float(t)


def main():
    man = pd.read_csv(MANIFEST_CSV)
    man = man[(man.dataset == "ds002336") & (man.modality == "fmri") &
              (man.task == TASK)]
    if man.empty:
        raise SystemExit(f"no ds002336 {TASK} fMRI rows in the manifest")

    from qc.qc_fmri import read_head_motion

    rows = []
    for r in man.to_dict("records"):
        subj = r["subject"]
        npz = ARRAY_DIR / "fmri_roi" / f"ds002336_{subj}_{r['session']}_{TASK}_fmri.npz"
        if not npz.exists():
            print(f"{subj}: no saved ROI array, skipped")
            continue
        d = np.load(npz, allow_pickle=True)
        ts = d["roi_timeseries"]                       # (n_vol, 48)
        design = d["design_matrix"]
        names = [str(x) for x in d["condition_names"]]
        labels = [str(x) for x in d["roi_labels"]]
        if design.size == 0 or not labels:
            print(f"{subj}: no design matrix / labels, skipped")
            continue

        n_vol = ts.shape[0]
        motion = read_head_motion(r.get("aux_path"))
        X, ti, _ = build_regressors(design, names, 2.0, motion, n_vol)
        if X is None:
            print(f"{subj}: no non-rest condition in {names}, skipped")
            continue

        motor = [i for i, nm in enumerate(labels) if any(k in nm for k in MOTOR_KEYS)]
        ctrl = [i for i, nm in enumerate(labels) if any(k in nm for k in CONTROL_KEYS)]

        def agg(idxs):
            out = [fit_roi(ts[:, i].astype(float), X, ti) for i in idxs
                   if ts[:, i].std() > 0]
            out = [o for o in out if np.isfinite(o[1])]
            if not out:
                return np.nan, np.nan
            return (float(np.mean([o[0] for o in out])),
                    float(np.max([o[1] for o in out])))

        m_psc, m_t = agg(motor)
        c_psc, c_t = agg(ctrl)
        rows.append({"subject": subj, "n_volumes": n_vol,
                     "n_motor_rois": len(motor), "n_control_rois": len(ctrl),
                     "motor_psc": m_psc, "motor_peak_t": m_t,
                     "control_psc": c_psc, "control_peak_t": c_t,
                     "motor_significant": bool(np.isfinite(m_t) and m_t > 3.0)})
        print(f"{subj}: motor PSC {m_psc:+.3f}%  peak t {m_t:6.2f}   |   "
              f"control PSC {c_psc:+.3f}%  peak t {c_t:6.2f}")

    if not rows:
        raise SystemExit("nothing fitted")

    df = pd.DataFrame(rows)
    out = TABLE_DIR / "glm_motorloc_check.csv"
    df.to_csv(out, index=False)

    n_sig = int(df.motor_significant.sum())
    print(f"\n{'='*66}")
    print(f"subjects fitted           : {len(df)}")
    print(f"motor peak t > 3.0        : {n_sig}/{len(df)}")
    print(f"median motor PSC          : {df.motor_psc.median():+.3f}%")
    print(f"median motor peak t       : {df.motor_peak_t.median():.2f}")
    print(f"median control PSC        : {df.control_psc.median():+.3f}%")
    print(f"median control peak t     : {df.control_peak_t.median():.2f}")
    print(f"{'='*66}")
    if n_sig >= max(1, len(df) // 2):
        print("VERDICT: the BOLD target DOES carry the motor response.")
        print("         The audit's near-zero event-locked values were a drift artifact.")
    else:
        print("VERDICT: the motor response is NOT detectable in the 48-ROI target.")
        print("         No EEG encoder can be expected to predict it as constructed.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
