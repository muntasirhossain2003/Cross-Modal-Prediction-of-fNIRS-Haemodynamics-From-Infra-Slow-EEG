"""Neurovascular coupling in ICH patients vs healthy controls (HEFMI).

HEFMI is 17 healthy + 20 intracerebral-haemorrhage patients recorded under the SAME
protocol with paired EEG-fNIRS. Every previous analysis in this project silently used the
Normals only, because `neurovascnet_loaders_v3.py` defaults to groups=("Normals",). The
patient half was audited (48/48 recordings passed QC) and then never used.

WHY THIS MATTERS WITHOUT NEEDING GOOD PREDICTION
Altered neurovascular coupling is itself a marker in stroke and ICH. So a clinical claim
here does not require the EEG->haemodynamic prediction to be accurate -- it requires the
COUPLING DESCRIPTORS to differ between groups. Three measures, none of which needs a
trained model:

  1. coupling strength   peak cross-correlation between the EEG band-power envelope
                         (sensorimotor channels) and HbO
  2. coupling lag        the lag at which that peak occurs -- the neurovascular delay,
                         measured directly rather than learned
  3. target quality      split-half reliability, HbO/HbR sign, evoked SNR per subject

CONFOUNDS THAT MUST BE STATED
Patients move more, and ICH can alter scalp/skull optical properties. A group difference in
coupling could reflect either physiology or acquisition. Motion rate and signal amplitude
are therefore reported per group as well, so a reviewer can see whether the coupling
difference tracks an artefact difference.

Usage:  python qc/clinical_coupling.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

ROOT = Path("dataset/HEFMI-ICH EEG+FNRI/epo_data")
CACHE = Path("neurovascnet_outputs/cache_v4")
EEG_FS_NATIVE, NIRS_FS = 256.0, 11.0
COUPLING_BAND = (13.0, 30.0)      # beta -- sensorimotor
MAX_LAG_S = 12.0


def _epo(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return m["epo"]


def _bp(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def envelope_at(x, fs, band, out_fs):
    """Band-power envelope of EEG, resampled to the fNIRS rate."""
    e = np.abs(sps.hilbert(_bp(x, fs, *band), axis=-1)) ** 2
    n_out = max(int(round(e.shape[-1] / fs * out_fs)), 4)
    return sps.resample(e, n_out, axis=-1)


def xcorr_lag(a, b, fs, max_lag_s=MAX_LAG_S):
    """Peak correlation of a vs b and the lag (s) at which it occurs.

    Positive lag = b follows a, i.e. the haemodynamic response trails the electrical one,
    which is the physiologically expected direction.
    """
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(max_lag_s * fs)
    best_r, best_l = -2.0, 0.0
    for lag in range(0, min(max_lag, n - 4) + 1):
        x, y = a[:n - lag], b[lag:]
        if len(x) < 4:
            break
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, lag / fs
    return best_r, best_l


def subject_files(group):
    eeg_root = ROOT / group / "EEG"
    if not eeg_root.exists():
        return []
    out = []
    for sd in sorted(eeg_root.iterdir()):
        if not sd.is_dir():
            continue
        for f in sorted(sd.glob("*_epo.mat")):
            n = ROOT / group / "NIRS" / sd.name / f.name
            if n.exists():
                out.append((sd.name, f, n))
    return out


def analyse():
    rows = []
    for group in ("Normals", "Patients"):
        files = subject_files(group)
        per_subject = {}
        for subj, ef, nf in files:
            try:
                e, n = _epo(ef), _epo(nf)
                X = np.asarray(e.X, np.float64)                 # (time, ch, trials)
                oxy = np.asarray(n.oxy.X, np.float64)
                dxy = np.asarray(n.doxy.X, np.float64)
            except Exception as exc:
                print(f"  {group}/{subj}: {type(exc).__name__}")
                continue

            n_tr = min(X.shape[2], oxy.shape[2])
            hbo = np.transpose(oxy[:, :, :n_tr], (2, 1, 0))     # (trial, ch, time)
            hbr = np.transpose(dxy[:, :, :n_tr], (2, 1, 0))
            eeg = np.transpose(X[:, :, :n_tr], (2, 1, 0))

            # --- coupling: beta envelope (all EEG ch) vs HbO (all optodes) ---
            rs, ls = [], []
            for t in range(n_tr):
                env = envelope_at(eeg[t], EEG_FS_NATIVE, COUPLING_BAND, NIRS_FS).mean(0)
                r, l = xcorr_lag(env, hbo[t].mean(0), NIRS_FS)
                if np.isfinite(r):
                    rs.append(r); ls.append(l)

            # --- target quality ---
            rng = np.random.default_rng(0)
            rel = []
            for _ in range(30):
                p = rng.permutation(n_tr)
                a_ = hbo[p[:n_tr // 2]].mean(0).ravel()
                b_ = hbo[p[n_tr // 2:]].mean(0).ravel()
                if a_.std() > 0 and b_.std() > 0:
                    rel.append(np.corrcoef(a_, b_)[0, 1])
            cc = [np.corrcoef(hbo[t].ravel(), hbr[t].ravel())[0, 1] for t in range(n_tr)]

            # --- artefact controls ---
            d = np.diff(hbo.reshape(n_tr, -1), axis=-1)
            motion = float((np.abs(d) > 5 * np.nanstd(d)).mean())
            amp = float(np.percentile(np.abs(hbo), 99.5))
            eeg_amp = float(np.percentile(np.abs(eeg), 99.5))

            rec = {"coupling_r": np.mean(rs) if rs else np.nan,
                   "coupling_lag_s": np.mean(ls) if ls else np.nan,
                   "reliability": np.mean(rel) if rel else np.nan,
                   "hbo_hbr": float(np.nanmedian(cc)),
                   "motion_frac": motion, "hbo_amp": amp, "eeg_amp": eeg_amp,
                   "n_trials": n_tr}
            per_subject.setdefault(subj, []).append(rec)

        for subj, recs in per_subject.items():
            m = {k: float(np.nanmean([r[k] for r in recs])) for k in recs[0]}
            rows.append({"group": group, "subject": subj, "n_sessions": len(recs), **m})
            print(f"  {group}/{subj:12s} coupling r={m['coupling_r']:+.3f} "
                  f"lag={m['coupling_lag_s']:.2f}s  rel={m['reliability']:+.3f}")
    return pd.DataFrame(rows)


def compare(df):
    print("\n" + "=" * 88)
    print("PATIENTS (ICH) vs CONTROLS")
    print("=" * 88)
    a = df[df.group == "Normals"]
    b = df[df.group == "Patients"]
    print(f"n = {len(a)} controls, {len(b)} patients\n")
    print(f"{'measure':22s}{'controls':>18s}{'patients':>18s}{'p (Mann-Whitney)':>20s}")
    print("-" * 88)
    out = []
    primary = ["coupling_r", "coupling_lag_s", "reliability", "hbo_hbr"]
    control = ["motion_frac", "hbo_amp", "eeg_amp", "n_trials"]
    for k in primary + control:
        x = a[k].dropna().values; y = b[k].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        tag = "" if k in primary else "   [artefact control]"
        star = " *" if p < 0.05 else ""
        print(f"{k:22s}{x.mean():>10.4f} +-{x.std():>5.3f}"
              f"{y.mean():>10.4f} +-{y.std():>5.3f}{p:>20.4f}{star}{tag}")
        out.append({"measure": k, "controls_mean": x.mean(), "controls_sd": x.std(),
                    "patients_mean": y.mean(), "patients_sd": y.std(), "p": p,
                    "is_primary": k in primary})
    res = pd.DataFrame(out)

    print("\n" + "-" * 88)
    sig_prim = res[(res.is_primary) & (res.p < 0.05)]
    sig_ctrl = res[(~res.is_primary) & (res.p < 0.05)]
    if len(sig_prim) == 0:
        print("No coupling measure separates the groups.")
    else:
        print(f"Group differences in: {', '.join(sig_prim.measure)}")
        if len(sig_ctrl):
            print(f"BUT artefact controls ALSO differ: {', '.join(sig_ctrl.measure)}")
            print("-> the coupling difference may reflect acquisition, not physiology.")
            print("   Any claim needs these confounds regressed out first.")
        else:
            print("Artefact controls do NOT differ -> the coupling difference is not")
            print("explained by motion or signal amplitude.")
    return res


def main():
    print("HEFMI: 17 healthy controls + 20 ICH patients, same protocol\n")
    df = analyse()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "clinical_per_subject.csv", index=False)
    res = compare(df)
    res.to_csv(TABLE_DIR / "clinical_group_comparison.csv", index=False)
    print(f"\nwrote {TABLE_DIR/'clinical_per_subject.csv'}")
    print(f"wrote {TABLE_DIR/'clinical_group_comparison.csv'}")


if __name__ == "__main__":
    main()
