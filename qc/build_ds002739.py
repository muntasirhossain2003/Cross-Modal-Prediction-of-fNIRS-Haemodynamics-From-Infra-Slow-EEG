"""Build the ds002739 EEG-fMRI arm — 24 subjects, the cleanest data in the project.

ds002739 has been used only as a discarded cross-task test, despite the audit rating it the
best fMRI data available: 24 subjects (vs ds002336's 10), tSNR 190 (vs 24), 771 volumes x
2 runs, thousands of trials, and EEG that is already gradient/BCG corrected and clean by
every measure (39 uV, line ratio 0.11, 4% bad channels).

ALIGNMENT. The EEG and BOLD timelines have different origins (EEG starts ~33 s before the
scan), but no offset estimation is needed: every trial carries BOTH times.
    tstim[i]                       -> stimulus time in EEG milliseconds
    dot_stim_validtrials onset[i]  -> the same stimulus in fMRI seconds
158 trials in each, in the same order, so trial i is simply epoched at tstim[i] in the EEG
and at onset[i]/TR in the BOLD.

The task is random-dot motion, so the responsive cortex is visual (hMT+ / lateral
occipital), not motor. All 48 Harvard-Oxford ROIs are stored so the subset can be chosen at
evaluation time rather than baked in here.

Usage:  python qc/build_ds002739.py [n_subjects]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("dataset/External fMRI validation OpenNeuro ds002739")
CACHE = Path("neurovascnet_outputs/cache_v4")
CACHE.mkdir(parents=True, exist_ok=True)

TR, EPOCH_VOL, N_ROI = 2.0, 15, 48
EEG_TARGET_FS, EEG_SAMPLES = 200.0, 6000     # 30 s
TMIN, TMAX = -2.0, 28.0


def _clean(ts, motion):
    """Regress polynomial drift (and motion, when present) out of each ROI series."""
    n = ts.shape[0]
    t = np.linspace(-1, 1, n)
    cols = [np.ones(n), t, t ** 2, t ** 3]
    if motion is not None and motion.shape[0] >= 2:
        m = motion[:n] if motion.shape[0] >= n else np.vstack(
            [motion, np.repeat(motion[-1:], n - motion.shape[0], 0)])
        for k in range(min(6, m.shape[1])):
            v = m[:, k]; sd = v.std()
            cols.append((v - v.mean()) / (sd if sd > 0 else 1.0))
    X = np.column_stack(cols)
    return ts - X @ np.linalg.lstsq(X, ts, rcond=None)[0]


def motion_from_events(df, n_vol):
    """ds002739 stores motion parameters as rows of the events file."""
    cols = []
    for ax in ("translX", "translY", "translZ", "rotX", "rotY", "rotZ"):
        sub = df[df.trial_type == f"motionparams_{ax}"]
        if len(sub) < n_vol:
            return None
        cols.append(pd.to_numeric(sub["param_value"], errors="coerce")
                    .to_numpy()[:n_vol])
    m = np.column_stack(cols)
    return m if np.isfinite(m).all() else None


def main():
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import resample_to_img

    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out = CACHE / "ds002739.npz"
    if out.exists():
        print("ds002739: cached"); return

    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    amap = atlas.maps if hasattr(atlas, "maps") else atlas["maps"]
    labels = [str(x) for x in (atlas.labels if hasattr(atlas, "labels")
                               else atlas["labels"])[1:N_ROI + 1]]

    E_, H_, S_ = [], [], []
    subs = sorted(d.name for d in ROOT.glob("sub-*") if d.is_dir())[:n_max]
    for si, sub in enumerate(subs):
        n_sub = 0
        for run in ("01", "02"):
            nii = ROOT / sub / "func" / f"{sub}_task-main_run-{run}_bold.nii.gz"
            evt = ROOT / sub / "func" / f"{sub}_task-main_run-{run}_events.tsv"
            eeg_f = ROOT / sub / "EEG" / f"EEG_data_{sub}_run-{run}.mat"
            ev_f = ROOT / sub / "EEG" / f"EEG_events_{sub}_run-{run}.mat"
            if not all(p.exists() for p in (nii, evt, eeg_f, ev_f)):
                continue
            try:
                # ---------- BOLD -> ROI series ----------
                img = nib.load(str(nii))
                data = np.asarray(img.dataobj, dtype=np.float32)
                if data.ndim != 4:
                    continue
                ref = nib.Nifti1Image(data[..., 0], img.affine, img.header)
                lab = np.asarray(resample_to_img(amap, ref, interpolation="nearest",
                                                 force_resample=True,
                                                 copy_header=True).dataobj).astype(int)
                ts = np.zeros((data.shape[3], N_ROI), np.float32)
                for k in range(1, N_ROI + 1):
                    m = lab == k
                    if m.sum() >= 10:
                        ts[:, k - 1] = data[m].mean(0)
                n_vol = data.shape[3]
                del data

                df = pd.read_csv(evt, sep="\t")
                ts = _clean(ts, motion_from_events(df, n_vol))

                # ---------- per-trial times in BOTH clocks ----------
                ev = sio.loadmat(ev_f, squeeze_me=True, struct_as_record=False)
                tstim = np.atleast_1d(np.asarray(ev["tstim"], dtype=float))   # EEG ms
                excl = set(np.atleast_1d(np.asarray(
                    ev.get("excludedtrials", []), dtype=int)).ravel().tolist())
                dot = df[df.trial_type == "dot_stim_validtrials"]
                onsets = pd.to_numeric(dot["onset"], errors="coerce").to_numpy()
                n_tr = min(tstim.size, onsets.size)
                if n_tr < 10:
                    continue

                # ---------- EEG ----------
                mm = sio.loadmat(eeg_f, squeeze_me=True, struct_as_record=False)
                d = mm["EEGdata"]
                Y = np.asarray(d.Y, dtype=np.float32)          # (ch, T)
                fs = float(np.atleast_1d(mm.get("fs", d.fd)).ravel()[0])
                step = max(int(round(fs / EEG_TARGET_FS)), 1)

                for i in range(n_tr):
                    if (i + 1) in excl:          # excludedtrials is 1-indexed
                        continue
                    a = int(round((tstim[i] / 1000.0 + TMIN) * fs))
                    b = int(round((tstim[i] / 1000.0 + TMAX) * fs))
                    v = int(round((onsets[i] + TMIN) / TR))
                    if a < 0 or b > Y.shape[1] or v < 0 or v + EPOCH_VOL > ts.shape[0]:
                        continue
                    seg = Y[:, a:b:step]
                    seg = sps.resample(seg, EEG_SAMPLES, axis=-1) \
                        if seg.shape[1] != EEG_SAMPLES else seg
                    seg = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
                    sh = ts[v:v + EPOCH_VOL].T
                    sh = sh - sh[:, :1]
                    E_.append(seg.astype(np.float32)); H_.append(sh.astype(np.float32))
                    S_.append(si); n_sub += 1
                del Y, mm
            except Exception as exc:
                print(f"  {sub}/run-{run}: {type(exc).__name__}: {str(exc)[:70]}")
        print(f"  {sub}: {n_sub} trials")

    if not E_:
        raise SystemExit("nothing built")
    nc = min(x.shape[0] for x in E_)
    np.savez_compressed(out, eeg=np.stack([x[:nc] for x in E_]),
                        hemo=np.stack(H_), subject_ids=np.asarray(S_),
                        roi_labels=np.array(labels, dtype=object))
    print(f"\n{out.name}: {len(E_)} trials, {len(set(S_))} subjects, "
          f"eeg {nc}ch x {EEG_SAMPLES}, hemo {N_ROI} ROIs x {EPOCH_VOL}")


if __name__ == "__main__":
    main()
