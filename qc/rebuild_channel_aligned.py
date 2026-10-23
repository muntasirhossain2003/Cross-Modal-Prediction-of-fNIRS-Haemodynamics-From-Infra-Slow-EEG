"""Rebuild both fMRI caches on a COMMON, NAME-ALIGNED EEG montage.

External fMRI validation (train ds002336 -> test ds002739) was previously reported as
"not evaluable" because the caches carried no channel names and had different counts
(63 vs 58). Both halves of that are fixable:

  * ds002739 ships `additional_files/electrode_info.elp` with all 64 electrode names, and
    each recording lists `excludedchannels`, so Y's rows map back to names exactly:
    rows = the 64 names minus the excluded indices (1-based).
  * ds002336 is BrainVision and carries names in its header.

Intersecting the two montages gives ~53 shared 10-20 electrodes. This script rebuilds both
caches restricted to that shared set, in one canonical order, and STORES THE NAMES in the
npz so the correspondence can never be silently lost again.

The earlier cache was built with `x[:min_channels]`, which truncated by POSITION and
destroyed identity -- different subjects had different exclusions, so row i was not the
same electrode across subjects. That bug is what made external validation look impossible.

Usage:  python qc/rebuild_channel_aligned.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

R39 = Path("dataset/External fMRI validation OpenNeuro ds002739")
R36 = Path("dataset/OpenNeuro ds002336")
CACHE = Path("neurovascnet_outputs/cache_v4")
TR, EPOCH_VOL, N_ROI = 2.0, 15, 48
EEG_SAMPLES, TMIN, TMAX = 6000, -2.0, 28.0
MOTOR_KEYS = ("Precentral", "Postcentral", "Supplementary", "Juxtapositional")
MAX_FILLED_CHANNELS = 6   # tolerate a few provider-excluded channels per run
FMRI_TASKS_36 = ("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF")


def elp_names():
    txt = (R39 / "additional_files" / "electrode_info.elp").read_text(errors="ignore")
    return re.findall(r"([A-Za-z]+[0-9z]*)\s+-?\d+\s+-?\d+", txt)


def ds36_names():
    import mne
    v = next((R36 / "derivatives").rglob("*_eeg_pp.vhdr"))
    raw = mne.io.read_raw_brainvision(str(v), preload=False, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    return [raw.ch_names[i] for i in picks]


def common_montage():
    a, b = elp_names(), ds36_names()
    la = {n.lower(): n for n in a}
    keep = [la[n.lower()] for n in b if n.lower() in la]     # canonical order = ds002336's
    return keep, a, b


def _clean(ts, motion):
    n = ts.shape[0]; t = np.linspace(-1, 1, n)
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
    cols = []
    for ax in ("translX", "translY", "translZ", "rotX", "rotY", "rotZ"):
        sub = df[df.trial_type == f"motionparams_{ax}"]
        if len(sub) < n_vol:
            return None
        cols.append(pd.to_numeric(sub["param_value"], errors="coerce").to_numpy()[:n_vol])
    m = np.column_stack(cols)
    return m if np.isfinite(m).all() else None


# ------------------------------------------------------------------ ds002739
def build_39(keep, all39, n_max=24):
    out = CACHE / "ds002739_aligned2.npz"
    if out.exists():
        print("ds002739_aligned: cached"); return out
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import resample_to_img

    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    amap = atlas.maps if hasattr(atlas, "maps") else atlas["maps"]
    labels = [str(x) for x in (atlas.labels if hasattr(atlas, "labels")
                               else atlas["labels"])[1:N_ROI + 1]]
    lower39 = [n.lower() for n in all39]
    want = [lower39.index(k.lower()) for k in keep]          # index into the 64 names

    E_, H_, S_ = [], [], []
    filled_log = []
    for si, sub in enumerate(sorted(d.name for d in R39.glob("sub-*") if d.is_dir())[:n_max]):
        cnt = 0
        for run in ("01", "02"):
            nii = R39 / sub / "func" / f"{sub}_task-main_run-{run}_bold.nii.gz"
            evt = R39 / sub / "func" / f"{sub}_task-main_run-{run}_events.tsv"
            ef = R39 / sub / "EEG" / f"EEG_data_{sub}_run-{run}.mat"
            vf = R39 / sub / "EEG" / f"EEG_events_{sub}_run-{run}.mat"
            if not all(p.exists() for p in (nii, evt, ef, vf)):
                continue
            try:
                img = nib.load(str(nii)); data = np.asarray(img.dataobj, np.float32)
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
                n_vol = data.shape[3]; del data
                df = pd.read_csv(evt, sep="\t")
                ts = _clean(ts, motion_from_events(df, n_vol))

                ev = sio.loadmat(vf, squeeze_me=True, struct_as_record=False)
                tstim = np.atleast_1d(np.asarray(ev["tstim"], float))
                dot = df[df.trial_type == "dot_stim_validtrials"]
                onsets = pd.to_numeric(dot["onset"], errors="coerce").to_numpy()

                mm = sio.loadmat(ef, squeeze_me=True, struct_as_record=False)
                Y = np.asarray(mm["EEGdata"].Y, np.float32)
                fs = float(np.atleast_1d(mm.get("fs", mm["EEGdata"].fd)).ravel()[0])
                excl = np.atleast_1d(np.asarray(mm.get("excludedchannels", []),
                                                dtype=int)).ravel()
                # rows of Y == the 64 canonical names minus the excluded (1-based) indices
                present = [i for i in range(len(all39)) if (i + 1) not in set(excl.tolist())]
                if len(present) != Y.shape[0]:
                    print(f"  {sub}/{run}: channel bookkeeping mismatch "
                          f"({len(present)} vs {Y.shape[0]}), skipped")
                    continue
                pos = {ch: r for r, ch in enumerate(present)}
                # Earlier this dropped the whole run if ANY shared channel was excluded,
                # which cost 9 of 24 subjects. A channel excluded by the providers is a
                # bad channel, so zero-filling it is what interpolation would approximate
                # and it keeps the subject. Count of filled channels is reported so the
                # cost is visible rather than hidden.
                missing = [w for w in want if w not in pos]
                if len(missing) > MAX_FILLED_CHANNELS:
                    print(f"  {sub}/{run}: {len(missing)} shared channels excluded "
                          f"(limit {MAX_FILLED_CHANNELS}), skipped")
                    continue
                Yc = np.zeros((len(want), Y.shape[1]), dtype=Y.dtype)
                for j, w in enumerate(want):
                    if w in pos:
                        Yc[j] = Y[pos[w]]
                if missing:
                    filled_log.append((sub, run, len(missing)))
                Y = Yc

                step = max(int(round(fs / 200.0)), 1)
                for i in range(min(tstim.size, onsets.size)):
                    a = int(round((tstim[i] / 1000.0 + TMIN) * fs))
                    b = int(round((tstim[i] / 1000.0 + TMAX) * fs))
                    v = int(round((onsets[i] + TMIN) / TR))
                    if a < 0 or b > Y.shape[1] or v < 0 or v + EPOCH_VOL > ts.shape[0]:
                        continue
                    seg = Y[:, a:b:step]
                    if seg.shape[1] != EEG_SAMPLES:
                        seg = sps.resample(seg, EEG_SAMPLES, axis=-1)
                    seg = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
                    sh = ts[v:v + EPOCH_VOL].T; sh = sh - sh[:, :1]
                    E_.append(seg.astype(np.float32)); H_.append(sh.astype(np.float32))
                    S_.append(si); cnt += 1
                del Y, mm
            except Exception as exc:
                print(f"  {sub}/{run}: {type(exc).__name__}: {str(exc)[:60]}")
        print(f"  {sub}: {cnt} trials")
    np.savez_compressed(out, eeg=np.stack(E_), hemo=np.stack(H_),
                        subject_ids=np.asarray(S_),
                        ch_names=np.array(keep, dtype=object),
                        roi_labels=np.array(labels, dtype=object))
    if filled_log:
        tot = sum(n for _, _, n in filled_log)
        print(f"  zero-filled {tot} channel-instances across {len(filled_log)} runs "
              f"(max {max(n for _,_,n in filled_log)} in one run)")
    print(f"{out.name}: {len(E_)} trials, {len(set(S_))} subj, {len(keep)} aligned channels")
    return out


# ------------------------------------------------------------------ ds002336
def build_36(keep):
    out = CACHE / "ds002336_aligned.npz"
    if out.exists():
        print("ds002336_aligned: cached"); return out
    import mne, nibabel as nib
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import resample_to_img

    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    amap = atlas.maps if hasattr(atlas, "maps") else atlas["maps"]
    labels = [str(x) for x in (atlas.labels if hasattr(atlas, "labels")
                               else atlas["labels"])[1:N_ROI + 1]]
    motor = [i for i, nm in enumerate(labels) if any(k in nm for k in MOTOR_KEYS)]

    E_, H_, S_ = [], [], []
    for si, sub in enumerate(sorted(d.name for d in R36.glob("sub-*") if d.is_dir())):
        cnt = 0
        for task in FMRI_TASKS_36:
            nii = R36 / sub / "func" / f"{sub}_task-{task}_bold.nii.gz"
            vhdr = R36 / "derivatives" / sub / "eeg_pp" / f"{sub}_task-{task}_eeg_pp.vhdr"
            ev = R36 / f"task-{task}_events.tsv"
            if not all(p.exists() for p in (nii, vhdr, ev)):
                continue
            try:
                img = nib.load(str(nii)); data = np.asarray(img.dataobj, np.float32)
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
                del data
                hm = (R36 / "derivatives" / sub / "head_motion" /
                      f"d_{sub}_task-{task}_headmotion.tsv")
                motion = None
                if hm.exists():
                    dfm = pd.read_csv(hm, sep="\t").select_dtypes("number").dropna(axis=1, how="all")
                    if dfm.shape[1] >= 6:
                        motion = dfm.to_numpy()
                ts = _clean(ts, motion)[:, motor]

                raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
                picks = mne.pick_types(raw.info, eeg=True, exclude=[])
                names = [raw.ch_names[i] for i in picks]
                idx = {n.lower(): j for j, n in enumerate(names)}
                if not all(k.lower() in idx for k in keep):
                    print(f"  {sub}/{task}: missing shared channel, skipped"); continue
                eeg = raw.get_data(picks=picks)[[idx[k.lower()] for k in keep]] * 1e6
                fs = float(raw.info["sfreq"])
                trig = np.array([o for o, dd in zip(raw.annotations.onset,
                                                    raw.annotations.description)
                                 if "R128" in str(dd)])
                trig = (trig * fs).astype(int)
                if trig.size < EPOCH_VOL:
                    continue
                e = pd.read_csv(ev, sep="\t")
                if "trial_type" in e.columns:
                    e = e[~e["trial_type"].astype(str).str.lower().str.contains("rest")]
                for o in pd.to_numeric(e["onset"], errors="coerce").dropna().to_numpy():
                    v = int(round(o / TR))
                    if v < 0 or v + EPOCH_VOL > ts.shape[0] or v >= trig.size:
                        continue
                    s0 = trig[v]
                    if s0 + EEG_SAMPLES > eeg.shape[1]:
                        continue
                    sh = ts[v:v + EPOCH_VOL].T; sh = sh - sh[:, :1]
                    seg = eeg[:, s0:s0 + EEG_SAMPLES]
                    seg = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
                    E_.append(seg.astype(np.float32)); H_.append(sh.astype(np.float32))
                    S_.append(si); cnt += 1
                del eeg, raw
            except Exception as exc:
                print(f"  {sub}/{task}: {type(exc).__name__}: {str(exc)[:60]}")
        print(f"  {sub}: {cnt} trials")
    np.savez_compressed(out, eeg=np.stack(E_), hemo=np.stack(H_),
                        subject_ids=np.asarray(S_),
                        ch_names=np.array(keep, dtype=object),
                        roi_labels=np.array([labels[i] for i in motor], dtype=object))
    print(f"{out.name}: {len(E_)} trials, {len(set(S_))} subj, {len(keep)} aligned channels")
    return out


def main():
    keep, a39, a36 = common_montage()
    print(f"ds002739 montage: {len(a39)} electrodes")
    print(f"ds002336 montage: {len(a36)} electrodes")
    print(f"SHARED: {len(keep)} channels\n  {keep}\n")
    build_36(keep)
    build_39(keep, a39)


if __name__ == "__main__":
    main()
