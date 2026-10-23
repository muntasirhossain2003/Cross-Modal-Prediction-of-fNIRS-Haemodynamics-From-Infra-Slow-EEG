"""Rebuilt ds002336 EEG-fMRI loader — the fixed fMRI supervision target.

The v3 cache produced a target with split-half reliability of **-0.793**: it does not
reproduce against itself, so no fMRI number computed against it means anything. Three
things were wrong, and `qc/fix_fmri_target.py` measured the effect of each:

    variant                                  reliability
    v3-style sliding, no clean, 48 ROI          (leaked)   overlapping windows
    event-locked, no clean, 48 ROI               0.468     inflated by scanner drift
    event-locked, cleaned, 48 ROI                0.268
    event-locked, cleaned, MOTOR ROI             0.357     <- this build

So this loader:
  1. EPOCHS ON STIMULUS ONSETS instead of sliding windows (the decisive fix)
  2. REGRESSES OUT motion + polynomial drift before epoching
  3. RESTRICTS to sensorimotor ROIs instead of all 48 Harvard-Oxford regions

EEG/BOLD alignment uses the **R128 scanner triggers** in the BrainVision marker files,
which mark every volume acquisition (400 samples apart at 200 Hz = TR 2 s). Volume k of
the BOLD series is therefore EEG sample R128[k] exactly -- no assumed offset.

Output schema matches the notebook contract:
    eeg   [N, C, 6000]   30 s @ 200 Hz, z-scored per trial
    hemo  [N, n_motor, 15]   30 s @ TR=2 s, baseline-corrected at onset
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DS_ROOT = Path("dataset/OpenNeuro ds002336")
ROI_DIR = Path("qc_outputs/arrays/fmri_roi")
CACHE_DIR = Path("neurovascnet_outputs/cache_v4")

TR = 2.0
EPOCH_VOL = 15                      # 30 s
EEG_FS = 200.0
EEG_SAMPLES = 6000                  # 30 s
MOTOR_KEYS = ("Precentral", "Postcentral", "Supplementary", "Juxtapositional")

# motorloc / MIpre / MIpost cover only 7-9 subjects and yield ~83 training trials.
# The three neurofeedback runs use the SAME 20 s Rest/Task block design, Task-NF is
# motor-imagery neurofeedback (so task-consistent), and all 10 subjects have all three.
TASKS = ("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF")


@dataclass(frozen=True)
class FmriConfig:
    clean_nuisance: bool = True     # regress motion + drift (audit: required)
    motor_only: bool = True         # sensorimotor ROI subset (audit: 0.268 -> 0.357)
    n_val_subjects: int = 3
    tasks: tuple = TASKS


# ------------------------------------------------------------------ helpers
def _clean(ts: np.ndarray, motion: np.ndarray | None) -> np.ndarray:
    n = ts.shape[0]
    t = np.linspace(-1, 1, n)
    cols = [np.ones(n), t, t ** 2, t ** 3]
    if motion is not None and motion.shape[0] >= 2:
        m = motion[:n] if motion.shape[0] >= n else np.vstack(
            [motion, np.repeat(motion[-1:], n - motion.shape[0], axis=0)])
        for k in range(min(6, m.shape[1])):
            v = m[:, k]
            s = v.std()
            cols.append((v - v.mean()) / (s if s > 0 else 1.0))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
    return ts - X @ beta


def _motion(subject: str, task: str) -> np.ndarray | None:
    p = DS_ROOT / "derivatives" / subject / "head_motion" / \
        f"d_{subject}_task-{task}_headmotion.tsv"
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t").select_dtypes("number").dropna(axis=1, how="all")
    return df.to_numpy() if df.shape[1] >= 6 else None


def _onsets(task: str) -> np.ndarray:
    p = DS_ROOT / f"task-{task}_events.tsv"
    if not p.exists():
        return np.array([])
    df = pd.read_csv(p, sep="\t")
    if "onset" not in df.columns:
        return np.array([])
    if "trial_type" in df.columns:
        df = df[~df["trial_type"].astype(str).str.lower().str.contains("rest")]
    on = pd.to_numeric(df["onset"], errors="coerce").to_numpy(dtype=float)
    return on[np.isfinite(on)]


def _volume_triggers(vhdr: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read the pp EEG and return (data uV, R128 onsets in samples, fs)."""
    import mne
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks) * 1e6
    fs = float(raw.info["sfreq"])
    ann = raw.annotations
    trig = np.array([o for o, d in zip(ann.onset, ann.description) if "R128" in str(d)])
    return data, (trig * fs).astype(int), fs


# ------------------------------------------------------------------ loader
def load_ds002336_fixed(split: Literal["train", "val", "all"] = "train",
                        cfg: FmriConfig = FmriConfig()) -> dict:
    subjects = sorted([d.name for d in DS_ROOT.glob("sub-*") if d.is_dir()])
    val = set(subjects[-cfg.n_val_subjects:])
    keep = (subjects if split == "all" else
            [s for s in subjects if (s in val) == (split == "val")])

    eeg_out, hemo_out, sid_out, task_out, audit = [], [], [], [], []

    for si, subject in enumerate(subjects):
        if subject not in keep:
            continue
        for task in cfg.tasks:
            npz = ROI_DIR / f"ds002336_{subject}_ses-01_{task}_fmri.npz"
            vhdr = (DS_ROOT / "derivatives" / subject / "eeg_pp" /
                    f"{subject}_task-{task}_eeg_pp.vhdr")
            if not npz.exists() or not vhdr.exists():
                audit.append({"subject": subject, "task": task,
                              "status": "missing roi or eeg_pp", "trials": 0})
                continue
            try:
                d = np.load(npz, allow_pickle=True)
                ts = d["roi_timeseries"].astype(float)
                labels = [str(x) for x in d["roi_labels"]]
                if ts.size == 0 or not labels:
                    raise ValueError("empty ROI array")

                if cfg.clean_nuisance:
                    ts = _clean(ts, _motion(subject, task))
                if cfg.motor_only:
                    idx = [i for i, nm in enumerate(labels)
                           if any(k in nm for k in MOTOR_KEYS)]
                    if not idx:
                        raise ValueError("no motor ROIs matched")
                    ts = ts[:, idx]

                eeg, trig, fs = _volume_triggers(vhdr)
                if trig.size < EPOCH_VOL:
                    raise ValueError(f"only {trig.size} R128 triggers")

                n_trials = 0
                for o in _onsets(task):
                    v = int(round(o / TR))                    # BOLD volume index
                    if v < 0 or v + EPOCH_VOL > ts.shape[0] or v >= trig.size:
                        continue
                    seg_h = ts[v:v + EPOCH_VOL].T
                    seg_h = seg_h - seg_h[:, :1]              # baseline at onset

                    s0 = trig[v]                              # EEG sample of that volume
                    if s0 + EEG_SAMPLES > eeg.shape[1]:
                        continue
                    seg_e = eeg[:, s0:s0 + EEG_SAMPLES]
                    mu = seg_e.mean(-1, keepdims=True)
                    sd = seg_e.std(-1, keepdims=True) + 1e-6
                    seg_e = (seg_e - mu) / sd

                    eeg_out.append(seg_e.astype(np.float32))
                    hemo_out.append(seg_h.astype(np.float32))
                    sid_out.append(si)
                    task_out.append(task)
                    n_trials += 1

                audit.append({"subject": subject, "task": task,
                              "status": "ok" if n_trials else "no usable trials",
                              "trials": n_trials, "n_rois": ts.shape[1],
                              "n_triggers": int(trig.size)})
            except Exception as exc:
                audit.append({"subject": subject, "task": task,
                              "status": f"skip: {exc}", "trials": 0})

    if not eeg_out:
        raise RuntimeError(f"fixed fMRI loader produced no samples for split={split}")

    # channel counts can differ across subjects -> trim to the common minimum
    n_ch = min(e.shape[0] for e in eeg_out)
    n_roi = min(h.shape[0] for h in hemo_out)
    return {
        "eeg": np.stack([e[:n_ch] for e in eeg_out]),
        "hemo": np.stack([h[:n_roi] for h in hemo_out]),
        "labels": np.zeros(len(eeg_out), dtype=np.int64),
        "subject_ids": np.asarray(sid_out, dtype=np.int64),
        "tasks": np.asarray(task_out),
        "true_delays": np.zeros(len(eeg_out), dtype=np.float32),
        "audit": np.asarray(audit, dtype=object),
    }


def build_cache(cfg: FmriConfig = FmriConfig(), force: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val"):
        p = CACHE_DIR / f"ds002336_fixed_{split}.npz"
        if p.exists() and not force:
            print(f"{p.name}: cached")
            out[split] = p
            continue
        d = load_ds002336_fixed(split, cfg)
        np.savez_compressed(p, eeg=d["eeg"], hemo=d["hemo"], labels=d["labels"],
                            subject_ids=d["subject_ids"], true_delays=d["true_delays"])
        print(f"{p.name}: {d['eeg'].shape[0]} trials, "
              f"{np.unique(d['subject_ids']).size} subjects, "
              f"eeg {d['eeg'].shape}, hemo {d['hemo'].shape}")
        for row in d["audit"]:
            if row["status"] != "ok":
                print(f"    {row['subject']}/{row['task']}: {row['status']}")
        out[split] = p
    return out


if __name__ == "__main__":
    build_cache()
