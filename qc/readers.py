"""Unified readers for every raw format in the audit.

Each reader returns a plain dict so the metric code never has to know which
dataset it is looking at. Continuous data is always returned as
(n_channels, n_samples) in microvolts for EEG and micromolar for fNIRS
concentrations; epoched data adds a trailing trial axis.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

from qc.config import clean_path

warnings.filterwarnings("ignore")


# =========================================================================
# HEFMI (epoched MATLAB 'epo' structs)
# =========================================================================
def read_hefmi_eeg(path: Path) -> dict:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    epo = m["epo"]
    X = np.asarray(epo.X, dtype=np.float64)          # (time, ch, trials)
    y = np.asarray(epo.y)
    fs = float(np.atleast_1d(epo.fs).ravel()[0])
    # class labels are one-hot (n_classes, n_trials) or a vector
    if y.ndim == 2:
        labels = np.argmax(y, axis=0)
    else:
        labels = y.astype(int)
    return {
        "data": np.transpose(X, (1, 0, 2)),          # (ch, time, trials)
        "fs": fs,
        "n_channels": X.shape[1],
        "n_trials": X.shape[2],
        "labels": labels,
        "units": "uV",
        "epoched": True,
        # resting-state eyes-open/closed blocks shipped alongside the epochs
        "EO": np.asarray(getattr(epo, "EO", np.empty((0, 0))), dtype=np.float64).T,
        "EC": np.asarray(getattr(epo, "EC", np.empty((0, 0))), dtype=np.float64).T,
    }


def read_hefmi_fnirs(path: Path) -> dict:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    epo = m["epo"]
    out = {}
    for chrom in ("oxy", "doxy", "toxy"):
        if not hasattr(epo, chrom):
            continue
        s = getattr(epo, chrom)
        X = np.asarray(s.X, dtype=np.float64)        # (time, ch, trials)
        out[chrom] = np.transpose(X, (1, 0, 2))
    y = np.asarray(epo.oxy.y)
    labels = np.argmax(y, axis=0) if y.ndim == 2 else y.astype(int)
    fs_raw = np.atleast_1d(epo.oxy.fs).ravel()
    fs = float(fs_raw[0]) if fs_raw.size and np.isfinite(fs_raw[0]) else np.nan
    ref = out["oxy"]
    return {
        "chromophores": out,
        "data": ref,
        "fs": fs,
        "n_channels": ref.shape[0],
        "n_trials": ref.shape[2],
        "labels": labels,
        "units": "umol/L",
        "epoched": True,
        "has_raw_wavelengths": False,   # already Beer-Lambert converted
    }


# =========================================================================
# BrainVision (ds002336 raw + provider-corrected derivatives)
# =========================================================================
def read_brainvision(path: Path, preload: bool = True) -> dict:
    import mne
    raw = mne.io.read_raw_brainvision(str(path), preload=preload, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks) * 1e6            # MNE returns volts -> uV
    ch_names = [raw.ch_names[i] for i in picks]
    try:
        ann = raw.annotations
        events = [(float(o), str(d)) for o, d in zip(ann.onset, ann.description)]
    except Exception:
        events = []
    return {
        "data": data,
        "fs": float(raw.info["sfreq"]),
        "n_channels": data.shape[0],
        "ch_names": ch_names,
        "duration_s": data.shape[1] / float(raw.info["sfreq"]),
        "events": events,
        "units": "uV",
        "epoched": False,
    }


# =========================================================================
# TU Berlin (BBCI cnt/mrk structs)
# =========================================================================
def _bbci_struct(path: Path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    return m[keys[0]]


def read_tuberlin_eeg(path: Path) -> dict:
    s = _bbci_struct(path)
    x = np.asarray(s.x, dtype=np.float64)             # (time, ch)
    clab = [str(c) for c in np.atleast_1d(s.clab)]
    return {
        "data": x.T,
        "fs": float(np.atleast_1d(s.fs).ravel()[0]),
        "n_channels": x.shape[1],
        "ch_names": clab,
        "duration_s": x.shape[0] / float(np.atleast_1d(s.fs).ravel()[0]),
        "units": str(getattr(s, "yUnit", "uV")),
        "epoched": False,
    }


def read_tuberlin_fnirs(path: Path) -> dict:
    s = _bbci_struct(path)
    out, fs, clab = {}, np.nan, []
    for chrom, key in (("oxy", "oxy"), ("doxy", "deoxy")):
        if not hasattr(s, key):
            continue
        sub = getattr(s, key)
        x = np.asarray(sub.x, dtype=np.float64)       # (time, ch)
        out[chrom] = x.T
        fs = float(np.atleast_1d(sub.fs).ravel()[0])
        clab = [str(c) for c in np.atleast_1d(getattr(sub, "clab", []))]
    ref = out.get("oxy")
    return {
        "chromophores": out,
        "data": ref,
        "fs": fs,
        "n_channels": ref.shape[0] if ref is not None else 0,
        "ch_names": clab,
        "duration_s": (ref.shape[1] / fs) if ref is not None and fs == fs else np.nan,
        "units": "umol/L",
        "epoched": False,
        "has_raw_wavelengths": False,
    }


def read_tuberlin_markers(path: Path) -> dict:
    """mrk struct -> event onsets (seconds) and class labels."""
    p = clean_path(path)
    if p is None or not p.exists():
        return {"onsets_s": np.array([]), "labels": np.array([]), "class_names": []}
    s = _bbci_struct(p)
    time_ms = np.atleast_1d(np.asarray(s.time, dtype=np.float64))
    y = np.asarray(s.y)
    labels = np.argmax(y, axis=0) if y.ndim == 2 else y.astype(int)
    names = [str(c) for c in np.atleast_1d(getattr(s, "className", []))]
    return {"onsets_s": time_ms / 1000.0, "labels": labels, "class_names": names}


# =========================================================================
# ds002739 (large MATLAB v7 structs, already gradient/BCG corrected)
# =========================================================================
def read_ds002739_eeg(path: Path) -> dict:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    d = m["EEGdata"]
    Y = np.asarray(d.Y, dtype=np.float64)             # (ch, time)
    # Top-level `fs` is the rate of Y; d.fs is the original acquisition rate.
    fs = float(np.atleast_1d(m.get("fs", d.fd)).ravel()[0])
    excluded = np.atleast_1d(np.asarray(m.get("excludedchannels", []))).ravel()
    return {
        "data": Y,
        "fs": fs,
        "acquisition_fs": float(np.atleast_1d(d.fs).ravel()[0]),
        "n_channels": Y.shape[0],
        "duration_s": Y.shape[1] / fs,
        "excluded_channels": excluded.tolist(),
        "chnan": np.atleast_1d(np.asarray(getattr(d, "chnan", []))).ravel(),
        "nan_index": np.atleast_1d(np.asarray(getattr(d, "nanidx", []))).ravel(),
        "units": "uV",
        "epoched": False,
    }


def read_ds002739_events(path: Path) -> dict:
    p = clean_path(path)
    if p is None or not p.exists():
        return {"onsets_s": np.array([]), "labels": np.array([])}
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    out = {"raw_keys": keys}
    for k in keys:
        v = m[k]
        if hasattr(v, "_fieldnames"):
            out[k] = {f: np.asarray(getattr(v, f)) for f in v._fieldnames}
        else:
            out[k] = np.asarray(v)
    return out


# =========================================================================
# Events (BIDS tsv)
# =========================================================================
def read_bids_events(path: Path) -> "object":
    import pandas as pd
    p = clean_path(path)
    if p is None or not p.exists():
        return None
    return pd.read_csv(p, sep="\t")


# =========================================================================
# Dispatcher
# =========================================================================
def read_recording(row: dict):
    """Read one manifest row. Returns None for fMRI (handled by the fMRI module)."""
    ds, mod, role = row["dataset"], row["modality"], row["data_role"]
    p = Path(row["primary_path"])
    if ds == "HEFMI":
        return read_hefmi_eeg(p) if mod == "eeg" else read_hefmi_fnirs(p)
    if ds == "ds002336" and mod == "eeg":
        return read_brainvision(p)
    if ds == "TUBerlin":
        return read_tuberlin_eeg(p) if mod == "eeg" else read_tuberlin_fnirs(p)
    if ds == "ds002739" and mod == "eeg":
        return read_ds002739_eeg(p)
    raise ValueError(f"no reader for {ds}/{mod}/{role}")
