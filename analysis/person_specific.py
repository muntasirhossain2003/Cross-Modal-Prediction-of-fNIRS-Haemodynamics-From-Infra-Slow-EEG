"""FINAL_PLAN T5: is EEG->fNIRS coupling person-specific? (hBCI, ridge, CPU)

M[s, t] = score of person s's own model on person t's trials (diagonal = s's held-out trials).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")          # leave CPU for the GPU run's filtering
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
import time
from pathlib import Path

import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut

CACHE = Path(r"G:\research_bio_medical\merge\neurovascnet_merge_v1_cache\train_fnirs_a2133f5d6a71ec34")
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "trials"     # "trials" (shuffled KFold) | "runs" (leave-one-run-out)
OUT = Path(__file__).with_name(f"person_specific_results_{MODE}.json")
FS, SEED = 200.0, 1337
BP_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

eeg = np.load(CACHE / "eeg.npy", mmap_mode="r")          # (n, 30, 6000), unfiltered, z-scored per trial
hemo = np.load(CACHE / "hemo.npy").astype(np.float64)     # (n, 72, 300) = 36 HbO + 36 HbR
sid, tid = np.load(CACHE / "subject_ids.npy"), np.load(CACHE / "task_ids.npy")
rec = np.load(CACHE / "rec_ids.npy")
strata = sid.astype(np.int64) * 1000 + tid
n = len(eeg)

# the unfiltered cache must still hold sub-0.5 Hz power (the filtered one has none)
f, p = sps.welch(np.asarray(eeg[:40], np.float64), fs=FS, nperseg=4000, axis=-1)
ratio = p[..., (f > 0) & (f < 0.5)].mean() / p[..., (f > 8) & (f < 13)].mean()
print(f"sub-0.5 Hz / alpha power ratio = {ratio:.2f}")
assert ratio > 1, "this looks like the band-passed cache"


def _sos(lo, hi):
    if lo is None:
        return sps.butter(4, hi, btype="low", fs=FS, output="sos")
    if hi is None:
        return sps.butter(4, lo, btype="high", fs=FS, output="sos")
    return sps.butter(4, [lo, hi], btype="band", fs=FS, output="sos")


DETREND = MODE.endswith("detrend")


def _bins(x, nb):
    return x.reshape(*x.shape[:-1], nb, -1).mean(-1)


def _dt(x):
    return sps.detrend(x, axis=-1, type="linear") if DETREND else x


SOS_SLOW, SOS_HI = _sos(None, 0.5), _sos(45.0, None)
SOS_BP = [_sos(lo, hi) for lo, hi in BP_BANDS]
NB, NA = sps.iirnotch(50.0, 30.0, fs=FS)


def features(chunk=120):
    out = {"slow": [], "bandpower": [], "high": []}
    for i in range(0, n, chunk):
        x = np.asarray(eeg[i:i + chunk], np.float64)
        b = len(x)
        out["slow"].append(_bins(_dt(sps.sosfiltfilt(SOS_SLOW, x, axis=-1)), 30).reshape(b, -1))
        out["bandpower"].append(np.concatenate(
            [np.log(_bins(sps.sosfiltfilt(s, x, axis=-1) ** 2, 10) + 1e-8).reshape(b, -1) for s in SOS_BP], 1))
        h = sps.filtfilt(NB, NA, sps.sosfiltfilt(SOS_HI, x, axis=-1), axis=-1)
        out["high"].append(np.log(_bins(h ** 2, 10) + 1e-8).reshape(b, -1))
    return {k: np.concatenate(v) for k, v in out.items()}


def zscore_per_person(F):
    F = F.copy()
    for s in np.unique(sid):
        m = sid == s
        F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
    return F


def demean(Y, g):
    Y = Y.copy()
    for k in np.unique(g):
        m = g == k
        Y[m] -= Y[m].mean(0)
    return Y


def trial_corr(P, T):
    """notebook's mean_corr per trial: centre and normalise each trial, then dot."""
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    return (P * T).sum(1)


Y = _bins(_dt(hemo), 30).reshape(n, -1)                     # 1 s bins -> (n, 2160)
Yd = demean(Y, strata)
subs = np.unique(sid)
t0 = time.time()
FEATS = features()
print(f"features in {(time.time() - t0) / 60:.1f} min: " + ", ".join(f"{k} {v.shape[1]}" for k, v in FEATS.items()))

results = {}
for fname, F in FEATS.items():
    F = zscore_per_person(F)
    M = np.zeros((len(subs), len(subs)))
    t1 = time.time()
    for si, s in enumerate(subs):
        own = np.flatnonzero(sid == s)
        other = np.flatnonzero(sid != s)
        P_own = np.zeros((len(own), Y.shape[1]))
        P_other = np.zeros((len(other), Y.shape[1]))
        splits = (KFold(5, shuffle=True, random_state=SEED).split(own) if MODE == "trials"
                  else LeaveOneGroupOut().split(own, groups=rec[own]))
        n_split = 0
        for tr, te in splits:
            n_split += 1
            tr_i, te_i = own[tr], own[te]
            Ytr = demean(Y[tr_i], tid[tr_i])               # task means from the training trials only
            mdl = RidgeCV(alphas=np.logspace(0, 6, 13)).fit(F[tr_i], Ytr)
            P_own[te] = mdl.predict(F[te_i])
            P_other += mdl.predict(F[other])
        P_other /= n_split
        r_own = trial_corr(demean(P_own, strata[own]), Yd[own])
        M[si, si] = r_own.mean()
        r_oth = trial_corr(demean(P_other, strata[other]), Yd[other])
        for ti, t in enumerate(subs):
            if t != s:
                M[si, ti] = r_oth[sid[other] == t].mean()
    off = (M.sum(1) - np.diag(M)) / (len(subs) - 1)
    d = np.diag(M) - off
    rng = np.random.default_rng(SEED)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    flips = rng.choice([-1.0, 1.0], size=(10000, len(d)))
    p_sf = float((np.sum(np.abs((flips * d).mean(1)) >= abs(d.mean())) + 1) / 10001)
    ident = float(np.mean(M.argmax(0) == np.arange(len(subs))))
    results[fname] = {"diag_mean": float(np.diag(M).mean()), "offdiag_mean": float(off.mean()),
                      "diff_mean": float(d.mean()), "diff_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                      "p_signflip": p_sf, "n_diff_positive": int((d > 0).sum()), "n_subjects": len(subs),
                      "identification": ident, "identification_chance": 1 / len(subs),
                      "matrix": M.round(4).tolist(), "minutes": (time.time() - t1) / 60}
    r = results[fname]
    print(f"{fname:9s} diag {r['diag_mean']:+.4f} off {r['offdiag_mean']:+.4f} diff {r['diff_mean']:+.4f} "
          f"CI [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}] p {p_sf:.4f} ({r['n_diff_positive']}/{len(subs)}) "
          f"| ident {ident:.3f} (chance {1 / len(subs):.3f}) | {r['minutes']:.1f} min", flush=True)
    OUT.write_text(json.dumps(results, indent=1))
print("DONE")
