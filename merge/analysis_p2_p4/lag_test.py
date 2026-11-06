"""FINAL_PLAN T6: does slow EEG lead the fNIRS?  Positive L = EEG earlier than fNIRS."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import json
from pathlib import Path

import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

CACHE = Path(r"G:\research_bio_medical\merge\neurovascnet_merge_v1_cache\train_fnirs_a2133f5d6a71ec34")
OUT = Path(__file__).with_name("lag_results.json")
FS, SEED, T0, LEN = 200.0, 1337, 8, 22          # fNIRS segment = 1 s bins 8..29
LAGS = list(range(-8, 9))
ALPHAS = np.logspace(0, 6, 13)

eeg = np.load(CACHE / "eeg.npy", mmap_mode="r")
hemo = np.load(CACHE / "hemo.npy").astype(np.float64)
sid, tid, rec = (np.load(CACHE / f"{k}.npy") for k in ("subject_ids", "task_ids", "rec_ids"))
n = len(hemo)
strata = sid.astype(np.int64) * 1000 + tid
subs = np.unique(sid)

sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
slow = np.concatenate([sps.sosfiltfilt(sos, np.asarray(eeg[i:i + 120], np.float64), axis=-1)
                       .reshape(-1, eeg.shape[1], 30, 200).mean(-1) for i in range(0, n, 120)])
for s in subs:
    m = sid == s
    slow[m] = (slow[m] - slow[m].mean(0)) / (slow[m].std(0) + 1e-8)
Yfull = hemo.reshape(n, 72, 30, 10).mean(-1)
Y = Yfull[:, :, T0:T0 + LEN].reshape(n, -1)


def demean(A, g):
    A = A.copy()
    for k in np.unique(g):
        m = g == k
        A[m] -= A[m].mean(0)
    return A


def trial_corr(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    return ((P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)) *
            (T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12))).sum(1)


Yd = demean(Y, strata)
per_sub = {"within": {}, "across": {}}
for L in LAGS:
    F = slow[:, :, T0 - L:T0 - L + LEN].reshape(n, -1)   # EEG bins shifted L s earlier
    P = np.zeros_like(Y)
    for s in subs:
        own = np.flatnonzero(sid == s)
        for tr, te in LeaveOneGroupOut().split(own, groups=rec[own]):
            P[own[te]] = RidgeCV(alphas=ALPHAS).fit(F[own[tr]], demean(Y[own[tr]], tid[own[tr]])).predict(F[own[te]])
    Q = np.zeros_like(Y)
    for s in subs:
        te = sid == s
        Q[te] = RidgeCV(alphas=ALPHAS).fit(F[~te], Yd[~te]).predict(F[te])
    for key, pred in (("within", P), ("across", Q)):
        r = trial_corr(demean(pred, strata), Yd)
        per_sub[key][L] = np.array([r[sid == s].mean() for s in subs])
    print(f"L {L:+d} s  within {per_sub['within'][L].mean():+.4f}  across {per_sub['across'][L].mean():+.4f}", flush=True)

res = {}
rng = np.random.default_rng(SEED)
for key, d in per_sub.items():
    means = {L: float(v.mean()) for L, v in d.items()}
    peak = max(means, key=means.get)
    diff = d[peak] - d[0]
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
    res[key] = {"curve": means, "peak_lag_s": peak, "peak_minus_zero": float(diff.mean()),
                "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])]}
    print(f"{key}: peak at L = {peak:+d} s ({means[peak]:+.4f}); peak - zero lag = {diff.mean():+.4f} "
          f"CI [{res[key]['ci'][0]:+.4f}, {res[key]['ci'][1]:+.4f}]")
OUT.write_text(json.dumps(res, indent=1))
