"""FINAL_PLAN T3/T4 (ridge form): which channels carry the slow (< 0.5 Hz) EEG->fNIRS coupling?

hBCI is re-loaded through the notebook's own loader with VEOG/HEOG kept, and checked against the cache.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
import time
from pathlib import Path

import nbformat
import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

NB = Path(r"G:\research_bio_medical\merge\Neurovascnet_marge_v1.ipynb")
CACHE = Path(r"G:\research_bio_medical\merge\neurovascnet_merge_v1_cache\train_fnirs_a2133f5d6a71ec34")
HERE = Path(__file__).parent
OUT = HERE / "topo_eog_results.json"
FS, SEED = 200.0, 1337

# ---- the notebook's loader, EOG kept ----
nb = nbformat.read(NB, as_version=4)
nb.cells[4].source = nb.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")
ns = {"__name__": "__main__", "display": print}
os.chdir(NB.parent)
for c in nb.cells[:12]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
ns["EOG"] = ()
os.chdir(HERE)
t0 = time.time()
parts = [ns["load_hbci"](ns["PATHS"]["train_fnirs"], subjects=range(1, 30), task=t) for t in ("MI", "MA")]
eeg = np.concatenate([p["eeg"] for p in parts])
hemo_new = np.concatenate([p["hemo"] for p in parts])
names = parts[0]["ch_names"]
print(f"loaded with EOG in {(time.time() - t0) / 60:.1f} min: {eeg.shape}, channels {names}")

hemo = np.load(CACHE / "hemo.npy")
sid, tid, rec = (np.load(CACHE / f"{k}.npy") for k in ("subject_ids", "task_ids", "rec_ids"))
assert hemo_new.shape == hemo.shape and np.allclose(hemo_new, hemo, atol=1e-5), "trial order differs from cache"
cache_names = json.load(open(CACHE / "meta.json"))["info"]["ch_names"]
ce = np.load(CACHE / "eeg.npy", mmap_mode="r")
idx30 = [names.index(c) for c in cache_names]
assert np.allclose(eeg[:20][:, idx30], ce[:20], atol=1e-4), "EEG channels differ from cache"
print("trials and EEG channels match the cache")

GROUPS = {
    "all_eeg": cache_names,
    "frontal": ["AFp1", "AFp2", "AFF1h", "AFF2h", "AFF5h", "AFF6h", "F3", "F4", "F7", "F8"],
    "central_motor": ["Cz", "FCC3h", "FCC4h", "FCC5h", "FCC6h", "CCP3h", "CCP4h", "CCP5h", "CCP6h"],
    "parietal_occipital": ["Pz", "P3", "P4", "P7", "P8", "PPO1h", "PPO2h", "POO1", "POO2"],
    "temporal": ["T7", "T8"],
    "eog": ["VEOG", "HEOG"],
}
for g, ch in GROUPS.items():
    missing = [c for c in ch if c not in names]
    assert not missing, (g, missing)

# ---- slow features for every channel (< 0.5 Hz, 1 s bins), z-scored per person ----
sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
slow = np.concatenate([sps.sosfiltfilt(sos, eeg[i:i + 120].astype(np.float64), axis=-1)
                       .reshape(-1, eeg.shape[1], 30, 200).mean(-1) for i in range(0, len(eeg), 120)])
del eeg
for s in np.unique(sid):
    m = sid == s
    slow[m] = (slow[m] - slow[m].mean(0)) / (slow[m].std(0) + 1e-8)

n = len(hemo)
strata = sid.astype(np.int64) * 1000 + tid
Y = hemo.astype(np.float64).reshape(n, 72, 30, 10).mean(-1).reshape(n, -1)


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
subs = np.unique(sid)
ALPHAS = np.logspace(0, 6, 13)


def per_subject(r):
    return np.array([r[sid == s].mean() for s in subs])


def summary(v):
    rng = np.random.default_rng(SEED)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
    return {"mean": float(v.mean()), "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])]}


results = {}
for g, ch in GROUPS.items():
    F = slow[:, [names.index(c) for c in ch]].reshape(n, -1)
    # within person, leave-one-run-out
    P = np.zeros_like(Y)
    for s in subs:
        own = np.flatnonzero(sid == s)
        for tr, te in LeaveOneGroupOut().split(own, groups=rec[own]):
            tr_i, te_i = own[tr], own[te]
            P[te_i] = RidgeCV(alphas=ALPHAS).fit(F[tr_i], demean(Y[tr_i], tid[tr_i])).predict(F[te_i])
    within = per_subject(trial_corr(demean(P, strata), Yd))
    # across people, leave-one-person-out, one pooled model
    Q = np.zeros_like(Y)
    for s in subs:
        te = sid == s
        Q[te] = RidgeCV(alphas=ALPHAS).fit(F[~te], Yd[~te]).predict(F[te])
    across = per_subject(trial_corr(demean(Q, strata), Yd))
    results[g] = {"n_channels": len(ch), "within": summary(within), "across": summary(across)}
    print(f"{g:19s} ({len(ch):2d} ch)  within-person {within.mean():+.4f}  across-person {across.mean():+.4f}", flush=True)
    OUT.write_text(json.dumps(results, indent=1))
print("DONE")
