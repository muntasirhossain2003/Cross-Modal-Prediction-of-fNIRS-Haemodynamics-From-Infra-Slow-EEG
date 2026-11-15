"""FINAL_PLAN P5.6 (Claim 2, fMRI output): frozen slow-EEG model, ds002336 <-> ds002338."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import json
from pathlib import Path

import nbformat
import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]   # repository root

NB = Path(str(REPO / "pipeline/Neurovascnet_marge_v1.ipynb"))
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
FNIRS, FMRI = ROOT / "train_fnirs_a2133f5d6a71ec34", ROOT / "train_fmri_56be430c4098b1ee"
COH = os.environ.get("FNIRS_COHORT", "mental_workload")   # mental_workload | finemi | hbci
FNIRS = {"hbci": FNIRS, "mental_workload": ROOT / "replication_tuberlin_mw", "finemi": ROOT / "replication_finemi"}[COH]
SEED = int(os.environ.get("SEED", "1337"))
SCARCE = int(os.environ.get("SCARCE", "0"))               # P5.1b: fNIRS training people per fold
OUT = Path(__file__).with_name(f"claim1_fresh_{COH}" + (f"_scarce{SCARCE}" if SCARCE else "") + (f"_seed{SEED}" if SEED != 1337 else "") + "_results.json")
FS, RANK = 200.0, 32
ALPHAS = np.logspace(0, 6, 13)

nb = nbformat.read(NB, as_version=4)
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12] + [nb.cells[19]]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)
subject_level_test, interpolation_matrix, _center = ns["subject_level_test"], ns["interpolation_matrix"], ns["_center"]


def _bins(x, nb_):
    return x.reshape(*x.shape[:-1], nb_, -1).mean(-1)


def load(cache):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    F = np.concatenate([_bins(sps.sosfiltfilt(sos, np.asarray(eeg[i:i + 120], np.float64), axis=-1), 30)
                        for i in range(0, len(eeg), 120)])
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    if hemo.shape[-1] == 300:
        hemo = _bins(hemo, 30)
    sid, tid = np.load(cache / "subject_ids.npy"), np.load(cache / "task_ids.npy")
    meta = json.loads((cache / "meta.json").read_text())
    return {"F": F, "Y": hemo.reshape(len(hemo), -1), "sid": sid, "tid": tid, "rec": np.load(cache / "rec_ids.npy"),
            "g": sid.astype(np.int64) * 1000 + tid, "ch": meta["info"]["ch_names"] if "info" in meta else meta["ch_names"]}


def zs_person(F, sid):
    F = F.reshape(len(F), -1).copy()
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


def per_subject_r(P, T, sid):
    return {int(s): float(np.diag(_center(P[sid == s]) @ _center(T[sid == s]).T).mean()) for s in np.unique(sid)}


def boot_ci(v, rng):
    v = np.asarray(v)
    return [float(x) for x in np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)], [2.5, 97.5])]


verdict_text = ns["verdict_text"]
OUT = Path(__file__).with_name("external_fmri_results.json")
a = load(ROOT / "train_fnirs_a2133f5d6a71ec34")
D = {}
for name, cache in (("ds002336", FMRI), ("ds002338", ROOT / "replication_ds002338")):
    d_ = load(cache)
    W = interpolation_matrix(d_["ch"], a["ch"])
    d_["F"] = np.einsum("ij,njt->nit", W, d_["F"])
    D[name] = {"F": zs_person(d_["F"], d_["sid"]), "Y": demean(d_["Y"], d_["g"]), "sid": d_["sid"], "g": d_["g"]}
Fa, Ya = zs_person(a["F"], a["sid"]), demean(a["Y"], a["g"])


def basis(rank, *Bs):
    M = np.hstack([B / np.linalg.norm(B) for B in Bs])
    return np.linalg.svd(M, full_matrices=False)[0][:, :rank]


Ba = RidgeCV(alphas=ALPHAS).fit(Fa, Ya).coef_.T
res = {}
for src, dst in (("ds002336", "ds002338"), ("ds002338", "ds002336")):
    A, B = D[src], D[dst]
    m = RidgeCV(alphas=ALPHAS).fit(A["F"], A["Y"])
    U = basis(RANK, Ba, m.coef_.T)
    h = RidgeCV(alphas=ALPHAS).fit(A["F"] @ U, A["Y"])
    for key, P in (("ridge", m.predict(B["F"])), ("shared_space", h.predict(B["F"] @ U))):
        st = subject_level_test(demean(P, B["g"]), B["Y"], B["sid"])
        st["verdict"] = verdict_text(st)
        res[f"{src}_to_{dst}_{key}"] = st
        print(f"{src} -> {dst} [{key}]: r_subject {st['r_subject']:+.4f} [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] "
              f"p_sf {st['p_signflip']:.4f} ({st['n_positive']}/{st['n_subjects']}) -> {st['verdict']}", flush=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print("DONE")
