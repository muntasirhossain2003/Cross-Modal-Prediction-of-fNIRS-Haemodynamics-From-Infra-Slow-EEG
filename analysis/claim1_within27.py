"""FINAL_PLAN P3.1 (Claim 1): does fMRI supervision improve a shared EEG latent for fNIRS prediction? (linear)"""
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

NB = Path(str(REPO / "pipeline/CrossModal_InfraSlowEEG_fNIRS_fMRI.ipynb"))
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
FNIRS, FMRI = ROOT / "train_fnirs_a2133f5d6a71ec34", ROOT / "train_fmri_56be430c4098b1ee"
OUT = Path(__file__).with_name("claim1_within_27_" + os.environ.get("FMRI_SET", "pooled") + "_results.json")
FS, SEED, RANK, SCARCE = 200.0, 1337, 8, 5
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
            "g": sid.astype(np.int64) * 1000 + tid, "ch": meta["info"]["ch_names"]}


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


a = load(FNIRS)
WHICH = os.environ.get("FMRI_SET", "pooled")             # pooled | ds002338 | ds002336
parts = []
SETS = {"ds002336": [(0, FMRI)], "ds002338": [(100, ROOT / "replication_ds002338")]}
SETS["pooled"] = SETS["ds002336"] + SETS["ds002338"]
for off, cache in SETS[WHICH]:
    d_ = load(cache)
    W = interpolation_matrix(d_["ch"], a["ch"])
    assert W is not None, cache
    d_["F"] = np.einsum("ij,njt->nit", W, d_["F"])
    d_["sid"] = d_["sid"] + off
    d_["tid"] = d_["tid"] + off
    d_["g"] = d_["sid"].astype(np.int64) * 1000 + d_["tid"]
    parts.append(d_)
b = {k: np.concatenate([p[k] for p in parts]) for k in ("F", "Y", "sid", "tid", "rec", "g")}
print("fMRI arm:", WHICH, len(np.unique(b["sid"])), "people", len(b["sid"]), "trials", flush=True)
Fa, Fb = zs_person(a["F"], a["sid"]), zs_person(b["F"], b["sid"])
Ya, Yb = demean(a["Y"], a["g"]), demean(b["Y"], b["g"])
rng = np.random.default_rng(SEED)
Yb_shuf = Yb.copy()
for s in np.unique(b["sid"]):
    m = np.flatnonzero(b["sid"] == s)
    Yb_shuf[m] = Yb[rng.permutation(m)]
print(f"fNIRS {Fa.shape} -> {Ya.shape[1]} targets; fMRI {Fb.shape} -> {Yb.shape[1]} targets", flush=True)



RANK = 16


def topk(B, r=RANK):
    return np.linalg.svd(B, full_matrices=False)[0][:, :r]


U_fn = topk(RidgeCV(alphas=ALPHAS).fit(Fa, Ya).coef_.T)
U_rand = np.linalg.qr(np.random.default_rng(SEED).standard_normal((Fa.shape[1], RANK)))[0]
P = {k: np.zeros_like(Yb) for k in ("raw", "fnirs_latent", "fmri_pop_latent", "random")}
for s_ in np.unique(b["sid"]):
    me = np.flatnonzero(b["sid"] == s_)
    others = b["sid"] != s_
    U_pop = topk(RidgeCV(alphas=ALPHAS).fit(Fb[others], Yb[others]).coef_.T)
    runs = np.unique(b["rec"][me])
    for run in runs:
        te = me[b["rec"][me] == run]
        tr = me[b["rec"][me] != run]
        Ytr = demean(b["Y"][tr], b["g"][tr])
        for key, U in (("raw", None), ("fnirs_latent", U_fn), ("fmri_pop_latent", U_pop), ("random", U_rand)):
            Xtr, Xte = (Fb[tr], Fb[te]) if U is None else (Fb[tr] @ U, Fb[te] @ U)
            P[key][te] = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr).predict(Xte)
    print(f"sub {s_}: {len(runs)} runs, {len(me)} trials", flush=True)

out, per = {}, {}
for k, v in P.items():
    Pd = demean(v, b["g"])
    out[k] = subject_level_test(Pd, Yb, b["sid"])
    per[k] = per_subject_r(Pd, Yb, b["sid"])
people = sorted(per["raw"])
for x, y in (("fnirs_latent", "raw"), ("fnirs_latent", "random"), ("fnirs_latent", "fmri_pop_latent"), ("fmri_pop_latent", "raw")):
    v = [per[x][s] - per[y][s] for s in people]
    out[f"{x}_minus_{y}"] = {"mean": float(np.mean(v)), "ci": boot_ci(v, rng), "n_positive": int(np.sum(np.array(v) > 0)),
                             "per_person": v}
a_ok = out["raw"]["ci_low"] > 0 and out["raw"]["p_signflip"] < 0.05
b_ok = (out["fnirs_latent_minus_raw"]["mean"] > 0 and out["fnirs_latent_minus_raw"]["ci"][0] > 0
        and out["fnirs_latent_minus_random"]["ci"][0] > 0)
out["a_fmri_arm_within_person"], out["b_claim1_transfer"] = bool(a_ok), bool(b_ok)
f = lambda t: f"{t['r_subject']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] p_sf {t['p_signflip']:.4f} ({t['n_positive']}/{t['n_subjects']})"
for k in P:
    print(f"{k:16s} {f(out[k])}")
for k, v in out.items():
    if k.endswith(("_raw", "_random", "_latent")) and isinstance(v, dict) and "ci" in v:
        print(f"{k:36s} {v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] ({v['n_positive']}/10)")
print(f"(a) fMRI arm within person: {'WORKS' if a_ok else 'NOT SHOWN'} | (b) Claim 1 transfer: {'PASS' if b_ok else 'FAIL'}")
OUT.write_text(json.dumps(out, indent=1, default=float))
print("DONE")
