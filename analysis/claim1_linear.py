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

NB = Path(r"G:\research_bio_medical\merge\Neurovascnet_marge_v1.ipynb")
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
FNIRS, FMRI = ROOT / "train_fnirs_a2133f5d6a71ec34", ROOT / "train_fmri_56be430c4098b1ee"
OUT = Path(__file__).with_name("claim1_linear_results.json")
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


a, b = load(FNIRS), load(FMRI)
W = interpolation_matrix(b["ch"], a["ch"])
assert W is not None
b["F"] = np.einsum("ij,njt->nit", W, b["F"])
Fa, Fb = zs_person(a["F"], a["sid"]), zs_person(b["F"], b["sid"])
Ya, Yb = demean(a["Y"], a["g"]), demean(b["Y"], b["g"])
rng = np.random.default_rng(SEED)
Yb_shuf = Yb.copy()
for s in np.unique(b["sid"]):
    m = np.flatnonzero(b["sid"] == s)
    Yb_shuf[m] = Yb[rng.permutation(m)]
Bm = RidgeCV(alphas=ALPHAS).fit(Fb, Yb).coef_.T
Bm_s = RidgeCV(alphas=ALPHAS).fit(Fb, Yb_shuf).coef_.T
print(f"fNIRS {Fa.shape} -> {Ya.shape[1]} targets; fMRI {Fb.shape} -> {Yb.shape[1]} targets", flush=True)


def basis(*Bs):
    M = np.hstack([B / np.linalg.norm(B) for B in Bs])
    return np.linalg.svd(M, full_matrices=False)[0][:, :RANK]


res = {}
subs = np.unique(a["sid"])
folds = np.array_split(np.random.default_rng(SEED).permutation(subs), 5)
for regime in ("all_training_people", f"scarce_{SCARCE}_people"):
    P = {k: np.zeros_like(Ya) for k in ("single", "joint", "joint_shuffled", "full_ridge")}
    pick_rng = np.random.default_rng(SEED)
    for hold in folds:
        te = np.isin(a["sid"], hold)
        pool = np.setdiff1d(subs, hold)
        if regime != "all_training_people":
            pool = pick_rng.choice(pool, SCARCE, replace=False)
        tr = np.isin(a["sid"], pool)
        full = RidgeCV(alphas=ALPHAS).fit(Fa[tr], Ya[tr])
        P["full_ridge"][te] = full.predict(Fa[te])
        Ba = full.coef_.T
        for key, U in (("single", basis(Ba)), ("joint", basis(Ba, Bm)), ("joint_shuffled", basis(Ba, Bm_s))):
            head = RidgeCV(alphas=ALPHAS).fit(Fa[tr] @ U, Ya[tr])
            P[key][te] = head.predict(Fa[te] @ U)
    out = {}
    per = {}
    for k, Pk in P.items():
        Pd = demean(Pk, a["g"])
        out[k] = subject_level_test(Pd, Ya, a["sid"])
        per[k] = per_subject_r(Pd, Ya, a["sid"])
    people = sorted(per["single"])
    for x, y in (("joint", "single"), ("joint", "joint_shuffled"), ("joint_shuffled", "single"), ("single", "full_ridge")):
        v = [per[x][s] - per[y][s] for s in people]
        out[f"{x}_minus_{y}"] = {"mean": float(np.mean(v)), "ci": boot_ci(v, rng), "n_positive": int(np.sum(np.array(v) > 0))}
    ok = (out["joint_minus_single"]["ci"][0] > 0 and out["joint_minus_single"]["mean"] > 0
          and out["joint_minus_joint_shuffled"]["ci"][0] > 0)
    out["claim1_pass"] = bool(ok)
    res[regime] = out
    print(f"== {regime}: " + ", ".join(f"{k} {out[k]['r_subject']:+.4f}" for k in P) + " | " +
          ", ".join(f"{k} {v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" for k, v in out.items()
                    if k.endswith(("single", "shuffled", "ridge")) and isinstance(v, dict) and "ci" in v) +
          f" -> Claim 1 {'CONFIRMED' if ok else 'NOT CONFIRMED'}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
print("DONE")
