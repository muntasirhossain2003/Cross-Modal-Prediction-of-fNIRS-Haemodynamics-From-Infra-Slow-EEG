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

NB = Path(str(REPO / "pipeline/Neurovascnet_marge_v1.ipynb"))
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
FNIRS, FMRI = ROOT / "train_fnirs_a2133f5d6a71ec34", ROOT / "train_fmri_56be430c4098b1ee"
OUT = Path(__file__).with_name("claim1_shared_results.json")
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
print(f"fNIRS {Fa.shape} -> {Ya.shape[1]} targets; fMRI {Fb.shape} -> {Yb.shape[1]} targets", flush=True)



def basis(rank, *Bs):
    M = np.hstack([B / np.linalg.norm(B) for B in Bs])
    return np.linalg.svd(M, full_matrices=False)[0][:, :rank]


def coef(F, Y):
    return RidgeCV(alphas=ALPHAS).fit(F, Y).coef_.T


def boot90(v, rng):
    v = np.asarray(v)
    return [float(x) for x in np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)], [5, 95])]


RANKS = (4, 8, 16, 32, 64)
subs = np.unique(a["sid"])
folds = np.array_split(np.random.default_rng(SEED).permutation(subs), 5)
Bm_all = coef(Fb, Yb)
# fNIRS side: every rank, single / joint / full
P = {("full", 0): np.zeros_like(Ya)}
for r in RANKS:
    P[("single", r)], P[("joint", r)] = np.zeros_like(Ya), np.zeros_like(Ya)
for hold in folds:
    te = np.isin(a["sid"], hold)
    full = RidgeCV(alphas=ALPHAS).fit(Fa[~te], Ya[~te])
    P[("full", 0)][te] = full.predict(Fa[te])
    Ba = full.coef_.T
    for r in RANKS:
        for key, U in (("single", basis(r, Ba)), ("joint", basis(r, Ba, Bm_all))):
            head = RidgeCV(alphas=ALPHAS).fit(Fa[~te] @ U, Ya[~te])
            P[(key, r)][te] = head.predict(Fa[te] @ U)
fn = {k: subject_level_test(demean(v, a["g"]), Ya, a["sid"]) for k, v in P.items()}
per_fn = {k: per_subject_r(demean(v, a["g"]), Ya, a["sid"]) for k, v in P.items()}
full_r = fn[("full", 0)]["r_subject"]
print("fNIRS full ridge", round(full_r, 4), "| single by rank", {r: round(fn[("single", r)]["r_subject"], 4) for r in RANKS},
      "| joint by rank", {r: round(fn[("joint", r)]["r_subject"], 4) for r in RANKS}, flush=True)
R = next((r for r in RANKS if fn[("single", r)]["r_subject"] >= 0.95 * full_r), RANKS[-1])
print("chosen rank", R, flush=True)

# fMRI side: leave one ds002336 person out
Ba_all = coef(Fa, Ya)
Q = {k: np.zeros_like(Yb) for k in ("full", "specific", "joint")}
for s_ in np.unique(b["sid"]):
    te = b["sid"] == s_
    full = RidgeCV(alphas=ALPHAS).fit(Fb[~te], Yb[~te])
    Q["full"][te] = full.predict(Fb[te])
    Bm = full.coef_.T
    for key, U in (("specific", basis(R, Bm)), ("joint", basis(R, Ba_all, Bm))):
        head = RidgeCV(alphas=ALPHAS).fit(Fb[~te] @ U, Yb[~te])
        Q[key][te] = head.predict(Fb[te] @ U)
fm = {k: subject_level_test(demean(v, b["g"]), Yb, b["sid"]) for k, v in Q.items()}
per_fm = {k: per_subject_r(demean(v, b["g"]), Yb, b["sid"]) for k, v in Q.items()}

rng = np.random.default_rng(SEED)
d_fn = [per_fn[("joint", R)][s] - per_fn[("single", R)][s] for s in sorted(per_fn[("single", R)])]
d_fm = [per_fm["joint"][s] - per_fm["specific"][s] for s in sorted(per_fm["specific"])]
ci_fn, ci_fm = boot90(d_fn, rng), boot90(d_fm, rng)
j_fn, j_fm = fn[("joint", R)], fm["joint"]
both_work = (j_fn["ci_low"] > 0 and j_fn["p_signflip"] < 0.05 and j_fn["r_subject"] >= 0.05
             and j_fm["ci_low"] > 0 and j_fm["p_signflip"] < 0.05)
no_cost = ci_fn[0] > -0.02 and ci_fm[0] > -0.02
res = {"rank": R, "fnirs": {f"{k[0]}_{k[1]}": v for k, v in fn.items()}, "fmri": fm,
       "fnirs_joint_minus_specific": {"mean": float(np.mean(d_fn)), "ci90": ci_fn},
       "fmri_joint_minus_specific": {"mean": float(np.mean(d_fm)), "ci90": ci_fm},
       "both_outputs_work": bool(both_work), "no_cost_from_sharing": bool(no_cost),
       "claim1_shared_space_holds": bool(both_work and no_cost)}
f = lambda t: f"{t['r_subject']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] p_sf {t['p_signflip']:.4f}"
print(f"fNIRS (rank {R}): full {f(fn[('full', 0)])} | specific {f(fn[('single', R)])} | joint {f(j_fn)}")
print(f"fMRI  (rank {R}): full {f(fm['full'])} | specific {f(fm['specific'])} | joint {f(j_fm)}")
print(f"joint - specific: fNIRS {np.mean(d_fn):+.4f} 90% CI {ci_fn}, fMRI {np.mean(d_fm):+.4f} 90% CI {ci_fm}")
print(f"both outputs work: {both_work} | no cost from sharing: {no_cost} -> Claim 1 (shared space) "
      f"{'HOLDS' if both_work and no_cost else 'DOES NOT HOLD'}")
OUT.write_text(json.dumps(res, indent=1, default=float))
print("DONE")
