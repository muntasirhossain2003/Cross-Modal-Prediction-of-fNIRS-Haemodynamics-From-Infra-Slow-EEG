"""FINAL_PLAN P5.1 (Claim 1): confirmatory shared-space test at fixed rank 32 with a fresh fNIRS cohort."""
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
LOCAL = os.environ.get("TARGET", "") == "local"           # P6.3: fNIRS channels minus channel mean
UNCUED = bool(os.environ.get("UNCUED"))                   # P14.5: fNIRS side from 30 s windows not aligned to cues
OUT = Path(__file__).with_name(f"claim1_fresh_{COH}" + (f"_scarce{SCARCE}" if SCARCE else "") + (f"_seed{SEED}" if SEED != 1337 else "") + ("_local" if LOCAL else "") + ("_uncued" if UNCUED else "") + "_results.json")
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


def load(cache, local=False):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    F = np.concatenate([_bins(sps.sosfiltfilt(sos, np.asarray(eeg[i:i + 120], np.float64), axis=-1), 30)
                        for i in range(0, len(eeg), 120)])
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    if hemo.shape[-1] == 300:
        hemo = _bins(hemo, 30)
    if local:
        k_ = hemo.shape[1] // 2
        hemo = np.concatenate([hemo[:, :k_] - hemo[:, :k_].mean(1, keepdims=True),
                               hemo[:, k_:] - hemo[:, k_:].mean(1, keepdims=True)], 1)
    sid, tid = np.load(cache / "subject_ids.npy"), np.load(cache / "task_ids.npy")
    meta = json.loads((cache / "meta.json").read_text())
    return {"F": F, "Y": hemo.reshape(len(hemo), -1), "sid": sid, "tid": tid, "rec": np.load(cache / "rec_ids.npy"),
            "g": sid.astype(np.int64) * 1000 + tid, "ch": meta["info"]["ch_names"] if "info" in meta else meta["ch_names"]}


def load_windows(name):
    z = np.load(ROOT / "p14_windows" / f"{name}.npz")
    sid = z["sid"].astype(np.int64)
    return {"F": z["F"].astype(np.float64), "Y": z["Y"].astype(np.float64), "sid": sid, "tid": np.zeros(len(sid), int), "rec": z["rec"],
            "g": sid * 1000, "ch": [str(c) for c in z["ch"]]}


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


a = load_windows(COH) if UNCUED else load(FNIRS, local=LOCAL)
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



def basis(rank, *Bs):
    M = np.hstack([B / np.linalg.norm(B) for B in Bs])
    return np.linalg.svd(M, full_matrices=False)[0][:, :rank]


def coef(F, Y):
    return RidgeCV(alphas=ALPHAS).fit(F, Y).coef_.T


def boot90(v, rng):
    v = np.asarray(v)
    return [float(x) for x in np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)], [5, 95])]


subs = np.unique(a["sid"])
folds = np.array_split(np.random.default_rng(SEED).permutation(subs), 5)
Bm_all, Bm_shuf = coef(Fb, Yb), coef(Fb, Yb_shuf)
R = RANK
P = {k: np.zeros_like(Ya) for k in ("full", "single", "joint", "joint_shuffled")}
for i, hold in enumerate(folds):
    te = np.isin(a["sid"], hold)
    trn = ~te
    if SCARCE:
        keep = np.random.default_rng(SEED).choice(np.setdiff1d(subs, hold), SCARCE, replace=False)
        trn = np.isin(a["sid"], keep)
    full = RidgeCV(alphas=ALPHAS).fit(Fa[trn], Ya[trn])
    P["full"][te] = full.predict(Fa[te])
    Ba = full.coef_.T
    for key, U in (("single", basis(R, Ba)), ("joint", basis(R, Ba, Bm_all)), ("joint_shuffled", basis(R, Ba, Bm_shuf))):
        head = RidgeCV(alphas=ALPHAS).fit(Fa[trn] @ U, Ya[trn])
        P[key][te] = head.predict(Fa[te] @ U)
    print("fold", i, "done", flush=True)
fn = {k: subject_level_test(demean(v, a["g"]), Ya, a["sid"]) for k, v in P.items()}
per_fn = {k: per_subject_r(demean(v, a["g"]), Ya, a["sid"]) for k, v in P.items()}
print("fNIRS", {k: round(v["r_subject"], 4) for k, v in fn.items()}, flush=True)

if SCARCE:
    import sys as _s
    rng = np.random.default_rng(SEED)
    d_fn = [per_fn["joint"][q] - per_fn["single"][q] for q in sorted(per_fn["single"])]
    d_js = [per_fn["joint"][q] - per_fn["joint_shuffled"][q] for q in sorted(per_fn["single"])]
    c1, c2 = boot_ci(d_fn, rng), boot_ci(d_js, rng)
    ok = bool(np.mean(d_fn) > 0 and c1[0] > 0 and np.mean(d_js) > 0 and c2[0] > 0)
    print(f"SCARCE {SCARCE} {COH}: joint - single {np.mean(d_fn):+.4f} {c1}; joint - shuffled {np.mean(d_js):+.4f} {c2} -> {'PASS' if ok else 'FAIL'}")
    OUT.write_text(json.dumps({"fnirs": fn, "joint_minus_single": [float(np.mean(d_fn)), c1], "joint_minus_shuffled": [float(np.mean(d_js)), c2], "pass": ok}, indent=1, default=float))
    print("DONE"); _s.exit(0)
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
d_fn = [per_fn["joint"][s] - per_fn["single"][s] for s in sorted(per_fn["single"])]
d_js = [per_fn["joint"][s] - per_fn["joint_shuffled"][s] for s in sorted(per_fn["single"])]
ci95_fn, ci95_js = boot_ci(d_fn, rng), boot_ci(d_js, rng)
p31 = bool(np.mean(d_fn) > 0 and ci95_fn[0] > 0 and np.mean(d_js) > 0 and ci95_js[0] > 0)
d_fm = [per_fm["joint"][s] - per_fm["specific"][s] for s in sorted(per_fm["specific"])]
ci_fn, ci_fm = boot90(d_fn, rng), boot90(d_fm, rng)
j_fn, j_fm = fn["joint"], fm["joint"]
both_work = (j_fn["ci_low"] > 0 and j_fn["p_signflip"] < 0.05 and j_fn["r_subject"] >= 0.05
             and j_fm["ci_low"] > 0 and j_fm["p_signflip"] < 0.05)
no_cost = ci_fn[0] > -0.02 and ci_fm[0] > -0.02
res = {"fnirs_cohort": COH, "rank": R, "fnirs": fn, "fmri": fm,
       "p31_joint_minus_single_ci95": {"mean": float(np.mean(d_fn)), "ci": ci95_fn},
       "p31_joint_minus_shuffled_ci95": {"mean": float(np.mean(d_js)), "ci": ci95_js}, "p31_form_pass": p31,
       "fnirs_joint_minus_specific": {"mean": float(np.mean(d_fn)), "ci90": ci_fn},
       "fmri_joint_minus_specific": {"mean": float(np.mean(d_fm)), "ci90": ci_fm},
       "both_outputs_work": bool(both_work), "no_cost_from_sharing": bool(no_cost),
       "claim1_shared_space_holds": bool(both_work and no_cost)}
f = lambda t: f"{t['r_subject']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] p_sf {t['p_signflip']:.4f}"
print(f"fNIRS {COH} (rank {R}): full {f(fn['full'])} | specific {f(fn['single'])} | joint {f(j_fn)} | joint-shuffled {f(fn['joint_shuffled'])}")
print(f"P3.1 form: joint - single {np.mean(d_fn):+.4f} 95% CI {ci95_fn}; joint - shuffled {np.mean(d_js):+.4f} {ci95_js} -> {'PASS' if p31 else 'FAIL'}")
print(f"fMRI  (rank {R}): full {f(fm['full'])} | specific {f(fm['specific'])} | joint {f(j_fm)}")
print(f"joint - specific: fNIRS {np.mean(d_fn):+.4f} 90% CI {ci_fn}, fMRI {np.mean(d_fm):+.4f} 90% CI {ci_fm}")
print(f"both outputs work: {both_work} | no cost from sharing: {no_cost} -> Claim 1 (shared space) "
      f"{'HOLDS' if both_work and no_cost else 'DOES NOT HOLD'}")
OUT.write_text(json.dumps(res, indent=1, default=float))
print("DONE")
