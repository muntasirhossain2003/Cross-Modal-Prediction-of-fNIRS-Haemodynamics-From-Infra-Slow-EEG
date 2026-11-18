"""FINAL_PLAN P6.2 (Claim 2, calibrated-transfer form): does a model frozen in another lab add to a person's own
two-run calibration model? One gain per model per person, fitted on the calibration runs (personal part cross-fitted)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
from pathlib import Path

import nbformat
import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]   # repository root

NB = Path(str(REPO / "pipeline/CrossModal_InfraSlowEEG_fNIRS_fMRI.ipynb"))
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
TAG = os.environ.get("FINEMI_TAG", "")
CACHES = {"hbci": ROOT / "train_fnirs_a2133f5d6a71ec34", "mental_workload": ROOT / "replication_tuberlin_mw",
          "finemi": ROOT / ("replication_finemi" + TAG)}
OUT = Path(__file__).with_name(f"calibrated_transfer{TAG}_results.json")
FS, SEED = 200.0, 1337
ALPHAS = np.logspace(0, 6, 13)

nb = nbformat.read(NB, as_version=4)
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12] + [nb.cells[19]]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)
interpolation_matrix = ns["interpolation_matrix"]


def _bins(x, nb_):
    return x.reshape(*x.shape[:-1], nb_, -1).mean(-1)


def load(name):
    cache = CACHES[name]
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    F = np.concatenate([_bins(sps.sosfiltfilt(sos, np.asarray(eeg[i:i + 120], np.float64), axis=-1), 30)
                        for i in range(0, len(eeg), 120)])
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    k_ = hemo.shape[1] // 2
    Y = _bins(np.stack([hemo[:, :k_].mean(1), hemo[:, k_:].mean(1)], 1), 30).reshape(len(hemo), -1)
    meta = json.loads((cache / "meta.json").read_text())
    return {"F": F, "Y": Y, "ch": meta["info"]["ch_names"] if "info" in meta else meta["ch_names"],
            "sid": np.load(cache / "subject_ids.npy"), "tid": np.load(cache / "task_ids.npy"),
            "rec": np.load(cache / "rec_ids.npy")}


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


def trial_corr(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    return (P * T).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-12)


def task_means(Y, tid, ref_tid):
    return np.stack([Y[tid == t].mean(0) if np.any(tid == t) else Y.mean(0) for t in ref_tid])


def run(src, dst):
    a, b = D[src], D[dst]
    W = interpolation_matrix(b["ch"], a["ch"])
    Fb = np.einsum("ij,njt->nit", W, b["F"]).reshape(len(b["F"]), -1)
    mdl = RidgeCV(alphas=ALPHAS).fit(zs_person(a["F"], a["sid"]), demean(a["Y"], a["sid"] * 1000 + a["tid"]))
    rng = np.random.default_rng(SEED)
    per = {}
    for s in np.unique(b["sid"]):
        m = np.flatnonzero(b["sid"] == s)
        runs = np.unique(b["rec"][m])
        if len(runs) < 3:
            continue
        cal = list(rng.choice(runs, 2, replace=False))
        ca, te = m[np.isin(b["rec"][m], cal)], m[~np.isin(b["rec"][m], cal)]
        mu, sd = Fb[ca].mean(0), Fb[ca].std(0) + 1e-8
        Z = (Fb - mu) / sd
        base_ca, base_te = task_means(b["Y"][ca], b["tid"][ca], b["tid"][ca]), task_means(b["Y"][ca], b["tid"][ca], b["tid"][te])
        R = b["Y"][ca] - base_ca
        # personal model, cross-fitted between the two calibration runs
        pers_cf = np.zeros_like(R)
        for r_ in cal:
            i_fit, i_pred = b["rec"][ca] != r_, b["rec"][ca] == r_
            pers_cf[i_pred] = RidgeCV(alphas=ALPHAS).fit(Z[ca][i_fit], R[i_fit]).predict(Z[ca][i_pred])
        pers_te = RidgeCV(alphas=ALPHAS).fit(Z[ca], R).predict(Z[te])
        ext_ca, ext_te = mdl.predict(Z[ca]), mdl.predict(Z[te])
        e_mean_ca = task_means(ext_ca, b["tid"][ca], b["tid"][ca])
        ext_ca_c, ext_te_c = ext_ca - e_mean_ca, ext_te - task_means(ext_ca, b["tid"][ca], b["tid"][te])
        g_p = np.linalg.lstsq(pers_cf.reshape(-1, 1), R.ravel(), rcond=None)[0]
        g_b = np.linalg.lstsq(np.stack([pers_cf.ravel(), ext_ca_c.ravel()], 1), R.ravel(), rcond=None)[0]
        P_pers = base_te + g_p[0] * pers_te
        P_both = base_te + g_b[0] * pers_te + g_b[1] * ext_te_c
        Yt = b["Y"][te]
        per[int(s)] = {"baseline": float(trial_corr(np.broadcast_to(base_te, Yt.shape), Yt).mean()),
                       "personal": float(trial_corr(P_pers, Yt).mean()),
                       "personal_plus_external": float(trial_corr(P_both, Yt).mean()),
                       "ext_gain_sign": float(np.sign(g_b[1]))}
    d = np.array([v["personal_plus_external"] - v["personal"] for v in per.values()])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    out = {k: float(np.mean([v[k] for v in per.values()])) for k in ("baseline", "personal", "personal_plus_external")}
    out.update({"diff": float(d.mean()), "ci": ci, "n_better": int(np.sum(d > 0)), "n": len(d),
                "frac_negative_external_gain": float(np.mean([v["ext_gain_sign"] < 0 for v in per.values()])),
                "pass": bool(d.mean() > 0 and ci[0] > 0), "people": per})
    print(f"{src} -> {dst}: baseline {out['baseline']:.3f} personal {out['personal']:.3f} personal+external "
          f"{out['personal_plus_external']:.3f} | diff {out['diff']:+.4f} {ci} ({out['n_better']}/{out['n']}), "
          f"negative external gain in {out['frac_negative_external_gain']:.2f} -> {'PASS' if out['pass'] else 'FAIL'}", flush=True)
    return out


D = {k: load(k) for k in CACHES}
res = {}
for src, dst in (("hbci", "finemi"), ("mental_workload", "finemi"), ("hbci", "mental_workload")):
    res[f"{src}_to_{dst}"] = run(src, dst)
    OUT.write_text(json.dumps(res, indent=1))
print("DONE")
