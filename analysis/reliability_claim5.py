"""FINAL_PLAN P3.5 (Claim 5): the notebook's reliability machinery on the slow-EEG ridge pipeline."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import ast
import json
from pathlib import Path

import nbformat
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

HERE = Path(__file__).parent
src = (HERE / "external_claim2.py").read_text(encoding="utf-8")
src = src[:src.index("res = {}")]                       # notebook helpers, loaders, D = both cohorts
ns = {"__name__": "rel", "__file__": str(HERE / "external_claim2.py")}
exec(src, ns)
D, zs_person, interpolation_matrix = ns["D"], ns["zs_person"], ns["interpolation_matrix"]
nbns = ns["ns"]
nb = nbformat.read(ns["NB"], as_version=4)
WANT = {"expected_calibration_error", "trial_error", "_logit", "fit_calibrator", "apply_calibrator",
        "reliability_report"}
for c in nb.cells[20:40]:
    if c.cell_type != "code":
        continue
    s = "\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!")))
    try:
        tree = ast.parse(s)
    except SyntaxError:
        continue
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in WANT:
            exec(ast.get_source_segment(s, n), nbns)
exec("from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; "
     "from sklearn.isotonic import IsotonicRegression; from scipy.optimize import minimize_scalar; "
     "from scipy import stats; import pandas as pd", nbns)
fit_calibrator, apply_calibrator, reliability_report, trial_error = (
    nbns["fit_calibrator"], nbns["apply_calibrator"], nbns["reliability_report"], nbns["trial_error"])
CFG = nbns["CFG"]
OUT = HERE / "reliability_claim5_results.json"
SEED, NENS = 1337, 10
ALPHAS = np.logspace(0, 6, 13)
print("abstain_coverage", CFG.abstain_coverage, "n_folds", CFG.n_folds)


def prep(d, F):
    Fz = zs_person(F, d["sid"])
    g = d["sid"].astype(np.int64) * 1000 + d["tid"]
    Y = d["Y"] / d["Y"].std()                          # cohort-level unit harmonisation
    return Fz, Y, g


def demean(Y, g):
    Y = Y.copy()
    for k in np.unique(g):
        m = g == k
        Y[m] -= Y[m].mean(0)
    return Y


def fit_models(Fz, Y, g, sid, rng):
    Yd = demean(Y, g)
    main = RidgeCV(alphas=ALPHAS).fit(Fz, Yd)
    people = np.unique(sid)
    ens = []
    for _ in range(NENS):
        pick = rng.choice(people, len(people), replace=True)
        idx = np.concatenate([np.flatnonzero(sid == p) for p in pick])
        ens.append(Ridge(alpha=main.alpha_).fit(Fz[idx], Yd[idx]))
    mu, sd = Fz.mean(0), Fz.std(0) + 1e-8
    return {"main": main, "ens": ens, "mu": mu, "sd": sd}


def predict_with_features(M, Fz, F_raw):
    pred = M["main"].predict(Fz)
    E = np.stack([m.predict(Fz) for m in M["ens"]])
    spread = E.std(0).mean(1) / (np.abs(E.mean(0)).mean(1) + 1e-8)
    ood = (((Fz - M["mu"]) / M["sd"]) ** 2).mean(1)
    energy = np.log((F_raw.reshape(len(F_raw), -1) ** 2).mean(1) + 1e-12)
    return pred, np.column_stack([spread, ood, energy])


def as3(Y):
    return Y.reshape(len(Y), 2, 30)


rng = np.random.default_rng(SEED)
a = D["hbci"]
Fz, Y, g = prep(a, a["F"])
people = np.unique(a["sid"])
folds = np.array_split(np.random.default_rng(SEED).permutation(people), CFG.n_folds)
X_all, err_all, fold_of = np.zeros((len(Y), 3)), np.zeros(len(Y)), np.zeros(len(Y), int)
for fi, hold in enumerate(folds):
    te = np.isin(a["sid"], hold)
    M = fit_models(Fz[~te], Y[~te], g[~te], a["sid"][~te], rng)
    pred, X = predict_with_features(M, Fz[te], a["F"][te])
    X_all[te], fold_of[te] = X, fi
    err_all[te] = trial_error(as3(pred), as3(Y[te]), g[te])
    print(f"fold {fi}: median err {np.median(err_all[te]):.3f}", flush=True)

# internal: calibrator cross-fitted over folds
scores = {k: np.zeros(len(Y)) for k in ("raw", "temp", "iso")}
thr = np.zeros(len(Y))
cal_tols = []
for fi in range(len(folds)):
    te = fold_of == fi
    cal = fit_calibrator(X_all[~te], err_all[~te])
    sc = apply_calibrator(cal, X_all[te])
    for k in scores:
        scores[k][te] = sc[k]
    thr[te] = cal["threshold"]
    cal_tols.append(cal["tol"])
cal_int = {"tol": float(np.median(err_all)), "threshold": float(np.median(thr))}
internal = reliability_report(scores, err_all, cal_int, a["sid"])
# keep the per-fold thresholds for the coverage/risk numbers
keep = scores["temp"] >= thr
internal["coverage"] = float(keep.mean())
internal["risk_answered"] = float(err_all[keep].mean())
print("internal:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in internal.items()}, flush=True)

# external: everything fitted on all hBCI, applied to mental workload
b = D["mental_workload"]
W = interpolation_matrix(b["ch"], a["ch"])
Fb = np.einsum("ij,njt->nit", W, b["F"])
Fbz, Yb, gb = prep(b, Fb)
M = fit_models(Fz, Y, g, a["sid"], rng)
CAL = fit_calibrator(X_all, err_all)
pred_b, Xb = predict_with_features(M, Fbz, Fb)
err_b = trial_error(as3(pred_b), as3(Yb), gb)
external = reliability_report(apply_calibrator(CAL, Xb), err_b, CAL, b["sid"])
print("external:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in external.items()}, flush=True)

ranks_int = internal["rank_trial"] > 0.2 or internal["rank_subject"] > 0.2
cal_ok = internal["ece_iso"] < internal["ece_raw"]
ranks_ext = external["rank_trial"] > 0.2
verdict = "HOLDS" if (ranks_int and cal_ok and ranks_ext) else ("PARTIAL (internal only)" if ranks_int else "FAILS")
print(f"CLAIM 5 (ridge pipeline): {verdict}  (rank internal trial {internal['rank_trial']:+.3f} / subject "
      f"{internal['rank_subject']:+.3f}, external trial {external['rank_trial']:+.3f}; ECE raw {internal['ece_raw']:.3f} "
      f"-> iso {internal['ece_iso']:.3f})")
OUT.write_text(json.dumps({"internal": internal, "external": external, "verdict": verdict}, indent=1, default=float))
print("DONE")
