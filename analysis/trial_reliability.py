"""FINAL_PLAN P5.7 (Claim 5, trial-level calibrated form): do per-trial scores rank the hybrid model's trial errors?"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).parent
src = (HERE / "calibration.py").read_text()
ns = {"__name__": "calib", "__file__": str(HERE / "calibration.py")}
exec(src[:src.index("def run_cohort")], ns)
COHORTS, ALPHAS, SEED = ns["COHORTS"], ns["ALPHAS"], 1337
trial_corr, slow_features, zs, task_mean, _bins = (ns[k] for k in ("trial_corr", "slow_features", "zs", "task_mean", "_bins"))
OUT = HERE / "trial_reliability_results.json"


def run(cache):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    sid, tid, rec = np.load(cache / "subject_ids.npy"), np.load(cache / "task_ids.npy"), np.load(cache / "rec_ids.npy")
    F = slow_features(eeg)
    Y = _bins(hemo, 30).reshape(len(hemo), -1)
    Fz, R = np.zeros_like(F), np.zeros_like(Y)
    for s in np.unique(sid):
        m = sid == s
        Fz[m] = zs(F[m], F[m])
        for t in np.unique(tid[m]):
            mt = m & (tid == t)
            R[mt] = Y[mt] - Y[mt].mean(0)
    rng = np.random.default_rng(SEED)
    per = {}
    for s in np.unique(sid):
        m = np.flatnonzero(sid == s)
        runs = np.unique(rec[m])
        if len(runs) < 3:
            continue
        others = np.flatnonzero(sid != s)
        pop = RidgeCV(alphas=ALPHAS).fit(Fz[others], R[others])
        cal = list(rng.choice(runs, 2, replace=False))
        ca = m[np.isin(rec[m], cal)]
        te = m[~np.isin(rec[m], cal)]
        base = np.stack([task_mean(Y[ca], tid[ca], t) for t in tid[te]])
        Rca = Y[ca] - np.stack([task_mean(Y[ca], tid[ca], t) for t in tid[ca]])
        Fca, Fte = zs(F[ca], F[ca]), zs(F[te], F[ca])
        dev = 0.5 * (RidgeCV(alphas=ALPHAS).fit(Fca, Rca).predict(Fte) + pop.predict(Fte))
        err = 1 - trial_corr(base + dev, Y[te])
        ood = -(Fte ** 2).mean(1)
        energy = np.linalg.norm(dev, axis=1)
        per[int(s)] = {"ood": float(spearmanr(ood, -err)[0]), "energy": float(spearmanr(energy, -err)[0]),
                       "n_trials": int(len(te))}
        print(f"  s{int(s)}: -OOD {per[int(s)]['ood']:+.3f} energy {per[int(s)]['energy']:+.3f}", flush=True)
    out = {"people": per}
    for k in ("ood", "energy"):
        v = np.array([p[k] for p in per.values()])
        b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
        out[k] = {"mean": float(v.mean()), "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])]}
    out["passes"] = bool(out["ood"]["mean"] > 0.2 and out["ood"]["ci"][0] > 0)
    return out


res = {}
for name, cache in COHORTS.items():
    print(name, flush=True)
    res[name] = run(cache)
    r = res[name]
    print(f"== {name}: -OOD {r['ood']['mean']:+.3f} {r['ood']['ci']} | energy {r['energy']['mean']:+.3f} {r['energy']['ci']} "
          f"-> {'PASS' if r['passes'] else 'FAIL'}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
res["claim5_trial_form_holds"] = bool(res["hbci"]["passes"] and (res["mental_workload"]["passes"] or res["finemi"]["passes"]))
print("Claim 5 trial-level calibrated form:", "HOLDS" if res["claim5_trial_form_holds"] else "DOES NOT HOLD")
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
