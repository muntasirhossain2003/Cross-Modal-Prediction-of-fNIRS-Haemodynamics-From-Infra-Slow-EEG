"""FINAL_PLAN P5.4 (Claim 5, person-level calibrated form): does a person's own calibration-run score predict how
well the k = 2 hybrid model will do on their remaining runs?"""
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
COHORTS, ALPHAS = ns["COHORTS"], ns["ALPHAS"]
SEED = int(os.environ.get("SEED", "1337"))
trial_corr, slow_features, zs, task_mean, _bins = (ns[k] for k in ("trial_corr", "slow_features", "zs", "task_mean", "_bins"))
TARGET = ns["TARGET"]
GAIN = bool(os.environ.get("GAIN"))                  # P5.4 EEG-specific check: gain over the personal mean
OUT = HERE / ("calib_reliability" + ("" if TARGET == "raw" else f"_{TARGET}") + ("_gain" if GAIN else "") + (f"_seed{SEED}" if SEED != 1337 else "") + "_results.json")


def hybrid(F, Y, tid, ca, te, pop):
    base = np.stack([task_mean(Y[ca], tid[ca], t) for t in tid[te]])
    Rca = Y[ca] - np.stack([task_mean(Y[ca], tid[ca], t) for t in tid[ca]])
    Fca, Fte = zs(F[ca], F[ca]), zs(F[te], F[ca])
    pers = RidgeCV(alphas=ALPHAS).fit(Fca, Rca).predict(Fte)
    h = trial_corr(base + 0.5 * (pers + pop.predict(Fte)), Y[te]).mean()
    if GAIN:
        return h - trial_corr(base, Y[te]).mean()
    return h


def run(cache):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    if TARGET != "raw":
        k_ = hemo.shape[1] // 2
        g = np.stack([hemo[:, :k_].mean(1), hemo[:, k_:].mean(1)], 1)
        hemo = g if TARGET == "global" else hemo - np.repeat(g, k_, axis=1)
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
        c0, c1 = (m[rec[m] == r] for r in cal)
        score = 0.5 * (hybrid(F, Y, tid, c0, c1, pop) + hybrid(F, Y, tid, c1, c0, pop))
        ca = np.concatenate([c0, c1])
        test = [hybrid(F, Y, tid, ca, m[rec[m] == r], pop) for r in runs if r not in cal]
        per[int(s)] = {"cal_score": float(score), "test_r": float(np.mean(test)), "error": float(1 - np.mean(test))}
        print(f"  s{int(s)}: calibration score {score:+.3f} test r {np.mean(test):+.3f}", flush=True)
    sc = np.array([v["cal_score"] for v in per.values()])
    te = np.array([v["test_r"] for v in per.values()])
    rho = float(spearmanr(sc, te)[0])
    boot = []
    for _ in range(5000):
        i = rng.integers(0, len(sc), len(sc))
        boot.append(spearmanr(sc[i], te[i])[0])
    ci = [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])]
    return {"people": per, "n": len(per), "spearman_score_vs_minus_error": rho, "ci95": ci,
            "passes": bool(rho > 0.2 and ci[0] > 0)}


res = json.loads(OUT.read_text()) if OUT.exists() else {}
for name, cache in COHORTS.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    print(name, flush=True)
    res[name] = run(cache)
    r = res[name]
    print(f"== {name}: n {r['n']}, Spearman {r['spearman_score_vs_minus_error']:+.3f} CI {r['ci95']} -> "
          f"{'PASS' if r['passes'] else 'FAIL'}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
ext = [res[k]["passes"] for k in ("mental_workload", "finemi") if k in res]
if "hbci" in res and ext:
    res["claim5_calibrated_form_holds"] = bool(res["hbci"]["passes"] and any(ext))
    print("Claim 5 (person-level calibrated form):", "HOLDS" if res["claim5_calibrated_form_holds"] else "DOES NOT HOLD")
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
