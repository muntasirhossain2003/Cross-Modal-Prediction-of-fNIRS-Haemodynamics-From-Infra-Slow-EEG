"""FINAL_PLAN P2.5: personalised calibration vs the no-EEG personal mean (hBCI + mental workload, ridge, CPU)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]   # repository root

ROOT = Path(str(REPO / "pipeline/neurovascnet_merge_v1_cache"))
COHORTS = {"hbci": ROOT / "train_fnirs_a2133f5d6a71ec34", "mental_workload": ROOT / "replication_tuberlin_mw",
           "finemi": ROOT / "replication_finemi"}
TARGET = os.environ.get("TARGET", "raw")                      # raw | local | global (T2 split, per chromophore)
OUT = Path(__file__).with_name("calibration_results.json" if TARGET == "raw" else f"calibration_results_{TARGET}.json")
FS, SEED, MAXSUB, PRIMARY_K = 200.0, 1337, 10, 2
ALPHAS = np.logspace(0, 6, 13)


def _bins(x, nb):
    return x.reshape(*x.shape[:-1], nb, -1).mean(-1)


def trial_corr(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    return (P * T).sum(1)


def slow_features(eeg, chunk=120):
    sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    out = []
    for i in range(0, len(eeg), chunk):
        x = np.asarray(eeg[i:i + chunk], np.float64)
        out.append(_bins(sps.sosfiltfilt(sos, x, axis=-1), 30).reshape(len(x), -1))
    return np.concatenate(out)


def zs(F, ref):
    return (F - ref.mean(0)) / (ref.std(0) + 1e-8)


def task_mean(Y, tid, task):
    m = tid == task
    return Y[m].mean(0) if m.any() else Y.mean(0)


def subject_stats(v, rng):
    v = np.asarray(v)
    boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
    flips = rng.choice([-1.0, 1.0], size=(10000, len(v)))
    p = float((np.sum(np.abs((flips * v).mean(1)) >= abs(v.mean())) + 1) / 10001)
    return {"mean": float(v.mean()), "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "p_signflip": p,
            "n_positive": int((v > 0).sum()), "n": len(v)}


def run_cohort(cache):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    if TARGET != "raw":
        k_ = hemo.shape[1] // 2
        g = np.stack([hemo[:, :k_].mean(1), hemo[:, k_:].mean(1)], 1)      # (n, 2, T)
        hemo = g if TARGET == "global" else hemo - np.repeat(g, k_, axis=1)
    sid, tid, rec = np.load(cache / "subject_ids.npy"), np.load(cache / "task_ids.npy"), np.load(cache / "rec_ids.npy")
    t0 = time.time()
    F = slow_features(eeg)
    Y = _bins(hemo, 30).reshape(len(hemo), -1)
    print(f"  features {F.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)
    subs = np.unique(sid)
    # population residuals: every person's features z-scored with their own stats, targets minus own task mean
    Fz_own, R_own = np.zeros_like(F), np.zeros_like(Y)
    for s in subs:
        m = sid == s
        Fz_own[m] = zs(F[m], F[m])
        for t in np.unique(tid[m]):
            mt = m & (tid == t)
            R_own[mt] = Y[mt] - Y[mt].mean(0)
    rng_sub = np.random.default_rng(SEED)
    per = {}                                   # person -> {key: mean score}
    for s in subs:
        m = np.flatnonzero(sid == s)
        others = np.flatnonzero(sid != s)
        pop = RidgeCV(alphas=ALPHAS).fit(Fz_own[others], R_own[others])
        runs = np.unique(rec[m])
        acc = {}

        def add(key, val):
            acc.setdefault(key, []).append(val)

        for test_run in runs:
            te = m[rec[m] == test_run]
            task = tid[te][0]
            pool = [r for r in runs if r != test_run]
            Yte = Y[te]
            Yte_c = Yte - Yte.mean(0)
            # k = 0: population task mean + population ridge; EEG z-scored with the person's other runs (EEG only)
            ref0 = F[m[rec[m] != test_run]]
            pt = task_mean(Y[others], tid[others], task)
            P0 = pt + pop.predict(zs(F[te], ref0))
            add("k0_population", trial_corr(P0, Yte).mean())
            add("k0_population_trial", trial_corr(P0 - P0.mean(0), Yte_c).mean())
            for k in range(1, len(pool) + 1):
                subsets = list(itertools.combinations(pool, k))
                if len(subsets) > MAXSUB:
                    subsets = [subsets[i] for i in rng_sub.choice(len(subsets), MAXSUB, replace=False)]
                for cal_runs in subsets:
                    ca = m[np.isin(rec[m], cal_runs)]
                    base = task_mean(Y[ca], tid[ca], task)
                    Rca = Y[ca] - np.stack([task_mean(Y[ca], tid[ca], t) for t in tid[ca]])
                    Fca, Fte = zs(F[ca], F[ca]), zs(F[te], F[ca])
                    pers = RidgeCV(alphas=ALPHAS).fit(Fca, Rca).predict(Fte)
                    Pb = np.broadcast_to(base, Yte.shape)
                    Pp = base + pers
                    Ph = base + 0.5 * (pers + pop.predict(Fte))
                    for name, P in (("baseline", Pb), ("personal", Pp), ("hybrid", Ph)):
                        add(f"k{k}_{name}", trial_corr(P, Yte).mean())
                        add(f"k{k}_{name}_trial", trial_corr(P - P.mean(0), Yte_c).mean())
        per[int(s)] = {key: float(np.mean(v)) for key, v in acc.items()}
        print(f"  s{int(s):02d}: k{PRIMARY_K} base {per[int(s)][f'k{PRIMARY_K}_baseline']:.3f} "
              f"pers {per[int(s)][f'k{PRIMARY_K}_personal']:.3f} hyb {per[int(s)][f'k{PRIMARY_K}_hybrid']:.3f} "
              f"| k0 pop {per[int(s)]['k0_population']:.3f}", flush=True)
    rng = np.random.default_rng(SEED)
    keys = sorted(set().union(*(v.keys() for v in per.values())))
    summary = {key: subject_stats([per[s][key] for s in per if key in per[s]], rng) for key in keys}
    ks = sorted({int(key.split("_")[0][1:]) for key in keys} - {0})
    gains = {}
    for k in ks:
        for name in ("personal", "hybrid"):
            for suffix in ("", "_trial"):
                g = [per[s][f"k{k}_{name}{suffix}"] - per[s][f"k{k}_baseline{suffix}"] for s in per
                     if f"k{k}_baseline{suffix}" in per[s]]
                gains[f"k{k}_{name}{suffix}_minus_baseline"] = subject_stats(g, rng)
    return {"n_people": len(per), "people": per, "summary": summary, "gains": gains}


import sys
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for name, cache in COHORTS.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    print(name, flush=True)
    res[name] = run_cohort(cache)
    g = res[name]["gains"][f"k{PRIMARY_K}_personal_minus_baseline"]
    verdict = "EEG ADDS DEPLOYMENT VALUE" if g["mean"] >= 0.03 and g["ci"][0] > 0 else "no deployment value shown"
    res[name]["verdict"] = verdict
    print(f"== {name}: k{PRIMARY_K} personal - baseline {g['mean']:+.4f} CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] "
          f"p {g['p_signflip']:.4f} ({g['n_positive']}/{g['n']}) -> {verdict}", flush=True)
    for key, v in res[name]["gains"].items():
        print(f"   {key:40s} {v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p_signflip']:.4f}")
    for key, v in res[name]["summary"].items():
        print(f"   {key:30s} {v['mean']:+.4f}")
    OUT.write_text(json.dumps(res, indent=1))
print("DONE")
