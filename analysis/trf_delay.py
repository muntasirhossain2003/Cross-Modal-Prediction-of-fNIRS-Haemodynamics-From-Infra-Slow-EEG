"""FINAL_PLAN P3.3 (Claim 3): delay from a multichannel lagged ridge on continuous recordings (positive = EEG earlier)."""
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"          # "shard K N" | "merge" | "all"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "shard" else (0, 1)
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "3"
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "continuous_lag.py").read_text()
src = src[:src.index("rng = np.random.default_rng(SEED)\nres = {}")]       # loaders and helpers only
ns = {"__name__": "trf", "__file__": str(HERE / "continuous_lag.py")}
exec(src, ns)

import json

import numpy as np
from scipy.stats import pearsonr, wilcoxon
from sklearn.linear_model import RidgeCV

COH = os.environ.get("COHORT", "")                         # "" = hbci + mental workload, "finemi" = P4.3
EEG_HP = float(os.environ.get("EEG_HP", "0"))              # P4.A: 0.01 = match the HbO band
TAG = (COH or "") + os.environ.get("FINEMI_TAG", "") + ("_lm" if os.environ.get("REREF") else "") + ("_hp" if EEG_HP else "")
OUT = HERE / ("trf_delay_results.json" if not TAG else f"trf_delay_{TAG}_results.json")
SHARDP = "trf_shard" if not TAG else f"trf_{TAG}_shard"
LAGS = np.arange(-10, 21)
NSURR, MINSHIFT, SEED = 100, 60, 1337
ALPHAS = np.logspace(0, 4, 5)
to_1hz, MAXLAG = ns["to_1hz"], ns["MAXLAG"]


def run_pair_mc(eeg, e_fs, hbo, n_fs, offset_s):
    """Multichannel version of P2.3's run_pair: (T, EEG ch) and (T, HbO ch) at 1 Hz, aligned."""
    e = to_1hz(eeg, e_fs, None, 0.1).T
    h = to_1hz(hbo, n_fs, 0.01, 0.1).T
    sh = int(round(offset_s))
    a0, b0 = max(0, -sh), min(len(e), len(h) - sh)
    if b0 - a0 < 3 * MAXLAG + MINSHIFT:
        return None
    return e[a0:b0], h[a0 + sh:b0 + sh]


ns["run_pair"] = run_pair_mc                    # the cohort generators look it up at call time


def lagged(E, H, L):
    if L >= 0:
        return E[:len(E) - L], H[L:]
    return E[-L:], H[:len(H) + L]


def colcorr(P, T):
    P, T = P.reshape(len(P), -1), T.reshape(len(T), -1)      # RidgeCV returns 1-D for a single target
    P = P - P.mean(0)
    T = T - T.mean(0)
    return float(np.mean((P * T).sum(0) / (np.linalg.norm(P, axis=0) * np.linalg.norm(T, axis=0) + 1e-12)))


def curve(pairs, target):
    """r(L), leave-one-run-out. target: 'local' (channels minus mean) or 'global' (channel mean)."""
    prep = []
    for E, H in pairs:
        if EEG_HP:
            import scipy.signal as _sps
            E = _sps.sosfiltfilt(_sps.butter(4, EEG_HP, btype="high", fs=1.0, output="sos"), E, axis=0)
        E = (E - E.mean(0)) / (E.std(0) + 1e-8)
        g = H.mean(1, keepdims=True)
        prep.append((E, H - g if target == "local" else g))
    out = []
    for L in LAGS:
        lp = [lagged(E, Y, L) for E, Y in prep]
        rs = []
        for i in range(len(lp)):
            Xtr = np.concatenate([x for j, (x, _) in enumerate(lp) if j != i])
            Ytr = np.concatenate([y for j, (_, y) in enumerate(lp) if j != i])
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr)
            rs.append(colcorr(m.predict(lp[i][0]), lp[i][1]))
        out.append(np.mean(rs))
    return np.array(out)


def person(pairs, rng):
    c = curve(pairs, "local")
    delay, peak = int(LAGS[np.argmax(c)]), float(c.max())
    null = []
    for _ in range(NSURR):
        sp = [(E, np.roll(H, rng.integers(MINSHIFT, max(MINSHIFT + 1, len(H) - MINSHIFT)), axis=0)) for E, H in pairs]
        null.append(curve(sp, "local").max())
    p = float((np.sum(np.array(null) >= peak) + 1) / (NSURR + 1))
    halves = []
    for h in (0, 1):
        hp = []
        for E, H in pairs:
            n = len(E) // 2
            sl = slice(0, n) if h == 0 else slice(n, None)
            hp.append((E[sl], H[sl]))
        halves.append(int(LAGS[np.argmax(curve(hp, "local"))]))
    cg = curve(pairs, "global")
    return {"delay_s": delay, "peak_r": peak, "p_null": p, "half_delays_s": halves, "curve_local": c.round(4).tolist(),
            "delay_global_s": int(LAGS[np.argmax(cg)]), "peak_r_global": float(cg.max()),
            "curve_global": cg.round(4).tolist(), "n_runs": len(pairs)}




def finemi_people():
    z = np.load(Path("G:/research_bio_medical/merge/neurovascnet_merge_v1_cache/replication_finemi" + os.environ.get("FINEMI_TAG", "") + ("_lm" if os.environ.get("REREF") else "") + "/continuous.npz"))
    subs = sorted({int(k.split("_")[0][1:]) for k in z.files})
    for s_ in subs:
        runs = sorted({int(k.split("_")[1][1:]) for k in z.files if k.startswith(f"s{s_}_")})
        yield f"finemi_s{s_:02d}", [(z[f"s{s_}_r{r}_E"].astype(float), z[f"s{s_}_r{r}_H"].astype(float)) for r in runs]


COHORTS = ((("hbci", ns["hbci_people"]), ("mental_workload", ns["mw_people"])) if not COH
           else ((COH, finemi_people),))

if MODE in ("shard", "all"):
    k, n = SHARD
    done = {}
    for ci, (cohort, gen) in enumerate(COHORTS):
        for pi, (pid, pairs) in enumerate(gen()):
            if (ci * 100 + pi) % n != k:
                continue
            if len(pairs) < 2:
                print(pid, "fewer than 2 usable runs"); continue
            v = person(pairs, np.random.default_rng([SEED, ci, pi]))
            done[pid] = dict(v, cohort=cohort)
            print(f"{pid}: local delay {v['delay_s']:+d} s r {v['peak_r']:.3f} p {v['p_null']:.3f} halves {v['half_delays_s']} "
                  f"| global delay {v['delay_global_s']:+d} s r {v['peak_r_global']:.3f}", flush=True)
            (HERE / f"{SHARDP}_{k}_of_{n}.json").write_text(json.dumps(done, indent=1))
    if MODE == "shard":
        print("SHARD DONE"); sys.exit(0)

allp = {}
for f in sorted(HERE.glob(f"{SHARDP}_*_of_*.json")):
    allp.update(json.loads(f.read_text()))
rng = np.random.default_rng(SEED)
res = {}
for cohort, _ in COHORTS:
    people = {p: v for p, v in allp.items() if v["cohort"] == cohort}
    d = np.array([v["delay_s"] for v in people.values()])
    h = np.array([v["half_delays_s"] for v in people.values()], float)
    ps = np.array([v["p_null"] for v in people.values()])
    r_half = float(pearsonr(h[:, 0], h[:, 1])[0]) if h[:, 0].std() > 0 and h[:, 1].std() > 0 else 0.0
    perm = np.array([pearsonr(h[:, 0], rng.permutation(h[:, 1]))[0] for _ in range(10000)])
    p_half = float((np.sum(perm >= r_half) + 1) / 10001)
    try:
        w = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w = 1.0
    exists = float(np.mean(ps < 0.05)) > 0.5
    reproducible = r_half > 0 and p_half < 0.05
    physio = 2 <= np.median(d) <= 10 and w < 0.05
    res[cohort] = {"people": people, "n": len(people), "frac_beating_null": float(np.mean(ps < 0.05)),
                   "median_delay_s": float(np.median(d)), "iqr_delay_s": [float(x) for x in np.percentile(d, [25, 75])],
                   "wilcoxon_p": w, "split_half_r": r_half, "split_half_perm_p": p_half,
                   "median_global_delay_s": float(np.median([v["delay_global_s"] for v in people.values()])),
                   "exists": bool(exists), "reproducible": bool(reproducible),
                   "claim3a_pass": bool(exists and reproducible), "physiological": bool(physio)}
    r = res[cohort]
    print(f"== {cohort}: n {r['n']}, beat null {r['frac_beating_null']:.2f}, median delay {r['median_delay_s']:+.1f} s "
          f"IQR {r['iqr_delay_s']} Wilcoxon p {w:.4f}; split-half r {r_half:+.3f} (perm p {p_half:.4f}); "
          f"global median {r['median_global_delay_s']:+.1f} s -> Claim 3a {'PASS' if r['claim3a_pass'] else 'FAIL'}, "
          f"physiological {'YES' if physio else 'NO'}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
print("DONE")
