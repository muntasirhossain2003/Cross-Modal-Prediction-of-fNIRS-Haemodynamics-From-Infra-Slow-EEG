"""FINAL_PLAN P9.1 (Claim 3 diagnostic): is the personal-delay failure outside hBCI a recording-length effect?
hBCI recordings are cut to the per-person length of mental workload / FineMI and the P7.1 + P8.1 analysis is repeated.
Modes: "durations" (measure MW / FineMI lengths) | "shard K N" | "merge"."""
import json
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "merge"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "shard" else (0, 1)
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "nvc_bandpower.py").read_text()
src = src[:src.index("COHORTS = tuple(")]
ns = {"__name__": "nvc_pow", "__file__": str(HERE / "nvc_bandpower.py")}
exec(src, ns)

import numpy as np
from scipy.stats import pearsonr, wilcoxon

curve, LAGS, SEED, GENS = ns["curve"], ns["LAGS"], ns["SEED"], ns["GENS"]
NK, MINSHIFT = 20, 60
DUR = HERE / "nvc_power_durations.json"
OUT = HERE / "nvc_power_results.json"

src_d = (HERE / "nvc_delay_smooth.py").read_text()
smooth_ns = {"__file__": str(HERE / "nvc_delay_smooth.py")}
exec(src_d[:src_d.index("rng = np.random.default_rng(1337)")], smooth_ns)   # smooth(), stats()


def halves(pairs):
    a, b = [], []
    for E, H in pairs:
        n = len(E) // 2
        a.append((E[:n], H[:n]))
        b.append((E[n:], H[n:]))
    return a, b


def person(pairs, rng):                                     # identical to nvc_kernel.person
    ha, hb = halves(pairs)
    ca, cb = curve(ha, "local"), curve(hb, "local")
    k = float(pearsonr(ca, cb)[0])
    kn = []
    for _ in range(NK):
        sp = [(E, np.roll(H, rng.integers(MINSHIFT, max(MINSHIFT + 1, len(H) - MINSHIFT)), axis=0)) for E, H in hb]
        kn.append(float(pearsonr(ca, curve(sp, "local"))[0]))
    return {"k": k, "k_null": kn, "d": k - float(np.mean(kn)),
            "p_person": float((np.sum(np.array(kn) >= k) + 1) / (NK + 1)),
            "delay_A_s": int(LAGS[np.argmax(ca)]), "delay_B_s": int(LAGS[np.argmax(cb)]),
            "curve_A": ca.round(4).tolist(), "curve_B": cb.round(4).tolist(), "n_runs": len(pairs)}


if MODE == "durations":
    out = {}
    for cohort in ("mental_workload", "finemi", "hbci"):
        tot, nr = [], []
        for pid, pairs in GENS[cohort]():
            if len(pairs) >= 2:
                tot.append(sum(len(E) for E, _ in pairs)); nr.append(len(pairs))
                print(pid, nr[-1], tot[-1], flush=True)
        out[cohort] = {"median_total_s": float(np.median(tot)), "median_runs": float(np.median(nr)), "n": len(tot),
                       "totals_s": tot, "runs": nr}
        print(cohort, out[cohort]["median_total_s"], out[cohort]["median_runs"], flush=True)
        DUR.write_text(json.dumps(out, indent=1))
    print("DONE"); sys.exit(0)

dur = json.loads(DUR.read_text())
LEVELS = {}
for c in ("mental_workload", "finemi"):
    n = int(min(6, round(dur[c]["median_runs"])))
    LEVELS[c] = (n, int(dur[c]["median_total_s"] // n))

if MODE == "shard":
    k_, n_ = SHARD
    sf = HERE / f"nvc_power_shard_{k_}_of_{n_}.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    for pi, (pid, pairs) in enumerate(GENS["hbci"]()):
        if pi % n_ != k_:
            continue
        for c, (n, L) in LEVELS.items():
            key = f"{pid} | {c}"
            if key in done:
                continue
            cut = [(E[:L], H[:L]) for E, H in pairs[:n]]
            if len(cut) < 2 or min(len(E) for E, _ in cut) < 3 * 20 + MINSHIFT:
                print(key, "too short"); continue
            v = person(cut, np.random.default_rng([SEED, pi, 9, len(key)]))
            done[key] = dict(v, cohort="hbci", level=c, runs_used=n, s_per_run=L)
            print(f"{key}: k {v['k']:+.3f} null {np.mean(v['k_null']):+.3f} delays {v['delay_A_s']:+d}/{v['delay_B_s']:+d}",
                  flush=True)
            sf.write_text(json.dumps(done, indent=1))
    print("SHARD DONE"); sys.exit(0)

allp = {}
for f in sorted(HERE.glob("nvc_power_shard_*_of_*.json")):
    allp.update(json.loads(f.read_text()))
rng = np.random.default_rng(SEED)
res = {"durations": {c: {k: dur[c][k] for k in ("median_total_s", "median_runs", "n")} for c in dur},
       "levels": {c: {"runs": n, "s_per_run": L} for c, (n, L) in LEVELS.items()}}
for c in LEVELS:
    people = {p: v for p, v in allp.items() if v["level"] == c}
    d = np.array([v["d"] for v in people.values()])
    w = float(wilcoxon(d, alternative="greater").pvalue)
    s = smooth_ns["stats"](people, rng)
    s.update({"median_d": float(np.median(d)), "wilcoxon_p": w, "kernel_reproducible": bool(np.median(d) > 0 and w < 0.05)})
    res[f"hbci matched to {c}"] = s
    n, L = LEVELS[c]
    print(f"== hBCI cut to {c} length ({n} runs x {L} s): kernel median d {s['median_d']:+.3f} Wilcoxon p {w:.4f} "
          f"-> {'reproducible' if s['kernel_reproducible'] else 'not'}; smoothed delay within 2 s {s['agree_2s']:.2f} "
          f"(chance {s['chance']:.2f}, p {s['p_agree']:.4f}) -> delay {'REPRODUCIBLE' if s['reproducible'] else 'NOT reproducible'}",
          flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
