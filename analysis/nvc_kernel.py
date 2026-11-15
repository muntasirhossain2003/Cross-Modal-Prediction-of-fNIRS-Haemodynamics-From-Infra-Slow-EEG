"""FINAL_PLAN P7.1 (Claim 3, kernel form): does a person's band-power -> local HbO lag curve repeat between the
first and second half of their recording, more than after circularly shifting the second-half HbO?
env RESID=1 runs the same test on the P6.4b task-residual data."""
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"          # "shard K N" | "merge" | "all"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "shard" else (0, 1)
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
RESID = os.environ.get("RESID") == "1"
if RESID:
    src = (HERE / "nvc_residual.py").read_text()
    src = src[:src.index("if MODE in (")]                     # helpers + residual generators
    ns = {"__name__": "nvc_k", "__file__": str(HERE / "nvc_residual.py")}
else:
    src = (HERE / "nvc_bandpower.py").read_text()
    src = src[:src.index("COHORTS = tuple(")]                  # helpers + P6.4 generators
    ns = {"__name__": "nvc_k", "__file__": str(HERE / "nvc_bandpower.py")}
exec(src, ns)

import json

import numpy as np
from scipy.stats import pearsonr, wilcoxon

base = ns["ns"] if RESID else ns                           # the nvc_bandpower namespace
curve, LAGS, SEED, GENS = base["curve"], base["LAGS"], base["SEED"], ns["GENS"]
NK, MINSHIFT = int(os.environ.get("NK", "20")), 60
COHORTS_ENV = os.environ.get("COHORTS", "hbci,mental_workload,finemi").split(",")
TAG = "_resid" if RESID else ""
OUT = HERE / f"nvc_kernel{TAG}_results.json"
SHARDP = f"nvc_kernel{TAG}_smoke" if os.environ.get("SMOKE") else f"nvc_kernel{TAG}_shard"


def halves(pairs):
    a, b = [], []
    for E, H in pairs:
        n = len(E) // 2
        a.append((E[:n], H[:n]))
        b.append((E[n:], H[n:]))
    return a, b


def person(pairs, rng):
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


if MODE in ("shard", "all"):
    k_, n_ = SHARD
    sf = HERE / f"{SHARDP}_{k_}_of_{n_}.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    for cohort in COHORTS_ENV:
        ci = ("hbci", "mental_workload", "finemi").index(cohort)
        for pi, (pid, pairs) in enumerate(GENS[cohort]()):
            if (ci * 100 + pi) % n_ != k_ or pid in done:
                continue
            if len(pairs) < 2:
                print(pid, "fewer than 2 usable runs"); continue
            v = person(pairs, np.random.default_rng([SEED, ci, pi, 7]))
            done[pid] = dict(v, cohort=cohort)
            print(f"{pid}: k {v['k']:+.3f} null {np.mean(v['k_null']):+.3f} d {v['d']:+.3f} p {v['p_person']:.3f} "
                  f"delays {v['delay_A_s']:+d}/{v['delay_B_s']:+d}", flush=True)
            sf.write_text(json.dumps(done, indent=1))
    if MODE == "shard":
        print("SHARD DONE"); sys.exit(0)

allp = {}
for f in sorted(HERE.glob(f"{SHARDP}_*_of_*.json")):
    allp.update(json.loads(f.read_text()))
p64 = json.loads((HERE / ("nvc_residual_results.json" if RESID else "nvc_bandpower_results.json")).read_text())
rng = np.random.default_rng(SEED)
res = {}
for cohort in ("hbci", "mental_workload", "finemi"):
    people = {p: v for p, v in allp.items() if v["cohort"] == cohort}
    if not people:
        continue
    d = np.array([v["d"] for v in people.values()])
    try:
        w = float(wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        w = 1.0
    dA = np.array([v["delay_A_s"] for v in people.values()])
    dB = np.array([v["delay_B_s"] for v in people.values()])
    agree = float(np.mean(np.abs(dA - dB) <= 2))
    chance = float(np.mean([np.mean(np.abs(dA[i] - np.delete(dB, i)) <= 2) for i in range(len(dA))]))
    exists = bool(p64[cohort]["frac_beating_null"] > 0.5)
    repro = bool(np.median(d) > 0 and w < 0.05)
    res[cohort] = {"people": people, "n": len(people), "median_k": float(np.median([v["k"] for v in people.values()])),
                   "median_k_null": float(np.median([np.mean(v["k_null"]) for v in people.values()])),
                   "median_d": float(np.median(d)), "wilcoxon_p_greater": w,
                   "frac_person_p05": float(np.mean([v["p_person"] < 0.05 for v in people.values()])),
                   "delay_agree_2s": agree, "delay_agree_2s_chance": chance,
                   "exists_p64": exists, "kernel_reproducible": repro, "claim3a_kernel_pass": bool(exists and repro)}
    r = res[cohort]
    print(f"== KERNEL{TAG} {cohort}: n {r['n']}, median k {r['median_k']:+.3f} vs null {r['median_k_null']:+.3f}, "
          f"median d {r['median_d']:+.3f}, Wilcoxon p {w:.4f}, {r['frac_person_p05']:.2f} of people p<0.05; "
          f"delays within 2 s {agree:.2f} (chance {chance:.2f}) -> exists {exists}, reproducible {repro} "
          f"-> 3a' {'PASS' if r['claim3a_kernel_pass'] else 'FAIL'}", flush=True)
res["claim3_kernel_supported"] = bool(res.get("hbci", {}).get("claim3a_kernel_pass") and any(
    res.get(c, {}).get("claim3a_kernel_pass") for c in ("mental_workload", "finemi")))
print(f"Claim 3 (kernel form{', beyond the task' if RESID else ''}):",
      "SUPPORTED" if res["claim3_kernel_supported"] else "NOT SUPPORTED")
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
