"""FINAL_PLAN P10.2 (Claim 3 diagnostic): hBCI local HbO degraded to the MW / FineMI coupling strength, then the P8.1
smoothed-delay statistics. Modes, in order: "calibrate K N" | "pick" | "shard K N" | "merge"."""
import json
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "merge"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE in ("shard", "calibrate") else (0, 1)
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "nvc_bandpower.py").read_text(encoding="utf-8")
src = src[:src.index("COHORTS = tuple(")]
ns = {"__name__": "nvc_snr", "__file__": str(HERE / "nvc_bandpower.py")}
exec(src, ns)

import numpy as np

curve, SEED, GENS = ns["curve"], ns["SEED"], ns["GENS"]
MINSHIFT = 60
GRID = (0.5, 1, 1.5, 2, 3, 4, 6, 8)
LEVELS = {"mental_workload": (3, 626, 0.135), "finemi": (6, None, 0.053)}    # runs, s per run, target median peak r
CAL = HERE / "nvc_snr_calibration.json"
OUT = HERE / "nvc_snr_results.json"
sd = (HERE / "nvc_delay_smooth.py").read_text(encoding="utf-8")
smooth_ns = {"__file__": str(HERE / "nvc_delay_smooth.py")}
exec(sd[:sd.index("rng = np.random.default_rng(1337)")], smooth_ns)


def cut(pairs, n, L):
    return [(E[:L], H[:L]) if L else (E, H) for E, H in pairs[:n]]


def degrade(pairs, c, rng):
    out = []
    for E, H in pairs:
        s = rng.integers(MINSHIFT, max(MINSHIFT + 1, len(H) - MINSHIFT))
        out.append((E, H + c * np.roll(H, s, axis=0)))
    return out


if MODE in ("calibrate", "shard"):
    k_, n_ = SHARD
    tag = "cal" if MODE == "calibrate" else "shard"
    sf = HERE / f"nvc_snr_{tag}_{k_}_of_{n_}.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    cs = json.loads(CAL.read_text())["c"] if MODE == "shard" else None
    for pi, (pid, pairs) in enumerate(GENS["hbci"]()):
        if pi % n_ != k_:
            continue
        for lv, (n, L, _) in LEVELS.items():
            cp = cut(pairs, n, L)
            if MODE == "calibrate":
                for c in GRID:
                    key = f"{pid} | {lv} | {c}"
                    if key not in done:
                        done[key] = float(curve(degrade(cp, c, np.random.default_rng([SEED, pi, 11])), "local").max())
                        sf.write_text(json.dumps(done, indent=1))
                print(pid, lv, "calibrated", flush=True)
            else:
                key = f"{pid} | {lv}"
                if key in done:
                    continue
                dg = degrade(cp, cs[lv], np.random.default_rng([SEED, pi, 13]))
                a = [(E[:len(E) // 2], H[:len(H) // 2]) for E, H in dg]
                b = [(E[len(E) // 2:], H[len(H) // 2:]) for E, H in dg]
                done[key] = {"curve_A": curve(a, "local").round(4).tolist(), "curve_B": curve(b, "local").round(4).tolist(),
                             "level": lv, "c": cs[lv]}
                print(key, flush=True)
                sf.write_text(json.dumps(done, indent=1))
    print("SHARD DONE")
    sys.exit(0)

if MODE == "pick":
    allc = {}
    for f in sorted(HERE.glob("nvc_snr_cal_*_of_*.json")):
        allc.update(json.loads(f.read_text()))
    out = {"c": {}, "median_peak": {}}
    for lv, (_, _, target) in LEVELS.items():
        med = {c: float(np.median([v for k, v in allc.items() if k.split(" | ")[1] == lv and float(k.split(" | ")[2]) == c]))
               for c in GRID}
        out["median_peak"][lv] = {str(c): m for c, m in med.items()}
        out["c"][lv] = min(GRID, key=lambda c: abs(med[c] - target))
        print(lv, "target", target, "grid", {c: round(m, 3) for c, m in med.items()}, "-> c", out["c"][lv], flush=True)
    CAL.write_text(json.dumps(out, indent=1))
    sys.exit(0)

allp = {}
for f in sorted(HERE.glob("nvc_snr_shard_*_of_*.json")):
    allp.update(json.loads(f.read_text()))
rng = np.random.default_rng(SEED)
cal = json.loads(CAL.read_text())
res = {"calibration": cal}
for lv, (_, _, target) in LEVELS.items():
    s = smooth_ns["stats"]({p: v for p, v in allp.items() if v["level"] == lv}, rng)
    res[f"hbci at {lv} strength"] = s
    achieved = cal["median_peak"][lv][str(cal["c"][lv])]
    print(f"== hBCI degraded to {lv} strength (c {cal['c'][lv]}, achieved peak {achieved:.3f}, target {target}): "
          f"delay within 2 s {s['agree_2s']:.2f} (chance {s['chance']:.2f}, p {s['p_agree']:.4f}) -> "
          f"{'REPRODUCIBLE' if s['reproducible'] else 'NOT reproducible'}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
