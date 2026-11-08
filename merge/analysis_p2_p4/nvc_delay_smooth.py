"""FINAL_PLAN P8.1 (Claim 3): personal delay from the P7.1 half curves after a 5-lag centred moving average."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

HERE = Path(__file__).parent
LAGS = np.arange(-10, 21)
OUT = HERE / "nvc_delay_smooth_results.json"


def smooth(c, w=5):
    c = np.asarray(c, float)
    h = w // 2
    return np.array([c[max(0, i - h):i + h + 1].mean() for i in range(len(c))])


def stats(people, rng):
    dA = np.array([LAGS[np.argmax(smooth(v["curve_A"]))] for v in people.values()])
    dB = np.array([LAGS[np.argmax(smooth(v["curve_B"]))] for v in people.values()])
    agree = float(np.mean(np.abs(dA - dB) <= 2))
    chance = float(np.mean([np.mean(np.abs(dA[i] - np.delete(dB, i)) <= 2) for i in range(len(dA))]))
    perm = np.array([np.mean(np.abs(dA - rng.permutation(dB)) <= 2) for _ in range(10000)])
    p_agree = float((np.sum(perm >= agree) + 1) / 10001)
    r = float(pearsonr(dA, dB)[0]) if dA.std() > 0 and dB.std() > 0 else 0.0
    pr = np.array([pearsonr(dA, rng.permutation(dB))[0] for _ in range(10000)])
    p_r = float((np.sum(pr >= r) + 1) / 10001)
    return {"n": len(dA), "agree_2s": agree, "chance": chance, "p_agree": p_agree, "split_half_r": r, "p_r": p_r,
            "median_delay_s": float(np.median(np.concatenate([dA, dB]))),
            "iqr_delay_s": [float(x) for x in np.percentile(np.concatenate([dA, dB]), [25, 75])],
            "reproducible": bool(agree > chance and p_agree < 0.05)}


rng = np.random.default_rng(1337)
res = {}
for tag, label in (("", "raw"), ("_resid", "task removed")):
    allp = {}
    for f in sorted(HERE.glob(f"nvc_kernel{tag}_shard_*_of_*.json")):
        allp.update(json.loads(f.read_text()))
    for cohort in ("hbci", "mental_workload", "finemi"):
        s = stats({p: v for p, v in allp.items() if v["cohort"] == cohort}, rng)
        res[f"{cohort} | {label}"] = s
        print(f"{cohort:16s} {label:12s}: within 2 s {s['agree_2s']:.2f} (chance {s['chance']:.2f}, p {s['p_agree']:.4f}); "
              f"split-half r {s['split_half_r']:+.3f} (p {s['p_r']:.4f}); median delay {s['median_delay_s']:+.1f} s "
              f"IQR {s['iqr_delay_s']} -> {'REPRODUCIBLE' if s['reproducible'] else 'not reproducible'}")
OUT.write_text(json.dumps(res, indent=1))
