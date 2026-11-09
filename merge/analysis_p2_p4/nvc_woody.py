"""FINAL_PLAN P11 (Claim 3): personal delay by template alignment of the saved half-recording curves, with a positive control."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAGS = np.arange(-10, 21)
SHIFTS = np.arange(-8, 9)
ALPHA = 0.05 / 14
OUT = HERE / "nvc_woody_results.json"


def smooth(c, w=5):
    c = np.asarray(c, float)
    h = w // 2
    return np.array([c[max(0, i - h):i + h + 1].mean() for i in range(len(c))])


def z(c):
    return (c - c.mean()) / (c.std() + 1e-12)


def align(curve, template):
    best, best_r = 0, -np.inf
    for s in SHIFTS:
        i = np.arange(max(0, s), min(len(curve), len(curve) + s))
        a, b = curve[i], template[i - s]
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else -np.inf
        if r > best_r:
            best, best_r = int(s), r
    return best


def delays(people):
    S = {p: (z(smooth(v["curve_A"])), z(smooth(v["curve_B"]))) for p, v in people.items()}
    dA, dB = [], []
    for p in S:
        T = smooth(np.mean([c for q in S if q != p for c in S[q]], 0))
        peak = LAGS[np.argmax(T)]
        dA.append(peak + align(S[p][0], T))
        dB.append(peak + align(S[p][1], T))
    return np.array(dA), np.array(dB)


def stats(people, rng):
    dA, dB = delays(people)
    agree = float(np.mean(np.abs(dA - dB) <= 2))
    chance = float(np.mean([np.mean(np.abs(dA[i] - np.delete(dB, i)) <= 2) for i in range(len(dA))]))
    perm = np.array([np.mean(np.abs(dA - rng.permutation(dB)) <= 2) for _ in range(10000)])
    p = float((np.sum(perm >= agree) + 1) / 10001)
    return {"n": len(dA), "agree_2s": agree, "chance": chance, "p_agree": p, "median_delay_s": float(np.median(np.concatenate([dA, dB]))),
            "passes_p05": bool(agree > chance and p < 0.05), "passes_corrected": bool(agree > chance and p < ALPHA)}


def load(pattern):
    d = {}
    for f in sorted(HERE.glob(pattern)):
        d.update(json.loads(f.read_text()))
    return d


rng = np.random.default_rng(1337)
snr = load("nvc_snr_shard_*_of_*.json")
ctrl = {k: v for k, v in snr.items() if v["level"] == "finemi"}
res = {"control | hbci degraded to finemi strength": stats(ctrl, rng),
       "control | hbci degraded to mental_workload strength": stats({k: v for k, v in snr.items() if v["level"] == "mental_workload"}, rng)}
valid = res["control | hbci degraded to finemi strength"]["passes_p05"]
res["estimator_valid"] = valid
for tag, label in (("", "raw"), ("_resid", "task removed")):
    K = load(f"nvc_kernel{tag}_shard_*_of_*.json")
    for c in ("hbci", "mental_workload", "finemi"):
        res[f"{c} | {label}"] = stats({p: v for p, v in K.items() if v["cohort"] == c}, rng)
for k, v in res.items():
    if isinstance(v, dict):
        print(f"== {k}: within 2 s {v['agree_2s']:.2f} (chance {v['chance']:.2f}, p {v['p_agree']:.4f}), median delay {v['median_delay_s']:+.1f} s, p<0.05 {v['passes_p05']}, corrected {v['passes_corrected']}", flush=True)
print("Estimator valid (positive control):", valid)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
