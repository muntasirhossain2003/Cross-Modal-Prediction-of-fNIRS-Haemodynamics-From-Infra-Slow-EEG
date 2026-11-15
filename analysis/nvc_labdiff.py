"""FINAL_PLAN P9.4 (Claim 2 diagnostic): (a) pooled FineMI kernel scored on hBCI and MW; (b) broadband EEG power vs
global HbO correlation per person, lags 0-14 s, per cohort."""
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "nvc_bands.py").read_text()
src = src[:src.index("rng = np.random.default_rng(SEED)")]
NSB = {"__name__": "nvc_labdiff", "__file__": str(HERE / "nvc_bands.py")}
exec(src, NSB)

import json

import numpy as np

load, score, show, fit, NF, LAGS, SEED = (NSB[k] for k in ("load", "score", "show", "fit", "NF", "LAGS", "SEED"))
OUT = HERE / "nvc_labdiff_results.json"


def broadband(people, rng):
    per = {}
    for pid, runs in people.items():
        rs = []
        for X, ys in runs:
            y = ys["hbo"]
            for li in range(len(LAGS)):
                b = X[:, li * NF:(li + 1) * NF].mean(1)
                rs.append(float(np.corrcoef(b, y)[0, 1]))
        per[pid] = float(np.mean(rs))
    v = np.array(list(per.values()))
    boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    return {"mean_r": float(v.mean()), "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "n_positive": int(np.sum(v > 0)), "n": len(v), "people": per}


rng = np.random.default_rng(SEED)
res = {}
F = load("finemi")
fk = fit(F)
res["broadband | finemi"] = broadband(F, rng)
del F
for cohort in ("hbci", "mental_workload"):
    P = load(cohort)
    res[f"finemi_to_{cohort}"] = score(fk, P, "hbo", rng)
    show(f"P9.4a FineMI kernel -> {cohort}", res[f"finemi_to_{cohort}"])
    res[f"broadband | {cohort}"] = broadband(P, rng)
    del P
    OUT.write_text(json.dumps(res, indent=1))
for c in ("hbci", "mental_workload", "finemi"):
    b = res[f"broadband | {c}"]
    print(f"== P9.4b broadband power vs global HbO, {c}: mean r {b['mean_r']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] "
          f"({b['n_positive']}/{b['n']} > 0)", flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
