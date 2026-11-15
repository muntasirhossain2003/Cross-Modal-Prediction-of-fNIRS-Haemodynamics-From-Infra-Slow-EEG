"""FINAL_PLAN P9.2 (Claim 2 diagnostic: per-band kernel signs, single-band transfer, FineMI LOPO) and P9.3
(confirmatory: frozen hBCI alpha + beta kernel -> FineMI, P7.2 rule; P8.3 calibrated rule secondary)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "nvc_transfer2.py").read_text()
src = src[:src.index("CAL_SEEDS = [")]
NS2 = {"__name__": "nvc_bands", "__file__": str(HERE / "nvc_transfer2.py")}
exec(src, NS2)

import json

import numpy as np
from sklearn.linear_model import RidgeCV

load, score, calibrated, show, show_cal = (NS2[k] for k in ("load", "score", "calibrated", "show", "show_cal"))
ALPHAS, LAGS, SEED = NS2["ALPHAS"], NS2["LAGS"], NS2["SEED"]
BANDS = ("theta", "alpha", "beta", "gamma")
NCH = len(NS2["CH"]["hbci"])
OUT = HERE / "nvc_bands_results.json"
NF = len(BANDS) * NCH                                         # features per lag
band_of = (np.arange(NF * len(LAGS)) % NF) // NCH


class Sub:
    def __init__(self, idx, m):
        self.idx, self.m = idx, m

    def predict(self, X):
        return self.m.predict(X[:, self.idx])


def fit(people, idx=None):
    X = np.concatenate([x for runs in people.values() for x, _ in runs])
    y = np.concatenate([ys["hbo"] for runs in people.values() for _, ys in runs])
    assert X.shape[1] == NF * len(LAGS), X.shape
    idx = np.arange(X.shape[1]) if idx is None else idx
    return Sub(idx, RidgeCV(alphas=ALPHAS).fit(X[:, idx], y))


def band_sums(model):
    c = model.m.coef_
    full = np.zeros(NF * len(LAGS)); full[model.idx] = c
    return {b: float(full[band_of == i].sum()) for i, b in enumerate(BANDS)}


def lopo(people, rng):
    """Leave-one-person-out kernel within a cohort, scored with the P7.2 score."""
    preds = {}
    for pid in people:
        m = fit({p: v for p, v in people.items() if p != pid})
        preds[pid] = m
    out = {}
    for pid, runs in people.items():
        out.update(score(preds[pid], {pid: runs}, "hbo", rng)["people"])
    v = np.array([q["r"] for q in out.values()])
    boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    beat = float(np.mean([q["p_null"] < 0.05 for q in out.values()]))
    return {"mean_r": float(v.mean()), "ci": ci, "n_positive": int(np.sum(v > 0)), "n": len(v),
            "frac_beating_null": beat, "pass": bool(v.mean() > 0 and ci[0] > 0 and beat > 0.5), "people": out}


rng = np.random.default_rng(SEED)
res = {}
AB = np.where(np.isin(band_of, [1, 2]))[0]
H = load("hbci")
kern = {"all": fit(H), "alpha+beta": fit(H, AB)}
for i, b in enumerate(BANDS):
    kern[b] = fit(H, np.where(band_of == i)[0])
res["band_sums | hbci"] = band_sums(kern["all"])
print("hBCI pooled kernel band sums:", {k: round(v, 4) for k, v in res["band_sums | hbci"].items()}, flush=True)
del H
for cohort in ("mental_workload", "finemi"):
    P = load(cohort)
    own = fit(P)
    res[f"band_sums | {cohort}"] = band_sums(own)
    print(f"{cohort} pooled kernel band sums:", {k: round(v, 4) for k, v in res[f"band_sums | {cohort}"].items()}, flush=True)
    for name, m in kern.items():
        res[f"hbci_to_{cohort} | {name}"] = score(m, P, "hbo", rng)
        show(f"hBCI [{name}] -> {cohort}", res[f"hbci_to_{cohort} | {name}"])
    res[f"calibrated | {cohort} | alpha+beta"] = calibrated(kern["alpha+beta"], P, np.random.default_rng(SEED))
    show_cal(f"P9.3 calibrated alpha+beta -> {cohort}", res[f"calibrated | {cohort} | alpha+beta"])
    if cohort == "finemi":
        res["lopo | finemi"] = lopo(P, rng)
        show("P9.2 FineMI leave-one-person-out", res["lopo | finemi"])
    del P
    OUT.write_text(json.dumps(res, indent=1))
print("P9.3 primary (frozen alpha+beta -> FineMI):", "PASS" if res["hbci_to_finemi | alpha+beta"]["pass"] else "FAIL")
print("P9.3 secondary (calibrated alpha+beta -> FineMI):", "PASS" if res["calibrated | finemi | alpha+beta"]["pass"] else "FAIL")
print("DONE")
