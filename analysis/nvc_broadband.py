"""FINAL_PLAN P10.1 (Claim 2): frozen hBCI band-power kernel with the broadband term removed, and broadband-only."""
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "nvc_bands.py").read_text(encoding="utf-8")
src = src[:src.index("rng = np.random.default_rng(SEED)")]
NSB = {"__name__": "nvc_broadband", "__file__": str(HERE / "nvc_bands.py")}
exec(src, NSB)

import json

import numpy as np
from sklearn.linear_model import RidgeCV

load, score, calibrated, show, show_cal = (NSB[k] for k in ("load", "score", "calibrated", "show", "show_cal"))
NF, LAGS, SEED, ALPHAS = (NSB[k] for k in ("NF", "LAGS", "SEED", "ALPHAS"))
OUT = HERE / "nvc_broadband_results.json"
NL = len(LAGS)


def same(X):
    return X


def removed(X):
    X = X.copy()
    for i in range(NL):
        b = X[:, i * NF:(i + 1) * NF]
        b -= b.mean(1, keepdims=True)
    return X


def only(X):
    return np.stack([X[:, i * NF:(i + 1) * NF].mean(1) for i in range(NL)], 1)


class Wrapped:
    def __init__(self, f, m):
        self.f, self.m = f, m

    def predict(self, X):
        return self.m.predict(self.f(X))


def fit(people, f):
    X = np.concatenate([f(x) for runs in people.values() for x, _ in runs])
    y = np.concatenate([ys["hbo"] for runs in people.values() for _, ys in runs])
    return Wrapped(f, RidgeCV(alphas=ALPHAS).fit(X, y))


def signflip(d, rng, n=200000):
    d = np.asarray(d)
    return float((np.sum((rng.choice([-1, 1], size=(n, len(d))) * d).mean(1) >= d.mean()) + 1) / (n + 1))


rng = np.random.default_rng(SEED)
res = {}
H = load("hbci")
K = {"all": fit(H, same), "removed": fit(H, removed), "only": fit(H, only)}
del H
for cohort in ("mental_workload", "finemi"):
    P = load(cohort)
    s = {}
    for name, m in K.items():
        s[name] = score(m, P, "hbo", rng)
        res[f"hbci_to_{cohort} | {name}"] = s[name]
        show(f"hBCI [{name}] -> {cohort}", s[name])
    d = np.array([s["removed"]["people"][p]["r"] - s["all"]["people"][p]["r"] for p in s["all"]["people"]])
    ci = [float(x) for x in np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])]
    pf = signflip([q["r"] for q in s["removed"]["people"].values()], rng)
    res[f"paired removed-all | {cohort}"] = {"diff": float(d.mean()), "ci": ci, "signflip_p_removed": pf}
    print(f"== paired removed - all, {cohort}: {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]; sign-flip p of removed {pf:.5f}", flush=True)
    c = calibrated(K["removed"], P, np.random.default_rng(SEED))
    show_cal(f"calibrated removed -> {cohort}", c)
    c["signflip_p"] = signflip([q["personal_plus_external"] - q["personal"] for q in c["people"].values()], rng)
    res[f"calibrated | {cohort} | removed"] = c
    print(f"== calibrated sign-flip p, {cohort}: {c['signflip_p']:.5f}", flush=True)
    del P
    OUT.write_text(json.dumps(res, indent=1))
f, o, pr = res["hbci_to_finemi | removed"], res["hbci_to_finemi | only"], res["paired removed-all | finemi"]
sup = bool(pr["ci"][0] > 0 and f["ci"][1] >= 0 and o["mean_r"] < 0 and o["ci"][1] < 0)
print("P10.1 broadband explanation:", "SUPPORTED" if sup else "NOT SUPPORTED")
print("P10.1 Claim 2 primary (frozen removed -> FineMI):", "PASS" if f["pass"] else "FAIL")
print("DONE")
