"""FINAL_PLAN P7.2 (Claim 2, coupling-transfer form): a band-power -> global HbO lagged ridge (EEG lags 0..14 s),
fitted once on all hBCI people and frozen, applied to mental workload (same lab) and FineMI (other lab) after
spherical-spline mapping of their raw EEG onto the hBCI electrodes."""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "nvc_bandpower.py").read_text()
src = src[:src.index("COHORTS = tuple(")]
NS = {"__name__": "nvc_tr", "__file__": str(HERE / "nvc_bandpower.py")}
exec(src, NS)

import json

import nbformat
import numpy as np
from sklearn.linear_model import RidgeCV

inner = NS["ns"]
nb = nbformat.read(inner["NB"], as_version=4)
nbns = inner["ns"]                                 # the notebook namespace (cells 0-11 already run)
cwd = os.getcwd(); os.chdir(inner["NB"].parent)
exec("\n".join(l for l in nb.cells[19].source.splitlines() if not l.lstrip().startswith(("%", "!"))), nbns)
os.chdir(cwd)
interpolation_matrix = nbns["interpolation_matrix"]

ROOT = inner["NB"].parent / "neurovascnet_merge_v1_cache"
META = {"hbci": "train_fnirs_a2133f5d6a71ec34", "mental_workload": "replication_tuberlin_mw", "finemi": "replication_finemi"}


def chans(c):
    m = json.loads((ROOT / META[c] / "meta.json").read_text())
    return m["info"]["ch_names"] if "info" in m else m["ch_names"]


CH = {c: chans(c) for c in META}
SEED, NNULL, MINSHIFT = 1337, 200, 60
LAGS = np.arange(0, 15, 2)
ALPHAS = np.logspace(0, 6, 7)
OUT = HERE / "nvc_transfer_results.json"
orig_bp = NS["run_pair_bp"]
CUR = {"Ws": {}}


def run_pair_tr(eeg, e_fs, hbo, n_fs, offset_s):
    """Band power of the EEG mapped onto each source montage; global HbO."""
    eeg = np.asarray(eeg, float)
    out, y = {}, None
    for key, W in CUR["Ws"].items():
        pr = orig_bp(eeg if W is None else W @ eeg, e_fs, hbo, n_fs, offset_s)
        if pr is None:
            return None
        out[key], y = pr[0], pr[1].mean(1)
    return out, y


NS["run_pair_bp"] = run_pair_tr                    # FineMI generator
inner["run_pair"] = run_pair_tr                    # hBCI / MW generators


def design(E, y):
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    t0 = LAGS.max()
    X = np.concatenate([E[t0 - L:len(E) - L] for L in LAGS], 1)
    return X.astype(np.float32), y[t0:].astype(np.float32)


def load(cohort, keys):
    CUR["Ws"] = {k: (None if k == cohort else interpolation_matrix(CH[cohort], CH[k])) for k in keys}
    people = {}
    for pid, pairs in NS["GENS"][cohort]():
        if len(pairs) < 1:
            print(pid, "no usable runs"); continue
        people[pid] = {k: [design(E[k], y) for E, y in pairs] for k in keys}
        print(f"loaded {pid}: {len(pairs)} runs", flush=True)
    return people


def fit(people, key):
    X = np.concatenate([x for p in people.values() for x, _ in p[key]])
    y = np.concatenate([y for p in people.values() for _, y in p[key]])
    m = RidgeCV(alphas=ALPHAS).fit(X, y)
    print(f"fitted on {len(people)} people, {len(y)} s, alpha {m.alpha_:g}", flush=True)
    return m


def r(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def score(pred_fn, people, key, rng):
    per = {}
    for pid, p in people.items():
        runs = [(pred_fn(x), y) for x, y in p[key]]
        obs = float(np.mean([r(a, b) for a, b in runs]))
        null = np.array([np.mean([r(a, np.roll(b, rng.integers(MINSHIFT, max(MINSHIFT + 1, len(b) - MINSHIFT))))
                                  for a, b in runs]) for _ in range(NNULL)])
        per[pid] = {"r": obs, "p_null": float((np.sum(null >= obs) + 1) / (NNULL + 1)), "n_runs": len(runs)}
    v = np.array([q["r"] for q in per.values()])
    boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    beat = float(np.mean([q["p_null"] < 0.05 for q in per.values()]))
    return {"mean_r": float(v.mean()), "ci": ci, "n_positive": int(np.sum(v > 0)), "n": len(v),
            "frac_beating_null": beat, "pass": bool(v.mean() > 0 and ci[0] > 0 and beat > 0.5), "people": per}


def show(name, s):
    print(f"== {name}: mean r {s['mean_r']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] ({s['n_positive']}/{s['n']} > 0), "
          f"beat null {s['frac_beating_null']:.2f} -> {'PASS' if s['pass'] else 'FAIL'}", flush=True)


rng = np.random.default_rng(SEED)
res = {}
H = load("hbci", ["hbci"])
m_h = fit(H, "hbci")
res["lag_weight_share"] = dict(zip([int(L) for L in LAGS], (lambda w: (w / w.sum()).round(3).tolist())(
    np.abs(m_h.coef_.reshape(len(LAGS), -1)).sum(1))))
print("hBCI kernel weight share by EEG lag (s):", res["lag_weight_share"], flush=True)

# descriptive: leave-one-person-out within hBCI at the frozen model's alpha
XtX = {p: sum(x.astype(np.float64).T @ x for x, _ in v["hbci"]) for p, v in H.items()}
Xty = {p: sum(x.astype(np.float64).T @ y for x, y in v["hbci"]) for p, v in H.items()}
A, B = sum(XtX.values()), sum(Xty.values())
lopo = {}
for p in H:
    w = np.linalg.solve(A - XtX[p] + m_h.alpha_ * np.eye(len(A)), B - Xty[p])
    lopo[p] = w
per_lopo = {}
for p, v in H.items():
    s1 = score(lambda x, w=lopo[p]: x @ w, {p: v}, "hbci", rng)
    per_lopo[p] = s1["people"][p]
v_ = np.array([q["r"] for q in per_lopo.values()])
res["within_hbci_lopo"] = {"mean_r": float(v_.mean()), "frac_beating_null": float(np.mean([q["p_null"] < 0.05 for q in per_lopo.values()])),
                           "people": per_lopo}
print(f"== within hBCI (leave one person out): mean r {v_.mean():+.4f}, beat null "
      f"{res['within_hbci_lopo']['frac_beating_null']:.2f}", flush=True)
del XtX, Xty, A, B

M = load("mental_workload", ["hbci", "mental_workload"])
res["hbci_to_mental_workload"] = score(m_h.predict, M, "hbci", rng)
show("hBCI -> mental workload (same lab, reference)", res["hbci_to_mental_workload"])
m_m = fit(M, "mental_workload")
del M
OUT.write_text(json.dumps(res, indent=1))

F = load("finemi", ["hbci", "mental_workload"])
res["hbci_to_finemi"] = score(m_h.predict, F, "hbci", rng)
show("hBCI -> FineMI (other lab, PRIMARY)", res["hbci_to_finemi"])
res["mental_workload_to_finemi"] = score(m_m.predict, F, "mental_workload", rng)
show("MW -> FineMI (descriptive)", res["mental_workload_to_finemi"])
res["claim2_coupling_transfer_cross_lab"] = res["hbci_to_finemi"]["pass"]
print("Claim 2 (coupling-transfer form, cross-lab):", "PASS" if res["claim2_coupling_transfer_cross_lab"] else "FAIL")
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
