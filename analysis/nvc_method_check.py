"""FINAL_PLAN P13: can the P12.1 method detect the slow-EEG advantage? Usage: nvc_method_check.py COHORT [COHORT ...] | merge."""
import json
import os
import sys
from pathlib import Path

ARGS = sys.argv[1:] or ["merge"]
SMOKE = int(os.environ.get("SMOKE", "0"))
HERE = Path(__file__).parent
sys.argv = [sys.argv[0], "none"]
src = (HERE / "nvc_mendeley.py").read_text(encoding="utf-8")
src = src[:src.index('if MODE == "coupling":')]
M12 = {"__name__": "nvc_m13", "__file__": str(HERE / "nvc_mendeley.py")}
exec(src, M12)

import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

threadpool_limits(limits=int(os.environ.get("THREADS", "10")))
BP, LC, NS2 = M12["BP"], M12["LC"], M12["NS2"]
nb = LC["ns"]
to_1hz, bandpower_1hz, MAXLAG, MINSHIFT, SEED, signflip, corr = LC["to_1hz"], BP["bandpower_1hz"], LC["MAXLAG"], BP["MINSHIFT"], BP["SEED"], M12["signflip"], M12["r"]
resample_to, zscore_trial, baseline_correct, bandpass = (nb[k] for k in ("resample_to", "zscore_trial", "baseline_correct", "bandpass"))
ALPHA = 0.05 / 7
LG = np.arange(-12, 13, 4)
T0 = int(LG.max())
WIN = 30.0
FS = 200.0
ALPHAS_C = M12["ALPHAS"]
ALPHAS_W = np.logspace(0, 6, 13)
S05 = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
BANDS = [sps.butter(4, [lo, hi], btype="band", fs=FS, output="sos") for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))]


def feats(E):
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    return np.concatenate([E[T0 - L: len(E) - T0 - L] for L in LG], 1).astype(np.float32)


def bins(x, nb_):
    return x.reshape(*x.shape[:-1], nb_, -1).mean(-1)


def windows(eeg, e_fs, hbo, n_fs, off):
    a = max(0.0, -off)
    b = min(eeg.shape[1] / e_fs, hbo.shape[1] / n_fs - off)
    hf = bandpass(hbo, n_fs, 0.01, 0.2)
    X, Y = [], []
    for i in range(int((b - a) // WIN)):
        t = a + i * WIN
        ie, in_ = int(round(t * e_fs)), int(round((t + off) * n_fs))
        X.append(zscore_trial(resample_to(eeg[:, ie:ie + int(round(WIN * e_fs))], 6000)))
        Y.append(baseline_correct(resample_to(hf[:, in_:in_ + int(round(WIN * n_fs))], 300), 20))
    if not X:
        return None
    X = np.stack(X).astype(np.float64)
    slow = bins(sps.sosfiltfilt(S05, X, axis=-1), 30).reshape(len(X), -1)
    band = np.concatenate([np.log(bins(sps.sosfiltfilt(s, X, axis=-1) ** 2, 10) + 1e-8).reshape(len(X), -1) for s in BANDS], 1)
    return slow.astype(np.float32), band.astype(np.float32), bins(np.stack(Y), 30).reshape(len(X), -1).astype(np.float32)


def hook(eeg, e_fs, hbo, n_fs, off):
    eeg, hbo = np.asarray(eeg, float), np.asarray(hbo, float)
    sl, bp, g = to_1hz(eeg, e_fs, None, 0.5).T, bandpower_1hz(eeg, e_fs), to_1hz(hbo, n_fs, 0.01, 0.1).mean(0)
    sh = int(round(off))
    a0, b0 = max(0, -sh), min(len(sl), len(bp), len(g) - sh)
    if b0 - a0 < 3 * MAXLAG + MINSHIFT:
        return None
    sl, bp, g = sl[a0:b0], bp[a0:b0], g[a0 + sh:b0 + sh]
    y = ((g - g.mean()) / (g.std() + 1e-8))[T0: len(g) - T0].astype(np.float32)
    return {"cont": (feats(sl), feats(bp), y), "win": windows(eeg, e_fs, hbo, n_fs, off)}


LC["run_pair"] = hook
BP["run_pair_bp"] = hook


def mendeley_people():
    for sid in M12["INCL"]:
        x, names, hbo, hbr, n_fs, q = M12["load_person"](M12["ALL_S"][sid])
        yield sid, [hook(e, 200.0, h, n_fs, 0.0) for e, h, _ in M12["segments"](x, hbo, hbr, n_fs, q["offset"])]


GENS = dict(BP["GENS"], mendeley=mendeley_people)


def summary(per, rng):
    keys = sorted(per["slow"])
    v, b = np.array([per["slow"][k] for k in keys]), np.array([per["bandpower"][k] for k in keys])
    d = v - b
    boot = lambda z: [float(q) for q in np.percentile([z[rng.integers(0, len(z), len(z))].mean() for _ in range(5000)], [2.5, 97.5])]
    out = {"slow": {"mean_r": float(v.mean()), "ci": boot(v), "p": signflip(v, rng), "n_positive": int((v > 0).sum())},
           "bandpower": {"mean_r": float(b.mean()), "ci": boot(b), "p": signflip(b, rng), "n_positive": int((b > 0).sum())},
           "slow_minus_bandpower": {"mean": float(d.mean()), "ci": boot(d), "p": signflip(d, rng)}, "n": len(keys), "people": per}
    out["pass"] = bool(out["slow"]["mean_r"] > 0 and out["slow"]["p"] < ALPHA and out["slow_minus_bandpower"]["ci"][0] > 0)
    return out


def continuous(D):
    per = {}
    for f, name in ((0, "slow"), (1, "bandpower")):
        v = {}
        for sid in D:
            X = np.concatenate([q["cont"][f] for o, rows in D.items() if o != sid for q in rows])
            Y = np.concatenate([q["cont"][2] for o, rows in D.items() if o != sid for q in rows])
            m = RidgeCV(alphas=ALPHAS_C).fit(X, Y)
            v[sid] = float(np.mean([corr(m.predict(q["cont"][f]), q["cont"][2]) for q in D[sid]]))
            del X, Y
        per[name] = v
        print("  continuous", name, round(float(np.mean(list(v.values()))), 4), flush=True)
    return per


def trial_corr(P, T):
    P, T = P - P.mean(1, keepdims=True), T - T.mean(1, keepdims=True)
    return ((P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)) * (T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12))).sum(1)


def windowed(D):
    W = {s: [q["win"] for q in rows if q["win"] is not None] for s, rows in D.items()}
    W = {s: r for s, r in W.items() if r}
    ids = np.concatenate([[s] * sum(len(q[2]) for q in r) for s, r in W.items()])
    Y = np.concatenate([q[2] for r in W.values() for q in r]).astype(np.float64)
    per = {}
    for f, name in ((0, "slow"), (1, "bandpower")):
        F = np.concatenate([q[f] for r in W.values() for q in r]).astype(np.float64)
        Yd = Y.copy()
        for s in W:
            m = ids == s
            F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
            Yd[m] -= Yd[m].mean(0)
        v = {}
        for s in W:
            te = ids == s
            Q = RidgeCV(alphas=ALPHAS_W).fit(F[~te], Yd[~te]).predict(F[te])
            v[s] = float(trial_corr(Q - Q.mean(0), Yd[te]).mean())
        per[name] = v
        print("  windowed", name, round(float(np.mean(list(v.values()))), 4), flush=True)
    return per, int(len(Y))


if ARGS != ["merge"]:
    for cohort in ARGS:
        D = {}
        for i, (pid, rows) in enumerate(GENS[cohort]()):
            if SMOKE and i >= SMOKE:
                break
            rows = [q for q in rows if q is not None]
            if len(rows) >= 2:
                D[pid] = rows
            print(cohort, pid, len(rows), flush=True)
        shape = lambda rows: (rows[0]["cont"][0].shape[1], rows[0]["cont"][1].shape[1], rows[0]["win"][2].shape[1])
        vals = [shape(r) for r in D.values()]
        mode = max(set(vals), key=vals.count)
        dropped = [p for p, r in D.items() if shape(r) != mode]
        D = {p: r for p, r in D.items() if p not in dropped}
        rng = np.random.default_rng(SEED)
        res = {"cohort": cohort, "dropped_channel_mismatch": dropped, "continuous": summary(continuous(D), rng)}
        pw, nw = windowed(D)
        res["windowed"] = summary(pw, rng)
        res["n_windows"] = nw
        name = f"nvc_method_check_{cohort}{'_smoke' if SMOKE else ''}.json"
        (HERE / name).write_text(json.dumps(res, indent=1))
        for k in ("continuous", "windowed"):
            q = res[k]
            print(cohort, k, "slow", round(q["slow"]["mean_r"], 4), q["slow"]["ci"], "p", q["slow"]["p"], "band", round(q["bandpower"]["mean_r"], 4), "diff", round(q["slow_minus_bandpower"]["mean"], 4), q["slow_minus_bandpower"]["ci"], "pass", q["pass"], flush=True)
    print("DONE")
    sys.exit(0)

res = {c: json.loads((HERE / f"nvc_method_check_{c}.json").read_text()) for c in ("hbci", "mental_workload", "finemi", "mendeley")}
ref = json.loads((HERE / "nvc_mendeley_founding.json").read_text())
check = max(abs(res["mendeley"]["continuous"]["people"][m][k] - ref["people"][m][k]) for m in ("slow", "bandpower") for k in ref["people"][m])
n_pass = sum(res[c]["continuous"]["pass"] for c in ("hbci", "mental_workload", "finemi"))
summ = {"mendeley_reproduces_P12.1_max_abs_diff": check, "P13.1_passes_of_3": n_pass, "P12.1_method_valid": n_pass >= 2, "alpha": ALPHA,
        "cohorts": {c: {k: {"slow": v[k]["slow"], "bandpower": v[k]["bandpower"], "diff": v[k]["slow_minus_bandpower"], "pass": v[k]["pass"], "n": v[k]["n"]} for k in ("continuous", "windowed")} for c, v in res.items()}}
(HERE / "nvc_method_check_results.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
print("DONE")
