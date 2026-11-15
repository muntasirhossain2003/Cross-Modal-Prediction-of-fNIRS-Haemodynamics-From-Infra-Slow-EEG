"""FINAL_PLAN P12: confirmation on the Mendeley Learning Capability cohort (third lab). Nothing is tuned on it.
Modes, in order: "sync" | "coupling K N" | "founding" | "transfer" | "merge"."""
import json
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "merge"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "coupling" else (0, 1)
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "nvc_transfer2.py").read_text(encoding="utf-8")
src = src[:src.index("CAL_SEEDS = [")]
NS2 = {"__name__": "nvc_mend", "__file__": str(HERE / "nvc_transfer2.py")}
exec(src, NS2)

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.linear_model import RidgeCV

mne.set_log_level("ERROR")
BP = NS2["NS"]
LC = NS2["inner"]
to_1hz, bandpower_1hz, curve, person_p64, run_pair_bp, pair = LC["to_1hz"], BP["bandpower_1hz"], BP["curve"], BP["person"], BP["run_pair_bp"], NS2["pair"]
LAGS_C, SEED, CH, interpolation_matrix = BP["LAGS"], BP["SEED"], NS2["CH"], NS2["interpolation_matrix"]
score, calibrated, show, show_cal, ALPHAS, MINSHIFT = (NS2[k] for k in ("score", "calibrated", "show", "show_cal", "ALPHAS", "MINSHIFT"))
M = Path(r"G:\research_bio_medical\dataset\mendeley Simultaneous EEG-fNIRS Data on Learning Capability")
DROP = {"DC09", "EEG Mark1", "EEG Mark2", "Events/Markers", "E"}
SYNC = HERE / "nvc_mendeley_sync.json"
OUT = HERE / "nvc_mendeley_results.json"
ALPHA = 0.05 / 6
NK, NSEG, MIN_S = 20, 4, 900
P = {1: {1: 1, 3: 1, 4: -1, 7: 1, 8: -1, 10: -1, 11: -1, 12: -1, 13: 1, 14: 1, 16: -1, 18: 1, 21: -1, 23: 1}}
P[2] = dict(P[1])


def people():
    return [s for g in sorted(M.iterdir()) if g.is_dir() and "Learning" in g.name and "Capability" not in g.name for s in sorted(g.iterdir()) if s.is_dir()]


def read_probe(path):
    with open(path, errors="replace") as f:
        lines = f.readlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("Probe"))
    period = next((float(l.split(",")[1]) for l in lines[:head] if l.startswith("Sampling Period")), 0.1)
    d = pd.read_csv(path, skiprows=head, encoding_errors="replace", index_col=False)
    return d, 1.0 / period


def load_nirs(s):
    hbo, hbr, mark, fs = [], [], None, None
    for pr in (1, 2):
        ox, fs = read_probe(next((s / "fNIRS").glob(f"*Probe{pr}_Oxy.csv")))
        de, _ = read_probe(next((s / "fNIRS").glob(f"*Probe{pr}_Deoxy.csv")))
        if mark is None:
            mark = ox["Mark"].to_numpy(float)
        hbo += [ox[f"CH{c}"].to_numpy(float) * sg for c, sg in P[pr].items()]
        hbr += [de[f"CH{c}"].to_numpy(float) * sg for c, sg in P[pr].items()]
    n = min(map(len, hbo + hbr))
    return np.array([x[:n] for x in hbo]), np.array([x[:n] for x in hbr]), mark[:n], fs


def align(te, tn):
    best = None
    for d in range(-8, 9):
        a, b = te[max(0, d):], tn[max(0, -d):]
        n = min(len(a), len(b))
        if n < 20:
            continue
        diff = b[:n] - a[:n]
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med)))
        if best is None or mad < best["mad"]:
            good = np.abs(diff - med) < 0.3
            drift = float(np.polyval(np.polyfit(a[:n][good], diff[good], 1), [a[:n][-1], a[:n][0]]).ptp()) if good.sum() > 10 else np.inf
            best = {"shift": d, "offset": med, "mad": mad, "frac_within": float(good.mean()), "drift_s": drift, "n_events": n}
    return best


def load_person(s):
    raw = mne.io.read_raw_edf(next((s / "EEG").glob("Raw_*.edf")), preload=True)
    fs = raw.info["sfreq"]
    tr = raw.get_data(picks=["DC09"])[0]
    b = tr > (tr.max() + tr.min()) / 2
    te = np.flatnonzero(b[1:] & ~b[:-1]) / fs
    real = mne.io.read_raw_edf(next((s / "EEG").glob("Truncated_*.edf")), preload=False).ch_names
    assert len(real) == len(raw.ch_names)
    raw.rename_channels(dict(zip(raw.ch_names, real)))
    names = [c for c in raw.ch_names if c not in DROP]
    assert len(names) == 32, len(names)
    raw.pick(names).resample(200.0, npad="auto")
    hbo, hbr, mark, n_fs = load_nirs(s)
    tn = np.flatnonzero((mark != 0) & (np.r_[0, mark[:-1]] != mark)) / n_fs
    q = align(te, tn)
    return raw.get_data(), names, hbo, hbr, n_fs, q


def usable(q, eeg_s, nirs_s):
    overlap = min(eeg_s, nirs_s - q["offset"]) - max(0.0, -q["offset"])
    reasons = [r for r, bad in (("mad", q["mad"] > 0.15), ("match", q["frac_within"] < 0.9), ("drift", abs(q["drift_s"]) > 0.5), ("short", overlap < MIN_S)) if bad]
    return reasons, overlap


def segments(x, hbo, hbr, n_fs, off):
    """Equal consecutive segments, in the EEG (200 Hz) and NIRS sample grids."""
    a = max(0.0, -off)
    b = min(x.shape[1] / 200.0, hbo.shape[1] / n_fs - off)
    edges = np.linspace(a, b, NSEG + 1)
    for s0, s1 in zip(edges[:-1], edges[1:]):
        n_s = s1 - s0
        ia, na = int(s0 * 200), int((s0 + off) * n_fs)
        yield x[:, ia:ia + int(n_s * 200)], hbo[:, na:na + int(n_s * n_fs)], hbr[:, na:na + int(n_s * n_fs)]


if MODE == "sync":
    res = {}
    for s in people():
        try:
            raw = mne.io.read_raw_edf(next((s / "EEG").glob("Raw_*.edf")), preload=False)
            fs = raw.info["sfreq"]
            tr = raw.get_data(picks=["DC09"])[0]
            b = tr > (tr.max() + tr.min()) / 2
            te = np.flatnonzero(b[1:] & ~b[:-1]) / fs
            ox, n_fs = read_probe(next((s / "fNIRS").glob("*Probe1_Oxy.csv")))
            mark = ox["Mark"].to_numpy(float)
            tn = np.flatnonzero((mark != 0) & (np.r_[0, mark[:-1]] != mark)) / n_fs
            q = align(te, tn)
            reasons, overlap = usable(q, raw.n_times / fs, len(mark) / n_fs)
            res[s.name] = dict(q, group=s.parent.name, n_fs=n_fs, overlap_s=overlap, excluded=reasons, n_te=len(te), n_tn=len(tn))
        except Exception as e:
            res[s.name] = {"excluded": [f"unreadable {type(e).__name__}: {e}"], "group": s.parent.name}
        v = res[s.name]
        print(s.name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    SYNC.write_text(json.dumps(res, indent=1))
    print("included", sum(not v["excluded"] for v in res.values()), "of", len(res))
    print("DONE")
    sys.exit(0)

INCL = sorted(k for k, v in json.loads(SYNC.read_text()).items() if not v["excluded"])
ALL_S = {s.name: s for s in people()}


def signflip(d, rng, n=200000):
    d = np.asarray(d, float)
    return float((np.sum((rng.choice([-1, 1], size=(n, len(d))) * d).mean(1) >= d.mean()) + 1) / (n + 1))


def r(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def delays_halves(ca, cb):
    return int(LAGS_C[np.argmax(ca)]), int(LAGS_C[np.argmax(cb)])


if MODE == "coupling":
    k_, n_ = SHARD
    sf = HERE / f"nvc_mendeley_shard_{k_}_of_{n_}.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    for pi, sid in enumerate(INCL):
        if pi % n_ != k_ or sid in done:
            continue
        x, names, hbo, hbr, n_fs, q = load_person(ALL_S[sid])
        pairs = [pr for pr in (run_pair_bp(e, 200.0, h, n_fs, 0.0) for e, h, _ in segments(x, hbo, hbr, n_fs, q["offset"])) if pr is not None]
        v = person_p64(pairs, np.random.default_rng([SEED, 12, pi]))
        hA, hB = [(E[:len(E) // 2], H[:len(E) // 2]) for E, H in pairs], [(E[len(E) // 2:], H[len(E) // 2:]) for E, H in pairs]
        ca, cb = curve(hA, "local"), curve(hB, "local")
        g = np.random.default_rng([SEED, 13, pi])
        kk = float(pearsonr(ca, cb)[0])
        kn = [float(pearsonr(ca, curve([(E, np.roll(H, g.integers(MINSHIFT, max(MINSHIFT + 1, len(H) - MINSHIFT)), axis=0)) for E, H in hB], "local"))[0]) for _ in range(NK)]
        done[sid] = dict(v, k=kk, k_null=kn, d=kk - float(np.mean(kn)), curve_A=ca.round(4).tolist(), curve_B=cb.round(4).tolist())
        print(sid, f"beat p {v['p_null']:.3f} delay {v['delay_s']:+d} peak {v['peak_r']:.3f} k {kk:+.3f} d {done[sid]['d']:+.3f}", flush=True)
        sf.write_text(json.dumps(done, indent=1))
    print("SHARD DONE")
    sys.exit(0)

if MODE == "founding":
    LG = np.arange(-12, 13, 4)
    t0 = int(LG.max())

    def feats(E):
        E = (E - E.mean(0)) / (E.std(0) + 1e-8)
        return np.concatenate([E[t0 - L: len(E) - t0 - L] for L in LG], 1).astype(np.float32)

    D = {}
    for sid in INCL:
        x, names, hbo, hbr, n_fs, q = load_person(ALL_S[sid])
        rows = []
        for e, h, _ in segments(x, hbo, hbr, n_fs, q["offset"]):
            sl = to_1hz(e, 200.0, None, 0.5).T
            bp = bandpower_1hz(e, 200.0)
            g = to_1hz(h, n_fs, 0.01, 0.1).mean(0)
            n = min(len(sl), len(bp), len(g))
            y = g[:n]
            y = ((y - y.mean()) / (y.std() + 1e-8))[t0: n - t0].astype(np.float32)
            rows.append((feats(sl[:n]), feats(bp[:n]), y))
        D[sid] = rows
        print("loaded", sid, flush=True)
    res = {"n": len(D)}
    out = {}
    for f, name in ((0, "slow"), (1, "bandpower")):
        per = {}
        for sid in D:
            X = np.concatenate([r_[f] for q, rows in D.items() if q != sid for r_ in rows])
            Y = np.concatenate([r_[2] for q, rows in D.items() if q != sid for r_ in rows])
            m = RidgeCV(alphas=ALPHAS).fit(X, Y)
            per[sid] = float(np.mean([r(m.predict(r_[f]), r_[2]) for r_ in D[sid]]))
        out[name] = per
        print(name, "mean r", round(float(np.mean(list(per.values()))), 4), flush=True)
    rng = np.random.default_rng(SEED)
    v, b = np.array(list(out["slow"].values())), np.array([out["bandpower"][k] for k in out["slow"]])
    d = v - b
    boot = lambda z: [float(q) for q in np.percentile([z[rng.integers(0, len(z), len(z))].mean() for _ in range(5000)], [2.5, 97.5])]
    res.update({"slow": {"mean_r": float(v.mean()), "ci": boot(v), "p": signflip(v, rng), "n_positive": int((v > 0).sum())},
                "bandpower": {"mean_r": float(b.mean()), "ci": boot(b), "p": signflip(b, rng), "n_positive": int((b > 0).sum())},
                "slow_minus_bandpower": {"mean": float(d.mean()), "ci": boot(d), "p": signflip(d, rng)}, "people": out})
    res["pass"] = bool(res["slow"]["mean_r"] > 0 and res["slow"]["p"] < ALPHA and res["slow_minus_bandpower"]["ci"][0] > 0)
    (HERE / "nvc_mendeley_founding.json").write_text(json.dumps(res, indent=1))
    print({k: res[k] for k in ("slow", "bandpower", "slow_minus_bandpower", "pass")})
    print("DONE")
    sys.exit(0)

if MODE == "transfer":
    H = NS2["load"]("hbci")
    Xt = np.concatenate([x for runs in H.values() for x, _ in runs])
    yt = np.concatenate([ys["hbo"] for runs in H.values() for _, ys in runs])
    model = RidgeCV(alphas=ALPHAS).fit(Xt, yt)
    print("hBCI kernel alpha", model.alpha_, flush=True)
    del H, Xt, yt
    PPL = {}
    for sid in INCL:
        x, names, hbo, hbr, n_fs, q = load_person(ALL_S[sid])
        W = interpolation_matrix(names, CH["hbci"])
        runs = [pr for pr in (pair(e, 200.0, h, hr, n_fs, 0.0, W) for e, h, hr in segments(x, hbo, hbr, n_fs, q["offset"])) if pr is not None]
        if len(runs) >= 3:
            PPL[sid] = runs
        print("loaded", sid, len(runs), flush=True)
    rng = np.random.default_rng(SEED)
    fr = score(model, PPL, "hbo", rng)
    show("P12.5 frozen hBCI kernel -> Mendeley", fr)
    fr["signflip_p"] = signflip([q["r"] for q in fr["people"].values()], rng)
    fr["pass_all"] = bool(fr["pass"] and fr["signflip_p"] < ALPHA)
    ca = calibrated(model, PPL, np.random.default_rng(SEED))
    show_cal("P12.6 calibrated hBCI kernel -> Mendeley", ca)
    ca["signflip_p"] = signflip([q["personal_plus_external"] - q["personal"] for q in ca["people"].values()], rng)
    ca["pass_all"] = bool(ca["pass"] and ca["signflip_p"] < ALPHA)
    print("frozen sign-flip p", fr["signflip_p"], "calibrated sign-flip p", ca["signflip_p"], flush=True)
    (HERE / "nvc_mendeley_transfer.json").write_text(json.dumps({"frozen": fr, "calibrated": ca}, indent=1))
    print("DONE")
    sys.exit(0)

allp = {}
for f in sorted(HERE.glob("nvc_mendeley_shard_*_of_*.json")):
    allp.update(json.loads(f.read_text()))
rng = np.random.default_rng(SEED)
V = list(allp.values())
d = np.array([v["d"] for v in V])
w = float(wilcoxon(d, alternative="greater").pvalue)
sm = lambda c: np.array([np.asarray(c, float)[max(0, i - 2):i + 3].mean() for i in range(len(c))])
dA = np.array([LAGS_C[np.argmax(sm(v["curve_A"]))] for v in V])
dB = np.array([LAGS_C[np.argmax(sm(v["curve_B"]))] for v in V])
agree = float(np.mean(np.abs(dA - dB) <= 2))
chance = float(np.mean([np.mean(np.abs(dA[i] - np.delete(dB, i)) <= 2) for i in range(len(dA))]))
perm = np.array([np.mean(np.abs(dA - rng.permutation(dB)) <= 2) for _ in range(10000)])
p_a = float((np.sum(perm >= agree) + 1) / 10001)
frac = float(np.mean([v["p_null"] < 0.05 for v in V]))
res = {"n": len(V), "P12.2 coupling": {"frac_beating_null": frac, "median_delay_s": float(np.median([v["delay_s"] for v in V])), "iqr_delay_s": [float(q) for q in np.percentile([v["delay_s"] for v in V], [25, 75])],
                                       "median_peak_r": float(np.median([v["peak_r"] for v in V])), "pass": bool(frac > 0.5)},
       "P12.3 kernel": {"median_d": float(np.median(d)), "median_k": float(np.median([v["k"] for v in V])), "wilcoxon_p": w, "pass": bool(np.median(d) > 0 and w < ALPHA)},
       "P12.4 delay": {"agree_2s": agree, "chance": chance, "p_agree": p_a, "median_delay_s": float(np.median(np.concatenate([dA, dB]))), "pass": bool(agree > chance and p_a < ALPHA)}}
print(json.dumps(res, indent=1))
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
