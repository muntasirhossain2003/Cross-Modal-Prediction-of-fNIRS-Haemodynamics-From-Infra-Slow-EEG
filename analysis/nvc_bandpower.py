"""FINAL_PLAN P6.4 (Claim 3, classical form): EEG band-power envelopes -> local HbO, lagged ridge (positive = EEG earlier)."""
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"          # "shard K N" | "merge" | "all"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "shard" else (0, 1)
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "continuous_lag.py").read_text()
src = src[:src.index("rng = np.random.default_rng(SEED)\nres = {}")]       # loaders and helpers only
ns = {"__name__": "nvc", "__file__": str(HERE / "continuous_lag.py")}
exec(src, ns)

import json
import warnings

import numpy as np
import scipy.signal as sps
from scipy.stats import pearsonr, wilcoxon
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
COHORTS_ENV = os.environ.get("COHORTS", "hbci,mental_workload,finemi").split(",")
OUT = HERE / "nvc_bandpower_results.json"
SHARDP = "nvc_smoke" if os.environ.get("SMOKE") else "nvc_shard"
LAGS = np.arange(-10, 21)
NSURR, MINSHIFT, SEED = int(os.environ.get("NSURR", "100")), 60, 1337
ALPHAS = np.logspace(0, 4, 5)
BANDS = (("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 45))
to_1hz, MAXLAG = ns["to_1hz"], ns["MAXLAG"]
SOS_SLOW = sps.butter(4, [0.01, 0.1], btype="band", fs=1.0, output="sos")


def bandpower_1hz(x, fs):
    """(ch, T) raw EEG -> (n_s, ch * 4) log band power, 1 s means, band-passed 0.01-0.1 Hz at 1 Hz."""
    step = int(round(fs))
    n = x.shape[-1] // step
    out = []
    for _, lo, hi in BANDS:
        y = sps.sosfiltfilt(sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos"), x, axis=-1)
        p = np.log((y[:, :n * step] ** 2).reshape(x.shape[0], n, step).mean(-1) + 1e-20)
        out.append(sps.sosfiltfilt(SOS_SLOW, p, axis=-1))
    return np.concatenate(out, 0).T


def run_pair_bp(eeg, e_fs, hbo, n_fs, offset_s):
    e = bandpower_1hz(np.asarray(eeg, float), e_fs)
    h = to_1hz(hbo, n_fs, 0.01, 0.1).T
    sh = int(round(offset_s))
    a0, b0 = max(0, -sh), min(len(e), len(h) - sh)
    if b0 - a0 < 3 * MAXLAG + MINSHIFT:
        return None
    return e[a0:b0], h[a0 + sh:b0 + sh]


ns["run_pair"] = run_pair_bp                    # the cohort generators look it up at call time


def lagged(E, H, L):
    if L >= 0:
        return E[:len(E) - L], H[L:]
    return E[-L:], H[:len(H) + L]


def colcorr(P, T):
    P, T = P.reshape(len(P), -1), T.reshape(len(T), -1)
    P = P - P.mean(0)
    T = T - T.mean(0)
    return float(np.mean((P * T).sum(0) / (np.linalg.norm(P, axis=0) * np.linalg.norm(T, axis=0) + 1e-12)))


def prep(pairs, target):
    out = []
    for E, H in pairs:
        E = (E - E.mean(0)) / (E.std(0) + 1e-8)
        g = H.mean(1, keepdims=True)
        out.append((E, H - g if target == "local" else g))
    return out


def curve(pairs, target):
    pp = prep(pairs, target)
    out = []
    for L in LAGS:
        lp = [lagged(E, Y, L) for E, Y in pp]
        rs = []
        for i in range(len(lp)):
            Xtr = np.concatenate([x for j, (x, _) in enumerate(lp) if j != i])
            Ytr = np.concatenate([y for j, (_, y) in enumerate(lp) if j != i])
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr)
            rs.append(colcorr(m.predict(lp[i][0]), lp[i][1]))
        out.append(np.mean(rs))
    return np.array(out)


def band_share(pairs, L):
    lp = [lagged(E, Y, L) for E, Y in prep(pairs, "local")]
    m = RidgeCV(alphas=ALPHAS).fit(np.concatenate([x for x, _ in lp]), np.concatenate([y for _, y in lp]))
    w = np.abs(m.coef_.reshape(-1, len(BANDS), lp[0][0].shape[1] // len(BANDS))).sum((0, 2))
    return {b[0]: float(v) for b, v in zip(BANDS, w / w.sum())}


def person(pairs, rng):
    c = curve(pairs, "local")
    delay, peak = int(LAGS[np.argmax(c)]), float(c.max())
    null = []
    for _ in range(NSURR):
        sp = [(E, np.roll(H, rng.integers(MINSHIFT, max(MINSHIFT + 1, len(H) - MINSHIFT)), axis=0)) for E, H in pairs]
        null.append(curve(sp, "local").max())
    p = float((np.sum(np.array(null) >= peak) + 1) / (NSURR + 1))
    halves = []
    for h in (0, 1):
        hp = []
        for E, H in pairs:
            n = len(E) // 2
            sl = slice(0, n) if h == 0 else slice(n, None)
            hp.append((E[sl], H[sl]))
        halves.append(int(LAGS[np.argmax(curve(hp, "local"))]))
    cg = curve(pairs, "global")
    return {"delay_s": delay, "peak_r": peak, "p_null": p, "half_delays_s": halves, "curve_local": c.round(4).tolist(),
            "delay_global_s": int(LAGS[np.argmax(cg)]), "peak_r_global": float(cg.max()),
            "curve_global": cg.round(4).tolist(), "band_share": band_share(pairs, delay), "n_runs": len(pairs)}


def finemi_people():
    """Raw FineMI CNT + NIRx, native reference, the P4 run split (gap > 40 s)."""
    import mne
    mne.set_log_level("ERROR")
    root = Path(r"G:\research_bio_medical\dataset\Multi-joint Upper-Limb EEG+FNIR\FineMI\FineMI")
    drop = {"M1", "M2", "CB1", "CB2", "EKG", "EMG", "HEO", "VEO"}

    def events(raw):
        a = raw.annotations
        keep = [i for i, d in enumerate(a.description) if d.replace(".0", "").strip().isdigit()]
        return np.asarray(a.onset)[keep], [int(float(a.description[i])) for i in keep]

    for s in sorted(int(p.name[7:]) for p in root.glob("subject*") if p.is_dir()):
        pairs = []
        for name in sorted(p.stem for p in (root / f"subject{s}" / "EEG").glob("*.cnt")):
            try:
                e = mne.io.read_raw_cnt(root / f"subject{s}" / "EEG" / f"{name}.cnt", preload=True)
                te, le = events(e)
                e.pick([c for c in e.ch_names if c not in drop])
                e.resample(200.0, npad="auto")
                x = e.get_data()
                n = mne.io.read_raw_nirx(root / f"subject{s}" / "fNIRS" / name, preload=True)
                tn, ln = events(n)
                hb = mne.preprocessing.nirs.beer_lambert_law(mne.preprocessing.nirs.optical_density(n), ppf=6.0)
                hbo = hb.get_data(picks=[i for i, c in enumerate(hb.ch_names) if c.endswith("hbo")])
                n_fs = float(n.info["sfreq"])
            except Exception as exc:
                print(f"finemi s{s} {name}: unreadable {type(exc).__name__}"); continue
            if le != ln:
                continue
            run = np.concatenate([[0], np.cumsum(np.diff(te) > 40.0)])
            off = tn - te
            for r in np.unique(run):
                m = run == r
                if np.ptp(off[m]) > 1.0:
                    continue
                o = float(np.median(off[m]))
                a, bb = max(0.0, te[m].min() - 20.0), te[m].max() + 60.0
                ia, ib = int(a * 200), min(x.shape[1], int(bb * 200))
                na, nb_ = max(0, int((a + o) * n_fs)), min(hbo.shape[1], int((bb + o) * n_fs))
                n_s = min((ib - ia) / 200.0, (nb_ - na) / n_fs)
                pr = run_pair_bp(x[:, ia:ia + int(n_s * 200)], 200.0, hbo[:, na:na + int(n_s * n_fs)], n_fs, 0.0)
                if pr is not None:
                    pairs.append(pr)
            del x, e, n, hb
        yield f"finemi_s{s:02d}", pairs


GENS = {"hbci": ns["hbci_people"], "mental_workload": ns["mw_people"], "finemi": finemi_people}
COHORTS = tuple((c, GENS[c]) for c in COHORTS_ENV)

if MODE in ("shard", "all"):
    k, n = SHARD
    sf = HERE / f"{SHARDP}_{k}_of_{n}.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    for ci, (cohort, gen) in enumerate(COHORTS):
        ci = ("hbci", "mental_workload", "finemi").index(cohort)
        for pi, (pid, pairs) in enumerate(gen()):
            if (ci * 100 + pi) % n != k or pid in done:
                continue
            if len(pairs) < 2:
                print(pid, "fewer than 2 usable runs"); continue
            v = person(pairs, np.random.default_rng([SEED, ci, pi]))
            done[pid] = dict(v, cohort=cohort)
            print(f"{pid}: local delay {v['delay_s']:+d} s r {v['peak_r']:.3f} p {v['p_null']:.3f} halves {v['half_delays_s']} "
                  f"| global delay {v['delay_global_s']:+d} s r {v['peak_r_global']:.3f} | bands "
                  + " ".join(f"{b} {w:.2f}" for b, w in v["band_share"].items()), flush=True)
            sf.write_text(json.dumps(done, indent=1))
    if MODE == "shard":
        print("SHARD DONE"); sys.exit(0)

allp = {}
for f in sorted(HERE.glob(f"{SHARDP}_*_of_*.json")):
    allp.update(json.loads(f.read_text()))
rng = np.random.default_rng(SEED)
res = {}
for cohort in ("hbci", "mental_workload", "finemi"):
    people = {p: v for p, v in allp.items() if v["cohort"] == cohort}
    if not people:
        continue
    d = np.array([v["delay_s"] for v in people.values()])
    h = np.array([v["half_delays_s"] for v in people.values()], float)
    ps = np.array([v["p_null"] for v in people.values()])
    r_half = float(pearsonr(h[:, 0], h[:, 1])[0]) if h[:, 0].std() > 0 and h[:, 1].std() > 0 else 0.0
    perm = np.array([pearsonr(h[:, 0], rng.permutation(h[:, 1]))[0] for _ in range(10000)])
    p_half = float((np.sum(perm >= r_half) + 1) / 10001)
    try:
        w = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w = 1.0
    exists = float(np.mean(ps < 0.05)) > 0.5
    reproducible = r_half > 0 and p_half < 0.05
    physio = bool(2 <= np.median(d) <= 10 and w < 0.05)
    edge = int(np.sum(np.isin(h, [LAGS[0], LAGS[-1]])))
    bands = {b[0]: float(np.mean([v["band_share"][b[0]] for v in people.values()])) for b in BANDS}
    res[cohort] = {"people": people, "n": len(people), "frac_beating_null": float(np.mean(ps < 0.05)),
                   "median_delay_s": float(np.median(d)), "iqr_delay_s": [float(x) for x in np.percentile(d, [25, 75])],
                   "wilcoxon_p": w, "split_half_r": r_half, "split_half_perm_p": p_half, "half_estimates_at_edge": edge,
                   "median_global_delay_s": float(np.median([v["delay_global_s"] for v in people.values()])),
                   "median_peak_r": float(np.median([v["peak_r"] for v in people.values()])),
                   "band_share": bands, "exists": bool(exists), "reproducible": bool(reproducible),
                   "claim3a_pass": bool(exists and reproducible), "physiological": physio}
    r = res[cohort]
    print(f"== {cohort}: n {r['n']}, beat null {r['frac_beating_null']:.2f}, median delay {r['median_delay_s']:+.1f} s "
          f"IQR {r['iqr_delay_s']} Wilcoxon p {w:.4f}; split-half r {r_half:+.3f} (perm p {p_half:.4f}, {edge} at edge); "
          f"median peak r {r['median_peak_r']:.3f}; global median {r['median_global_delay_s']:+.1f} s; bands {bands} "
          f"-> Claim 3a {'PASS' if r['claim3a_pass'] else 'FAIL'}, physiological {'YES' if physio else 'NO'}", flush=True)
res["coupling_classical_supported"] = bool(res.get("hbci", {}).get("claim3a_pass") and any(
    res.get(c, {}).get("claim3a_pass") for c in ("mental_workload", "finemi")))
print("Classical coupling claim:", "SUPPORTED" if res["coupling_classical_supported"] else "NOT SUPPORTED")
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
