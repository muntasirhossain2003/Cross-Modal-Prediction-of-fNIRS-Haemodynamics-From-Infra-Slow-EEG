"""FINAL_PLAN P8.2 / P8.3 (Claim 2): frozen hBCI band-power kernel with a CBSI target (P8.2), and calibrated
transfer: does the frozen kernel add to a person's own two-run model (P8.3)?"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]   # repository root

sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "nvc_bandpower.py").read_text()
src = src[:src.index("COHORTS = tuple(")]
NS = {"__name__": "nvc_tr2", "__file__": str(HERE / "nvc_bandpower.py")}
exec(src, NS)

import json

import nbformat
import numpy as np
import scipy.io as sio
from sklearn.linear_model import RidgeCV

inner = NS["ns"]
nbns = inner["ns"]
nb = nbformat.read(inner["NB"], as_version=4)
cwd = os.getcwd(); os.chdir(inner["NB"].parent)
exec("\n".join(l for l in nb.cells[19].source.splitlines() if not l.lstrip().startswith(("%", "!"))), nbns)
os.chdir(cwd)
interpolation_matrix = nbns["interpolation_matrix"]
bandpower_1hz, to_1hz = NS["bandpower_1hz"], inner["to_1hz"]
_runs, beer_lambert, split_wl, HBCI, MW, EOG = (inner[k] for k in ("_runs", "beer_lambert", "split_wl", "HBCI", "MW", "EOG"))
MAXLAG = inner["MAXLAG"]

ROOT = inner["NB"].parent / "neurovascnet_merge_v1_cache"
META = {"hbci": "train_fnirs_a2133f5d6a71ec34", "mental_workload": "replication_tuberlin_mw", "finemi": "replication_finemi"}
CH = {}
for c, d in META.items():
    m = json.loads((ROOT / d / "meta.json").read_text())
    CH[c] = m["info"]["ch_names"] if "info" in m else m["ch_names"]
SEED, NNULL, MINSHIFT = 1337, 200, 60
LAGS = np.arange(0, 15, 2)
ALPHAS = np.logspace(0, 6, 7)
OUT = HERE / "nvc_transfer2_results.json"


def pair(eeg, e_fs, hbo, hbr, n_fs, off, W):
    """-> (X, {"hbo": y, "cbsi": y}) for one run, or None."""
    eeg = np.asarray(eeg, float)
    E = bandpower_1hz(eeg if W is None else W @ eeg, e_fs)
    ho, hr = to_1hz(np.asarray(hbo, float), n_fs, 0.01, 0.1).T, to_1hz(np.asarray(hbr, float), n_fs, 0.01, 0.1).T
    sh = int(round(off))
    a0, b0 = max(0, -sh), min(len(E), len(ho) - sh)
    if b0 - a0 < 3 * MAXLAG + MINSHIFT:
        return None
    E, ho, hr = E[a0:b0], ho[a0 + sh:b0 + sh], hr[a0 + sh:b0 + sh]
    alpha = ho.std(0) / (hr.std(0) + 1e-12)
    ys = {"hbo": ho.mean(1), "cbsi": ((ho - alpha * hr) / 2).mean(1)}
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    t0 = LAGS.max()
    X = np.concatenate([E[t0 - L:len(E) - L] for L in LAGS], 1).astype(np.float32)
    return X, {k: ((y - y.mean()) / (y.std() + 1e-8))[t0:].astype(np.float32) for k, y in ys.items()}


def hbci_people(W):
    eeg_dir, nirs_dir = HBCI / "EEG data", HBCI / "NRI data" / "NIRS_01-29"
    for s in range(1, 30):
        ed = next((g / f"subject {s:02d}" / "with occular artifact" for g in sorted(eeg_dir.glob("EEG_*"))
                   if (g / f"subject {s:02d}" / "with occular artifact").exists()), None)
        er_all, em_all = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
        nr_all, nm_all = _runs(nirs_dir / f"subject {s:02d}" / "cnt.mat", "cnt"), _runs(nirs_dir / f"subject {s:02d}" / "mrk.mat", "mrk")
        runs = []
        for r in range(min(er_all.size, nr_all.size, em_all.size, nm_all.size)):
            er, nr, em, nm = er_all[r], nr_all[r], em_all[r], nm_all[r]
            clab = [str(c) for c in np.atleast_1d(er.clab)]
            ex = np.asarray(er.x, float)[:, [i for i, c in enumerate(clab) if c not in EOG]].T
            lo, hi = split_wl(np.asarray(nr.x, float), nr.clab)
            hbo, hbr = beer_lambert(lo, hi)
            off = float(np.atleast_1d(nm.time)[0] - np.atleast_1d(em.time)[0]) / 1000.0
            pr = pair(ex, float(np.atleast_1d(er.fs).ravel()[0]), hbo, hbr, float(np.atleast_1d(nr.fs).ravel()[0]), off, W)
            if pr is not None:
                runs.append(pr)
        yield f"hbci_s{s:02d}", runs


def mw_people(W):
    L = lambda p, k: sio.loadmat(p, squeeze_me=True, struct_as_record=False)[k]
    for s in range(1, 27):
        vp = f"VP{s:03d}"
        ce, me = L(MW / "EEG" / f"{vp}-EEG" / "cnt_nback.mat", "cnt_nback"), L(MW / "EEG" / f"{vp}-EEG" / "mrk_nback.mat", "mrk_nback")
        cn, mn = L(MW / "NIRS" / f"{vp}-NIRS" / "cnt_nback.mat", "cnt_nback"), L(MW / "NIRS" / f"{vp}-NIRS" / "mrk_nback.mat", "mrk_nback")
        names = [str(c) for c in np.atleast_1d(me.className)]
        ye = np.asarray(me.y)
        sess = [i for i, c in enumerate(names) if "session" in c]
        te = np.asarray(me.time, float)[ye[sess].sum(0) > 0] / 1000.0
        tn = np.asarray(mn.time, float) / 1000.0
        off = tn - te
        run = np.concatenate([[0], np.cumsum(np.abs(np.diff(off)) > 1.0)])
        clab = [str(c) for c in np.atleast_1d(ce.clab)]
        ex = np.asarray(ce.x, float)[:, [i for i, c in enumerate(clab) if c not in EOG]].T
        hbo, hbr = np.asarray(cn.oxy.x, float).T, np.asarray(cn.deoxy.x, float).T
        e_fs, n_fs = float(ce.fs), float(cn.oxy.fs)
        runs = []
        for r in np.unique(run):
            m = run == r
            a, b = te[m].min() - 20.0, te[m].max() + 60.0
            ia, ib = max(0, int(a * e_fs)), min(ex.shape[1], int(b * e_fs))
            o = float(np.median(off[m]))
            na, nb_ = max(0, int((a + o) * n_fs)), min(hbo.shape[1], int((b + o) * n_fs))
            n = min((ib - ia) / e_fs, (nb_ - na) / n_fs)
            pr = pair(ex[:, ia:ia + int(n * e_fs)], e_fs, hbo[:, na:na + int(n * n_fs)], hbr[:, na:na + int(n * n_fs)], n_fs, 0.0, W)
            if pr is not None:
                runs.append(pr)
        yield f"mw_{vp}", runs


def finemi_people(W):
    import mne
    mne.set_log_level("ERROR")
    root = Path(str(REPO / "dataset/Multi-joint Upper-Limb EEG+FNIR/FineMI/FineMI"))
    drop = {"M1", "M2", "CB1", "CB2", "EKG", "EMG", "HEO", "VEO"}

    def events(raw):
        a = raw.annotations
        keep = [i for i, d in enumerate(a.description) if d.replace(".0", "").strip().isdigit()]
        return np.asarray(a.onset)[keep], [int(float(a.description[i])) for i in keep]

    for s in sorted(int(p.name[7:]) for p in root.glob("subject*") if p.is_dir()):
        runs = []
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
                hbr = hb.get_data(picks=[i for i, c in enumerate(hb.ch_names) if c.endswith("hbr")])
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
                pr = pair(x[:, ia:ia + int(n_s * 200)], 200.0, hbo[:, na:na + int(n_s * n_fs)],
                          hbr[:, na:na + int(n_s * n_fs)], n_fs, 0.0, W)
                if pr is not None:
                    runs.append(pr)
            del x, e, n, hb
        yield f"finemi_s{s:02d}", runs


GENS = {"hbci": hbci_people, "mental_workload": mw_people, "finemi": finemi_people}


def load(cohort):
    W = None if cohort == "hbci" else interpolation_matrix(CH[cohort], CH["hbci"])
    people = {}
    for pid, runs in GENS[cohort](W):
        if runs:
            people[pid] = runs
            print(f"loaded {pid}: {len(runs)} runs", flush=True)
    return people


def r(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def score(model, people, tgt, rng):
    per = {}
    for pid, runs in people.items():
        pr = [(model.predict(X), ys[tgt]) for X, ys in runs]
        obs = float(np.mean([r(a, b) for a, b in pr]))
        null = np.array([np.mean([r(a, np.roll(b, rng.integers(MINSHIFT, max(MINSHIFT + 1, len(b) - MINSHIFT))))
                                  for a, b in pr]) for _ in range(NNULL)])
        per[pid] = {"r": obs, "p_null": float((np.sum(null >= obs) + 1) / (NNULL + 1))}
    v = np.array([q["r"] for q in per.values()])
    boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    beat = float(np.mean([q["p_null"] < 0.05 for q in per.values()]))
    return {"mean_r": float(v.mean()), "ci": ci, "n_positive": int(np.sum(v > 0)), "n": len(v),
            "frac_beating_null": beat, "pass": bool(v.mean() > 0 and ci[0] > 0 and beat > 0.5), "people": per}


def calibrated(model, people, rng):
    """P8.3: personal two-run model vs personal + frozen external, stacking gains fitted on the calibration runs."""
    per = {}
    for pid, runs in people.items():
        if len(runs) < 3:
            continue
        cal = sorted(rng.choice(len(runs), 2, replace=False).tolist())
        te = [i for i in range(len(runs)) if i not in cal]
        Xc = [runs[i][0] for i in cal]
        yc = [runs[i][1]["hbo"] for i in cal]
        pers_cf = [RidgeCV(alphas=ALPHAS).fit(Xc[1 - j], yc[1 - j]).predict(Xc[j]) for j in (0, 1)]
        ext_c = [model.predict(x) for x in Xc]
        Y = np.concatenate(yc)
        P, E = np.concatenate(pers_cf), np.concatenate(ext_c)
        g_p = np.linalg.lstsq(P[:, None], Y, rcond=None)[0]
        g_b = np.linalg.lstsq(np.stack([P, E], 1), Y, rcond=None)[0]
        g_e = np.linalg.lstsq(E[:, None], Y, rcond=None)[0]
        pm = RidgeCV(alphas=ALPHAS).fit(np.concatenate(Xc), Y)
        rp, rb, re_ = [], [], []
        for i in te:
            X, y = runs[i][0], runs[i][1]["hbo"]
            pt, et = pm.predict(X), model.predict(X)
            rp.append(r(g_p[0] * pt, y))
            rb.append(r(g_b[0] * pt + g_b[1] * et, y))
            re_.append(r(g_e[0] * et, y))
        per[pid] = {"personal": float(np.mean(rp)), "personal_plus_external": float(np.mean(rb)),
                    "external_calibrated": float(np.mean(re_)), "external_gain": float(g_b[1])}
    d = np.array([v["personal_plus_external"] - v["personal"] for v in per.values()])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    out = {k: float(np.mean([v[k] for v in per.values()])) for k in ("personal", "personal_plus_external", "external_calibrated")}
    out.update({"diff": float(d.mean()), "ci": ci, "n_better": int(np.sum(d > 0)), "n": len(d),
                "frac_negative_external_gain": float(np.mean([v["external_gain"] < 0 for v in per.values()])),
                "pass": bool(d.mean() > 0 and ci[0] > 0), "people": per})
    return out


def show(name, s):
    print(f"== {name}: mean r {s['mean_r']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] ({s['n_positive']}/{s['n']} > 0), "
          f"beat null {s['frac_beating_null']:.2f} -> {'PASS' if s['pass'] else 'FAIL'}", flush=True)


def show_cal(name, s):
    print(f"== {name}: personal {s['personal']:+.4f} personal+external {s['personal_plus_external']:+.4f} "
          f"external calibrated {s['external_calibrated']:+.4f} | diff {s['diff']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] "
          f"({s['n_better']}/{s['n']}), negative external gain {s['frac_negative_external_gain']:.2f} "
          f"-> {'PASS' if s['pass'] else 'FAIL'}", flush=True)


CAL_SEEDS = [int(s) for s in os.environ.get("CAL_SEEDS", "").split(",") if s]    # P8.3b
if CAL_SEEDS:
    OUT = HERE / "nvc_transfer2_seeds_results.json"
rng = np.random.default_rng(SEED)
res = {}
H = load("hbci")
models = {}
for tgt in (("hbo",) if CAL_SEEDS else ("hbo", "cbsi")):
    X = np.concatenate([x for runs in H.values() for x, _ in runs])
    y = np.concatenate([ys[tgt] for runs in H.values() for _, ys in runs])
    models[tgt] = RidgeCV(alphas=ALPHAS).fit(X, y)
    print(f"hBCI {tgt} kernel: {len(y)} s, alpha {models[tgt].alpha_:g}", flush=True)
    del X, y
del H
if CAL_SEEDS:
    for cohort in ("mental_workload", "finemi"):
        P = load(cohort)
        for sd in CAL_SEEDS:
            res[f"calibrated | {cohort} | seed {sd}"] = calibrated(models["hbo"], P, np.random.default_rng(sd))
            show_cal(f"P8.3b seed {sd} calibrated transfer into {cohort}", res[f"calibrated | {cohort} | seed {sd}"])
        del P
        OUT.write_text(json.dumps(res, indent=1))
    n_pass = sum(res[f"calibrated | finemi | seed {sd}"]["pass"] for sd in CAL_SEEDS)
    print(f"P8.3b FineMI passes {n_pass}/{len(CAL_SEEDS)} seeds -> {'ROBUST' if n_pass >= 4 else 'NOT ROBUST'}")
    print("DONE")
    sys.exit(0)
for cohort in ("mental_workload", "finemi"):
    P = load(cohort)
    for tgt in ("hbo", "cbsi"):
        res[f"hbci_to_{cohort} | {tgt}"] = score(models[tgt], P, tgt, rng)
        show(f"P8.2 hBCI -> {cohort} [{tgt}]", res[f"hbci_to_{cohort} | {tgt}"])
    res[f"calibrated | {cohort}"] = calibrated(models["hbo"], P, np.random.default_rng(SEED))
    show_cal(f"P8.3 calibrated transfer into {cohort}", res[f"calibrated | {cohort}"])
    del P
    OUT.write_text(json.dumps(res, indent=1))
print("Claim 2 P8.2 (CBSI, cross-lab):", "PASS" if res["hbci_to_finemi | cbsi"]["pass"] else "FAIL")
print("Claim 2 P8.3 (calibrated, cross-lab):", "PASS" if res["calibrated | finemi"]["pass"] else "FAIL")
print("DONE")
