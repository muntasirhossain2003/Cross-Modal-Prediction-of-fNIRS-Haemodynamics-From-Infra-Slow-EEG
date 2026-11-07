"""FINAL_PLAN P6.4b: P6.4 band-power coupling after removing the cue-locked average response (FIR 0-29 s) per run."""
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "shard" else (0, 1)
HERE = Path(__file__).parent
src = (HERE / "nvc_bandpower.py").read_text()
src = src[:src.index("GENS = {")]                    # helpers only (runs the continuous_lag loaders' set-up)
sys.argv = [sys.argv[0]]
ns = {"__name__": "nvc_res", "__file__": str(HERE / "nvc_bandpower.py")}
exec(src, ns)

import json

import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, wilcoxon

person, run_pair_bp, LAGS, SEED, BANDS = ns["person"], ns["run_pair_bp"], ns["LAGS"], ns["SEED"], ns["BANDS"]
inner = ns["ns"]                                   # the continuous_lag namespace
_runs, beer_lambert, split_wl, HBCI, MW = inner["_runs"], inner["beer_lambert"], inner["split_wl"], inner["HBCI"], inner["MW"]
EOG = inner["EOG"]
COHORTS_ENV = os.environ.get("COHORTS", "hbci,mental_workload,finemi").split(",")
OUT = HERE / "nvc_residual_results.json"
SHARDP = "nvc_res_smoke" if os.environ.get("SMOKE") else "nvc_res_shard"
NFIR = 30


def residualise(E, H, cues):
    """Remove the pooled cue-locked FIR response (0..NFIR-1 s) + intercept from every column of E and H."""
    T = len(E)
    X = np.zeros((T, NFIR + 1))
    X[:, -1] = 1.0
    for c in cues:
        for l in range(NFIR):
            if 0 <= c + l < T:
                X[c + l, l] += 1.0
    Z = np.hstack([E, H])
    beta = np.linalg.lstsq(X, Z, rcond=None)[0]
    R = Z - X @ beta
    return R[:, :E.shape[1]], R[:, E.shape[1]:]


def with_cues(pr, cues_s, start_s):
    """pr = (E, H) from run_pair_bp; cue times in EEG seconds; start_s = EEG time of E[0]."""
    if pr is None:
        return None
    idx = [int(round(t - start_s)) for t in cues_s]
    return residualise(pr[0], pr[1], [i for i in idx if -NFIR < i < len(pr[0])])


def hbci_people():
    eeg_dir, nirs_dir = HBCI / "EEG data", HBCI / "NRI data" / "NIRS_01-29"
    for s in range(1, 30):
        ed = next((g / f"subject {s:02d}" / "with occular artifact" for g in sorted(eeg_dir.glob("EEG_*"))
                   if (g / f"subject {s:02d}" / "with occular artifact").exists()), None)
        er_all, em_all = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
        nr_all, nm_all = _runs(nirs_dir / f"subject {s:02d}" / "cnt.mat", "cnt"), \
            _runs(nirs_dir / f"subject {s:02d}" / "mrk.mat", "mrk")
        pairs = []
        for r in range(min(er_all.size, nr_all.size, em_all.size, nm_all.size)):
            er, nr, em, nm = er_all[r], nr_all[r], em_all[r], nm_all[r]
            clab = [str(c) for c in np.atleast_1d(er.clab)]
            keep = [i for i, c in enumerate(clab) if c not in EOG]
            ex = np.asarray(er.x, float)[:, keep].T
            lo, hi = split_wl(np.asarray(nr.x, float), nr.clab)
            hbo, _ = beer_lambert(lo, hi)
            off = float(np.atleast_1d(nm.time)[0] - np.atleast_1d(em.time)[0]) / 1000.0
            pr = run_pair_bp(ex, float(np.atleast_1d(er.fs).ravel()[0]), np.asarray(hbo, float),
                             float(np.atleast_1d(nr.fs).ravel()[0]), off)
            a0 = max(0, -int(round(off)))
            pr = with_cues(pr, np.atleast_1d(em.time).astype(float) / 1000.0, a0)
            if pr is not None:
                pairs.append(pr)
        yield f"hbci_s{s:02d}", pairs


def mw_people():
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
        keep = [i for i, c in enumerate(clab) if c not in EOG]
        ex = np.asarray(ce.x, float)[:, keep].T
        hbo = np.asarray(cn.oxy.x, float).T
        e_fs, n_fs = float(ce.fs), float(cn.oxy.fs)
        pairs = []
        for r in np.unique(run):
            m = run == r
            a, b = te[m].min() - 20.0, te[m].max() + 60.0
            ia, ib = max(0, int(a * e_fs)), min(ex.shape[1], int(b * e_fs))
            o = float(np.median(off[m]))
            na, nb_ = max(0, int((a + o) * n_fs)), min(hbo.shape[1], int((b + o) * n_fs))
            seg_e, seg_n = ex[:, ia:ib], hbo[:, na:nb_]
            n = min(seg_e.shape[1] / e_fs, seg_n.shape[1] / n_fs)
            pr = run_pair_bp(seg_e[:, :int(n * e_fs)], e_fs, seg_n[:, :int(n * n_fs)], n_fs, 0.0)
            pr = with_cues(pr, te[m], ia / e_fs)
            if pr is not None:
                pairs.append(pr)
        yield f"mw_{vp}", pairs


def finemi_people():
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
                pr = with_cues(pr, te[m], ia / 200.0)
                if pr is not None:
                    pairs.append(pr)
            del x, e, n, hb
        yield f"finemi_s{s:02d}", pairs


GENS = {"hbci": hbci_people, "mental_workload": mw_people, "finemi": finemi_people}
if MODE in ("shard", "all"):
    k, n = SHARD
    sf = HERE / f"{SHARDP}_{k}_of_{n}.json"
    done = json.loads(sf.read_text()) if sf.exists() else {}
    for cohort in COHORTS_ENV:
        ci = ("hbci", "mental_workload", "finemi").index(cohort)
        for pi, (pid, pairs) in enumerate(GENS[cohort]()):
            if (ci * 100 + pi) % n != k or pid in done:
                continue
            if len(pairs) < 2:
                print(pid, "fewer than 2 usable runs"); continue
            v = person(pairs, np.random.default_rng([SEED, ci, pi]))
            done[pid] = dict(v, cohort=cohort)
            print(f"RES {pid}: local delay {v['delay_s']:+d} s r {v['peak_r']:.3f} p {v['p_null']:.3f} halves {v['half_delays_s']} "
                  f"| global delay {v['delay_global_s']:+d} s r {v['peak_r_global']:.3f}", flush=True)
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
    res[cohort] = {"people": people, "n": len(people), "frac_beating_null": float(np.mean(ps < 0.05)),
                   "median_delay_s": float(np.median(d)), "iqr_delay_s": [float(x) for x in np.percentile(d, [25, 75])],
                   "wilcoxon_p": w, "split_half_r": r_half, "split_half_perm_p": p_half,
                   "half_estimates_at_edge": int(np.sum(np.isin(h, [LAGS[0], LAGS[-1]]))),
                   "median_peak_r": float(np.median([v["peak_r"] for v in people.values()])),
                   "median_global_delay_s": float(np.median([v["delay_global_s"] for v in people.values()])),
                   "band_share": {b[0]: float(np.mean([v["band_share"][b[0]] for v in people.values()])) for b in BANDS},
                   "exists": bool(exists), "reproducible": bool(reproducible),
                   "claim3a_pass": bool(exists and reproducible), "physiological": physio}
    r = res[cohort]
    print(f"== RES {cohort}: n {r['n']}, beat null {r['frac_beating_null']:.2f}, median delay {r['median_delay_s']:+.1f} s "
          f"IQR {r['iqr_delay_s']} Wilcoxon p {w:.4f}; split-half r {r_half:+.3f} (perm p {p_half:.4f}, "
          f"{r['half_estimates_at_edge']} at edge); median peak r {r['median_peak_r']:.3f} -> Claim 3a "
          f"{'PASS' if r['claim3a_pass'] else 'FAIL'}, physiological {'YES' if physio else 'NO'}", flush=True)
res["coupling_beyond_task_supported"] = bool(res.get("hbci", {}).get("claim3a_pass") and any(
    res.get(c, {}).get("claim3a_pass") for c in ("mental_workload", "finemi")))
print("Coupling beyond the task:", "SUPPORTED" if res["coupling_beyond_task_supported"] else "NOT SUPPORTED")
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
