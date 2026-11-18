"""FINAL_PLAN P2.3: lag between infra-slow EEG and HbO on continuous recordings (positive = EEG earlier)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
from pathlib import Path

import nbformat
import numpy as np
import scipy.io as sio
import scipy.signal as sps
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]   # repository root

NB = Path(str(REPO / "pipeline/CrossModal_InfraSlowEEG_fNIRS_fMRI.ipynb"))
MW = Path(str(REPO / "dataset/TU Berlin EEG + NIRS"))
OUT = Path(__file__).with_name("continuous_lag_results.json")
MAXLAG, MINSHIFT, NSURR, SEED = 20, 60, 200, 1337
EOG = ("VEOG", "HEOG")

nb = nbformat.read(NB, as_version=4)
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)
_runs, beer_lambert, split_wl = ns["_runs"], ns["beer_lambert"], ns["_split_wavelengths"]
HBCI = ns["PATHS"]["train_fnirs"]


def to_1hz(x, fs, lo, hi):
    """(ch, T) -> band-limited, 1 Hz."""
    if lo is None:
        sos = sps.butter(4, hi, btype="low", fs=fs, output="sos")
    else:
        sos = sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    y = sps.sosfiltfilt(sos, x, axis=-1)
    step = int(round(fs))
    n = y.shape[-1] // step
    return y[:, :n * step].reshape(y.shape[0], n, step).mean(-1)


def run_pair(eeg, e_fs, hbo, n_fs, offset_s):
    """EEG (ch, T) and HbO (ch, T) of one run; NIRS time = EEG time + offset. Returns aligned 1 Hz series."""
    e = to_1hz(eeg, e_fs, None, 0.1)
    e = e - e.mean(1, keepdims=True)
    u, s, vt = np.linalg.svd(e, full_matrices=False)
    pc = vt[0] * s[0]
    h = to_1hz(hbo, n_fs, 0.01, 0.1).mean(0)
    sh = int(round(offset_s))               # EEG second k <-> NIRS second k + sh
    a0, b0 = max(0, -sh), min(len(pc), len(h) - sh)
    if b0 - a0 < 3 * MAXLAG + MINSHIFT:
        return None
    return pc[a0:b0], h[a0 + sh:b0 + sh]


def xcorr(a, b):
    """r at lags -MAXLAG..MAXLAG; positive lag = a (EEG) earlier than b (HbO)."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    out = []
    for L in range(-MAXLAG, MAXLAG + 1):
        if L >= 0:
            x, y = a[:len(a) - L], b[L:]
        else:
            x, y = a[-L:], b[:len(b) + L]
        out.append(np.corrcoef(x, y)[0, 1])
    return np.abs(np.array(out))


def person_stats(pairs, rng):
    curves = np.array([xcorr(a, b) for a, b in pairs])
    curve = curves.mean(0)
    lag = int(np.argmax(curve)) - MAXLAG
    peak = curve.max()
    null = []
    for _ in range(NSURR):
        c = []
        for a, b in pairs:
            k = rng.integers(MINSHIFT, max(MINSHIFT + 1, len(b) - MINSHIFT))
            c.append(xcorr(a, np.roll(b, k)))
        null.append(np.mean(c, 0).max())
    p = float((np.sum(np.array(null) >= peak) + 1) / (NSURR + 1))
    return {"lag_s": lag, "peak_abs_r": float(peak), "p_null": p, "curve": curve.round(4).tolist(),
            "n_runs": len(pairs)}


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
            pr = run_pair(ex, float(np.atleast_1d(er.fs).ravel()[0]), np.asarray(hbo, float),
                          float(np.atleast_1d(nr.fs).ravel()[0]), off)
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
        assert len(te) == len(tn), vp
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
            a, b = te[m].min() - 20.0, te[m].max() + 60.0          # this run's stretch of the EEG timeline
            ia, ib = max(0, int(a * e_fs)), min(ex.shape[1], int(b * e_fs))
            o = float(np.median(off[m]))
            na, nb_ = max(0, int((a + o) * n_fs)), min(hbo.shape[1], int((b + o) * n_fs))
            seg_e, seg_n = ex[:, ia:ib], hbo[:, na:nb_]
            n = min(seg_e.shape[1] / e_fs, seg_n.shape[1] / n_fs)
            pr = run_pair(seg_e[:, :int(n * e_fs)], e_fs, seg_n[:, :int(n * n_fs)], n_fs, 0.0)
            if pr is not None:
                pairs.append(pr)
        yield f"mw_{vp}", pairs


rng = np.random.default_rng(SEED)
res = {}
for cohort, gen in (("hbci", hbci_people), ("mental_workload", mw_people)):
    people = {}
    for pid, pairs in gen():
        if not pairs:
            print(pid, "no usable runs"); continue
        people[pid] = person_stats(pairs, rng)
        print(f"{pid}: lag {people[pid]['lag_s']:+d} s |r| {people[pid]['peak_abs_r']:.3f} p {people[pid]['p_null']:.3f} "
              f"({people[pid]['n_runs']} runs)", flush=True)
    lags = np.array([v["lag_s"] for v in people.values()])
    ps = np.array([v["p_null"] for v in people.values()])
    try:
        w = float(wilcoxon(lags).pvalue) if np.any(lags != 0) else 1.0
    except ValueError:
        w = 1.0
    grand = np.mean([v["curve"] for v in people.values()], 0)
    res[cohort] = {"people": people, "n": len(people), "frac_beating_null": float(np.mean(ps < 0.05)),
                   "median_lag_s": float(np.median(lags)), "iqr_lag_s": [float(x) for x in np.percentile(lags, [25, 75])],
                   "wilcoxon_p_vs_0": w, "grand_curve": grand.round(4).tolist(),
                   "grand_peak_lag_s": int(np.argmax(grand)) - MAXLAG}
    r = res[cohort]
    print(f"== {cohort}: n {r['n']}, beat null {r['frac_beating_null']:.2f}, median lag {r['median_lag_s']:+.1f} s "
          f"IQR {r['iqr_lag_s']}, Wilcoxon p {w:.4f}, grand-curve peak {r['grand_peak_lag_s']:+d} s", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
print("DONE")
