"""FINAL_PLAN R1-R3: replication on TU Berlin mental workload (26 people, not the hBCI people).

Step 1 builds trials with the notebook's own helpers (same window/resampling/z-score/band-pass/baseline
as load_hbci) and caches them. Step 2 runs the ridge tests.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import json
import time
from pathlib import Path

import nbformat
import numpy as np
import scipy.io as sio
import scipy.signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

NB = Path(r"G:\research_bio_medical\merge\Neurovascnet_marge_v1.ipynb")
ROOT = Path(r"G:\research_bio_medical\dataset\TU Berlin EEG + NIRS")
CACHE = Path(r"G:\research_bio_medical\merge\neurovascnet_merge_v1_cache\replication_finemi")
HERE = Path(__file__).parent
OUT = HERE / "replication_finemi_results.json"
FS, SEED = 200.0, 1337
EOG = ("HEOG", "VEOG")
PARADIGMS = [("nback", ("nback",), {}), ("wg", ("vf", "wg"), {"WG": "VF"}), ("dsr", ("gonogo", "dsr"), {})]

# ---------------- step 1: trials ----------------
if not (CACHE / "meta.json").exists():
    nb = nbformat.read(NB, as_version=4)
    nb.cells[4].source = nb.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")
    ns = {"__name__": "__main__", "display": print}
    cwd = os.getcwd(); os.chdir(NB.parent)
    for c in nb.cells[:12]:
        if c.cell_type == "code":
            exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
    os.chdir(cwd)
    cfg = ns["CFG"]
    resample_to, zscore_trial, baseline_correct, bandpass = (ns[k] for k in
                                                             ("resample_to", "zscore_trial", "baseline_correct", "bandpass"))
    n_base = int(cfg.baseline_s * cfg.hemo_fs)

    def L(p, key):
        return sio.loadmat(p, squeeze_me=True, struct_as_record=False)[key]

    def events(mrk, rename=None):
        names = [str(c) for c in np.atleast_1d(mrk.className)]
        y = np.asarray(mrk.y)
        if y.ndim == 1 or len(names) != y.shape[0]:
            return None
        lab = [names[i] for i in np.argmax(y, 0)]
        if rename:
            lab = [rename.get(x, x) for x in lab]
        return np.atleast_1d(np.asarray(mrk.time, float)) / 1000.0, lab

    eeg_l, hemo_l, eog_l, sid_l, tid_l, rec_l, lab_l, audit = [], [], [], [], [], [], [], []
    ch_names = None
    t0 = time.time()
    for s in range(1, 27):
        vp = f"VP{s:03d}"
        for ti, (etask, ntasks, rename) in enumerate(PARADIGMS):
            nfile = next((t for t in ntasks if (ROOT / "NIRS" / f"{vp}-NIRS" / f"cnt_{t}.mat").exists()), None)
            if nfile is None:
                audit.append((vp, etask, "no NIRS file")); continue
            ce, me = L(ROOT / "EEG" / f"{vp}-EEG" / f"cnt_{etask}.mat", f"cnt_{etask}"), \
                L(ROOT / "EEG" / f"{vp}-EEG" / f"mrk_{etask}.mat", f"mrk_{etask}")
            cn, mn = L(ROOT / "NIRS" / f"{vp}-NIRS" / f"cnt_{nfile}.mat", f"cnt_{nfile}"), \
                L(ROOT / "NIRS" / f"{vp}-NIRS" / f"mrk_{nfile}.mat", f"mrk_{nfile}")
            ev_n = events(mn)
            ev_e = events(me, rename)
            if ev_n is None or ev_e is None:
                audit.append((vp, etask, "unreadable markers")); continue
            # EEG markers also carry stimulus-level events; keep the classes NIRS has
            keep = [i for i, x in enumerate(ev_e[1]) if x in set(ev_n[1])]
            te, le = ev_e[0][keep], [ev_e[1][i] for i in keep]
            tn, ln = ev_n
            if le != list(ln):
                audit.append((vp, etask, f"label sequences differ ({len(le)} vs {len(ln)})")); continue
            off = tn - te
            run = np.concatenate([[0], np.cumsum(np.abs(np.diff(off)) > 1.0)])
            clab = [str(c) for c in np.atleast_1d(ce.clab)]
            k_eeg = [i for i, c in enumerate(clab) if c not in EOG]
            k_eog = [clab.index(c) for c in EOG if c in clab]
            ch_names = ch_names or [clab[i] for i in k_eeg]
            assert [clab[i] for i in k_eeg] == ch_names
            ex = np.asarray(ce.x, float).T
            e_fs = float(ce.fs)
            n_fs = float(cn.oxy.fs)
            hbo = bandpass(np.asarray(cn.oxy.x, float).T, n_fs, .01, .2)
            hbr = bandpass(np.asarray(cn.deoxy.x, float).T, n_fs, .01, .2)
            n_ok = 0
            for k in range(len(te)):
                a_e, b_e = int(round((te[k] + cfg.tmin_s) * e_fs)), int(round((te[k] + cfg.tmax_s) * e_fs))
                a_n, b_n = int(round((tn[k] + cfg.tmin_s) * n_fs)), int(round((tn[k] + cfg.tmax_s) * n_fs))
                if a_e < 0 or b_e > ex.shape[1] or a_n < 0 or b_n > hbo.shape[1]:
                    continue
                seg = resample_to(ex[:, a_e:b_e], cfg.eeg_samples)
                eeg_l.append(zscore_trial(seg[k_eeg]).astype(np.float32))
                eog_l.append(zscore_trial(seg[k_eog]).astype(np.float32))
                so = baseline_correct(resample_to(hbo[:, a_n:b_n], cfg.hemo_samples), n_base)
                sr = baseline_correct(resample_to(hbr[:, a_n:b_n], cfg.hemo_samples), n_base)
                hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
                sid_l.append(s); tid_l.append(ti); rec_l.append(f"{vp}/{etask}_run{run[k]}"); lab_l.append(le[k])
                n_ok += 1
            audit.append((vp, etask, f"ok {n_ok} trials, {run.max() + 1} runs"))
        print(vp, [a[2] for a in audit if a[0] == vp], f"{(time.time() - t0) / 60:.1f} min", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {"eeg": np.stack(eeg_l), "eog": np.stack(eog_l), "hemo": np.stack(hemo_l),
              "subject_ids": np.asarray(sid_l), "task_ids": np.asarray(tid_l), "rec_ids": np.asarray(rec_l),
              "labels": np.asarray(lab_l)}
    for k, v in arrays.items():
        np.save(CACHE / f"{k}.npy", v)
    (CACHE / "meta.json").write_text(json.dumps({"ch_names": ch_names, "audit": audit,
                                                  "paradigms": [p[0] for p in PARADIGMS]}, indent=1))

# ---------------- step 2: R1-R3 ----------------
meta = json.loads((CACHE / "meta.json").read_text())
print("audit problems:", [a for a in meta["audit"] if not a[2].startswith("ok")])
eeg = np.load(CACHE / "eeg.npy", mmap_mode="r")
eog = np.load(CACHE / "eog.npy")
hemo = np.load(CACHE / "hemo.npy").astype(np.float64)
sid, tid, rec = (np.load(CACHE / f"{k}.npy") for k in ("subject_ids", "task_ids", "rec_ids"))
n, n_ch = len(hemo), hemo.shape[1]
k = n_ch // 2
subs = np.unique(sid)
strata = sid.astype(np.int64) * 1000 + tid
print(f"{n} trials, {len(subs)} people, paradigms {np.bincount(tid)}, EEG {eeg.shape}, hemo {hemo.shape}")


def sos(lo, hi):
    if lo is None:
        return sps.butter(4, hi, btype="low", fs=FS, output="sos")
    return sps.butter(4, [lo, hi], btype="band", fs=FS, output="sos")


def bins(x, nb):
    return x.reshape(*x.shape[:-1], nb, -1).mean(-1)


def feat_chunks(src, fn, chunk=120):
    return np.concatenate([fn(np.asarray(src[i:i + chunk], np.float64)) for i in range(0, len(src), chunk)])


S05, S01, S0105 = sos(None, 0.5), sos(None, 0.1), sos(0.1, 0.5)
BP = [sos(lo, hi) for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]]
FEATS = {
    "slow_lt0.5": feat_chunks(eeg, lambda x: bins(sps.sosfiltfilt(S05, x, axis=-1), 30).reshape(len(x), -1)),
    "slow_lt0.1": feat_chunks(eeg, lambda x: bins(sps.sosfiltfilt(S01, x, axis=-1), 30).reshape(len(x), -1)),
    "slow_0.1-0.5": feat_chunks(eeg, lambda x: bins(sps.sosfiltfilt(S0105, x, axis=-1), 30).reshape(len(x), -1)),
    "bandpower_1-45": feat_chunks(eeg, lambda x: np.concatenate(
        [np.log(bins(sps.sosfiltfilt(b, x, axis=-1) ** 2, 10) + 1e-8).reshape(len(x), -1) for b in BP], 1)),
    "eog_slow_lt0.5": feat_chunks(eog, lambda x: bins(sps.sosfiltfilt(S05, x, axis=-1), 30).reshape(len(x), -1)),
}
for name, F in FEATS.items():
    for s in subs:
        m = sid == s
        F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)

Y = bins(hemo, 30).reshape(n, -1)


def demean(A, g):
    A = A.copy()
    for kk in np.unique(g):
        m = g == kk
        A[m] -= A[m].mean(0)
    return A


def trial_corr(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    return ((P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)) *
            (T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12))).sum(1)


def per_subject(r, ids=None):
    ids = sid if ids is None else ids
    return np.array([r[ids == s].mean() for s in subs])


rng = np.random.default_rng(SEED)


def ci(v):
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def signflip(v):
    f = rng.choice([-1.0, 1.0], size=(10000, len(v)))
    return float((np.sum(np.abs((f * v).mean(1)) >= abs(v.mean())) + 1) / 10001)


Yd = demean(Y, strata)
ALPHAS = np.logspace(0, 6, 13)
res = {"n_trials": int(n), "n_people": int(len(subs)), "trials_per_paradigm": np.bincount(tid).tolist()}

# R1 cross-person
r1, preds = {}, {}
for name, F in FEATS.items():
    Q = np.zeros_like(Y)
    for s in subs:
        te = sid == s
        Q[te] = RidgeCV(alphas=ALPHAS).fit(F[~te], Yd[~te]).predict(F[te])
    preds[name] = Q
    v = per_subject(trial_corr(demean(Q, strata), Yd))
    r1[name] = v
    res.setdefault("R1", {})[name] = {"r_subject": float(v.mean()), "ci": ci(v), "p_signflip": signflip(v),
                                      "n_positive": int((v > 0).sum())}
    print(f"R1 {name:15s} cross-person r {v.mean():+.4f} CI {res['R1'][name]['ci']}", flush=True)
d = r1["slow_lt0.5"] - r1["bandpower_1-45"]
res["R1"]["slow_minus_bandpower"] = {"mean": float(d.mean()), "ci": ci(d), "p_signflip": signflip(d)}
print(f"R1 slow - bandpower {d.mean():+.4f} CI {res['R1']['slow_minus_bandpower']['ci']}")

# R3 global/local of the slow cross-person predictions
P = preds["slow_lt0.5"]
Yc = Y.reshape(n, n_ch, 30)
Pc = P.reshape(n, n_ch, 30)
for chrom, sl in (("HbO", slice(0, k)), ("HbR", slice(k, None))):
    T_, P_ = Yc[:, sl], Pc[:, sl]
    Tg, Pg = T_.mean(1, keepdims=True), P_.mean(1, keepdims=True)
    for part, (pp, tt) in (("global", (Pg, Tg)), ("local", (P_ - Pg, T_ - Tg))):
        v = per_subject(trial_corr(demean(pp.reshape(n, -1), strata), demean(tt.reshape(n, -1), strata)))
        res.setdefault("R3", {})[f"{chrom}_{part}"] = {"r_subject": float(v.mean()), "ci": ci(v)}
        print(f"R3 {chrom}_{part:6s} r {v.mean():+.4f} CI {res['R3'][f'{chrom}_{part}']['ci']}")

# R2 person-specificity, leave-one-run-out
for name in ("slow_lt0.5", "bandpower_1-45"):
    F = FEATS[name]
    M = np.zeros((len(subs), len(subs)))
    for si, s in enumerate(subs):
        own, other = np.flatnonzero(sid == s), np.flatnonzero(sid != s)
        Po, Pt, ns_ = np.zeros((len(own), Y.shape[1])), np.zeros((len(other), Y.shape[1])), 0
        for tr, te in LeaveOneGroupOut().split(own, groups=rec[own]):
            mdl = RidgeCV(alphas=ALPHAS).fit(F[own[tr]], demean(Y[own[tr]], tid[own[tr]]))
            Po[te] = mdl.predict(F[own[te]])
            Pt += mdl.predict(F[other]); ns_ += 1
        Pt /= ns_
        M[si, si] = trial_corr(demean(Po, strata[own]), Yd[own]).mean()
        ro = trial_corr(demean(Pt, strata[other]), Yd[other])
        for ti_, t in enumerate(subs):
            if t != s:
                M[si, ti_] = ro[sid[other] == t].mean()
    off = (M.sum(1) - np.diag(M)) / (len(subs) - 1)
    dd = np.diag(M) - off
    res.setdefault("R2", {})[name] = {"diag": float(np.diag(M).mean()), "offdiag": float(off.mean()),
                                      "diff": float(dd.mean()), "ci": ci(dd), "p_signflip": signflip(dd),
                                      "n_positive": int((dd > 0).sum()),
                                      "identification": float(np.mean(M.argmax(0) == np.arange(len(subs)))),
                                      "chance": 1 / len(subs)}
    r = res["R2"][name]
    print(f"R2 {name:15s} same {r['diag']:+.4f} other {r['offdiag']:+.4f} diff {r['diff']:+.4f} CI {r['ci']} "
          f"ident {r['identification']:.3f} (chance {r['chance']:.3f})", flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
