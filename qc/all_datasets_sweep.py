"""Every usable dataset, every combination: the exhaustive cross-subject sweep.

Prior rounds tested preprocessing, model class, target dimensionality, data volume and
physiology. This tests the remaining axis: DATA. Every fNIRS corpus on disk is built,
gated, and evaluated the same way, then pooled to see whether more subjects help.

Datasets
--------
  hBCI        TU Berlin hBCI       29 subj  motor imagery      (current supervision)
  hBCI_MA     TU Berlin hBCI       29 subj  mental arithmetic  (same subjects, other task)
  TUB_WL      TU Berlin workload   26 subj  n-back/DSR/word-gen (never used for training)
  HEFMI       HEFMI-ICH            37 subj  motor imagery      (audit-rejected, as contrast)

TU Berlin workload pairs EEG to NIRS by task: dsr<->gonogo, nback<->nback, wg<->vf, using
the NIRS session-level markers (block design) as onsets.

Everything is scored the SAME way: subject-wise 5-fold CV, subject-demeaned (so an EEG-free
constant predictor scores exactly 0), exact permutation null.

Usage:  python qc/all_datasets_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

CACHE = Path("neurovascnet_outputs/cache_v4")
CACHE.mkdir(parents=True, exist_ok=True)
WL_ROOT = Path("dataset/TU Berlin EEG + NIRS")
EEG_FS, HEMO_FS = 200.0, 10.0
TMIN, TMAX, EEG_SAMPLES, HEMO_LEN = -2.0, 28.0, 6000, 300
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS, N_PERM, K_FOLDS = 6, 1000, 5
WL_PAIRS = [("dsr", "gonogo"), ("nback", "nback"), ("wg", "vf")]


def _bp(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def _rs(x, n):
    return x if x.shape[-1] == n else sps.resample(x, n, axis=-1)


def _one(p):
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    return m[[k for k in m if not k.startswith("__")][0]]


# ------------------------------------------------------------ TU Berlin workload
def build_wl():
    out = CACHE / "tubwl.npz"
    if out.exists():
        print("TUB_WL: cached")
        return out
    E_, H_, S_ = [], [], []
    nb = int(2.0 * HEMO_FS)
    subs = sorted(d.name.replace("-EEG", "") for d in (WL_ROOT / "EEG").glob("VP*-EEG"))
    for s in subs:
        n_sub = 0
        for etask, ntask in WL_PAIRS:
            ep = WL_ROOT / "EEG" / f"{s}-EEG" / f"cnt_{etask}.mat"
            np_ = WL_ROOT / "NIRS" / f"{s}-NIRS" / f"cnt_{ntask}.mat"
            mp = WL_ROOT / "NIRS" / f"{s}-NIRS" / f"mrk_{ntask}.mat"
            if not (ep.exists() and np_.exists() and mp.exists()):
                continue
            try:
                ec, nc, mk = _one(ep), _one(np_), _one(mp)
                efs = float(np.atleast_1d(ec.fs).ravel()[0])
                ex = np.asarray(ec.x, np.float64).T                    # (ch, T)
                nfs = float(np.atleast_1d(nc.oxy.fs).ravel()[0])
                hbo = _bp(np.asarray(nc.oxy.x, np.float64).T, nfs, 0.01, 0.2)
                hbr = _bp(np.asarray(nc.deoxy.x, np.float64).T, nfs, 0.01, 0.2)
                onsets = np.atleast_1d(np.asarray(mk.time, float)) / 1000.0
                for o in onsets:
                    ae, be = int(round((o + TMIN) * efs)), int(round((o + TMAX) * efs))
                    an, bn = int(round((o + TMIN) * nfs)), int(round((o + TMAX) * nfs))
                    if ae < 0 or be > ex.shape[1] or an < 0 or bn > hbo.shape[1]:
                        continue
                    se = _rs(ex[:, ae:be], EEG_SAMPLES)
                    se = ((se - se.mean(-1, keepdims=True)) /
                          (se.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
                    so, sr = _rs(hbo[:, an:bn], HEMO_LEN), _rs(hbr[:, an:bn], HEMO_LEN)
                    so = so - so[:, :nb].mean(-1, keepdims=True)
                    sr = sr - sr[:, :nb].mean(-1, keepdims=True)
                    E_.append(se); H_.append(np.concatenate([so, sr], 0).astype(np.float32))
                    S_.append(int("".join(c for c in s if c.isdigit()))); n_sub += 1
            except Exception as exc:
                print(f"  {s}/{etask}: {type(exc).__name__}: {str(exc)[:50]}")
        print(f"  {s}: {n_sub} trials")
    if not E_:
        print("TUB_WL: nothing built")
        return None
    nc_ = min(x.shape[0] for x in E_); nr = min(x.shape[0] for x in H_)
    np.savez_compressed(out, eeg=np.stack([x[:nc_] for x in E_]),
                        hemo=np.stack([x[:nr] for x in H_]), subject_ids=np.asarray(S_))
    print(f"{out.name}: {len(E_)} trials, {len(set(S_))} subjects, "
          f"eeg {nc_}ch hemo {nr}ch")
    return out


# ------------------------------------------------------------ metrics
def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def subject_reliability(t, n_repeat=50, seed=0):
    rng = np.random.default_rng(seed); n = t.shape[0]
    if n < 6:
        return np.nan
    rs = [concat_r(t[p[:n // 2]].mean(0), t[p[n // 2:]].mean(0))
          for p in (rng.permutation(n) for _ in range(n_repeat))]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else np.nan


def perm_p(pred, true, n=N_PERM, seed=0):
    P = np.asarray(pred, np.float32).reshape(len(pred), -1)
    T = np.asarray(true, np.float32).reshape(len(true), -1)
    Pc, Tc = P - P.mean(), T - T.mean()
    nP = np.sqrt((Pc.astype(np.float64) ** 2).sum())
    nT = np.sqrt((Tc.astype(np.float64) ** 2).sum())
    if nP <= 0 or nT <= 0:
        return np.nan, np.nan, np.nan
    D = (Pc @ Tc.T).astype(np.float64); N = len(D); ar = np.arange(N)
    obs = float(D[ar, ar].sum() / (nP * nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N), ar].sum() for _ in range(n)]) / (nP * nT)
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (null.size + 1))


def features(eeg):
    N, C, T = eeg.shape; L = T // N_BINS
    x = eeg[:, :, :L * N_BINS].reshape(N, C, N_BINS, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    bp = np.log10(np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1) + 1e-12)
    return bp.reshape(N, -1).astype(np.float32)


def ridge(Xtr, Ytr, Xte, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0); Yc = Ytr - ym
    n = int(Xtr.shape[0] * 0.8); best, best_r = alphas[len(alphas) // 2], -np.inf
    if Xtr.shape[0] - n >= 4:
        G = Xtr[:n].T @ Xtr[:n]
        for al in alphas:
            W = np.linalg.solve(G + al * np.eye(G.shape[0]), Xtr[:n].T @ Yc[:n])
            r = concat_r(Xtr[n:] @ W, Yc[n:])
            if np.isfinite(r) and r > best_r:
                best_r, best = r, al
    G = Xtr.T @ Xtr
    return Xte @ np.linalg.solve(G + best * np.eye(G.shape[0]), Xtr.T @ Yc) + ym


def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid):
        out[sid == s] -= out[sid == s].mean(0, keepdims=True)
    return out


def evaluate(eeg, hemo, sid, label, rows, extra=None):
    X = features(eeg); Y = hemo.reshape(len(hemo), -1)
    subs = np.unique(sid)
    if subs.size < K_FOLDS:
        print(f"{label:24s} too few subjects"); return
    rel = float(np.nanmean([subject_reliability(hemo[sid == s]) for s in subs]))
    rng = np.random.default_rng(0); sh = subs.copy(); rng.shuffle(sh)
    folds = [sh[i::K_FOLDS] for i in range(K_FOLDS)]
    P, T, S = [], [], []
    for test in folds:
        te = np.isin(sid, test); tr = ~te
        if te.sum() < 6 or tr.sum() < 20:
            continue
        P.append(ridge(X[tr], Y[tr], X[te])); T.append(Y[te]); S.append(sid[te])
    if not P:
        print(f"{label:24s} no usable folds"); return
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    obs, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
    ceil = np.sqrt(max(rel, 0))
    row = {"dataset": label, "n_trials": len(hemo), "n_subjects": int(subs.size),
           "reliability": rel, "ceiling": ceil, "r_demeaned": obs, "null": null, "p": p,
           "pct_ceiling": (100 * obs / ceil) if ceil > 0 else np.nan}
    if extra: row.update(extra)
    rows.append(row)
    star = " *" if (np.isfinite(p) and p < 0.05) else ""
    print(f"{label:24s} n={len(hemo):5d} subj={subs.size:3d}  rel {rel:+.3f}  "
          f"r {obs:+.4f}  p={p:.4f}{star}")


# ------------------------------------------------------------ main
def main():
    rows = []
    sets = {}

    for name, fn in [("hBCI_MI", CACHE / "hbci_train_MI.npz"),
                     ("hBCI_MA", CACHE / "hbci_train_MA.npz")]:
        # train+val were saved separately; merge them
        va = Path(str(fn).replace("_train_", "_val_"))
        if fn.exists() and va.exists():
            a, b = np.load(fn, allow_pickle=True), np.load(va, allow_pickle=True)
            sets[name] = (np.concatenate([a["eeg"], b["eeg"]]),
                          np.concatenate([a["hemo"], b["hemo"]]).astype(np.float64),
                          np.concatenate([a["subject_ids"],
                                          b["subject_ids"] + a["subject_ids"].max() + 1]))

    wl = build_wl()
    if wl is not None:
        z = np.load(wl, allow_pickle=True)
        sets["TUB_WL"] = (z["eeg"], z["hemo"].astype(np.float64), z["subject_ids"])

    for name in ("fnirs_train.npz",):
        p = Path("neurovascnet_outputs/cache_v3") / name
        v = Path("neurovascnet_outputs/cache_v3/fnirs_val.npz")
        if p.exists() and v.exists():
            a, b = np.load(p, allow_pickle=True), np.load(v, allow_pickle=True)
            sets["HEFMI"] = (np.concatenate([a["eeg"], b["eeg"]]),
                             np.concatenate([a["hemo"], b["hemo"]]).astype(np.float64),
                             np.concatenate([a["subject_ids"],
                                             b["subject_ids"] + a["subject_ids"].max() + 1]))

    print(f"\n{'-'*74}\nINDIVIDUAL DATASETS\n{'-'*74}")
    for name, (e, h, s) in sets.items():
        evaluate(e, h, s, name, rows)

    # ---------------- pooling: only where geometry matches ----------------
    print(f"\n{'-'*74}\nPOOLED (same EEG channel count AND same target shape)\n{'-'*74}")
    combos = [("hBCI_MI", "hBCI_MA"), ("hBCI_MI", "TUB_WL"),
              ("hBCI_MI", "hBCI_MA", "TUB_WL")]
    for combo in combos:
        if not all(k in sets for k in combo):
            continue
        shapes = {(sets[k][0].shape[1], sets[k][1].shape[1]) for k in combo}
        if len(shapes) > 1:
            print(f"{'+'.join(combo):24s} SKIP - incompatible geometry {shapes}")
            continue
        E = np.concatenate([sets[k][0] for k in combo])
        H = np.concatenate([sets[k][1] for k in combo])
        off, S = 0, []
        for k in combo:
            S.append(sets[k][2] + off); off = S[-1].max() + 1
        evaluate(E, H, np.concatenate(S), "+".join(combo), rows,
                 extra={"pooled": True})

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "all_datasets_sweep.csv", index=False)
    print("\n" + "=" * 78)
    print(df.round(4).to_string(index=False))
    print("=" * 78)
    ok = df[df.p < 0.05].sort_values("r_demeaned", ascending=False)
    if len(ok):
        b = ok.iloc[0]
        print(f"\nbest significant: {b.dataset}  r={b.r_demeaned:+.4f}  p={b.p:.4f}  "
              f"({b.n_subjects} subjects, {b.n_trials} trials)")
    print(f"\nwrote {TABLE_DIR/'all_datasets_sweep.csv'}")


if __name__ == "__main__":
    main()
