"""Is the unreachable variance vascular? Regress physiology out of the fNIRS target.

Target reliability is 0.44 (ceiling ~0.66) but prediction reaches only ~0.07. The gap is
either (a) neural signal the model fails to extract, or (b) haemodynamic variance that is
real and reproducible but NOT electrical in origin -- blood pressure, arousal, respiration --
which EEG cannot see in principle.

This distinguishes them. Physiological regressors are removed from the target and we watch
two numbers:

    reliability   falls -> the removed variance WAS the reproducible part (i.e. vascular)
    r_demeaned    rises -> what remains is more neural, so EEG predicts it better

Both outcomes are informative:
  * r_demeaned up   -> a better target; the number improves
  * ceiling collapses toward r_demeaned -> the shortfall is QUANTIFIED as vascular,
                                           which turns the negative into a real finding

Why envelopes: the target is bandpassed 0.01-0.2 Hz, so the cardiac fundamental (~1 Hz) is
already gone. But the slow AMPLITUDE ENVELOPE of the cardiac and respiratory bands sits
inside 0.01-0.2 Hz and contaminates it directly. Mayer waves (~0.1 Hz) are in-band already.

No short-separation channels exist in any dataset here, so this is the best available
systemic estimate rather than the ideal one.

Usage:  python qc/physio_decompose.py
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

HBCI = Path("dataset/TU Berlin hBCI EEG+FNRI")
CACHE = Path("neurovascnet_outputs/cache_v4")
EXT = {760: {"HbO": 1486.59, "HbR": 3843.71}, 850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}
SD_CM, EOG = 3.0, ("VEOG", "HEOG")
TMIN, TMAX, EEG_SAMPLES, HEMO_LEN = -2.0, 28.0, 6000, 300
EEG_FS = 200.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS, N_PERM, K_FOLDS = 6, 1000, 5

CARDIAC = (0.8, 2.0)
RESP    = (0.15, 0.40)
MAYER   = (0.08, 0.12)
TARGET_BAND = (0.01, 0.20)


# ------------------------------------------------------------------ signal utils
def _bp(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def _lp(x, fs, hi, order=3):
    b, a = sps.butter(order, min(hi / (fs / 2), 0.95), btype="low")
    return sps.filtfilt(b, a, x, axis=-1)


def envelope(x, fs, band):
    """Slow amplitude envelope of a band -- this is what leaks into the target band."""
    e = np.abs(sps.hilbert(_bp(x, fs, *band), axis=-1))
    return _lp(e, fs, TARGET_BAND[1])


def beer_lambert(lo_wl, hi_wl):
    def od(x):
        base = np.median(x, -1, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        return -np.log(np.clip(x / base, 1e-9, None))
    a1, a2 = od(lo_wl), od(hi_wl)
    L1, L2 = SD_CM * DPF[760], SD_CM * DPF[850]
    E = np.array([[EXT[760]["HbO"] * L1, EXT[760]["HbR"] * L1],
                  [EXT[850]["HbO"] * L2, EXT[850]["HbR"] * L2]])
    Ei = np.linalg.inv(E)
    return (Ei[0, 0] * a1 + Ei[0, 1] * a2) * 1e6, (Ei[1, 0] * a1 + Ei[1, 1] * a2) * 1e6


def regress_out(y, regs):
    """Remove regressors (n_reg, T) from every channel of y (n_ch, T)."""
    if regs.size == 0:
        return y
    X = np.column_stack([np.ones(y.shape[1])] + [r - r.mean() for r in regs])
    beta, *_ = np.linalg.lstsq(X, y.T, rcond=None)
    return y - (X @ beta).T


# ------------------------------------------------------------------ build variants
def _runs(p, hint=None):
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    ks = [k for k in m if not k.startswith("__")]
    return np.atleast_1d(m[hint if hint in ks else ks[0]]).ravel()


def _split_wl(x, clab):
    n = [str(c) for c in np.atleast_1d(clab)]
    lo = {v.replace("lowWL", ""): i for i, v in enumerate(n) if "lowWL" in v}
    hi = {v.replace("highWL", ""): i for i, v in enumerate(n) if "highWL" in v}
    pr = sorted(set(lo) & set(hi))
    return np.stack([x[:, lo[p]] for p in pr]), np.stack([x[:, hi[p]] for p in pr])


def _rs(x, n):
    return x if x.shape[-1] == n else sps.resample(x, n, axis=-1)


VARIANTS = {
    "baseline":        (),                        # bandpass only -- current pipeline
    "minus_cardiac":   ("cardiac",),
    "minus_resp":      ("resp",),
    "minus_mayer":     ("mayer",),
    "minus_all_physio": ("cardiac", "resp", "mayer"),
    "minus_gsr":       ("gsr",),                  # known-harmful, kept as the contrast
}


def build_variants():
    out = CACHE / "hbci_physio_variants.npz"
    if out.exists():
        print("physio variants: cached")
        return out
    store = {k: [] for k in VARIANTS}
    eeg_all, sid_all = [], []
    nb = int(2.0 * 10)

    for s in range(1, 30):
        ed = next((g / f"subject {s:02d}" / "with occular artifact"
                   for g in sorted((HBCI / "EEG data").glob("EEG_*"))
                   if (g / f"subject {s:02d}" / "with occular artifact").exists()), None)
        nd = HBCI / "NRI data" / "NIRS_01-29" / f"subject {s:02d}"
        if ed is None or not nd.exists():
            continue
        try:
            er, em = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
            nr, nm = _runs(nd / "cnt.mat", "cnt"), _runs(nd / "mrk.mat", "mrk")
        except Exception as e:
            print(f"  subj {s:02d}: {e}")
            continue

        cnt = 0
        for r in range(min(er.size, nr.size)):
            Er, Nr = er[r], nr[r]
            if not str(getattr(Er, "title", "")).upper().startswith("MI"):
                continue
            efs = float(np.atleast_1d(Er.fs).ravel()[0])
            clab = [str(c) for c in np.atleast_1d(Er.clab)]
            keep = [i for i, c in enumerate(clab) if c not in EOG]
            ex = np.asarray(Er.x, np.float64)[:, keep].T
            nfs = float(np.atleast_1d(Nr.fs).ravel()[0])
            lo, hi = _split_wl(np.asarray(Nr.x, np.float64), Nr.clab)
            hbo_full, hbr_full = beer_lambert(lo, hi)

            # --- physiological regressors, from the FULL-band signal ---
            g_hbo = hbo_full.mean(0)                       # global (all optodes)
            regs = {
                "cardiac": envelope(g_hbo, nfs, CARDIAC)[None, :],
                "resp":    envelope(g_hbo, nfs, RESP)[None, :],
                "mayer":   _bp(g_hbo, nfs, *MAYER)[None, :],
                "gsr":     _bp(g_hbo, nfs, *TARGET_BAND)[None, :],
            }

            hbo = _bp(hbo_full, nfs, *TARGET_BAND)
            hbr = _bp(hbr_full, nfs, *TARGET_BAND)

            variants_ch = {}
            for name, keys in VARIANTS.items():
                R = np.vstack([regs[k] for k in keys]) if keys else np.empty((0, hbo.shape[1]))
                variants_ch[name] = (regress_out(hbo, R), regress_out(hbr, R))

            if r >= em.size or r >= nm.size:
                continue
            eon = np.atleast_1d(np.asarray(em[r].time, float)) / 1000.0
            non = np.atleast_1d(np.asarray(nm[r].time, float)) / 1000.0
            for k in range(min(eon.size, non.size)):
                ae, be = int(round((eon[k] + TMIN) * efs)), int(round((eon[k] + TMAX) * efs))
                an, bn = int(round((non[k] + TMIN) * nfs)), int(round((non[k] + TMAX) * nfs))
                if ae < 0 or be > ex.shape[1] or an < 0 or bn > hbo.shape[1]:
                    continue
                se = _rs(ex[:, ae:be], EEG_SAMPLES)
                se = ((se - se.mean(-1, keepdims=True)) /
                      (se.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
                eeg_all.append(se); sid_all.append(s); cnt += 1
                for name, (o, d_) in variants_ch.items():
                    so, sr = _rs(o[:, an:bn], HEMO_LEN), _rs(d_[:, an:bn], HEMO_LEN)
                    so = so - so[:, :nb].mean(-1, keepdims=True)
                    sr = sr - sr[:, :nb].mean(-1, keepdims=True)
                    store[name].append(np.concatenate([so, sr], 0).astype(np.float32))
        print(f"  subj {s:02d}: {cnt} trials")

    payload = {"eeg": np.stack(eeg_all), "sid": np.asarray(sid_all)}
    for k, v in store.items():
        payload[f"hemo_{k}"] = np.stack(v)
    np.savez_compressed(out, **payload)
    print(f"\nwrote {out.name}: {len(eeg_all)} trials, {len(set(sid_all))} subjects")
    return out


# ------------------------------------------------------------------ evaluation
def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def subject_reliability(t, n_repeat=50, seed=0):
    rng = np.random.default_rng(seed)
    n = t.shape[0]
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
    D = (Pc @ Tc.T).astype(np.float64)
    N = len(D); ar = np.arange(N)
    obs = float(D[ar, ar].sum() / (nP * nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N), ar].sum() for _ in range(n)]) / (nP * nT)
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (null.size + 1))


def features(eeg):
    N, C, T = eeg.shape
    L = T // N_BINS
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


def main():
    path = build_variants()
    z = np.load(path, allow_pickle=True)
    eeg, sid = z["eeg"], z["sid"]
    X = features(eeg)
    subs = np.unique(sid)
    rng = np.random.default_rng(0); rng.shuffle(subs)
    folds = [subs[i::K_FOLDS] for i in range(K_FOLDS)]

    print(f"\n{len(eeg)} trials, {subs.size} subjects, {X.shape[1]} features\n")
    print(f"{'variant':20s}{'reliability':>13s}{'ceiling':>9s}{'r_demeaned':>12s}"
          f"{'%ceiling':>10s}{'p':>9s}")
    print("-" * 74)

    rows = []
    for name in VARIANTS:
        H = z[f"hemo_{name}"]
        rel = float(np.nanmean([subject_reliability(H[sid == s]) for s in subs]))
        ceil = np.sqrt(max(rel, 0))
        Y = H.reshape(len(H), -1)
        P, T, S = [], [], []
        for test in folds:
            te = np.isin(sid, test); tr = ~te
            if te.sum() < 6 or tr.sum() < 20:
                continue
            P.append(ridge(X[tr], Y[tr], X[te])); T.append(Y[te]); S.append(sid[te])
        Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
        obs, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
        pct = f"{100*obs/ceil:.0f}%" if ceil > 0 else "n/a"
        rows.append({"variant": name, "reliability": rel, "ceiling": ceil,
                     "r_demeaned": obs, "pct_of_ceiling": pct, "null": null, "p": p})
        star = " *" if (np.isfinite(p) and p < 0.05) else ""
        print(f"{name:20s}{rel:>13.4f}{ceil:>9.3f}{obs:>12.4f}{pct:>10s}{p:>9.4f}{star}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "physio_decompose.csv", index=False)

    base = df[df.variant == "baseline"].iloc[0]
    allp = df[df.variant == "minus_all_physio"].iloc[0]
    print("\n" + "=" * 74)
    print(f"baseline          reliability {base.reliability:.3f}  r {base.r_demeaned:+.4f}")
    print(f"minus all physio  reliability {allp.reliability:.3f}  r {allp.r_demeaned:+.4f}")
    print(f"change            reliability {allp.reliability-base.reliability:+.3f}  "
          f"r {allp.r_demeaned-base.r_demeaned:+.4f}")
    print("=" * 74)
    if allp.r_demeaned > base.r_demeaned + 0.01:
        print("VERDICT: removing physiology IMPROVES prediction -> better target, higher r.")
    elif allp.reliability < base.reliability - 0.05 and abs(allp.r_demeaned-base.r_demeaned) < 0.01:
        print("VERDICT: physiology carried the reproducible variance but NOT the predictable")
        print("         part -> the ceiling was inflated by vascular signal EEG cannot see.")
        print("         The shortfall is now quantified rather than unexplained.")
    else:
        print("VERDICT: physiological regression does not account for the gap.")
    print(f"\nwrote {TABLE_DIR/'physio_decompose.csv'}")


if __name__ == "__main__":
    main()
