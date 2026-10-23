"""The last untested axis: how EEG is represented.

Every result so far uses log band-power per channel per time bin. That representation
throws away two things entirely:

    * PHASE            - only amplitude survives the |FFT|^2
    * SPATIAL STRUCTURE - each channel is treated independently, so inter-channel
                          covariance and connectivity are discarded

Both matter for neurovascular coupling, which is thought to track network-level activity
rather than local power alone. This sweeps representations that keep them:

  bandpower        baseline: log band-power, 5 bands x 6 bins        (what we have used)
  covariance       log-euclidean TANGENT SPACE of the spatial covariance matrix. This is
                   the modern standard in EEG-BCI decoding and routinely beats band-power
                   by a wide margin. Never tried in this project.
  connectivity     per-band inter-channel correlation (upper triangle)
  bp+cov           band-power and tangent space concatenated
  temporal         band-power at finer time resolution (12 bins)
  envelope_pca     PCA of the band-limited amplitude envelopes -- keeps time course
                   rather than collapsing it into bins

Also tested: KERNEL ridge (RBF) instead of linear, in case the mapping is nonlinear in a
way a linear read-out cannot express.

Scored identically to everything else: subject-wise 5-fold CV, subject-demeaned (constant
predictor = 0), exact permutation null.

Usage:  python qc/feature_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V4 = Path("neurovascnet_outputs/cache_v4")
EEG_FS = 200.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_PERM, K_FOLDS = 1000, 5
MAX_FMRI = 2500


# ------------------------------------------------------------------ metrics
def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


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


def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid):
        out[sid == s] -= out[sid == s].mean(0, keepdims=True)
    return out


# ------------------------------------------------------------------ features
def _bandfilt(eeg, lo, hi):
    b, a = sps.butter(4, [lo / (EEG_FS / 2), min(hi / (EEG_FS / 2), 0.99)], btype="band")
    return sps.filtfilt(b, a, eeg, axis=-1)


def f_bandpower(eeg, n_bins=6):
    N, C, T = eeg.shape; L = T // n_bins
    x = eeg[:, :, :L * n_bins].reshape(N, C, n_bins, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    bp = np.log10(np.stack([P[..., (f >= lo) & (f < hi)].sum(-1)
                            for lo, hi in BANDS], -1) + 1e-12)
    return bp.reshape(N, -1).astype(np.float32)


def f_temporal(eeg):
    return f_bandpower(eeg, n_bins=12)


def _tangent(covs, eps=1e-6):
    """Log-euclidean tangent-space mapping of SPD covariance matrices.

    Covariances live on a curved manifold; Euclidean models on the raw entries perform
    badly. Projecting through the matrix logarithm flattens the manifold so a linear model
    is appropriate. This is the standard trick in EEG-BCI decoding.
    """
    out = []
    for c in covs:
        c = c + eps * np.trace(c) / c.shape[0] * np.eye(c.shape[0])
        w, V = eigh(c)
        w = np.clip(w, 1e-10, None)
        L = V @ np.diag(np.log(w)) @ V.T
        iu = np.triu_indices(L.shape[0])
        v = L[iu]
        off = L.shape[0]
        v[off:] *= np.sqrt(2)          # preserve the norm for off-diagonal terms
        out.append(v)
    return np.stack(out).astype(np.float32)


def f_covariance(eeg):
    """Tangent space of the broadband spatial covariance, per trial."""
    x = _bandfilt(eeg, 1, 45)
    x = x - x.mean(-1, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
    return _tangent(covs)


def f_connectivity(eeg):
    """Inter-channel correlation within alpha and beta (upper triangle)."""
    feats = []
    for lo, hi in [(8, 13), (13, 30)]:
        x = _bandfilt(eeg, lo, hi)
        x = x - x.mean(-1, keepdims=True)
        sd = x.std(-1, keepdims=True) + 1e-9
        xn = x / sd
        c = np.einsum("nct,ndt->ncd", xn, xn) / x.shape[-1]
        iu = np.triu_indices(c.shape[1], k=1)
        feats.append(c[:, iu[0], iu[1]])
    return np.concatenate(feats, 1).astype(np.float32)


def f_bp_cov(eeg):
    return np.concatenate([f_bandpower(eeg), f_covariance(eeg)], 1)


def f_envelope_pca(eeg, n_comp=150, out_hz=4.0):
    """Amplitude envelopes kept as time courses, then PCA -- retains temporal shape."""
    from sklearn.decomposition import PCA
    step = int(EEG_FS / out_hz)
    outs = []
    for lo, hi in BANDS:
        e = np.abs(sps.hilbert(_bandfilt(eeg, lo, hi), axis=-1))
        T = e.shape[-1] // step * step
        outs.append(e[..., :T].reshape(e.shape[0], e.shape[1], -1, step).mean(-1))
    x = np.log10(np.concatenate(outs, 1) + 1e-12)
    x = x.reshape(len(x), -1)
    d = min(n_comp, x.shape[0] - 1, x.shape[1])
    return PCA(n_components=d).fit_transform(x).astype(np.float32)


FEATURES = {
    "bandpower": f_bandpower,
    "temporal": f_temporal,
    "covariance": f_covariance,
    "connectivity": f_connectivity,
    "bp+cov": f_bp_cov,
    "envelope_pca": f_envelope_pca,
}


# ------------------------------------------------------------------ models
def ridge(Xtr, Ytr, Xte, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)):
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


def kernel_ridge(Xtr, Ytr, Xte, gammas=(1e-4, 1e-3), alphas=(1e-1, 1e0, 1e1)):
    """RBF kernel ridge -- tests whether the mapping is nonlinear."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0); Yc = Ytr - ym
    n = int(len(A) * 0.8)
    def K(P, Q, g):
        d = (P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None, :] - 2 * P @ Q.T
        return np.exp(-g * np.maximum(d, 0))
    best, best_r = (gammas[0], alphas[0]), -np.inf
    for g in gammas:
        Kin = K(A[:n], A[:n], g); Kva = K(A[n:], A[:n], g)
        for al in alphas:
            W = np.linalg.solve(Kin + al * np.eye(n), Yc[:n])
            r = concat_r(Kva @ W, Yc[n:])
            if np.isfinite(r) and r > best_r:
                best_r, best = r, (g, al)
    g, al = best
    Kt = K(A, A, g)
    W = np.linalg.solve(Kt + al * np.eye(len(A)), Yc)
    return K(B, A, g) @ W + ym


# ------------------------------------------------------------------ eval
def evaluate(X, Y, sid, label, rows, model=ridge):
    subs = np.unique(sid)
    rng = np.random.default_rng(0); sh = subs.copy(); rng.shuffle(sh)
    folds = [sh[i::K_FOLDS] for i in range(K_FOLDS)]
    P, T, S = [], [], []
    for f in folds:
        te = np.isin(sid, f); tr = ~te
        if te.sum() < 6 or tr.sum() < 20:
            continue
        P.append(model(X[tr], Y[tr], X[te])); T.append(Y[te]); S.append(sid[te])
    if not P:
        return None
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    obs, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
    rows.append({"arm": label.split("|")[0].strip(), "features": label.split("|")[1].strip(),
                 "n_features": X.shape[1], "r_demeaned": obs, "null": null, "p": p})
    star = " *" if (np.isfinite(p) and p < 0.05) else ""
    print(f"  {label:34s} {X.shape[1]:6d} feats   r {obs:+.4f}  p={p:.4f}{star}")
    return obs


def merge(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    return (np.concatenate([x["eeg"], y["eeg"]]),
            np.concatenate([x["hemo"], y["hemo"]]).astype(np.float32),
            np.concatenate([x["subject_ids"],
                            y["subject_ids"] + x["subject_ids"].max() + 1]))


def main():
    arms = {}
    e, h, s = merge(V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz")
    arms["fNIRS"] = (e, h.reshape(len(h), -1), s)
    if (V4 / "ds002739.npz").exists():
        z = np.load(V4 / "ds002739.npz", allow_pickle=True)
        E, H, S = z["eeg"], z["hemo"].astype(np.float32), z["subject_ids"]
        if len(E) > MAX_FMRI:
            k = np.random.default_rng(0).choice(len(E), MAX_FMRI, replace=False); k.sort()
            E, H, S = E[k], H[k], S[k]
        arms["fMRI"] = (E, H.reshape(len(H), -1), S)

    rows = []
    for arm, (eeg, Y, sid) in arms.items():
        print(f"\n{'='*72}\n{arm}  ({len(eeg)} trials, {np.unique(sid).size} subjects, "
              f"{eeg.shape[1]} channels)\n{'='*72}")
        cache = {}
        for name, fn in FEATURES.items():
            try:
                X = fn(eeg)
            except Exception as exc:
                print(f"  {name:20s} FAILED {type(exc).__name__}: {str(exc)[:50]}")
                continue
            cache[name] = X
            evaluate(X, Y, sid, f"{arm} | {name}", rows)
        if "bandpower" in cache:
            print("  --- nonlinear read-out ---")
            evaluate(cache["bandpower"], Y, sid, f"{arm} | bandpower+RBF", rows,
                     model=kernel_ridge)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "feature_sweep.csv", index=False)
    print("\n" + "=" * 82)
    for arm in arms:
        sub = df[df.arm == arm].sort_values("r_demeaned", ascending=False)
        if not len(sub):
            continue
        base = sub[sub.features == "bandpower"].r_demeaned
        base = float(base.iloc[0]) if len(base) else np.nan
        print(f"\n{arm}   (baseline band-power = {base:+.4f})")
        print(sub[["features", "n_features", "r_demeaned", "p"]].round(4).to_string(index=False))
        b = sub.iloc[0]
        if np.isfinite(base) and b.r_demeaned > base:
            print(f"  -> best: {b.features}  {base:+.4f} -> {b.r_demeaned:+.4f} "
                  f"({b.r_demeaned-base:+.4f}, {100*(b.r_demeaned-base)/abs(base):+.0f}%)")
        else:
            print("  -> no representation beats band-power")
    print("=" * 82)
    print(f"\nwrote {TABLE_DIR/'feature_sweep.csv'}")


if __name__ == "__main__":
    main()
