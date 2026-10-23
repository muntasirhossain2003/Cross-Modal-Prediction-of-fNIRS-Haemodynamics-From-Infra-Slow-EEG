"""Low-rank target: predict only the components of the response that reproduce.

The v4 notebook predicts every sample of the hemodynamic response -- 72 x 300 = 21,600
numbers for fNIRS -- from 900 EEG features with 1380 training trials. That is badly
underdetermined, and target reliability of 0.44 means over half of those 21,600 values are
noise *by definition*. Model capacity is being spent fitting them.

The reliable structure should be low-rank: a few spatial/temporal components. This script

  1. PCAs the target (fit on training trials only -- no leakage),
  2. measures split-half reliability of EACH component separately,
  3. regresses EEG -> component scores for the top-k most reliable components,
  4. reconstructs the full waveform and scores it with the SAME concat-Pearson and
     permutation null as the notebook, so the numbers are directly comparable.

If the headline r rises, the bottleneck was target dimensionality, not EEG.

Usage:  python qc/lowrank_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V3 = Path("neurovascnet_outputs/cache_v3")
V4 = Path("neurovascnet_outputs/cache_v4")
EEG_FS = 200.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS = 6
N_PERM = 200
K_GRID = (1, 2, 3, 5, 8, 12, 20, 35, 60, 100)

DATASETS = {
    "fNIRS hBCI (v4)": (V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz"),
    "fMRI v4 (fixed)": (V4 / "ds002336_fixed_train.npz", V4 / "ds002336_fixed_val.npz"),
}


def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def perm_null(pred, true, n=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    obs = concat_r(pred, true)
    null = [concat_r(pred[rng.permutation(pred.shape[0])], true) for _ in range(n)]
    null = np.asarray([x for x in null if np.isfinite(x)])
    if null.size == 0:
        return obs, np.nan, np.nan
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (null.size + 1))


def eeg_features(eeg):
    N, C, T = eeg.shape
    L = T // N_BINS
    x = eeg[:, :, :L * N_BINS].reshape(N, C, N_BINS, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    out = [P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS]
    return np.log10(np.stack(out, -1) + 1e-12).reshape(N, -1).astype(np.float32)


def ridge(Xtr, Ytr, Xte, alphas=(1e1, 1e2, 1e3, 1e4, 1e5)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    Yc = Ytr - ym
    n = int(Xtr.shape[0] * 0.8)
    best, best_r = alphas[0], -np.inf
    if Xtr.shape[0] - n >= 5:
        G = Xtr[:n].T @ Xtr[:n]
        for al in alphas:
            W = np.linalg.solve(G + al * np.eye(G.shape[0]), Xtr[:n].T @ Yc[:n])
            r = concat_r(Xtr[n:] @ W, Yc[n:])
            if np.isfinite(r) and r > best_r:
                best_r, best = r, al
    G = Xtr.T @ Xtr
    W = np.linalg.solve(G + best * np.eye(G.shape[0]), Xtr.T @ Yc)
    return Xte @ W + ym


def component_reliability(scores, sid, n_repeat=30):
    """Split-half reliability of EACH component's per-trial scores, per subject."""
    rng = np.random.default_rng(0)
    n_comp = scores.shape[1]
    out = np.full(n_comp, np.nan)
    for c in range(n_comp):
        rs = []
        for s in np.unique(sid):
            idx = np.flatnonzero(sid == s)
            if idx.size < 6:
                continue
            for _ in range(n_repeat):
                p = rng.permutation(idx)
                a = scores[p[: idx.size // 2], c].mean()
                b = scores[p[idx.size // 2:], c].mean()
                rs.append((a, b))
        if len(rs) > 3:
            A = np.array([x[0] for x in rs])
            B = np.array([x[1] for x in rs])
            if A.std() > 0 and B.std() > 0:
                out[c] = np.corrcoef(A, B)[0, 1]
    return out


def main():
    rows, comp_rows = [], []
    for name, (ptr, pva) in DATASETS.items():
        if not ptr.exists() or not pva.exists():
            print(f"{name}: cache missing")
            continue
        tr, va = np.load(ptr, allow_pickle=True), np.load(pva, allow_pickle=True)
        Xtr, Xva = eeg_features(tr["eeg"]), eeg_features(va["eeg"])
        Ytr = tr["hemo"].reshape(tr["hemo"].shape[0], -1).astype(np.float64)
        Yva = va["hemo"].reshape(va["hemo"].shape[0], -1).astype(np.float64)
        sid_tr = tr["subject_ids"]

        print(f"\n=== {name} ===")
        print(f"  target dims {Ytr.shape[1]},  train {Ytr.shape[0]},  val {Yva.shape[0]}")

        # baseline: predict every sample (what the notebook does)
        base_obs, base_null, base_p = perm_null(ridge(Xtr, Ytr, Xva), Yva)
        print(f"  FULL target      r={base_obs:+.4f}  null {base_null:+.4f}  p={base_p:.3f}")

        # PCA fitted on TRAIN only
        n_comp = int(min(max(K_GRID), Ytr.shape[0] - 1, Ytr.shape[1]))
        pca = PCA(n_components=n_comp).fit(Ytr)
        Str, Sva = pca.transform(Ytr), pca.transform(Yva)

        rel = component_reliability(Str, sid_tr)
        order = np.argsort(-np.nan_to_num(rel, nan=-9))     # most reliable first
        evr = pca.explained_variance_ratio_

        print(f"  top components by reliability:")
        for c in order[:6]:
            comp_rows.append({"dataset": name, "component": int(c),
                              "reliability": float(rel[c]),
                              "explained_var": float(evr[c])})
            print(f"     PC{c:<3d} reliability {rel[c]:+.3f}   "
                  f"explains {100*evr[c]:5.2f}% of variance")

        best = (None, -9)
        for k in K_GRID:
            if k > n_comp:
                continue
            sel = order[:k]
            pred_s = ridge(Xtr, Str[:, sel], Xva)
            # reconstruct into full waveform space so the metric is comparable
            recon = pred_s @ pca.components_[sel] + pca.mean_
            obs, nul, p = perm_null(recon, Yva)
            rows.append({"dataset": name, "k": k, "r": obs, "null": nul, "p": p,
                         "mean_component_reliability": float(np.nanmean(rel[sel])),
                         "baseline_full_r": base_obs, "baseline_full_p": base_p})
            flag = ""
            if obs > best[1]:
                best = (k, obs)
            if p < 0.05:
                flag = " *"
            print(f"  top-{k:<4d} components  r={obs:+.4f}  null {nul:+.4f}  p={p:.3f}{flag}")

        print(f"  BEST k={best[0]}  r={best[1]:+.4f}   vs full-target {base_obs:+.4f}"
              f"   ({100*(best[1]-base_obs)/abs(base_obs):+.0f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "lowrank_target.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(TABLE_DIR / "lowrank_components.csv", index=False)
    print(f"\nwrote {TABLE_DIR/'lowrank_target.csv'}")


if __name__ == "__main__":
    main()
