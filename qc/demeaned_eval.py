"""Subject-demeaned cross-subject evaluation -- can r reach 0.2-0.3 honestly?

The raw concat-Pearson is dominated by the stereotyped HRF shape that every trial shares.
That is why an EEG-free constant predictor scores 0.076 while the model scores 0.072: both
are mostly "emit an HRF". The shared shape swamps the trial-specific part.

Subject-demeaning removes it. Subtract each subject's own mean response from both the
prediction and the truth, then correlate the residuals. What is left is exactly the
trial-to-trial variation -- the part that is actually about EEG. Critically, the constant
predictor's residual becomes ZERO, so the group-mean baseline collapses to 0 by
construction and any positive r is a genuine EEG contribution.

This is the protocol `NeuroVascNet_CrossSubj_TrialLevel_Verification.ipynb` used to report
fNIRS r = 0.33, and it is the standard fix for the subject-ranking confound.

HONEST CAVEAT, to be stated in any write-up: demeaning uses the held-out subject's OWN mean
response. That is a mild form of per-subject calibration -- you must have measured that
subject's average response to apply it. It is NOT fully zero-shot. Both variants are
reported below:
    own-mean      : demean with the test subject's own mean   (mild calibration)
    train-mean    : demean with the TRAINING group mean       (fully zero-shot)

Usage:  python qc/demeaned_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V4 = Path("neurovascnet_outputs/cache_v4")
EEG_FS = 200.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS, N_PERM = 6, 200
K_GRID = (1, 2, 4, 8)

DATASETS = {
    "fNIRS hBCI": (V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz"),
    "fMRI fixed": (V4 / "ds002336_fixed_train.npz", V4 / "ds002336_fixed_val.npz"),
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


def ridge(Xtr, Ytr, Xte, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    Yc = Ytr - ym
    n = int(Xtr.shape[0] * 0.8)
    best, best_r = alphas[len(alphas) // 2], -np.inf
    if Xtr.shape[0] - n >= 4:
        G = Xtr[:n].T @ Xtr[:n]
        for al in alphas:
            W = np.linalg.solve(G + al * np.eye(G.shape[0]), Xtr[:n].T @ Yc[:n])
            r = concat_r(Xtr[n:] @ W, Yc[n:])
            if np.isfinite(r) and r > best_r:
                best_r, best = r, al
    G = Xtr.T @ Xtr
    W = np.linalg.solve(G + best * np.eye(G.shape[0]), Xtr.T @ Yc)
    return Xte @ W + ym


def demean_by_subject(A, sid):
    """Subtract each subject's own mean row-block."""
    out = A.copy()
    for s in np.unique(sid):
        m = sid == s
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def group_avg(X, Y, sid, k, seed=0):
    if k == 1:
        return X, Y, sid
    rng = np.random.default_rng(seed)
    Xo, Yo, So = [], [], []
    for s in np.unique(sid):
        idx = np.flatnonzero(sid == s)
        rng.shuffle(idx)
        for i in range(0, len(idx) - k + 1, k):
            g = idx[i:i + k]
            Xo.append(X[g].mean(0)); Yo.append(Y[g].mean(0)); So.append(s)
    if not Xo:
        return None, None, None
    return np.stack(Xo), np.stack(Yo), np.asarray(So)


def main():
    rows = []
    for name, (ptr, pva) in DATASETS.items():
        if not ptr.exists() or not pva.exists():
            continue
        tr, va = np.load(ptr, allow_pickle=True), np.load(pva, allow_pickle=True)
        Xtr0, Xva0 = eeg_features(tr["eeg"]), eeg_features(va["eeg"])
        Ytr0 = tr["hemo"].reshape(len(tr["hemo"]), -1).astype(np.float64)
        Yva0 = va["hemo"].reshape(len(va["hemo"]), -1).astype(np.float64)
        str_, sva0 = tr["subject_ids"], va["subject_ids"]

        print(f"\n=== {name} ===  (training is CROSS-SUBJECT in every row)")
        print(f"{'k':>3s}{'mode':>14s}{'r_model':>10s}{'r_const':>9s}"
              f"{'null':>9s}{'p':>8s}   n_val")
        print("-" * 62)

        for k in K_GRID:
            Xtr, Ytr, _ = group_avg(Xtr0, Ytr0, str_, k)
            Xva, Yva, sva = group_avg(Xva0, Yva0, sva0, k, seed=1)
            if Xva is None or len(Yva) < 6:
                continue
            pred = ridge(Xtr, Ytr, Xva)
            const = np.repeat(Ytr.mean(0)[None, :], len(Yva), axis=0)

            for mode in ("raw", "own-mean", "train-mean"):
                if mode == "raw":
                    p_, t_, c_ = pred, Yva, const
                elif mode == "own-mean":
                    p_ = demean_by_subject(pred, sva)
                    t_ = demean_by_subject(Yva, sva)
                    c_ = demean_by_subject(const, sva)
                else:                                    # fully zero-shot
                    gm = Ytr.mean(0)
                    p_, t_, c_ = pred - gm, Yva - gm, const - gm

                obs, nul, pv = perm_null(p_, t_)
                r_const = concat_r(c_, t_)
                rows.append({"dataset": name, "k": k, "mode": mode,
                             "r_model": obs, "r_const": r_const,
                             "null": nul, "p": pv, "n_val": len(Yva)})
                star = " *" if (np.isfinite(pv) and pv < 0.05) else ""
                cs = f"{r_const:>9.4f}" if np.isfinite(r_const) else f"{'~0':>9s}"
                print(f"{k:>3d}{mode:>14s}{obs:>10.4f}{cs}"
                      f"{nul:>9.4f}{pv:>8.3f}{star}   {len(Yva)}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "demeaned_eval.csv", index=False)
    print("\n" + "=" * 74)
    print("raw         concat-Pearson on the response as-is; dominated by the shared HRF")
    print("own-mean    each subject's own mean removed -> constant predictor scores ~0,")
    print("            so r_model is purely the EEG contribution. Needs that subject's")
    print("            mean to have been measured => MILD PER-SUBJECT CALIBRATION.")
    print("train-mean  training group mean removed -> fully zero-shot, no calibration.")
    print("=" * 74)
    best = df[(df["mode"] == "own-mean")].sort_values("r_model", ascending=False).head(1)
    if len(best):
        b = best.iloc[0]
        print(f"\nbest own-mean: r={b.r_model:.3f} (p={b.p:.3f}) at k={int(b.k)}, {b.dataset}")
    bz = df[(df["mode"] == "train-mean")].sort_values("r_model", ascending=False).head(1)
    if len(bz):
        b = bz.iloc[0]
        print(f"best zero-shot: r={b.r_model:.3f} (p={b.p:.3f}) at k={int(b.k)}, {b.dataset}")
    print(f"\nwrote {TABLE_DIR/'demeaned_eval.csv'}")


if __name__ == "__main__":
    main()
