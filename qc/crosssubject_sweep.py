"""How high can CROSS-SUBJECT r go? An averaging sweep with an honest baseline.

Every setting here is cross-subject: the model never sees a validation subject's data.
The only thing that varies is how many trials are averaged before predicting.

Averaging is the one lever that has actually moved the number (0.072 -> 0.176 at k=8),
because the binding constraint is SNR: measured evoked SNR is below 1.0 in every fNIRS
dataset here, so a single trial's response is smaller than its own noise.

THE BASELINE THAT MATTERS
-------------------------
Averaged evoked responses are largely stereotyped -- everyone's mean HbO looks like an HRF.
So a high r can be obtained by ignoring EEG entirely and always emitting the *training-set
group mean*. Any honest claim must beat that. This script therefore reports, at every
averaging level:

    r_model       ridge on EEG band-power -> averaged response
    r_groupmean   constant prediction: the training group mean, EEG ignored
    r_gain        r_model - r_groupmean          <- the only number that is about EEG
    p             permutation null on the model

If r_model is high but r_gain ~ 0, the model has learned "output an HRF" and nothing more.
That is the failure mode that turned an earlier 0.21 in this project into 0.01 under LOSO.

Usage:  python qc/crosssubject_sweep.py
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
N_BINS = 6
N_PERM = 200
K_GRID = (1, 2, 4, 8, 15, 30)

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


def make_averages(X, Y, sid, k, seed=0):
    """Group k trials at a time WITHIN subject. k=1 returns the trials unchanged."""
    if k == 1:
        return X, Y, sid
    rng = np.random.default_rng(seed)
    Xo, Yo, So = [], [], []
    for s in np.unique(sid):
        idx = np.flatnonzero(sid == s)
        rng.shuffle(idx)
        for i in range(0, len(idx) - k + 1, k):
            g = idx[i:i + k]
            Xo.append(X[g].mean(0))
            Yo.append(Y[g].mean(0))
            So.append(s)
    if not Xo:
        return None, None, None
    return np.stack(Xo), np.stack(Yo), np.asarray(So)


def main():
    rows = []
    for name, (ptr, pva) in DATASETS.items():
        if not ptr.exists() or not pva.exists():
            continue
        tr, va = np.load(ptr, allow_pickle=True), np.load(pva, allow_pickle=True)
        Xtr_all, Xva_all = eeg_features(tr["eeg"]), eeg_features(va["eeg"])
        Ytr_all = tr["hemo"].reshape(tr["hemo"].shape[0], -1).astype(np.float64)
        Yva_all = va["hemo"].reshape(va["hemo"].shape[0], -1).astype(np.float64)
        str_, sva = tr["subject_ids"], va["subject_ids"]

        print(f"\n=== {name} ===  train {len(Ytr_all)} trials / "
              f"{np.unique(str_).size} subj,  val {len(Yva_all)} / "
              f"{np.unique(sva).size} subj   (CROSS-SUBJECT throughout)")
        print(f"{'k':>4s}{'n_train':>9s}{'n_val':>7s}{'r_model':>10s}"
              f"{'r_groupmean':>13s}{'r_gain':>9s}{'p':>8s}")
        print("-" * 62)

        for k in K_GRID:
            Xtr, Ytr, _ = make_averages(Xtr_all, Ytr_all, str_, k)
            Xva, Yva, _ = make_averages(Xva_all, Yva_all, sva, k, seed=1)
            if Xtr is None or Xva is None or len(Yva) < 4:
                continue

            pred = ridge(Xtr, Ytr, Xva)
            r_model, null_mu, p = perm_null(pred, Yva)

            # EEG-free baseline: always emit the training group mean
            gm = np.repeat(Ytr.mean(0)[None, :], len(Yva), axis=0)
            r_gm = concat_r(gm, Yva)

            rows.append({"dataset": name, "k": k, "n_train": len(Ytr),
                         "n_val": len(Yva), "r_model": r_model,
                         "r_groupmean": r_gm, "r_gain": r_model - r_gm,
                         "null_mean": null_mu, "p": p})
            star = " *" if (np.isfinite(p) and p < 0.05) else ""
            print(f"{k:>4d}{len(Ytr):>9d}{len(Yva):>7d}{r_model:>10.4f}"
                  f"{r_gm:>13.4f}{r_model - r_gm:>9.4f}{p:>8.3f}{star}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "crosssubject_sweep.csv", index=False)

    print("\n" + "=" * 72)
    print("r_model      raw correlation -- inflated by the stereotyped HRF shape")
    print("r_groupmean  what you get by IGNORING EEG entirely")
    print("r_gain       the part that is actually about EEG  <- judge on this")
    print("=" * 72)
    best = df.loc[df.r_model.idxmax()] if len(df) else None
    if best is not None:
        print(f"\nhighest raw r: {best.r_model:.3f} at k={int(best.k)} ({best.dataset}), "
              f"but group-mean alone gives {best.r_groupmean:.3f} "
              f"-> EEG contributes {best.r_gain:+.3f}")
    print(f"\nwrote {TABLE_DIR/'crosssubject_sweep.csv'}")


if __name__ == "__main__":
    main()
