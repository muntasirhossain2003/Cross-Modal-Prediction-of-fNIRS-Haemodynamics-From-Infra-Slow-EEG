"""Three untested levers, each judged on the demeaned metric with the right controls.

Baseline to beat: cross-subject, zero-shot, subject-demeaned  r = 0.0533 (p = 0.005).
Demeaning makes the EEG-free constant predictor score 0, so every number below is purely
the EEG contribution -- no HRF-shape inflation.

  LEVER 1  Neurovascular delay. Every analysis so far correlated a fixed 30 s window.
           The EEG->haemodynamic lag is 2-8 s and varies by subject, so misaligned
           waveforms attenuate the correlation. The optimal response window is chosen on
           TRAINING data only and applied unchanged to the held-out subjects.

  LEVER 2  Richer EEG features. Everything so far is log band-power, 5 bands x 6 bins.
           Tested: finer time bins, more bands, and ERD (change relative to the
           pre-stimulus baseline) which band-power alone discards.

  LEVER 3  More training data. The hBCI mental-arithmetic runs use the same subjects,
           montage and hardware as the motor-imagery runs. Adding them doubles the
           training set while the test set stays motor-imagery only.

Usage:  python qc/three_levers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V4 = Path("neurovascnet_outputs/cache_v4")
EEG_FS, HEMO_FS = 200.0, 10.0
N_PERM = 200
BANDS_BASE = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BANDS_FINE = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45), (45, 70)]


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


def features(eeg, bands=BANDS_BASE, n_bins=6, erd=False):
    N, C, T = eeg.shape
    L = T // n_bins
    x = eeg[:, :, :L * n_bins].reshape(N, C, n_bins, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    bp = np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in bands], -1)
    logbp = np.log10(bp + 1e-12)
    out = [logbp.reshape(N, -1)]
    if erd:
        # event-related desynchronisation: each bin relative to the first (baseline) bin
        rel = logbp - logbp[:, :, :1, :]
        out.append(rel.reshape(N, -1))
    return np.concatenate(out, 1).astype(np.float32)


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


def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid):
        m = sid == s
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def evaluate(Xtr, Ytr, Xva, Yva, sva, label, rows):
    """Fit cross-subject, score subject-demeaned (constant baseline = 0 by construction)."""
    pred = ridge(Xtr, Ytr, Xva)
    obs, nul, p = perm_null(demean(pred, sva), demean(Yva, sva))
    rows.append({"variant": label, "r_demeaned": obs, "null": nul, "p": p,
                 "n_train": len(Ytr), "n_val": len(Yva), "n_features": Xtr.shape[1],
                 "target_dims": Ytr.shape[1]})
    star = " *" if (np.isfinite(p) and p < 0.05) else ""
    print(f"{label:46s} r={obs:+.4f}  null {nul:+.4f}  p={p:.3f}{star}")
    return obs


def main():
    tr = np.load(V4 / "hbci_train_MI.npz", allow_pickle=True)
    va = np.load(V4 / "hbci_val_MI.npz", allow_pickle=True)
    Etr, Eva = tr["eeg"], va["eeg"]
    Htr, Hva = tr["hemo"].astype(np.float64), va["hemo"].astype(np.float64)
    str_, sva = tr["subject_ids"], va["subject_ids"]
    rows = []

    print("BASELINE  (band-power 5x6, full 300-sample window, MI only)")
    Xtr, Xva = features(Etr), features(Eva)
    base = evaluate(Xtr, Htr.reshape(len(Htr), -1), Xva,
                    Hva.reshape(len(Hva), -1), sva, "baseline", rows)

    # ---------------------------------------------------------------- LEVER 1
    print("\nLEVER 1  neurovascular delay -- response window chosen on TRAIN only")
    win = 100                                    # 10 s window
    offsets = list(range(0, 201, 20))            # 0-20 s after epoch start
    best_off, best_tr = None, -np.inf
    for off in offsets:
        if off + win > Htr.shape[2]:
            continue
        ytr = Htr[:, :, off:off + win].reshape(len(Htr), -1)
        # score on TRAIN subjects only, so the choice never sees held-out data
        n = int(len(ytr) * 0.8)
        p_in = ridge(Xtr[:n], ytr[:n], Xtr[n:])
        r_in = concat_r(demean(p_in, str_[n:]), demean(ytr[n:], str_[n:]))
        if np.isfinite(r_in) and r_in > best_tr:
            best_tr, best_off = r_in, off
    print(f"  chosen offset {best_off/HEMO_FS - 2:.1f} s after onset "
          f"(train score {best_tr:+.4f})")
    ytr = Htr[:, :, best_off:best_off + win].reshape(len(Htr), -1)
    yva = Hva[:, :, best_off:best_off + win].reshape(len(Hva), -1)
    evaluate(Xtr, ytr, Xva, yva, sva,
             f"L1 delay-aligned window @{best_off/HEMO_FS - 2:.0f}s", rows)

    # ---------------------------------------------------------------- LEVER 2
    print("\nLEVER 2  richer EEG features")
    for label, kw in [("L2 finer bins (12)", dict(n_bins=12)),
                      ("L2 more bands (8)", dict(bands=BANDS_FINE)),
                      ("L2 + ERD", dict(erd=True)),
                      ("L2 all combined", dict(bands=BANDS_FINE, n_bins=12, erd=True))]:
        Xa, Xb = features(Etr, **kw), features(Eva, **kw)
        evaluate(Xa, Htr.reshape(len(Htr), -1), Xb,
                 Hva.reshape(len(Hva), -1), sva, label, rows)

    # ---------------------------------------------------------------- LEVER 3
    print("\nLEVER 3  more training data (add mental-arithmetic runs; test stays MI)")
    ma = V4 / "hbci_train_MA.npz"
    if ma.exists():
        m = np.load(ma, allow_pickle=True)
        Etr2 = np.concatenate([Etr, m["eeg"]])
        Htr2 = np.concatenate([Htr, m["hemo"].astype(np.float64)])
        X2 = features(Etr2)
        evaluate(X2, Htr2.reshape(len(Htr2), -1), Xva,
                 Hva.reshape(len(Hva), -1), sva, "L3 MI+MA training", rows)
    else:
        print("  hbci_train_MA.npz not found, skipped")

    # ---------------------------------------------------------------- combined
    print("\nCOMBINED  best of each lever")
    if ma.exists():
        kw = dict(bands=BANDS_FINE, n_bins=12, erd=True)
        Xc_tr, Xc_va = features(Etr2, **kw), features(Eva, **kw)
        ytr = Htr2[:, :, best_off:best_off + win].reshape(len(Htr2), -1)
        yva = Hva[:, :, best_off:best_off + win].reshape(len(Hva), -1)
        evaluate(Xc_tr, ytr, Xc_va, yva, sva, "L1+L2+L3 combined", rows)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "three_levers.csv", index=False)
    print("\n" + "=" * 78)
    print(df.sort_values("r_demeaned", ascending=False)[
        ["variant", "r_demeaned", "null", "p", "n_train"]].round(4).to_string(index=False))
    print("=" * 78)
    best = df.loc[df.r_demeaned.idxmax()]
    print(f"\nbaseline          {base:+.4f}")
    print(f"best variant      {best.r_demeaned:+.4f}  ({best.variant}, p={best.p:.3f})")
    print(f"change            {best.r_demeaned - base:+.4f}")
    print(f"\nwrote {TABLE_DIR/'three_levers.csv'}")


if __name__ == "__main__":
    main()
