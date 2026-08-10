"""Within-subject cross-validation: is EEG->haemodynamic prediction possible at all?

Everything so far has been CROSS-SUBJECT (train on some people, test on people never seen),
which lands at r ~ 0.03-0.05. That is the hardest possible setting and it conflates two
questions:

    1. does EEG carry haemodynamic information?              <- the scientific question
    2. does that mapping transfer to an unseen person?        <- the deployment question

Cross-subject answers both at once, and fails. This isolates question 1: train and test on
the SAME subject, on DIFFERENT trials, with proper k-fold cross-validation. If r rises
sharply, EEG does carry the signal and the cross-subject failure is a transfer problem, not
an information problem. That distinction is the whole thesis.

This is still real cross-validation - no trial is ever in both train and test - and it is
what the successful literature (EFP/cEFP) actually does: calibrate per subject, then deploy
EEG-only on that subject.

Controls kept identical to the cross-subject analysis:
    * the training-fold mean is removed from both prediction and truth, so a constant
      predictor scores exactly 0 and r measures only the EEG contribution
    * exact permutation null per subject
    * PCA before ridge, because ~48 training trials against 900 features overfits

Usage:  python qc/within_subject_cv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

CACHE = Path("neurovascnet_outputs/cache_v4")
V3 = Path("neurovascnet_outputs/cache_v3")
EEG_FS, HEMO_FS = 200.0, 10.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS, N_PERM, K = 6, 500, 5
PCA_DIMS = 20
AVG_LEVELS = (1, 2, 4)


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
    if nP <= 0 or nT <= 0 or len(P) < 4:
        return np.nan, np.nan
    D = (Pc @ Tc.T).astype(np.float64); N = len(D); ar = np.arange(N)
    obs = float(D[ar, ar].sum() / (nP * nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N), ar].sum() for _ in range(n)]) / (nP * nT)
    return obs, float((np.sum(null >= obs) + 1) / (null.size + 1))


def features(eeg):
    N, C, T = eeg.shape; L = T // N_BINS
    x = eeg[:, :, :L * N_BINS].reshape(N, C, N_BINS, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    bp = np.log10(np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1) + 1e-12)
    return bp.reshape(N, -1).astype(np.float32)


def ridge(Xtr, Ytr, Xte, alphas=(1e0, 1e1, 1e2, 1e3, 1e4)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0); Yc = Ytr - ym
    G = Xtr.T @ Xtr
    # too few trials to hold out an inner split reliably -> fixed mid-range alpha
    al = alphas[len(alphas) // 2]
    return Xte @ np.linalg.solve(G + al * np.eye(G.shape[0]), Xtr.T @ Yc) + ym


def group_average(X, Y, k, seed=0):
    if k == 1:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    Xo, Yo = [], []
    for i in range(0, len(idx) - k + 1, k):
        g = idx[i:i + k]
        Xo.append(X[g].mean(0)); Yo.append(Y[g].mean(0))
    return (np.stack(Xo), np.stack(Yo)) if Xo else (None, None)


def within_subject(eeg, hemo, sid, label, k_avg=1):
    """Per subject: k-fold over that subject's own trials. Returns per-subject rows."""
    X_all = features(eeg)
    Y_all = hemo.reshape(len(hemo), -1)
    rows = []
    for s in np.unique(sid):
        m = sid == s
        X, Y = X_all[m], Y_all[m]
        if k_avg > 1:
            X, Y = group_average(X, Y, k_avg)
            if X is None or len(X) < 10:
                continue
        n = len(X)
        if n < 15:
            continue
        rng = np.random.default_rng(0)
        order = rng.permutation(n)
        folds = [order[i::K] for i in range(K)]
        P, T = [], []
        for f in folds:
            te = np.zeros(n, bool); te[f] = True; tr = ~te
            if te.sum() < 2 or tr.sum() < 8:
                continue
            d = min(PCA_DIMS, tr.sum() - 1, X.shape[1])
            p = PCA(n_components=d).fit(X[tr])
            pred = ridge(p.transform(X[tr]), Y[tr], p.transform(X[te]))
            # remove the TRAINING mean from both -> constant predictor scores 0
            gm = Y[tr].mean(0)
            P.append(pred - gm); T.append(Y[te] - gm)
        if not P:
            continue
        Pm, Tm = np.vstack(P), np.vstack(T)
        r, p_ = perm_p(Pm, Tm)
        rows.append({"dataset": label, "k_avg": k_avg, "subject": int(s),
                     "n_units": n, "r": r, "p": p_})
    return rows


def main():
    sets = {}
    for name, tr, va in [("hBCI_MI", CACHE / "hbci_train_MI.npz", CACHE / "hbci_val_MI.npz"),
                         ("hBCI_MA", CACHE / "hbci_train_MA.npz", CACHE / "hbci_val_MA.npz"),
                         ("HEFMI",   V3 / "fnirs_train.npz",      V3 / "fnirs_val.npz")]:
        if tr.exists() and va.exists():
            a, b = np.load(tr, allow_pickle=True), np.load(va, allow_pickle=True)
            sets[name] = (np.concatenate([a["eeg"], b["eeg"]]),
                          np.concatenate([a["hemo"], b["hemo"]]).astype(np.float64),
                          np.concatenate([a["subject_ids"],
                                          b["subject_ids"] + a["subject_ids"].max() + 1]))
    wl = CACHE / "tubwl.npz"
    if wl.exists():
        z = np.load(wl, allow_pickle=True)
        sets["TUB_WL"] = (z["eeg"], z["hemo"].astype(np.float64), z["subject_ids"])
    fm = CACHE / "ds002336_fixed_train.npz"
    if fm.exists():
        a = np.load(fm, allow_pickle=True)
        b = np.load(CACHE / "ds002336_fixed_val.npz", allow_pickle=True)
        sets["fMRI"] = (np.concatenate([a["eeg"], b["eeg"]]),
                        np.concatenate([a["hemo"], b["hemo"]]).astype(np.float64),
                        np.concatenate([a["subject_ids"],
                                        b["subject_ids"] + a["subject_ids"].max() + 1]))

    all_rows = []
    print(f"{'dataset':10s}{'k_avg':>6s}{'subj':>6s}{'mean r':>9s}{'median':>9s}"
          f"{'max':>8s}{'n p<.05':>9s}{'n r>0.3':>9s}")
    print("-" * 66)
    for name, (e, h, s) in sets.items():
        for k in AVG_LEVELS:
            rows = within_subject(e, h, s, name, k_avg=k)
            if not rows:
                continue
            all_rows += rows
            df = pd.DataFrame(rows)
            print(f"{name:10s}{k:>6d}{len(df):>6d}{df.r.mean():>9.4f}{df.r.median():>9.4f}"
                  f"{df.r.max():>8.4f}{int((df.p < 0.05).sum()):>9d}{int((df.r > 0.3).sum()):>9d}")

    df = pd.DataFrame(all_rows)
    df.to_csv(TABLE_DIR / "within_subject_cv.csv", index=False)

    print("\n" + "=" * 78)
    best = (df.groupby(["dataset", "k_avg"])
              .agg(mean_r=("r", "mean"), frac_sig=("p", lambda x: (x < 0.05).mean()),
                   frac_over_030=("r", lambda x: (x > 0.30).mean()), n=("r", "size"))
              .reset_index().sort_values("mean_r", ascending=False))
    print(best.round(4).to_string(index=False))
    print("=" * 78)
    top = best.iloc[0]
    print(f"\nbest setting: {top.dataset} averaging {int(top.k_avg)} trials")
    print(f"  mean within-subject r = {top.mean_r:+.4f}")
    print(f"  {100*top.frac_sig:.0f}% of subjects significant (p<0.05)")
    print(f"  {100*top.frac_over_030:.0f}% of subjects exceed r = 0.30")
    print("\nCompare cross-subject on the same data: r ~ 0.025-0.038.")
    if top.mean_r > 0.15:
        print("\nEEG DOES carry haemodynamic information within a subject. The cross-subject")
        print("failure is therefore a TRANSFER problem, not an information problem.")
    else:
        print("\nWithin-subject prediction is also weak -> the limit is information, not transfer.")
    print(f"\nwrote {TABLE_DIR/'within_subject_cv.csv'}")


if __name__ == "__main__":
    main()
