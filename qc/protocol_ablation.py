"""How much does each evaluation shortcut inflate the result?

Same data, same model, same features. Only the EVALUATION PROTOCOL changes. Starting from
a conventional setup and adding one rigour element at a time shows exactly what each
shortcut is worth -- and where an apparently publishable result stops being one.

The elements, and what each is for:

  single split      one fixed hold-out instead of subject-wise cross-validation. Cheap, and
                    the estimate depends heavily on which subjects happen to be held out.
  trial averaging   averaging k trials raises SNR, but it also amplifies the stereotyped
                    response shape every trial shares -- which is not an EEG contribution.
  raw correlation   correlating the response as-is. Because every trial contains a similar
                    haemodynamic shape, a predictor that IGNORES EEG entirely already
                    correlates with the target. Without a constant-predictor baseline there
                    is no way to see this.
  single-shuffle    one permutation as the null, which is itself a noisy draw.
  no reliability    never asking whether the TARGET reproduces against itself. A target that
                    gate              cannot predict itself cannot be predicted by anything.

Panel A  cumulative waterfall on a GOOD target -- how much inflation each shortcut adds.
Panel B  the reliability gate on a BROKEN target -- a result that every conventional check
         passes, and which the gate rejects.

We describe these as conventional PRACTICES and quantify them on our own data. We do not
attribute them to specific publications.

Usage:  python qc/protocol_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V3 = Path("neurovascnet_outputs/cache_v3")
V4 = Path("neurovascnet_outputs/cache_v4")
EEG_FS = 200.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS, K_FOLDS, N_PERM = 6, 5, 1000
N_FMRI = 3000


# ------------------------------------------------------------------ basics
def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def demean_by_subject(A, sid):
    out = A.copy()
    for s in np.unique(sid):
        out[sid == s] -= out[sid == s].mean(0, keepdims=True)
    return out


def features(eeg, n_bins=N_BINS):
    N, C, T = eeg.shape; L = T // n_bins
    x = eeg[:, :, :L * n_bins].reshape(N, C, n_bins, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    bp = np.log10(np.stack([P[..., (f >= lo) & (f < hi)].sum(-1)
                            for lo, hi in BANDS], -1) + 1e-12)
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


def perm_null(pred, true, n, seed=0):
    """n=1 reproduces the common single-shuffle null; n=1000 is the exact test."""
    P = np.asarray(pred, np.float32).reshape(len(pred), -1)
    T = np.asarray(true, np.float32).reshape(len(true), -1)
    Pc, Tc = P - P.mean(), T - T.mean()
    nP = np.sqrt((Pc.astype(np.float64) ** 2).sum())
    nT = np.sqrt((Tc.astype(np.float64) ** 2).sum())
    if nP <= 0 or nT <= 0:
        return np.nan, np.nan
    D = (Pc @ Tc.T).astype(np.float64); N = len(D); ar = np.arange(N)
    obs = float(D[ar, ar].sum() / (nP * nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N), ar].sum() for _ in range(n)]) / (nP * nT)
    if n == 1:                       # "is it bigger than one shuffle?" -- the common test
        return obs, 0.0 if obs > null[0] else 1.0
    return obs, float((np.sum(null >= obs) + 1) / (null.size + 1))


def group_average(X, Y, sid, k, seed=0):
    if k == 1:
        return X, Y, sid
    rng = np.random.default_rng(seed)
    Xo, Yo, So = [], [], []
    for s in np.unique(sid):
        idx = np.flatnonzero(sid == s); rng.shuffle(idx)
        for i in range(0, len(idx) - k + 1, k):
            g = idx[i:i + k]
            Xo.append(X[g].mean(0)); Yo.append(Y[g].mean(0)); So.append(s)
    return np.stack(Xo), np.stack(Yo), np.asarray(So)


# ------------------------------------------------------------------ protocol
def run_protocol(X, Y, sid, split, average_k, demean, n_perm, seed=0):
    """Returns (r_model, r_constant, p). Only the protocol varies."""
    Xa, Ya, Sa = group_average(X, Y, sid, average_k, seed)
    subs = np.unique(Sa)
    if split == "single":
        rng = np.random.default_rng(seed); sh = subs.copy(); rng.shuffle(sh)
        test_sets = [sh[:max(1, int(round(0.2 * len(sh))))]]
    else:
        rng = np.random.default_rng(seed); sh = subs.copy(); rng.shuffle(sh)
        test_sets = [sh[i::K_FOLDS] for i in range(K_FOLDS)]

    P, T, S, C = [], [], [], []
    for ts in test_sets:
        te = np.isin(Sa, ts); tr = ~te
        if te.sum() < 3 or tr.sum() < 10:
            continue
        P.append(ridge(Xa[tr], Ya[tr], Xa[te])); T.append(Ya[te]); S.append(Sa[te])
        C.append(np.repeat(Ya[tr].mean(0)[None, :], te.sum(), 0))
    Pm, Tm, Sm, Cm = np.vstack(P), np.vstack(T), np.concatenate(S), np.vstack(C)
    if demean:
        Pm, Cm, Tm = (demean_by_subject(Pm, Sm), demean_by_subject(Cm, Sm),
                      demean_by_subject(Tm, Sm))
    r_model, p = perm_null(Pm, Tm, n_perm, seed)
    return r_model, concat_r(Cm, Tm), p


# steps are cumulative: each row adds one rigour element to the row above
STEPS = [
    ("0  conventional",              dict(split="single", average_k=8, demean=False, n_perm=1)),
    ("1  + subject-wise CV",         dict(split="cv",     average_k=8, demean=False, n_perm=1)),
    ("2  + exact permutation null",  dict(split="cv",     average_k=8, demean=False, n_perm=N_PERM)),
    ("3  + no trial averaging",      dict(split="cv",     average_k=1, demean=False, n_perm=N_PERM)),
    ("4  + subject-demeaned",        dict(split="cv",     average_k=1, demean=True,  n_perm=N_PERM)),
]


def panel_a(name, X, Y, sid, rows, seeds=(0, 1, 2)):
    print(f"\n{'='*94}\nPANEL A - protocol waterfall: {name}\n{'='*94}")
    print(f"{'protocol':32s}{'r_model':>10s}{'r_const':>10s}{'p':>9s}"
          f"{'beats const?':>14s}   reported as")
    print("-" * 94)
    for label, cfg in STEPS:
        rs = [run_protocol(X, Y, sid, seed=s, **cfg) for s in seeds]
        r = float(np.mean([a[0] for a in rs]))
        c = float(np.mean([a[1] for a in rs]))
        p = float(np.mean([a[2] for a in rs]))
        beats = r > c
        # what a reader would conclude at this level of rigour
        if "conventional" in label or "const" not in label:
            verdict = "POSITIVE" if (p < 0.05) else "null"
        if p < 0.05 and beats:
            verdict = "POSITIVE (survives)"
        elif p < 0.05 and not beats:
            verdict = "POSITIVE (but baseline wins)"
        else:
            verdict = "null"
        rows.append({"panel": "A", "dataset": name, "protocol": label.strip(),
                     "r_model": r, "r_const": c, "p": p, "beats_constant": beats,
                     "verdict": verdict})
        print(f"{label:32s}{r:>10.4f}{c:>10.4f}{p:>9.4f}"
              f"{('yes' if beats else 'NO'):>14s}   {verdict}")


def panel_b(rows):
    """The reliability gate on a broken target vs the same target repaired."""
    print(f"\n{'='*94}\nPANEL B - the reliability gate on a broken target\n{'='*94}")

    def load(tr, va):
        a, b = np.load(tr, allow_pickle=True), np.load(va, allow_pickle=True)
        return (np.concatenate([a["eeg"], b["eeg"]]),
                np.concatenate([a["hemo"], b["hemo"]]).astype(np.float32),
                np.concatenate([a["subject_ids"],
                                b["subject_ids"] + a["subject_ids"].max() + 1]))

    def reliability(H, sid, n_repeat=50, seed=0):
        rng = np.random.default_rng(seed); vals = []
        for s in np.unique(sid):
            t = H[sid == s]
            if len(t) < 6:
                continue
            rr = [concat_r(t[p[:len(t) // 2]].mean(0), t[p[len(t) // 2:]].mean(0))
                  for p in (rng.permutation(len(t)) for _ in range(n_repeat))]
            rr = [x for x in rr if np.isfinite(x)]
            if rr:
                vals.append(np.mean(rr))
        return float(np.mean(vals)) if vals else np.nan

    for tag, tr, va in [("BROKEN target (sliding windows, no nuisance regression)",
                         V3 / "ds002336_train.npz", V3 / "ds002336_val.npz"),
                        ("REPAIRED target (event-locked, motion+drift regressed)",
                         V4 / "ds002336_fixed_train.npz", V4 / "ds002336_fixed_val.npz")]:
        if not (tr.exists() and va.exists()):
            print(f"  {tag}: cache missing"); continue
        E, H, S = load(tr, va)
        X = features(E); Y = H.reshape(len(H), -1)
        rel = reliability(H, S)
        r, c, p = run_protocol(X, Y, S, split="single", average_k=8, demean=False, n_perm=1)
        rr, cc, pp = run_protocol(X, Y, S, split="cv", average_k=1, demean=True, n_perm=N_PERM)
        gate = "PASS" if rel > 0 else "FAIL - analysis should stop here"
        print(f"\n  {tag}")
        print(f"    target reliability            {rel:+.3f}   -> gate: {gate}")
        print(f"    conventional protocol         r = {r:+.4f}  p = {p:.4f}"
              f"   {'-> looks publishable' if p < 0.05 else ''}")
        print(f"    rigorous protocol             r = {rr:+.4f}  p = {pp:.4f}")
        rows.append({"panel": "B", "dataset": tag, "reliability": rel,
                     "conventional_r": r, "conventional_p": p,
                     "rigorous_r": rr, "rigorous_p": pp,
                     "gate": "pass" if rel > 0 else "fail"})


def main():
    rows = []

    a, b = np.load(V4 / "hbci_train_MI.npz", allow_pickle=True), \
           np.load(V4 / "hbci_val_MI.npz", allow_pickle=True)
    E = np.concatenate([a["eeg"], b["eeg"]])
    H = np.concatenate([a["hemo"], b["hemo"]]).astype(np.float32)
    S = np.concatenate([a["subject_ids"], b["subject_ids"] + a["subject_ids"].max() + 1])
    panel_a("fNIRS (TU Berlin hBCI)", features(E), H.reshape(len(H), -1), S, rows)
    del E, H

    if (V4 / "ds002739.npz").exists():
        z = np.load(V4 / "ds002739.npz", allow_pickle=True)
        E, H, S = z["eeg"], z["hemo"].astype(np.float32), z["subject_ids"]
        if len(E) > N_FMRI:
            k = np.random.default_rng(123).choice(len(E), N_FMRI, replace=False); k.sort()
            E, H, S = E[k], H[k], S[k]
        panel_a("fMRI (ds002739)", features(E), H.reshape(len(H), -1), S, rows)
        del E, H

    panel_b(rows)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "protocol_ablation.csv", index=False)
    print("\n" + "=" * 94)
    A = df[df.panel == "A"]
    for ds in A.dataset.unique():
        s = A[A.dataset == ds]
        first, last = s.iloc[0], s.iloc[-1]
        print(f"\n{ds}: conventional r = {first.r_model:+.4f} -> rigorous r = {last.r_model:+.4f}"
              f"   ({100*(first.r_model-last.r_model)/abs(last.r_model):+.0f}% inflation)")
        worst = s[~s.beats_constant]
        if len(worst):
            print(f"   at {len(worst)}/{len(s)} rigour levels the model LOSES to a "
                  f"predictor that ignores EEG entirely")
    print(f"\nwrote {TABLE_DIR/'protocol_ablation.csv'}")


if __name__ == "__main__":
    main()
