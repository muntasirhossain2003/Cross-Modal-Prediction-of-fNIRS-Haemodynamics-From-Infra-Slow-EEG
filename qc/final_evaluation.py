"""Final evaluation — every dataset, every role, one protocol.

Runs the project's intended four-dataset design plus everything the audit added:

    train fNIRS   TU Berlin hBCI (MI + MA pooled)      29 subj
    train fMRI    ds002739 (24 subj) and ds002336 (10 subj), compared
    external      TU Berlin workload (fNIRS)
    contrast      HEFMI -- audit-rejected target, kept to show what a bad target does

Every number is produced identically:
  * subject-wise K-fold CV, no subject in both train and test
  * subject-demeaned, so an EEG-free constant predictor scores exactly 0
  * exact permutation null (matmul trick: ~550x faster, numerically identical)
  * bootstrap CI over folds
  * split-half target reliability reported alongside, since it bounds everything

For ds002739 the visual / motor ROI dissociation is tested explicitly: the task is
random-dot motion, so a genuine result must appear in visual cortex and NOT in motor
cortex. A spurious result from drift, subject identity or scanner artifact would not
respect that anatomy.

Usage:  python qc/final_evaluation.py
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
N_BINS, N_PERM, K_FOLDS, N_BOOT = 6, 1000, 5, 2000

VIS = ("Occipital", "Lingual", "Intracalcarine", "Cuneal", "Supracalcarine")
MOT = ("Precentral", "Postcentral", "Supplementary", "Juxtapositional")


# ------------------------------------------------------------------ metrics
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


def boot_ci(v, n=N_BOOT, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, v.size, replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


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


def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid):
        out[sid == s] -= out[sid == s].mean(0, keepdims=True)
    return out


def evaluate(eeg, hemo, sid, label, rows, note=""):
    subs = np.unique(sid)
    if subs.size < K_FOLDS:
        print(f"{label:26s} only {subs.size} subjects, skipped")
        return
    X = features(eeg); Y = hemo.reshape(len(hemo), -1)
    rel = float(np.nanmean([subject_reliability(hemo[sid == s]) for s in subs]))
    ceil = np.sqrt(max(rel, 0))
    rng = np.random.default_rng(0); sh = subs.copy(); rng.shuffle(sh)
    folds = [sh[i::K_FOLDS] for i in range(K_FOLDS)]
    per, P, T, S = [], [], [], []
    for f in folds:
        te = np.isin(sid, f); tr = ~te
        if te.sum() < 6 or tr.sum() < 20:
            continue
        pred = ridge(X[tr], Y[tr], X[te])
        const = np.repeat(Y[tr].mean(0)[None, :], te.sum(), 0)
        per.append({"r_raw": concat_r(pred, Y[te]),
                    "r_const": concat_r(const, Y[te]),
                    "r_dm": concat_r(demean(pred, sid[te]), demean(Y[te], sid[te]))})
        P.append(pred); T.append(Y[te]); S.append(sid[te])
    if not per:
        print(f"{label:26s} no usable folds")
        return
    fd = pd.DataFrame(per)
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    obs, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
    m, lo, hi = boot_ci(fd.r_dm.values)
    rows.append({"dataset": label, "n_trials": len(hemo), "n_subjects": int(subs.size),
                 "reliability": rel, "ceiling": ceil,
                 "r_raw": fd.r_raw.mean(), "r_const": fd.r_const.mean(),
                 "r_gain": fd.r_raw.mean() - fd.r_const.mean(),
                 "r_demeaned": obs, "ci_low": lo, "ci_high": hi, "p": p,
                 "pct_ceiling": 100 * obs / ceil if ceil > 0 else np.nan, "note": note})
    star = " *" if (np.isfinite(p) and p < 0.05) else ""
    print(f"{label:26s} n={len(hemo):5d} subj={subs.size:3d}  rel {rel:+.3f}  "
          f"r {obs:+.4f} [{lo:+.4f},{hi:+.4f}]  p={p:.4f}{star}  {note}")


# ------------------------------------------------------------------ load
def merge(train, val):
    a, b = np.load(train, allow_pickle=True), np.load(val, allow_pickle=True)
    return (np.concatenate([a["eeg"], b["eeg"]]),
            np.concatenate([a["hemo"], b["hemo"]]).astype(np.float64),
            np.concatenate([a["subject_ids"],
                            b["subject_ids"] + a["subject_ids"].max() + 1]))


def main():
    rows = []
    print("=" * 90)
    print("fNIRS ARM")
    print("=" * 90)

    sets = {}
    if (V4 / "hbci_train_MI.npz").exists():
        sets["hBCI_MI"] = merge(V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz")
    if (V4 / "hbci_train_MA.npz").exists():
        sets["hBCI_MA"] = merge(V4 / "hbci_train_MA.npz", V4 / "hbci_val_MA.npz")
    if (V4 / "tubwl.npz").exists():
        z = np.load(V4 / "tubwl.npz", allow_pickle=True)
        sets["TUB_WL"] = (z["eeg"], z["hemo"].astype(np.float64), z["subject_ids"])
    if (V3 / "fnirs_train.npz").exists():
        sets["HEFMI"] = merge(V3 / "fnirs_train.npz", V3 / "fnirs_val.npz")

    for k in ("hBCI_MI", "hBCI_MA", "TUB_WL", "HEFMI"):
        if k in sets:
            note = "audit-rejected target (HbO/HbR +0.898)" if k == "HEFMI" else ""
            evaluate(*sets[k], k, rows, note)

    if "hBCI_MI" in sets and "hBCI_MA" in sets:
        E = np.concatenate([sets["hBCI_MI"][0], sets["hBCI_MA"][0]])
        H = np.concatenate([sets["hBCI_MI"][1], sets["hBCI_MA"][1]])
        S = np.concatenate([sets["hBCI_MI"][2],
                            sets["hBCI_MA"][2] + sets["hBCI_MI"][2].max() + 1])
        evaluate(E, H, S, "hBCI_MI+MA (pooled)", rows, "best fNIRS config")

    print()
    print("=" * 90)
    print("fMRI ARM")
    print("=" * 90)

    if (V4 / "ds002336_fixed_train.npz").exists():
        e, h, s = merge(V4 / "ds002336_fixed_train.npz", V4 / "ds002336_fixed_val.npz")
        evaluate(e, h, s, "ds002336 (motor ROIs)", rows, "10 subj, tSNR 24")

    p39 = V4 / "ds002739.npz"
    if p39.exists():
        z = np.load(p39, allow_pickle=True)
        E, H, S = z["eeg"], z["hemo"].astype(np.float64), z["subject_ids"]
        labels = [str(x) for x in z["roi_labels"]]
        idx = {"all48": list(range(H.shape[1])),
               "visual": [i for i, n in enumerate(labels) if any(k in n for k in VIS)],
               "motor":  [i for i, n in enumerate(labels) if any(k in n for k in MOT)]}
        for name, ii in idx.items():
            if not ii:
                continue
            note = ("task-relevant cortex" if name == "visual" else
                    "task-IRRELEVANT control" if name == "motor" else "")
            evaluate(E, H[:, ii], S, f"ds002739 ({name})", rows, note)
    else:
        print("ds002739 cache not found -- run qc/build_ds002739.py first")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "final_evaluation.csv", index=False)
    print("\n" + "=" * 100)
    print(df.round(4).to_string(index=False))
    print("=" * 100)

    sig = df[(df.p < 0.05) & (~df.note.str.contains("rejected", na=False))]
    if len(sig):
        b = sig.sort_values("r_demeaned", ascending=False).iloc[0]
        print(f"\nbest valid result: {b.dataset}")
        print(f"  r = {b.r_demeaned:+.4f} [{b.ci_low:+.4f}, {b.ci_high:+.4f}]  p = {b.p:.4f}")
        print(f"  {b.n_subjects} subjects, {b.n_trials} trials, "
              f"{b.pct_ceiling:.1f}% of a {b.ceiling:.3f} ceiling")

    v = df[df.dataset == "ds002739 (visual)"]
    m = df[df.dataset == "ds002739 (motor)"]
    if len(v) and len(m):
        print(f"\nANATOMICAL SPECIFICITY (ds002739, a visual task):")
        print(f"  visual cortex  r = {v.r_demeaned.iloc[0]:+.4f}  p = {v.p.iloc[0]:.4f}")
        print(f"  motor cortex   r = {m.r_demeaned.iloc[0]:+.4f}  p = {m.p.iloc[0]:.4f}")
        if v.p.iloc[0] < 0.05 and m.p.iloc[0] > 0.05:
            print("  -> predicts task-relevant cortex ONLY. A spurious result from drift,")
            print("     subject identity or scanner artifact would not respect this anatomy.")

    print(f"\nwrote {TABLE_DIR/'final_evaluation.csv'}")


if __name__ == "__main__":
    main()
