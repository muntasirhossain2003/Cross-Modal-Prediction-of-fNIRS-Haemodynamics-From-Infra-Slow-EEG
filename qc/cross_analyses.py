"""The three "cross" questions that have never actually been tested.

Everything so far evaluated ONE arm with ridge, which cannot test the project's central
claim. Three genuinely untested questions, in order of importance:

1. CROSS-MODAL  (the thesis)
   Does a shared EEG encoder trained on fNIRS *and* fMRI beat the same encoder trained on
   one modality alone? This is the entire novel claim of the project and ridge cannot test
   it -- it needs a shared encoder with two heads. Datasets have different channel counts
   (30 / 58 / 63), so each gets a 1x1 input adapter into a common space; the encoder after
   that point is genuinely shared.

2. CROSS-DATASET TRANSFER
   Train on hBCI, predict TU Berlin workload zero-shot. Same lab, same montage, different
   task. Never tried -- earlier work POOLED them, which is not the same thing.

3. CROSS-TASK TRANSFER
   Train on motor imagery, predict mental arithmetic in the same subjects. Isolates task
   transfer from subject transfer.

All scored the same way as everything else: subject-demeaned (constant predictor = 0),
exact permutation null.

Usage:  python qc/cross_analyses.py
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
N_BINS, N_PERM = 6, 1000
MAX_FMRI_TRIALS = 2500          # subsample ds002739 to keep memory sane


# ------------------------------------------------------------------ shared metrics
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


def merge(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    return (np.concatenate([x["eeg"], y["eeg"]]),
            np.concatenate([x["hemo"], y["hemo"]]).astype(np.float32),
            np.concatenate([x["subject_ids"],
                            y["subject_ids"] + x["subject_ids"].max() + 1]))


# ==================================================================== 2 & 3: transfer
def transfer(train, test, label, rows):
    Etr, Htr, _ = train
    Ete, Hte, Ste = test
    Xtr, Xte = features(Etr), features(Ete)
    if Xtr.shape[1] != Xte.shape[1]:
        print(f"{label:34s} SKIP - feature mismatch {Xtr.shape[1]} vs {Xte.shape[1]}")
        return
    if Htr.shape[1:] != Hte.shape[1:]:
        print(f"{label:34s} SKIP - target mismatch {Htr.shape[1:]} vs {Hte.shape[1:]}")
        return
    pred = ridge(Xtr, Htr.reshape(len(Htr), -1), Xte)
    Y = Hte.reshape(len(Hte), -1)
    obs, null, p = perm_p(demean(pred, Ste), demean(Y, Ste))
    rows.append({"analysis": "transfer", "setting": label, "n_train": len(Htr),
                 "n_test": len(Hte), "r_demeaned": obs, "null": null, "p": p})
    star = " *" if (np.isfinite(p) and p < 0.05) else ""
    print(f"{label:34s} train {len(Htr):5d} -> test {len(Hte):5d}   "
          f"r {obs:+.4f}  p={p:.4f}{star}")


# ==================================================================== 1: cross-modal
def cross_modal(fnirs, fmri, rows, epochs=40, seed=0):
    """Single-modality vs joint training with a genuinely shared encoder."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    def prep(d):
        E, H, S = d
        X = features(E)
        return X, H.reshape(len(H), -1).astype(np.float32), S

    Xa, Ya, Sa = prep(fnirs)
    Xb, Yb, Sb = prep(fmri)

    class Net(nn.Module):
        """One shared trunk; per-dataset input adapters and output heads."""
        def __init__(s, dims, outs, hid=256, lat=128):
            super().__init__()
            s.adapt = nn.ModuleList([nn.Linear(d, hid) for d in dims])
            s.trunk = nn.Sequential(nn.ELU(), nn.Dropout(0.3), nn.Linear(hid, lat),
                                    nn.ELU(), nn.Dropout(0.3), nn.Linear(lat, lat), nn.ELU())
            s.heads = nn.ModuleList([nn.Linear(lat, o) for o in outs])
        def forward(s, x, i):
            return s.heads[i](s.trunk(s.adapt[i](x)))

    def run(train_idx, eval_idx, tag):
        """train_idx: which arms to train on. eval_idx: which arm to score."""
        folds_a = np.array_split(np.random.default_rng(0).permutation(np.unique(Sa)), 5)
        folds_b = np.array_split(np.random.default_rng(0).permutation(np.unique(Sb)), 5)
        P, T, S = [], [], []
        for f in range(5):
            te_a, te_b = np.isin(Sa, folds_a[f]), np.isin(Sb, folds_b[f])
            data = [(Xa, Ya, ~te_a), (Xb, Yb, ~te_b)]
            mus = [(d[0][d[2]].mean(0), d[0][d[2]].std(0) + 1e-8) for d in data]
            net = Net([Xa.shape[1], Xb.shape[1]], [Ya.shape[1], Yb.shape[1]]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            tens = []
            for i, (X, Y, m) in enumerate(data):
                xm, xs = mus[i]
                ym = Y[m].mean(0)
                tens.append((torch.tensor((X[m] - xm) / xs), torch.tensor(Y[m] - ym), ym))
            for ep in range(epochs):
                net.train()
                for i in train_idx:
                    Xt, Yt, _ = tens[i]
                    perm = torch.randperm(len(Xt))
                    for j in range(0, len(Xt), 64):
                        k = perm[j:j + 64]
                        opt.zero_grad()
                        nn.functional.mse_loss(net(Xt[k].to(dev), i), Yt[k].to(dev)).backward()
                        opt.step()
            net.eval()
            i = eval_idx
            X, Y, m = data[i]
            xm, xs = mus[i]
            te = ~m
            with torch.no_grad():
                pr = net(torch.tensor((X[te] - xm) / xs).to(dev), i).cpu().numpy() + tens[i][2]
            P.append(pr); T.append(Y[te]); S.append((Sa if i == 0 else Sb)[te])
        Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
        obs, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
        print(f"  {tag:44s} r {obs:+.4f}  p={p:.4f}{' *' if p < 0.05 else ''}")
        return obs, p

    print("\nfNIRS arm:")
    r_solo_a, p_solo_a = run([0], 0, "fNIRS only (single-modality)")
    r_joint_a, p_joint_a = run([0, 1], 0, "fNIRS + fMRI (cross-modal, shared encoder)")
    print("\nfMRI arm:")
    r_solo_b, p_solo_b = run([1], 1, "fMRI only (single-modality)")
    r_joint_b, p_joint_b = run([0, 1], 1, "fMRI + fNIRS (cross-modal, shared encoder)")

    for arm, rs, ps, rj, pj in [("fNIRS", r_solo_a, p_solo_a, r_joint_a, p_joint_a),
                                ("fMRI", r_solo_b, p_solo_b, r_joint_b, p_joint_b)]:
        rows.append({"analysis": "cross_modal", "setting": f"{arm} single",
                     "r_demeaned": rs, "p": ps})
        rows.append({"analysis": "cross_modal", "setting": f"{arm} joint",
                     "r_demeaned": rj, "p": pj})
    return (r_solo_a, r_joint_a), (r_solo_b, r_joint_b)


# ==================================================================== main
def main():
    rows = []
    mi = merge(V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz")
    ma = merge(V4 / "hbci_train_MA.npz", V4 / "hbci_val_MA.npz")
    wl = None
    if (V4 / "tubwl.npz").exists():
        z = np.load(V4 / "tubwl.npz", allow_pickle=True)
        wl = (z["eeg"], z["hemo"].astype(np.float32), z["subject_ids"])

    print("=" * 78)
    print("2 & 3.  TRANSFER  (train on one corpus/task, predict another zero-shot)")
    print("=" * 78)
    transfer(mi, ma, "cross-task:    hBCI MI -> MA", rows)
    transfer(ma, mi, "cross-task:    hBCI MA -> MI", rows)
    if wl is not None:
        transfer(mi, wl, "cross-dataset: hBCI MI -> TUB_WL", rows)
        transfer(wl, mi, "cross-dataset: TUB_WL -> hBCI MI", rows)

    print()
    print("=" * 78)
    print("1.  CROSS-MODAL  (the thesis: does fMRI supervision help fNIRS, and vice versa?)")
    print("=" * 78)
    fm = None
    if (V4 / "ds002739.npz").exists():
        z = np.load(V4 / "ds002739.npz", allow_pickle=True)
        E, H, S = z["eeg"], z["hemo"].astype(np.float32), z["subject_ids"]
        if len(E) > MAX_FMRI_TRIALS:                       # keep memory manageable
            keep = np.random.default_rng(0).choice(len(E), MAX_FMRI_TRIALS, replace=False)
            keep.sort()
            E, H, S = E[keep], H[keep], S[keep]
        fm = (E, H, S)
        print(f"fMRI arm: ds002739, {len(E)} trials, {np.unique(S).size} subjects")
    elif (V4 / "ds002336_fixed_train.npz").exists():
        fm = merge(V4 / "ds002336_fixed_train.npz", V4 / "ds002336_fixed_val.npz")
        print(f"fMRI arm: ds002336, {len(fm[0])} trials")

    if fm is not None:
        print(f"fNIRS arm: hBCI MI, {len(mi[0])} trials, {np.unique(mi[2]).size} subjects")
        (rsa, rja), (rsb, rjb) = cross_modal(mi, fm, rows)
        print("\n" + "-" * 78)
        print(f"fNIRS: single {rsa:+.4f} -> joint {rja:+.4f}   ({rja-rsa:+.4f})")
        print(f"fMRI : single {rsb:+.4f} -> joint {rjb:+.4f}   ({rjb-rsb:+.4f})")
        if rja > rsa and rjb > rsb:
            print("\nVERDICT: cross-modal supervision helps BOTH arms -- the thesis holds.")
        elif rja > rsa or rjb > rsb:
            print("\nVERDICT: cross-modal helps ONE arm only -- partial support.")
        else:
            print("\nVERDICT: cross-modal supervision does NOT help. Sharing an encoder")
            print("         across modalities gains nothing over training each alone.")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "cross_analyses.csv", index=False)
    print(f"\nwrote {TABLE_DIR/'cross_analyses.csv'}")


if __name__ == "__main__":
    main()
