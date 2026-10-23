"""Confirmation run: are the two candidate gains real, or noise?

Two candidates emerged at the edge of the noise floor:

  A. FINER TIME BINS   band-power with 12 bins instead of 6
                       fNIRS +81% (0.0253 -> 0.0458), fMRI +40% (0.0427 -> 0.0595)
  B. SUBJECT-ADAPTER   shared trunk + per-subject embedding in the cross-modal model
                       fNIRS +38% (0.0297 -> 0.0412)

Neither is trustworthy yet. Three results already collapsed under proper testing today
(the anatomical dissociation, the single-split 0.053, the first cross-modal test), and the
same data swung 4x from subsampling alone. So the protocol here is fixed in advance:

    PAIRED       both conditions see the IDENTICAL fold assignment within a seed, so the
                 difference cannot be an artifact of lucky folds
    MULTI-SEED   5 seeds (A) / 3 seeds (B), each a different subject-to-fold assignment
    FIXED DATA   the same trial subsample for every condition and seed
    DIFFERENCE   we test the PAIRED DIFFERENCE, not each arm against its own null.
                 Both conditions already beat their nulls; that was never the question.

A gain counts as confirmed only if the paired difference is positive in the large majority
of seeds AND its bootstrap CI excludes zero.

Usage:  python qc/confirm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V4 = Path("neurovascnet_outputs/cache_v4")
EEG_FS = 200.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
K_FOLDS = 5
SEEDS_A = [0, 1, 2, 3, 4]
SEEDS_B = [0, 1, 2]
N_FMRI = 3000                      # fixed subsample, identical across all conditions
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, BS, LR = 120, 20, 64, 1e-3


def concat_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid):
        out[sid == s] -= out[sid == s].mean(0, keepdims=True)
    return out


def bandpower(eeg, n_bins):
    N, C, T = eeg.shape; L = T // n_bins
    x = eeg[:, :, :L * n_bins].reshape(N, C, n_bins, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1 / EEG_FS)
    bp = np.log10(np.stack([P[..., (f >= lo) & (f < hi)].sum(-1)
                            for lo, hi in BANDS], -1) + 1e-12)
    return bp.reshape(N, -1).astype(np.float32)


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


def folds_for(sid, seed):
    subs = np.unique(sid)
    rng = np.random.default_rng(seed); sh = subs.copy(); rng.shuffle(sh)
    return [sh[i::K_FOLDS] for i in range(K_FOLDS)]


def cv_ridge(X, Y, sid, folds):
    P, T, S = [], [], []
    for f in folds:
        te = np.isin(sid, f); tr = ~te
        if te.sum() < 6 or tr.sum() < 20:
            continue
        P.append(ridge(X[tr], Y[tr], X[te])); T.append(Y[te]); S.append(sid[te])
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    return concat_r(demean(Pm, Sm), demean(Tm, Sm))


def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, v.size, replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def report(name, diffs, rows):
    m, lo, hi = boot_ci(diffs)
    n_pos = int(np.sum(np.asarray(diffs) > 0))
    ok = (lo > 0) and (n_pos >= len(diffs) - 1)
    rows.append({"test": name, "mean_diff": m, "ci_low": lo, "ci_high": hi,
                 "n_seeds": len(diffs), "n_positive": n_pos, "confirmed": ok})
    print(f"\n  paired difference: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  positive in {n_pos}/{len(diffs)} seeds")
    print(f"  -> {'CONFIRMED' if ok else 'NOT CONFIRMED (consistent with noise)'}")
    return ok


# ==================================================================== A
def test_a(name, eeg, Y, sid, rows):
    print(f"\n{'='*72}\nTEST A - finer time bins ({name})\n{'='*72}")
    X6, X12 = bandpower(eeg, 6), bandpower(eeg, 12)
    diffs = []
    print(f"  {'seed':>5s}{'6 bins':>10s}{'12 bins':>10s}{'diff':>10s}")
    for s in SEEDS_A:
        fl = folds_for(sid, s)                     # identical folds for both conditions
        r6, r12 = cv_ridge(X6, Y, sid, fl), cv_ridge(X12, Y, sid, fl)
        diffs.append(r12 - r6)
        print(f"  {s:>5d}{r6:>10.4f}{r12:>10.4f}{r12-r6:>+10.4f}")
    return report(f"A: 12-bin vs 6-bin ({name})", diffs, rows)


# ==================================================================== B
class Net(nn.Module):
    """Shared trunk + per-arm heads, linear skip, optional per-subject embedding."""
    def __init__(s, dims, outs, n_subj, hid=256, lat=128, adapter=False, emb=16):
        super().__init__()
        s.adapter = adapter
        s.adapt = nn.ModuleList([nn.Linear(d, hid) for d in dims])
        s.trunk = nn.Sequential(nn.ELU(), nn.Dropout(0.3), nn.Linear(hid, lat),
                                nn.ELU(), nn.Dropout(0.3), nn.Linear(lat, lat), nn.ELU())
        s.heads = nn.ModuleList([nn.Linear(lat, o) for o in outs])
        s.skip = nn.ModuleList([nn.Linear(d, o) for d, o in zip(dims, outs)])
        if adapter:
            s.emb = nn.ModuleList([nn.Embedding(n, emb) for n in n_subj])
            s.sub = nn.ModuleList([nn.Linear(emb, o) for o in outs])

    def forward(s, x, i, subj=None):
        out = s.heads[i](s.trunk(s.adapt[i](x))) + s.skip[i](x)
        if s.adapter and subj is not None:
            out = out + s.sub[i](s.emb[i](subj))
        return out


def cv_net(arms, eval_i, folds_per_arm, adapter, seed):
    P, T, S = [], [], []
    for f in range(K_FOLDS):
        prep = []
        for i, (X, Y, sid) in enumerate(arms):
            te = np.isin(sid, folds_per_arm[i][f]); tr = ~te
            xm, xs, ym = X[tr].mean(0), X[tr].std(0) + 1e-8, Y[tr].mean(0)
            uniq = {v: j for j, v in enumerate(np.unique(sid))}
            rid = np.array([uniq[v] for v in sid])
            prep.append({"Xtr": torch.tensor((X[tr] - xm) / xs),
                         "Ytr": torch.tensor(Y[tr] - ym),
                         "Str": torch.tensor(rid[tr]),
                         "Xte": torch.tensor((X[te] - xm) / xs),
                         "Ste": torch.tensor(rid[te]), "ym": ym, "te": te})
        torch.manual_seed(seed)
        net = Net([a[0].shape[1] for a in arms], [a[1].shape[1] for a in arms],
                  [len(np.unique(a[2])) for a in arms], adapter=adapter).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3)
        best, bad, state = np.inf, 0, None
        for ep in range(EPOCHS):
            net.train()
            for i in range(len(arms)):
                Xt, Yt = prep[i]["Xtr"], prep[i]["Ytr"]
                perm = torch.randperm(len(Xt))
                for j in range(0, len(Xt), BS):
                    k = perm[j:j + BS]
                    sb = prep[i]["Str"][k].to(DEV) if adapter else None
                    opt.zero_grad()
                    nn.functional.mse_loss(net(Xt[k].to(DEV), i, sb),
                                           Yt[k].to(DEV)).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                v = sum(nn.functional.mse_loss(
                    net(prep[i]["Xtr"].to(DEV), i,
                        prep[i]["Str"].to(DEV) if adapter else None),
                    prep[i]["Ytr"].to(DEV)).item() for i in range(len(arms)))
            if v < best - 1e-6:
                best, bad = v, 0
                state = {k_: t.detach().clone() for k_, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(state); net.eval()
        i = eval_i
        with torch.no_grad():
            sb = prep[i]["Ste"].to(DEV) if adapter else None
            pr = net(prep[i]["Xte"].to(DEV), i, sb).cpu().numpy() + prep[i]["ym"]
        P.append(pr); T.append(arms[i][1][prep[i]["te"]]); S.append(arms[i][2][prep[i]["te"]])
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    return concat_r(demean(Pm, Sm), demean(Tm, Sm))


def test_b(arms, rows):
    print(f"\n{'='*72}\nTEST B - subject-adapter vs single-modality (fNIRS)\n{'='*72}")
    diffs = []
    print(f"  {'seed':>5s}{'single':>10s}{'adapter':>10s}{'diff':>10s}")
    for s in SEEDS_B:
        fl = [folds_for(a[2], s) for a in arms]     # identical folds for both conditions
        r_single = cv_net([arms[0]], 0, [fl[0]], adapter=False, seed=s)
        r_adapt = cv_net(arms, 0, fl, adapter=True, seed=s)
        diffs.append(r_adapt - r_single)
        print(f"  {s:>5d}{r_single:>10.4f}{r_adapt:>10.4f}{r_adapt-r_single:>+10.4f}")
    return report("B: subject-adapter vs single (fNIRS)", diffs, rows)


# ==================================================================== main
def merge(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    return (np.concatenate([x["eeg"], y["eeg"]]),
            np.concatenate([x["hemo"], y["hemo"]]).astype(np.float32),
            np.concatenate([x["subject_ids"],
                            y["subject_ids"] + x["subject_ids"].max() + 1]))


def main():
    print("device:", DEV)
    rows = []

    e_fn, h_fn, s_fn = merge(V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz")
    Y_fn = h_fn.reshape(len(h_fn), -1)
    print(f"fNIRS: {len(e_fn)} trials, {np.unique(s_fn).size} subjects")
    test_a("fNIRS", e_fn, Y_fn, s_fn, rows)

    arms = None
    if (V4 / "ds002739.npz").exists():
        z = np.load(V4 / "ds002739.npz", allow_pickle=True)
        E, H, S = z["eeg"], z["hemo"].astype(np.float32), z["subject_ids"]
        if len(E) > N_FMRI:                        # ONE fixed subsample, reused everywhere
            k = np.random.default_rng(123).choice(len(E), N_FMRI, replace=False); k.sort()
            E, H, S = E[k], H[k], S[k]
        Y_fm = H.reshape(len(H), -1)
        print(f"\nfMRI: {len(E)} trials (fixed subsample), {np.unique(S).size} subjects")
        test_a("fMRI", E, Y_fm, S, rows)
        arms = [(bandpower(e_fn, 6), Y_fn, s_fn), (bandpower(E, 6), Y_fm, S)]
        del E, H
    del e_fn

    if arms is not None:
        test_b(arms, rows)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "confirmation.csv", index=False)
    print("\n" + "=" * 84)
    print(df.round(4).to_string(index=False))
    print("=" * 84)
    n_ok = int(df.confirmed.sum())
    print(f"\n{n_ok}/{len(df)} candidate gains confirmed.")
    if n_ok == 0:
        print("None survived. The apparent improvements were noise -- which is itself the")
        print("finding: at this effect size, protocol choices dominate the results.")
    print(f"\nwrote {TABLE_DIR/'confirmation.csv'}")


if __name__ == "__main__":
    main()
