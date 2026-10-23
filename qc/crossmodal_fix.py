"""Can cross-modal supervision be made to work? A fair test, plus three principled fixes.

The first cross-modal test was HANDICAPPED: its MLP scored fNIRS r=0.0084 where plain
ridge scores 0.0253. A shared-encoder claim tested with a model 3x weaker than linear
regression proves nothing. This fixes that and then tests the fixes.

THE ENCODER FIX
    Every model here carries an explicit LINEAR SKIP path (output = Linear(x) + MLP(x)),
    so it can always represent ridge exactly and can only improve on it. If the single-
    modality arm does not now match ~0.025, the comparison is still broken and the result
    should be discarded.

THE DIAGNOSIS BEING TESTED
    Cross-modal supervision failed (fNIRS 0.0084->0.0140 n.s., fMRI 0.0244->0.0196 worse).
    The likely reason: the mapping is SUBJECT-SPECIFIC (we measured task transfer r=0.058
    vs subject transfer r=0.009), and the two arms share ZERO subjects -- so a shared
    encoder has nothing subject-general to transfer.

    Three fixes follow directly from that diagnosis:

    A. JOINT           plain shared trunk (the original claim)
    B. JOINT + ALIGN   moment-matching loss pulling the two latent distributions together.
                       CORAL was ruled out in v3.3 -- but that was on the BROKEN HEFMI
                       target, so it has never had a fair test.
    C. JOINT + ADAPTER shared trunk carries what is subject-GENERAL; a small per-subject
                       embedding carries what is subject-SPECIFIC. This is the architecture
                       our own finding implies.

Scored identically to everything else: subject-wise CV, subject-demeaned (constant
predictor = 0), exact permutation null.

Usage:  python qc/crossmodal_fix.py
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
N_BINS, N_PERM, K_FOLDS = 6, 1000, 5
EPOCHS, PATIENCE, BS, LR = 200, 25, 64, 1e-3
MAX_FMRI = 2500
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


# ------------------------------------------------------------------ model
class Net(nn.Module):
    """Shared trunk + per-arm adapters/heads, with a LINEAR SKIP so it can always
    represent ridge. Optional per-subject embedding carries subject-specific offset."""

    def __init__(s, dims, outs, n_subj, hid=256, lat=128, use_adapter=False, emb=16):
        super().__init__()
        s.use_adapter = use_adapter
        s.adapt = nn.ModuleList([nn.Linear(d, hid) for d in dims])
        s.trunk = nn.Sequential(nn.ELU(), nn.Dropout(0.3), nn.Linear(hid, lat),
                                nn.ELU(), nn.Dropout(0.3), nn.Linear(lat, lat), nn.ELU())
        s.heads = nn.ModuleList([nn.Linear(lat, o) for o in outs])
        s.skip = nn.ModuleList([nn.Linear(d, o) for d, o in zip(dims, outs)])
        if use_adapter:
            s.emb = nn.ModuleList([nn.Embedding(n, emb) for n in n_subj])
            s.sub_head = nn.ModuleList([nn.Linear(emb, o) for o in outs])

    def latent(s, x, i):
        return s.trunk(s.adapt[i](x))

    def forward(s, x, i, subj=None):
        out = s.heads[i](s.latent(x, i)) + s.skip[i](x)
        if s.use_adapter and subj is not None:
            out = out + s.sub_head[i](s.emb[i](subj))
        return out


def moment_loss(za, zb):
    md = (za.mean(0) - zb.mean(0)).pow(2).mean()
    cd = (torch.cov(za.T) - torch.cov(zb.T)).pow(2).mean()
    return md + cd


# ------------------------------------------------------------------ training
def run_variant(arms, train_on, eval_i, mode, tag, rows, w_align=0.0):
    """arms: list of (X, Y, sid). train_on: indices to train. eval_i: arm to score."""
    folds = [np.array_split(np.random.default_rng(0).permutation(np.unique(a[2])),
                            K_FOLDS) for a in arms]
    P, T, S = [], [], []
    for f in range(K_FOLDS):
        te_masks = [np.isin(a[2], folds[i][f]) for i, a in enumerate(arms)]
        prep, remap = [], []
        for i, (X, Y, sid) in enumerate(arms):
            m = ~te_masks[i]
            xm, xs = X[m].mean(0), X[m].std(0) + 1e-8
            ym = Y[m].mean(0)
            uniq = {s_: j for j, s_ in enumerate(np.unique(sid))}
            remap.append(np.array([uniq[s_] for s_ in sid]))
            prep.append({"Xtr": torch.tensor((X[m] - xm) / xs),
                         "Ytr": torch.tensor(Y[m] - ym),
                         "Str": torch.tensor(remap[i][m]),
                         "Xte": torch.tensor((X[te_masks[i]] - xm) / xs),
                         "Ste": torch.tensor(remap[i][te_masks[i]]),
                         "ym": ym})
        torch.manual_seed(0)
        net = Net([a[0].shape[1] for a in arms], [a[1].shape[1] for a in arms],
                  [len(np.unique(a[2])) for a in arms],
                  use_adapter=(mode == "adapter")).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3)
        best, bad, state = np.inf, 0, None
        for ep in range(EPOCHS):
            net.train()
            n = min(len(prep[i]["Xtr"]) for i in train_on)
            perm = {i: torch.randperm(len(prep[i]["Xtr"])) for i in train_on}
            for j in range(0, n, BS):
                opt.zero_grad()
                loss, lats = 0.0, {}
                for i in train_on:
                    k = perm[i][j:j + BS]
                    xb = prep[i]["Xtr"][k].to(DEV)
                    yb = prep[i]["Ytr"][k].to(DEV)
                    sb = prep[i]["Str"][k].to(DEV) if mode == "adapter" else None
                    loss = loss + nn.functional.mse_loss(net(xb, i, sb), yb)
                    if w_align > 0:
                        lats[i] = net.latent(xb, i)
                if w_align > 0 and len(lats) == 2:
                    a_, b_ = list(lats.values())
                    m = min(len(a_), len(b_))
                    loss = loss + w_align * moment_loss(a_[:m], b_[:m])
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = sum(nn.functional.mse_loss(
                    net(prep[i]["Xtr"].to(DEV), i,
                        prep[i]["Str"].to(DEV) if mode == "adapter" else None),
                    prep[i]["Ytr"].to(DEV)).item() for i in train_on)
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
            sb = prep[i]["Ste"].to(DEV) if mode == "adapter" else None
            pr = net(prep[i]["Xte"].to(DEV), i, sb).cpu().numpy() + prep[i]["ym"]
        P.append(pr); T.append(arms[i][1][te_masks[i]]); S.append(arms[i][2][te_masks[i]])
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    obs, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
    rows.append({"setting": tag, "r_demeaned": obs, "null": null, "p": p})
    print(f"  {tag:48s} r {obs:+.4f}  p={p:.4f}{' *' if p < 0.05 else ''}")
    return obs


def merge(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    return (np.concatenate([x["eeg"], y["eeg"]]),
            np.concatenate([x["hemo"], y["hemo"]]).astype(np.float32),
            np.concatenate([x["subject_ids"],
                            y["subject_ids"] + x["subject_ids"].max() + 1]))


def main():
    print("device:", DEV)
    fn = merge(V4 / "hbci_train_MI.npz", V4 / "hbci_val_MI.npz")
    z = np.load(V4 / "ds002739.npz", allow_pickle=True)
    E, H, S = z["eeg"], z["hemo"].astype(np.float32), z["subject_ids"]
    if len(E) > MAX_FMRI:
        k = np.random.default_rng(0).choice(len(E), MAX_FMRI, replace=False); k.sort()
        E, H, S = E[k], H[k], S[k]

    arms = [(features(fn[0]), fn[1].reshape(len(fn[1]), -1), fn[2]),
            (features(E), H.reshape(len(H), -1), S)]
    names = ["fNIRS", "fMRI"]
    print(f"fNIRS {arms[0][0].shape[0]} trials / {np.unique(arms[0][2]).size} subj")
    print(f"fMRI  {arms[1][0].shape[0]} trials / {np.unique(arms[1][2]).size} subj")

    rows = []
    print("\n" + "=" * 74)
    print("REFERENCE: ridge (what the encoder must at least match)")
    print("=" * 74)
    ridge_ref = {}
    for i, nm in enumerate(names):
        X, Y, sid = arms[i]
        fold = np.array_split(np.random.default_rng(0).permutation(np.unique(sid)), K_FOLDS)
        P, T, Sx = [], [], []
        for f in range(K_FOLDS):
            te = np.isin(sid, fold[f]); tr = ~te
            P.append(ridge(X[tr], Y[tr], X[te])); T.append(Y[te]); Sx.append(sid[te])
        o, _, p = perm_p(demean(np.vstack(P), np.concatenate(Sx)),
                         demean(np.vstack(T), np.concatenate(Sx)))
        ridge_ref[nm] = o
        rows.append({"setting": f"{nm} ridge", "r_demeaned": o, "p": p})
        print(f"  {nm+' ridge':48s} r {o:+.4f}  p={p:.4f}")

    results = {}
    for i, nm in enumerate(names):
        print("\n" + "=" * 74)
        print(f"{nm} ARM")
        print("=" * 74)
        results[(nm, "single")] = run_variant(arms, [i], i, "plain", f"{nm} single-modality", rows)
        results[(nm, "joint")] = run_variant(arms, [0, 1], i, "plain", f"{nm} + joint (A)", rows)
        results[(nm, "align")] = run_variant(arms, [0, 1], i, "plain", f"{nm} + joint + align (B)", rows, w_align=0.1)
        results[(nm, "adapter")] = run_variant(arms, [0, 1], i, "adapter", f"{nm} + joint + subject-adapter (C)", rows)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "crossmodal_fix.csv", index=False)

    print("\n" + "=" * 82)
    print(f"{'arm':8s}{'ridge':>10s}{'single':>10s}{'joint':>10s}{'+align':>10s}{'+adapter':>11s}")
    for nm in names:
        print(f"{nm:8s}{ridge_ref[nm]:>10.4f}{results[(nm,'single')]:>10.4f}"
              f"{results[(nm,'joint')]:>10.4f}{results[(nm,'align')]:>10.4f}"
              f"{results[(nm,'adapter')]:>11.4f}")
    print("=" * 82)

    fair = all(results[(nm, "single")] >= 0.6 * ridge_ref[nm] for nm in names)
    print("\nENCODER SANITY: single-modality vs ridge ->",
          "OK, comparison is fair" if fair else
          "STILL HANDICAPPED - the encoder underperforms ridge, discard the comparison")
    if fair:
        best = {nm: max(["joint", "align", "adapter"],
                        key=lambda k: results[(nm, k)]) for nm in names}
        for nm in names:
            b = best[nm]
            d = results[(nm, b)] - results[(nm, "single")]
            verdict = "HELPS" if d > 0 else "does NOT help"
            print(f"  {nm}: best cross-modal variant '{b}' {verdict} "
                  f"({results[(nm,'single')]:+.4f} -> {results[(nm,b)]:+.4f}, {d:+.4f})")
    print(f"\nwrote {TABLE_DIR/'crossmodal_fix.csv'}")


if __name__ == "__main__":
    main()
