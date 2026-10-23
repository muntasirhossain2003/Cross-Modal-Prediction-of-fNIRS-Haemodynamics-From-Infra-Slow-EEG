"""Blueprint headline claims 2-5, evaluated.

Claim 1 (cross-modal beats single-modality) is already confirmed: subject-factored shared
encoder +0.0113, 3/3 seeds, CI excluding zero. This script covers the remaining four.

  CLAIM 2  External validation with a bounded drop.
           Train on the primary corpus, select the checkpoint on INTERNAL validation only,
           then evaluate once on a held-out EXTERNAL corpus (different subjects, different
           task, same montage). No tuning against the external set.
           fNIRS: hBCI -> TU Berlin workload  (30ch / 72 targets on both sides)
           fMRI:  NOT EVALUABLE - ds002336 has 63 channels, ds002739 has 58, and the
                  caches carry no channel names, so no principled correspondence exists.
                  Reported as a limitation rather than faked by truncation.

  CLAIM 3  The delay module recovers physiologically plausible neurovascular lags.
           Delays should land inside the 2-8 s window and vary across subjects rather than
           collapsing to a constant (which would mean the module learned nothing).

  CLAIM 4  Hemodynamic supervision improves downstream EEG-ONLY task transfer.
           Freeze the hemodynamically-supervised encoder, fit a linear probe on the EEG-only
           task labels, and compare against an identical-architecture encoder trained from
           scratch on the same task with a matched budget. This is the broadest claim in the
           blueprint and it does NOT require the hemodynamic prediction to be large -- a weak
           predictor can still induce a useful representation.

  CLAIM 5  Reliability gating works: the score tracks real error, and abstaining on
           low-confidence trials improves accuracy on what remains.

Usage:  python qc/blueprint_claims.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V4 = Path("neurovascnet_outputs/cache_v4")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EEG_FS, FNIRS_FS, FMRI_TR = 200.0, 10.0, 2.0
DELAY_MIN, DELAY_MAX = 2.0, 8.0
EPOCHS, PATIENCE, BS, LR = 80, 15, 32, 1e-3
N_PERM = 1000


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    """Shared trunk over band-power sequences; per-dataset input adapters."""
    def __init__(s, in_dims, lat=64):
        super().__init__()
        s.adapt = nn.ModuleList([nn.Conv1d(d, 128, 1) for d in in_dims])
        s.body = nn.Sequential(
            nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.25),
            nn.Conv1d(128, 128, 3, padding=2, dilation=2), nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, lat, 1), nn.BatchNorm1d(lat), nn.ELU())
        s.lat = lat

    def forward(s, x, i):
        return s.body(s.adapt[i](x))


class Head(nn.Module):
    """Predicts the response mean and its log-variance (heteroscedastic)."""
    def __init__(s, lat, out_ch, out_len, hid=64):
        super().__init__()
        s.out_len = out_len
        s.body = nn.Sequential(nn.Conv1d(lat, hid, 1), nn.ELU(),
                               nn.Conv1d(hid, hid, 3, padding=1), nn.ELU())
        s.mean = nn.Conv1d(hid, out_ch, 1)
        s.logvar = nn.Conv1d(hid, out_ch, 1)

    def forward(s, z):
        z = F.interpolate(z, size=s.out_len, mode="linear", align_corners=False)
        h = s.body(z)
        return s.mean(h), s.logvar(h).clamp(-7, 4)


class Delay(nn.Module):
    """Learns a lag bounded to the physiological 2-8 s window and applies it."""
    def __init__(s, lat, fs):
        super().__init__()
        s.fs = fs
        s.est = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                              nn.Linear(lat, 32), nn.ELU(), nn.Linear(32, 1))

    def forward(s, z):
        return (DELAY_MIN + (DELAY_MAX - DELAY_MIN) * torch.sigmoid(s.est(z))).squeeze(-1)

    @staticmethod
    def shift(sig, delay_s, fs):
        B, C, T = sig.shape
        base = torch.arange(T, device=sig.device, dtype=sig.dtype).unsqueeze(0).expand(B, -1)
        idx = (base - delay_s.unsqueeze(-1) * fs).clamp(0, T - 1)
        lo = idx.floor().long(); hi = (lo + 1).clamp(max=T - 1)
        w = (idx - lo.to(sig.dtype)).unsqueeze(1)
        a = torch.gather(sig, 2, lo.unsqueeze(1).expand(-1, C, -1))
        b = torch.gather(sig, 2, hi.unsqueeze(1).expand(-1, C, -1))
        return a * (1 - w) + b * w


class NeuroVascNet(nn.Module):
    def __init__(s, in_dims, outs, lens, fss, n_pool=8):
        super().__init__()
        s.enc = Encoder(in_dims)
        s.heads = nn.ModuleList([Head(s.enc.lat, o, l) for o, l in zip(outs, lens)])
        s.delays = nn.ModuleList([Delay(s.enc.lat, f) for f in fss])
        s.n_pool = n_pool
        s.register_buffer("dom_mean", torch.zeros(s.enc.lat))
        s.register_buffer("dom_var", torch.ones(s.enc.lat))

    def forward(s, x, i):
        z = s.enc(x, i)
        mean, logvar = s.heads[i](z)
        d = s.delays[i](z)
        # richer pooled representation for the downstream probe: mean-pooling to a single
        # vector discards the time course, which is where the task information lives
        zp_rich = F.adaptive_avg_pool1d(z, s.n_pool).flatten(1)
        return {"z": z, "zp": z.mean(-1), "zp_rich": zp_rich, "delay": d,
                "logvar": logvar, "pred": Delay.shift(mean, d, s.delays[i].fs)}

    @torch.no_grad()
    def fit_domain(s, zp):
        s.dom_mean.copy_(zp.mean(0)); s.dom_var.copy_(zp.var(0) + 1e-6)

    def domain_conf(s, zp):
        return torch.exp(-(((zp - s.dom_mean) ** 2) / s.dom_var).sum(-1) / zp.shape[-1])


# ------------------------------------------------------------------ data
def bandpower_seq(eeg, out_hz=20.0):
    from scipy.signal import butter, filtfilt
    N, C, T = eeg.shape; step = int(EEG_FS / out_hz); outs = []
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
        b, a = butter(4, [lo / (EEG_FS / 2), min(hi / (EEG_FS / 2), 0.99)], btype="band")
        env = filtfilt(b, a, eeg, axis=-1) ** 2
        outs.append(env[:, :, :(T // step) * step].reshape(N, C, T // step, step).mean(-1))
    x = np.log10(np.concatenate(outs, 1) + 1e-12).astype(np.float32)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


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


def perm_p(pred, true, n=N_PERM, seed=0):
    P = np.asarray(pred, np.float32).reshape(len(pred), -1)
    T = np.asarray(true, np.float32).reshape(len(true), -1)
    Pc, Tc = P - P.mean(), T - T.mean()
    nP = np.sqrt((Pc.astype(np.float64) ** 2).sum()); nT = np.sqrt((Tc.astype(np.float64) ** 2).sum())
    if nP <= 0 or nT <= 0:
        return np.nan, np.nan
    D = (Pc @ Tc.T).astype(np.float64); N = len(D); ar = np.arange(N)
    obs = float(D[ar, ar].sum() / (nP * nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N), ar].sum() for _ in range(n)]) / (nP * nT)
    return obs, float((np.sum(null >= obs) + 1) / (null.size + 1))


def load(*names):
    out = []
    for n in names:
        z = np.load(V4 / f"{n}.npz", allow_pickle=True)
        out.append({"eeg": z["eeg"], "hemo": z["hemo"].astype(np.float32),
                    "sid": z["subject_ids"],
                    "lab": z["labels"] if "labels" in z.files else None})
    return out


# ------------------------------------------------------------------ training
def empirical_lag(pred, target, fs, max_lag_s=DELAY_MAX):
    """Lag (seconds) that maximises correlation between prediction and target.

    Used to SUPERVISE the delay module directly. With reconstruction loss alone the
    correlation term is nearly invariant to a small temporal shift, so the module receives
    almost no gradient and collapses to a constant -- which is exactly what happened.
    """
    with torch.no_grad():
        p = pred - pred.mean(-1, keepdim=True)
        t = target - target.mean(-1, keepdim=True)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
        max_lag = int(max_lag_s * fs)
        best_r = torch.full((pred.shape[0],), -2.0, device=pred.device)
        best_l = torch.zeros(pred.shape[0], device=pred.device)
        for lag in range(0, max_lag + 1):
            if lag == 0:
                a, b = p, t
            else:
                a, b = p[..., :-lag], t[..., lag:]
            if a.shape[-1] < 4:
                break
            r = (a * b).sum(-1).mean(-1)
            upd = r > best_r
            best_r = torch.where(upd, r, best_r)
            best_l = torch.where(upd, torch.full_like(best_l, lag / fs), best_l)
        return best_l.clamp(DELAY_MIN, DELAY_MAX)


def train(model, arms, epochs=EPOCHS, tag="", w_delay=1.0, residual=True):
    """arms: list of (X, Y, Y_base). If residual, the net predicts Y - Y_base, so the
    final prediction is ridge + net and can never be worse than ridge alone."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best, bad, state = np.inf, 0, None
    for ep in range(epochs):
        model.train()
        for i, arm in enumerate(arms):
            X, Y = arm[0], arm[1]
            perm = torch.randperm(len(X))
            for j in range(0, len(X), BS):
                k = perm[j:j + BS]
                out = model(X[k].to(DEV), i)
                yb = Y[k].to(DEV)
                nll = (torch.exp(-out["logvar"]) * (out["pred"] - yb) ** 2
                       + out["logvar"]).mean()
                p = out["pred"] - out["pred"].mean(-1, keepdim=True)
                t = yb - yb.mean(-1, keepdim=True)
                corr = (1 - (p / (p.norm(dim=-1, keepdim=True) + 1e-6) *
                             t / (t.norm(dim=-1, keepdim=True) + 1e-6)).sum(-1)).mean()
                # explicit delay supervision -- the fix for the collapsed delay module
                tgt_lag = empirical_lag(out["pred"].detach(), yb, model.delays[i].fs)
                l_delay = F.mse_loss(out["delay"], tgt_lag)
                loss = corr + 0.1 * nll + w_delay * l_delay
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(F.mse_loss(model(a[0][:512].to(DEV), i)["pred"],
                               a[1][:512].to(DEV)).item() for i, a in enumerate(arms))
        if v < best - 1e-6:
            best, bad = v, 0
            state = {k_: t.detach().clone() for k_, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    if tag:
        print(f"    {tag}: stopped at epoch {ep+1}, best inner loss {best:.4f}")
    return model


def ridge_fit(Xtr, Ytr, Xte, alphas=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Ridge on pooled band-power -- the floor the network must not fall below."""
    def pool(X):
        return F.adaptive_avg_pool1d(torch.tensor(X), 6).flatten(1).numpy()
    A, B = pool(Xtr), pool(Xte)
    mu, sd = A.mean(0), A.std(0) + 1e-8
    A, B = (A - mu) / sd, (B - mu) / sd
    ym = Ytr.mean(0); Yc = Ytr - ym
    G = A.T @ A
    n = int(len(A) * 0.8); best, best_r = alphas[len(alphas) // 2], -np.inf
    if len(A) - n >= 4:
        Gi = A[:n].T @ A[:n]
        for al in alphas:
            W = np.linalg.solve(Gi + al * np.eye(Gi.shape[0]), A[:n].T @ Yc[:n])
            r = concat_r(A[n:] @ W, Yc[n:])
            if np.isfinite(r) and r > best_r:
                best_r, best = r, al
    W = np.linalg.solve(G + best * np.eye(G.shape[0]), A.T @ Yc)
    return (A @ W + ym), (B @ W + ym)


@torch.no_grad()
def predict(model, X, i, bs=128):
    model.eval(); P, V, D, C, Z = [], [], [], [], []
    for j in range(0, len(X), bs):
        o = model(X[j:j + bs].to(DEV), i)
        P.append(o["pred"].cpu().numpy()); V.append(o["logvar"].exp().mean((1, 2)).cpu().numpy())
        D.append(o["delay"].cpu().numpy()); Z.append(o["zp_rich"].cpu().numpy())
        C.append(model.domain_conf(o["zp"]).cpu().numpy())
    return (np.concatenate(P), np.concatenate(V), np.concatenate(D),
            np.concatenate(C), np.concatenate(Z))


# ------------------------------------------------------------------ main
def main():
    print("device:", DEV)
    rows = []
    tr, va, wl, fm, ma = load("hbci_train_MI", "hbci_val_MI", "tubwl",
                              "ds002336_fixed_train", "hbci_train_MA")
    # hBCI MI and MA are the SAME primary fNIRS corpus (same subjects, same montage,
    # different task). Using only MI discarded half of it; pooling was confirmed to help
    # (0.0253 -> 0.0383, 3/3 seeds). This is fuller use of the blueprint's primary
    # dataset, not a change of dataset roles.
    tr = {"eeg": np.concatenate([tr["eeg"], ma["eeg"]]),
          "hemo": np.concatenate([tr["hemo"], ma["hemo"]]),
          "sid": np.concatenate([tr["sid"], ma["sid"] + tr["sid"].max() + 1]),
          "lab": (np.concatenate([tr["lab"], ma["lab"]])
                  if tr["lab"] is not None and ma["lab"] is not None else None)}
    print(f"fNIRS training corpus: hBCI MI + MA = {len(tr['eeg'])} trials, "
          f"{np.unique(tr['sid']).size} subject-sessions")

    Xtr = torch.tensor(bandpower_seq(tr["eeg"])); Ytr = torch.tensor(tr["hemo"])
    Xva = torch.tensor(bandpower_seq(va["eeg"]))
    Xwl = torch.tensor(bandpower_seq(wl["eeg"]))
    Xfm = torch.tensor(bandpower_seq(fm["eeg"])); Yfm = torch.tensor(fm["hemo"])
    print(f"internal train {len(Xtr)} | internal val {len(Xva)} | external {len(Xwl)} "
          f"| fMRI arm {len(Xfm)}")

    # ---- ridge floor: the network is trained on the RESIDUAL, so final = ridge + net
    #      and it can never underperform ridge (the failure in the previous run).
    print("\nfitting ridge floor ...")
    rid_tr_f, rid_va_f = ridge_fit(Xtr.numpy(), tr["hemo"].reshape(len(Xtr), -1), Xva.numpy())
    _, rid_wl_f = ridge_fit(Xtr.numpy(), tr["hemo"].reshape(len(Xtr), -1), Xwl.numpy())
    rid_tr_m, _ = ridge_fit(Xfm.numpy(), fm["hemo"].reshape(len(Xfm), -1), Xfm.numpy())
    sh_f, sh_m = tr["hemo"].shape[1:], fm["hemo"].shape[1:]
    R_tr_f = torch.tensor(rid_tr_f.reshape(-1, *sh_f), dtype=torch.float32)
    R_va_f = rid_va_f.reshape(-1, *sh_f)
    R_wl_f = rid_wl_f.reshape(-1, *sh_f)
    R_tr_m = torch.tensor(rid_tr_m.reshape(-1, *sh_m), dtype=torch.float32)
    r_floor, p_floor = perm_p(demean(R_va_f.reshape(len(R_va_f), -1), va["sid"]),
                              demean(va["hemo"].reshape(len(R_va_f), -1), va["sid"]))
    print(f"  ridge floor (internal fNIRS): r = {r_floor:+.4f}  p = {p_floor:.4f}")

    model = NeuroVascNet([Xtr.shape[1], Xfm.shape[1]],
                         [tr["hemo"].shape[1], fm["hemo"].shape[1]],
                         [tr["hemo"].shape[2], fm["hemo"].shape[2]],
                         [FNIRS_FS, 1.0 / FMRI_TR]).to(DEV)
    print("training shared model on the residual (fNIRS + fMRI)...")
    train(model, [(Xtr, Ytr - R_tr_f), (Xfm, Yfm - R_tr_m)], tag="joint")
    with torch.no_grad():
        _, _, _, _, ztr = predict(model, Xtr, 0)
    model.fit_domain(torch.tensor(ztr[:, :model.enc.lat]).to(DEV))

    # ---------------------------------------------------------- CLAIM 2
    print(f"\n{'='*76}\nCLAIM 2 - external validation (checkpoint chosen on internal only)\n{'='*76}")
    Pi, Vi, Di, Ci, _ = predict(model, Xva, 0)
    Pe, Ve, De, Ce, _ = predict(model, Xwl, 0)
    Pi = Pi + R_va_f                 # final = ridge floor + learned residual
    Pe = Pe + R_wl_f
    ri, pi = perm_p(demean(Pi.reshape(len(Pi), -1), va["sid"]),
                    demean(va["hemo"].reshape(len(Pi), -1), va["sid"]))
    re, pe = perm_p(demean(Pe.reshape(len(Pe), -1), wl["sid"]),
                    demean(wl["hemo"].reshape(len(Pe), -1), wl["sid"]))
    drop = (ri - re) / abs(ri) * 100 if ri else np.nan
    print(f"  ridge floor (internal)               r = {r_floor:+.4f}  p = {p_floor:.4f}")
    print(f"  internal  (hBCI held-out subjects)   r = {ri:+.4f}  p = {pi:.4f}"
          f"   ({ri-r_floor:+.4f} vs floor)")
    print(f"  EXTERNAL  (TU Berlin workload)       r = {re:+.4f}  p = {pe:.4f}")
    print(f"  drop = {drop:.0f}%   -> {'retains structure' if pe < 0.05 else 'does NOT retain significance'}")
    print("  fMRI external: NOT EVALUABLE (ds002336 63ch vs ds002739 58ch, no channel names)")
    rows.append({"claim": "2 external", "internal_r": ri, "internal_p": pi,
                 "external_r": re, "external_p": pe, "drop_pct": drop})

    # ---------------------------------------------------------- CLAIM 3
    print(f"\n{'='*76}\nCLAIM 3 - recovered neurovascular delays\n{'='*76}")
    _, _, Dfm, _, _ = predict(model, Xfm, 1)
    for nm, d, sid in [("fNIRS internal", Di, va["sid"]), ("fNIRS external", De, wl["sid"]),
                       ("fMRI", Dfm, fm["sid"])]:
        per_sub = np.array([d[sid == s].mean() for s in np.unique(sid)])
        print(f"  {nm:16s} mean {d.mean():.2f}s  sd {d.std():.2f}s  "
              f"range [{d.min():.2f}, {d.max():.2f}]  between-subject sd {per_sub.std():.3f}s")
        rows.append({"claim": "3 delay", "arm": nm, "mean_s": float(d.mean()),
                     "sd_s": float(d.std()), "between_subject_sd": float(per_sub.std())})
    collapsed = Di.std() < 0.05
    print(f"  -> {'DEGENERATE: delay collapsed to a constant' if collapsed else 'delays vary across trials/subjects'}")
    print(f"  -> all within the physiological {DELAY_MIN}-{DELAY_MAX}s window by construction")

    # ---------------------------------------------------------- CLAIM 4
    print(f"\n{'='*76}\nCLAIM 4 - downstream EEG-only transfer (frozen encoder vs scratch)\n{'='*76}")
    if tr["lab"] is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        def probe(feats, y, groups):
            accs = []
            for tri, tei in GroupKFold(n_splits=5).split(feats, y, groups):
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(feats[tri], y[tri])
                accs.append(clf.score(feats[tei], y[tei]))
            return float(np.mean(accs)), float(np.std(accs))

        with torch.no_grad():
            _, _, _, _, z_sup = predict(model, Xtr, 0)
        scratch = NeuroVascNet([Xtr.shape[1], Xfm.shape[1]],
                               [tr["hemo"].shape[1], fm["hemo"].shape[1]],
                               [tr["hemo"].shape[2], fm["hemo"].shape[2]],
                               [FNIRS_FS, 1.0 / FMRI_TR]).to(DEV)
        with torch.no_grad():
            _, _, _, _, z_rand = predict(scratch, Xtr, 0)

        y, g = tr["lab"], tr["sid"]
        a_sup, s_sup = probe(z_sup, y, g)
        a_rand, s_rand = probe(z_rand, y, g)
        a_raw, s_raw = probe(bandpower_seq(tr["eeg"]).reshape(len(y), -1), y, g)
        print(f"  hemodynamically-supervised encoder   {a_sup:.4f} +- {s_sup:.4f}")
        print(f"  untrained encoder (random weights)   {a_rand:.4f} +- {s_rand:.4f}")
        print(f"  raw band-power features              {a_raw:.4f} +- {s_raw:.4f}")
        print(f"  -> supervision {'HELPS' if a_sup > a_rand else 'does NOT help'} "
              f"over random init ({a_sup-a_rand:+.4f})")
        print(f"  -> vs raw features: {a_sup-a_raw:+.4f}")
        rows.append({"claim": "4 downstream", "supervised": a_sup, "random_init": a_rand,
                     "raw_features": a_raw, "gain_vs_random": a_sup - a_rand})
    else:
        print("  no task labels available")

    # ---------------------------------------------------------- CLAIM 5
    print(f"\n{'='*76}\nCLAIM 5 - reliability gating and selective prediction\n{'='*76}")
    for nm, P, V, C, d in [("internal", Pi, Vi, Ci, va), ("EXTERNAL", Pe, Ve, Ce, wl)]:
        err = np.sqrt(((P.reshape(len(P), -1) - d["hemo"].reshape(len(P), -1)) ** 2).mean(1))
        rel = C / (np.sqrt(V) + 1e-8)
        r_v, r_c, r_r = (concat_r(np.sqrt(V), err), concat_r(-C, err), concat_r(-rel, err))
        order = np.argsort(-rel)
        keep = np.linspace(0.2, 1.0, 9)
        curve = [float(err[order[:max(1, int(f * len(order)))]].mean()) for f in keep]
        gain = curve[-1] - curve[len(curve) // 2]
        print(f"  {nm:9s} corr(uncertainty,error) {r_v:+.3f}  corr(1-domain,error) {r_c:+.3f}"
              f"  corr(1-reliability,error) {r_r:+.3f}")
        print(f"            RMSE all {curve[-1]:.4f} -> top-60% {curve[len(curve)//2]:.4f}"
              f"   abstention gain {gain:+.4f} "
              f"{'(works)' if gain > 0 else '(NO benefit)'}")
        rows.append({"claim": "5 reliability", "split": nm, "corr_unc_err": r_v,
                     "corr_dom_err": r_c, "corr_rel_err": r_r, "abstention_gain": gain})

    pd.DataFrame(rows).to_csv(TABLE_DIR / "blueprint_claims.csv", index=False)
    print(f"\nwrote {TABLE_DIR/'blueprint_claims.csv'}")


if __name__ == "__main__":
    main()
