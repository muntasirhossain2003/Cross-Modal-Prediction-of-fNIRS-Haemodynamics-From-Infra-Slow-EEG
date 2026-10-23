"""Claim 3 (delay bound) and Claim 4 (shuffled control) -- the two open questions.

CLAIM 3 -- where does the delay actually want to sit?
    The module saturated at whichever bound it was given: 7.59 s with reconstruction loss
    alone, then 2.12 s after explicit lag supervision was added. Both times it pinned to an
    edge of the physiological [2, 8] s window, which means the WINDOW was binding rather
    than describing. Widening to [0, 12] s lets the data choose. If it then settles in the
    interior, the recovered lag is a measurement; if it pins again, the parameterisation is
    at fault, not the physiology.

CLAIM 4 -- is the degradation specific to HEMODYNAMIC supervision?
    A frozen hemodynamically-supervised encoder scored BELOW random initialisation on
    EEG-only task decoding (0.573 vs 0.588), consistently across three runs. But that
    result is ambiguous: it could mean
        (a) hemodynamic targets specifically damage the representation, or
        (b) training on ANY low-SNR target damages it.
    The control that separates them: train an identical encoder on SHUFFLED hemodynamic
    targets -- same statistics, same difficulty, no real EEG->haemodynamic relationship.
        shuffled ~ real      -> (b) generic low-SNR damage
        shuffled ~ random    -> (a) specific to hemodynamic supervision

Usage:  python qc/claims_3_4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

V4 = Path("neurovascnet_outputs/cache_v4")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EEG_FS, FNIRS_FS = 200.0, 10.0
EPOCHS, PATIENCE, BS, LR = 80, 15, 32, 1e-3
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    def __init__(s, in_dim, lat=64):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.25),
            nn.Conv1d(128, 128, 3, padding=2, dilation=2), nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, lat, 1), nn.BatchNorm1d(lat), nn.ELU())
        s.lat = lat

    def forward(s, x):
        return s.net(x)


class Model(nn.Module):
    def __init__(s, in_dim, out_ch, out_len, dmin, dmax, hid=64, n_pool=8):
        super().__init__()
        s.enc = Encoder(in_dim)
        s.dmin, s.dmax, s.out_len, s.n_pool = dmin, dmax, out_len, n_pool
        s.body = nn.Sequential(nn.Conv1d(s.enc.lat, hid, 1), nn.ELU(),
                               nn.Conv1d(hid, hid, 3, padding=1), nn.ELU())
        s.mean = nn.Conv1d(hid, out_ch, 1)
        s.est = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                              nn.Linear(s.enc.lat, 32), nn.ELU(), nn.Linear(32, 1))

    def forward(s, x):
        z = s.enc(x)
        d = (s.dmin + (s.dmax - s.dmin) * torch.sigmoid(s.est(z))).squeeze(-1)
        h = s.body(F.interpolate(z, size=s.out_len, mode="linear", align_corners=False))
        return {"pred": shift(s.mean(h), d, FNIRS_FS), "delay": d,
                "feat": F.adaptive_avg_pool1d(z, s.n_pool).flatten(1)}


def shift(sig, delay_s, fs):
    B, C, T = sig.shape
    base = torch.arange(T, device=sig.device, dtype=sig.dtype).unsqueeze(0).expand(B, -1)
    idx = (base - delay_s.unsqueeze(-1) * fs).clamp(0, T - 1)
    lo = idx.floor().long(); hi = (lo + 1).clamp(max=T - 1)
    w = (idx - lo.to(sig.dtype)).unsqueeze(1)
    a = torch.gather(sig, 2, lo.unsqueeze(1).expand(-1, C, -1))
    b = torch.gather(sig, 2, hi.unsqueeze(1).expand(-1, C, -1))
    return a * (1 - w) + b * w


def empirical_lag(pred, target, fs, dmin, dmax):
    with torch.no_grad():
        p = pred - pred.mean(-1, keepdim=True); t = target - target.mean(-1, keepdim=True)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
        best_r = torch.full((pred.shape[0],), -2.0, device=pred.device)
        best_l = torch.zeros(pred.shape[0], device=pred.device)
        for lag in range(0, int(dmax * fs) + 1):
            a, b = (p, t) if lag == 0 else (p[..., :-lag], t[..., lag:])
            if a.shape[-1] < 4:
                break
            r = (a * b).sum(-1).mean(-1)
            upd = r > best_r
            best_r = torch.where(upd, r, best_r)
            best_l = torch.where(upd, torch.full_like(best_l, lag / fs), best_l)
        return best_l.clamp(dmin, dmax)


def bandpower_seq(eeg, out_hz=20.0):
    from scipy.signal import butter, filtfilt
    N, C, T = eeg.shape; step = int(EEG_FS / out_hz); outs = []
    for lo, hi in BANDS:
        b, a = butter(4, [lo / (EEG_FS / 2), min(hi / (EEG_FS / 2), 0.99)], btype="band")
        env = filtfilt(b, a, eeg, axis=-1) ** 2
        outs.append(env[:, :, :(T // step) * step].reshape(N, C, T // step, step).mean(-1))
    x = np.log10(np.concatenate(outs, 1) + 1e-12).astype(np.float32)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def train(X, Y, dmin, dmax, w_delay=1.0, tag=""):
    m = Model(X.shape[1], Y.shape[1], Y.shape[2], dmin, dmax).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    best, bad, state = np.inf, 0, None
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xt))
        for j in range(0, len(Xt), BS):
            k = perm[j:j + BS]
            xb, yb = Xt[k].to(DEV), Yt[k].to(DEV)
            o = m(xb)
            p = o["pred"] - o["pred"].mean(-1, keepdim=True)
            t = yb - yb.mean(-1, keepdim=True)
            corr = (1 - (p / (p.norm(dim=-1, keepdim=True) + 1e-6) *
                         t / (t.norm(dim=-1, keepdim=True) + 1e-6)).sum(-1)).mean()
            lag = empirical_lag(o["pred"].detach(), yb, FNIRS_FS, dmin, dmax)
            loss = corr + w_delay * F.mse_loss(o["delay"], lag)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = F.mse_loss(m(Xt[:512].to(DEV))["pred"], Yt[:512].to(DEV)).item()
        if v < best - 1e-6:
            best, bad = v, 0
            state = {k_: t.detach().clone() for k_, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    if tag:
        print(f"    {tag}: stopped ep {ep+1}")
    return m


@torch.no_grad()
def infer(m, X, bs=128):
    D, Fh = [], []
    for j in range(0, len(X), bs):
        o = m(torch.tensor(X[j:j + bs]).to(DEV))
        D.append(o["delay"].cpu().numpy()); Fh.append(o["feat"].cpu().numpy())
    return np.concatenate(D), np.concatenate(Fh)


def probe(feats, y, groups):
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(feats, y, groups):
        clf = LogisticRegression(max_iter=2000).fit(feats[tr], y[tr])
        accs.append(clf.score(feats[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    print("device:", DEV)
    a = np.load(V4 / "hbci_train_MI.npz", allow_pickle=True)
    b = np.load(V4 / "hbci_train_MA.npz", allow_pickle=True)
    eeg = np.concatenate([a["eeg"], b["eeg"]])
    hemo = np.concatenate([a["hemo"], b["hemo"]]).astype(np.float32)
    sid = np.concatenate([a["subject_ids"], b["subject_ids"] + a["subject_ids"].max() + 1])
    lab = np.concatenate([a["labels"], b["labels"]])
    X = bandpower_seq(eeg)
    print(f"fNIRS corpus: {len(X)} trials, {np.unique(sid).size} subject-sessions")
    rows = []

    # ------------------------------------------------------------ CLAIM 3
    print(f"\n{'='*74}\nCLAIM 3 - where does the delay settle when the bound is widened?\n{'='*74}")
    print(f"{'window':>14s}{'mean':>9s}{'sd':>8s}{'min':>8s}{'max':>8s}"
          f"{'btw-subj sd':>13s}   verdict")
    for dmin, dmax in [(2.0, 8.0), (0.0, 12.0), (0.0, 20.0)]:
        m = train(X, hemo, dmin, dmax)
        D, _ = infer(m, X)
        per = np.array([D[sid == s].mean() for s in np.unique(sid)])
        span = dmax - dmin
        at_edge = (D.mean() - dmin) < 0.05 * span or (dmax - D.mean()) < 0.05 * span
        verdict = "PINNED at bound" if at_edge else "interior -- measurement"
        rows.append({"claim": 3, "dmin": dmin, "dmax": dmax, "mean_s": float(D.mean()),
                     "sd_s": float(D.std()), "between_subject_sd": float(per.std()),
                     "pinned": at_edge})
        print(f"  [{dmin:.0f},{dmax:.0f}] s{D.mean():>9.2f}{D.std():>8.2f}"
              f"{D.min():>8.2f}{D.max():>8.2f}{per.std():>13.3f}   {verdict}")
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ CLAIM 4
    print(f"\n{'='*74}\nCLAIM 4 - is the degradation specific to hemodynamic supervision?\n{'='*74}")
    rng = np.random.default_rng(0)
    hemo_shuf = hemo[rng.permutation(len(hemo))]          # same stats, no real relationship

    m_real = train(X, hemo, 0.0, 12.0, tag="real target")
    _, f_real = infer(m_real, X); del m_real
    m_shuf = train(X, hemo_shuf, 0.0, 12.0, tag="shuffled target")
    _, f_shuf = infer(m_shuf, X); del m_shuf
    m_rand = Model(X.shape[1], hemo.shape[1], hemo.shape[2], 0.0, 12.0).to(DEV).eval()
    _, f_rand = infer(m_rand, X); del m_rand
    if DEV == "cuda":
        torch.cuda.empty_cache()

    res = {}
    for nm, f in [("hemodynamic-supervised", f_real), ("SHUFFLED-target control", f_shuf),
                  ("random init (untrained)", f_rand),
                  ("raw band-power", X.reshape(len(X), -1))]:
        acc, sd = probe(f, lab, sid)
        res[nm] = acc
        rows.append({"claim": 4, "encoder": nm, "accuracy": acc, "sd": sd})
        print(f"  {nm:26s} {acc:.4f} +- {sd:.4f}")

    real, shuf, rand = (res["hemodynamic-supervised"], res["SHUFFLED-target control"],
                        res["random init (untrained)"])
    print(f"\n  real vs random     {real-rand:+.4f}")
    print(f"  shuffled vs random {shuf-rand:+.4f}")
    print(f"  real vs shuffled   {real-shuf:+.4f}")
    if abs(real - shuf) < 0.015:
        print("\n  -> shuffled behaves like real: the damage is from training on a")
        print("     LOW-SNR TARGET IN GENERAL, not from hemodynamic supervision as such.")
    elif shuf > real + 0.015:
        print("\n  -> shuffled is better than real: hemodynamic structure specifically")
        print("     pulls the representation away from task-discriminative features.")
    else:
        print("\n  -> real is better than shuffled: hemodynamic supervision adds something,")
        print("     even though both sit below random initialisation.")

    pd.DataFrame(rows).to_csv(TABLE_DIR / "claims_3_4.csv", index=False)
    print(f"\nwrote {TABLE_DIR/'claims_3_4.csv'}")


if __name__ == "__main__":
    main()
