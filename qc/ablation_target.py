"""Factorial ablation over fNIRS TARGET preprocessing.

The audit showed the training target passes through three destructive steps:
    bandpass 0.01-0.2 Hz  ->  global-signal regression  ->  per-channel z-score
and that GSR alone removes ~83% of the evoked HbO amplitude. This sweeps every
combination to find out how much of the "cross-subject prediction ~= 0" result
is caused by the pipeline rather than by EEG.

Factors
-------
    bandpass      : on / off
    GSR           : on / off            (clean_fnirs_systemic)
    normalisation : per_channel_z (current) / global_z / none
  = 12 target variants

For each variant we report
    target_reliability  split-half correlation of the trial-averaged target,
                        within subject. This is the CEILING -- if the target has
                        no reproducible structure, no model can predict it, and
                        the failure is preprocessing, not EEG.
    waveform_r          held-out-subject concat-Pearson, ridge on EEG band-power
    amplitude_r         held-out-subject Pearson on response magnitude
    *_null              same, with the EEG<->target pairing shuffled

Ridge (not the deep model) is used deliberately: it is fast, it is what produced
the project's one solid positive (amplitude r=0.33), and we are comparing
variants against each other, not chasing peak performance.

Usage:  python qc/ablation_target.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

CACHE = Path("neurovascnet_outputs/cache_v3")
EEG_FS = 200.0
HEMO_FS = 10.0                      # 300 samples over a 30 s window
RESP_WIN = (4.0, 12.0)              # HRF response window, seconds
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BINS = 6
RNG = np.random.default_rng(0)


# =====================================================================
# target preprocessing (mirrors NeuroVascNet_v3.ipynb cell 15)
# =====================================================================
def bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2
    b, a = sps.butter(order, [max(lo / nyq, 1e-6), min(hi / nyq, 0.99)], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def regress_out_global(block):
    g = block.mean(axis=-2, keepdims=True)
    gc = g - g.mean(axis=-1, keepdims=True)
    bc = block - block.mean(axis=-1, keepdims=True)
    beta = (bc * gc).sum(-1, keepdims=True) / ((gc ** 2).sum(-1, keepdims=True) + 1e-12)
    return bc - beta * gc


def clean_systemic(x):
    """x: (N, 180, T) -> GSR applied separately to the HbO and HbR halves."""
    n = x.shape[1] // 2
    return np.concatenate([regress_out_global(x[:, :n]),
                           regress_out_global(x[:, n:])], axis=1)


def make_target(hemo, do_bp, do_gsr, norm):
    x = hemo.astype(np.float64)
    if do_bp:
        x = bandpass(x, HEMO_FS, 0.01, 0.2)
    if do_gsr:
        x = clean_systemic(x)
    if norm == "per_channel_z":
        mu = x.mean(-1, keepdims=True); sd = x.std(-1, keepdims=True) + 1e-6
        x = (x - mu) / sd
    elif norm == "global_z":
        # one scale per window: preserves RELATIVE channel amplitude
        mu = x.mean((-2, -1), keepdims=True); sd = x.std((-2, -1), keepdims=True) + 1e-12
        x = (x - mu) / sd
    return x.astype(np.float32)


# =====================================================================
# EEG features: band-power per channel per time bin
# =====================================================================
def eeg_features(eeg):
    N, C, T = eeg.shape
    binlen = T // N_BINS
    x = eeg[:, :, :binlen * N_BINS].reshape(N, C, N_BINS, binlen)
    x = x - x.mean(-1, keepdims=True)
    win = np.hanning(binlen)
    F = np.fft.rfft(x * win, axis=-1)
    P = (np.abs(F) ** 2)
    freqs = np.fft.rfftfreq(binlen, 1 / EEG_FS)
    feats = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        feats.append(P[..., m].sum(-1))
    f = np.stack(feats, -1)                       # (N, C, BINS, BANDS)
    f = np.log10(f + 1e-12)
    return f.reshape(N, -1).astype(np.float32)


def amplitude_of(target):
    """Mean over the HRF response window, averaged across HbO channels."""
    a, b = int(RESP_WIN[0] * HEMO_FS), int(RESP_WIN[1] * HEMO_FS)
    n = target.shape[1] // 2
    return target[:, :n, a:b].mean(axis=(1, 2))


# =====================================================================
# metrics
# =====================================================================
def concat_pearson(pred, true):
    p, t = pred.ravel(), true.ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 10 or p[ok].std() == 0 or t[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def split_half_reliability(target, subject_ids):
    """Ceiling: how reproducible is the trial-averaged target within a subject?"""
    rs = []
    for s in np.unique(subject_ids):
        idx = np.flatnonzero(subject_ids == s)
        if idx.size < 6:
            continue
        perm = RNG.permutation(idx)
        h1, h2 = perm[: idx.size // 2], perm[idx.size // 2:]
        m1, m2 = target[h1].mean(0), target[h2].mean(0)
        r = concat_pearson(m1, m2)
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else np.nan


def ridge_fit_predict(Xtr, Ytr, Xte, alphas=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Ridge with a small alpha search on a held-out slice of the training set."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ym = Ytr.mean(0)
    Yc = Ytr - ym

    n_in = int(Xtr.shape[0] * 0.8)
    best, best_r = alphas[0], -np.inf
    A = Xtr[:n_in]; B = Xtr[n_in:]
    if B.shape[0] >= 5:
        G = A.T @ A
        for al in alphas:
            W = np.linalg.solve(G + al * np.eye(G.shape[0]), A.T @ Yc[:n_in])
            r = concat_pearson(B @ W, Yc[n_in:])
            if np.isfinite(r) and r > best_r:
                best_r, best = r, al
    G = Xtr.T @ Xtr
    W = np.linalg.solve(G + best * np.eye(G.shape[0]), Xtr.T @ Yc)
    return Xte @ W + ym


# =====================================================================
def main():
    tr = np.load(CACHE / "fnirs_train.npz", allow_pickle=True)
    va = np.load(CACHE / "fnirs_val.npz", allow_pickle=True)
    eeg_tr, hemo_tr, sid_tr = tr["eeg"], tr["hemo"], tr["subject_ids"]
    eeg_va, hemo_va, sid_va = va["eeg"], va["hemo"], va["subject_ids"]
    print(f"train {eeg_tr.shape[0]} windows / {np.unique(sid_tr).size} subjects")
    print(f"val   {eeg_va.shape[0]} windows / {np.unique(sid_va).size} subjects "
          f"(held out)\n")

    print("computing EEG band-power features ...")
    Xtr, Xte = eeg_features(eeg_tr), eeg_features(eeg_va)
    print(f"  features: {Xtr.shape[1]}\n")

    rows = []
    combos = list(itertools.product([True, False], [True, False],
                                    ["per_channel_z", "global_z", "none"]))
    for i, (bp, gsr, norm) in enumerate(combos, 1):
        Ttr = make_target(hemo_tr, bp, gsr, norm)
        Tte = make_target(hemo_va, bp, gsr, norm)

        rel = split_half_reliability(Tte, sid_va)

        # --- waveform ---
        ytr = Ttr.reshape(Ttr.shape[0], -1)
        yte = Tte.reshape(Tte.shape[0], -1)
        pw = ridge_fit_predict(Xtr, ytr, Xte)
        r_wave = concat_pearson(pw, yte)
        pw_n = ridge_fit_predict(Xtr, ytr[RNG.permutation(ytr.shape[0])], Xte)
        r_wave_null = concat_pearson(pw_n, yte)

        # --- amplitude ---
        atr, ate = amplitude_of(Ttr), amplitude_of(Tte)
        pa = ridge_fit_predict(Xtr, atr[:, None], Xte).ravel()
        r_amp = concat_pearson(pa, ate)
        pa_n = ridge_fit_predict(Xtr, atr[RNG.permutation(atr.size)][:, None], Xte).ravel()
        r_amp_null = concat_pearson(pa_n, ate)

        # how much evoked amplitude survived this variant, vs the raw target
        raw_amp = np.abs(amplitude_of(make_target(hemo_va, False, False, "none"))).mean()
        kept = float(np.abs(ate).mean() / raw_amp) if raw_amp > 0 else np.nan

        tag = f"bp={int(bp)} gsr={int(gsr)} norm={norm}"
        rows.append({"bandpass": bp, "gsr": gsr, "norm": norm,
                     "target_reliability": rel,
                     "waveform_r": r_wave, "waveform_null": r_wave_null,
                     "amplitude_r": r_amp, "amplitude_null": r_amp_null,
                     "amp_scale_vs_raw": kept,
                     "is_current_pipeline": bool(bp and gsr and norm == "per_channel_z")})
        print(f"[{i:2d}/12] {tag:38s} rel={rel:+.3f}  "
              f"wave={r_wave:+.3f} (null {r_wave_null:+.3f})  "
              f"amp={r_amp:+.3f} (null {r_amp_null:+.3f})")

    df = pd.DataFrame(rows)
    out = TABLE_DIR / "ablation_fnirs_target.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    cur = df[df.is_current_pipeline].iloc[0]
    print(f"CURRENT PIPELINE           reliability {cur.target_reliability:+.3f}   "
          f"waveform {cur.waveform_r:+.3f}   amplitude {cur.amplitude_r:+.3f}")
    b_rel = df.loc[df.target_reliability.idxmax()]
    b_wav = df.loc[df.waveform_r.idxmax()]
    b_amp = df.loc[df.amplitude_r.idxmax()]
    print(f"BEST reliability           bp={int(b_rel.bandpass)} gsr={int(b_rel.gsr)} "
          f"norm={b_rel.norm:14s} -> {b_rel.target_reliability:+.3f}")
    print(f"BEST waveform              bp={int(b_wav.bandpass)} gsr={int(b_wav.gsr)} "
          f"norm={b_wav.norm:14s} -> {b_wav.waveform_r:+.3f} "
          f"(null {b_wav.waveform_null:+.3f})")
    print(f"BEST amplitude             bp={int(b_amp.bandpass)} gsr={int(b_amp.gsr)} "
          f"norm={b_amp.norm:14s} -> {b_amp.amplitude_r:+.3f} "
          f"(null {b_amp.amplitude_null:+.3f})")
    print("=" * 78)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
