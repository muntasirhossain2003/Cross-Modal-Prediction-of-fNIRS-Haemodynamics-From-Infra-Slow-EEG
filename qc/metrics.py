"""Signal-quality metrics shared across modalities.

All functions are pure: array in, numbers out. They never decide pass/fail --
that is the job of rules.py -- so thresholds and measurements stay separable.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


# =========================================================================
# Generic amplitude / integrity
# =========================================================================
def robust_amplitude(x: np.ndarray, pct: float = 99.5) -> float:
    """Robust peak amplitude: the `pct`-th percentile of |x|.

    Preferred over max() because a single spike should not define the
    amplitude of an otherwise clean recording.
    """
    v = np.abs(x[np.isfinite(x)])
    return float(np.percentile(v, pct)) if v.size else np.nan


def amplitude_range(x: np.ndarray) -> dict:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return {k: np.nan for k in
                ("min", "max", "p0_5", "p99_5", "median", "iqr", "sd")}
    q = np.percentile(v, [0.5, 25, 50, 75, 99.5])
    return {
        "min": float(v.min()), "max": float(v.max()),
        "p0_5": float(q[0]), "p99_5": float(q[4]),
        "median": float(q[2]), "iqr": float(q[3] - q[1]),
        "sd": float(v.std()),
    }


def nan_fraction(x: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(x)))


def clipped_fraction(x: np.ndarray, tol_frac: float = 1e-6) -> float:
    """Fraction of samples pinned at the recording's own extremes (ADC rail)."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.nan
    lo, hi = v.min(), v.max()
    span = hi - lo
    if span <= 0:
        return 1.0
    tol = max(span * tol_frac, np.finfo(float).eps)
    return float(np.mean((v <= lo + tol) | (v >= hi - tol)))


def constant_fraction(x: np.ndarray, fs: float, run_ms: float = 100.0) -> float:
    """Fraction of samples inside runs of >= run_ms of identical values.

    Detects dropout / sample-and-hold, which a plain variance check misses when
    only part of the recording is dead.
    """
    if x.ndim == 1:
        x = x[None, :]
    min_run = max(int(round(run_ms / 1000.0 * fs)), 2)
    total, flagged = 0, 0
    for ch in x:
        ch = ch[np.isfinite(ch)]
        if ch.size < min_run:
            continue
        same = np.diff(ch) == 0
        # run-length encode the True stretches
        idx = np.flatnonzero(np.diff(np.concatenate(([0], same.view(np.int8), [0]))))
        starts, ends = idx[::2], idx[1::2]
        runs = ends - starts
        flagged += int(runs[runs >= min_run - 1].sum())
        total += ch.size
    return float(flagged / total) if total else np.nan


# =========================================================================
# Spectral
# =========================================================================
def welch_psd(x: np.ndarray, fs: float, nperseg_s: float = 4.0,
              fmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel. Returns (freqs, psd[n_ch, n_freq]) in units^2/Hz."""
    if x.ndim == 1:
        x = x[None, :]
    nperseg = int(min(x.shape[1], max(256, round(nperseg_s * fs))))
    f, p = sps.welch(np.nan_to_num(x), fs=fs, nperseg=nperseg,
                     noverlap=nperseg // 2, axis=-1)
    if fmax is not None:
        keep = f <= fmax
        f, p = f[keep], p[..., keep]
    return f, np.atleast_2d(p)


def band_power(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrated power in [lo, hi] per channel."""
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return np.full(psd.shape[0], np.nan)
    return np.trapz(psd[:, m], f[m], axis=-1)


def narrowband_ratio(f: np.ndarray, psd: np.ndarray, f0: float,
                     bw: float = 1.0, neighbourhood: float = 4.0) -> float:
    """Power at f0 relative to the surrounding spectral neighbourhood.

    Used for line noise and for scanner-gradient / BCG residual harmonics. A
    ratio of 1 means "indistinguishable from the local background".
    """
    if f0 <= 0 or f0 >= f[-1]:
        return np.nan
    peak = (f >= f0 - bw / 2) & (f <= f0 + bw / 2)
    side = (((f >= f0 - neighbourhood) & (f < f0 - bw)) |
            ((f > f0 + bw) & (f <= f0 + neighbourhood)))
    if not peak.any() or not side.any():
        return np.nan
    num = np.nanmean(psd[:, peak])
    den = np.nanmean(psd[:, side])
    return float(num / den) if den > 0 else np.nan


def detect_line_freq(f: np.ndarray, psd: np.ndarray,
                     candidates=(50.0, 60.0)) -> tuple[float, float]:
    """Pick whichever mains frequency actually dominates; return (freq, ratio)."""
    best_f, best_r = np.nan, np.nan
    for c in candidates:
        if c >= f[-1]:
            continue
        r = narrowband_ratio(f, psd, c)
        if np.isfinite(r) and (not np.isfinite(best_r) or r > best_r):
            best_f, best_r = c, r
    return best_f, best_r


def harmonic_residual(f: np.ndarray, psd: np.ndarray, fundamental: float,
                      n_harmonics: int = 5, bw: float = 1.0) -> dict:
    """Max/mean narrowband ratio across harmonics of `fundamental`.

    For simultaneous EEG-fMRI the gradient artifact appears at multiples of the
    slice frequency; a clean correction leaves ratios near 1.
    """
    ratios = []
    for k in range(1, n_harmonics + 1):
        fk = fundamental * k
        if fk >= f[-1]:
            break
        r = narrowband_ratio(f, psd, fk, bw=bw)
        if np.isfinite(r):
            ratios.append(r)
    if not ratios:
        return {"max": np.nan, "mean": np.nan, "n": 0}
    return {"max": float(np.max(ratios)), "mean": float(np.mean(ratios)),
            "n": len(ratios)}


# =========================================================================
# Filtering / transients
# =========================================================================
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float,
             order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-6), min(hi / nyq, 0.99)
    if lo_n >= hi_n:
        return np.nan_to_num(x)
    b, a = sps.butter(order, [lo_n, hi_n], btype="band")
    return sps.filtfilt(b, a, np.nan_to_num(x), axis=-1)


def transient_count(x: np.ndarray, fs: float, threshold: float) -> dict:
    """Count threshold excursions, reported per minute.

    Consecutive supra-threshold samples count once, so a single long excursion
    is one transient rather than hundreds.
    """
    if x.ndim == 1:
        x = x[None, :]
    over = np.abs(x) > threshold
    # rising edges only
    edges = np.diff(over.astype(np.int8), axis=-1) == 1
    n = int(edges.sum()) + int(over[:, 0].sum())
    minutes = x.shape[-1] / fs / 60.0
    return {"count": n, "per_min": float(n / minutes) if minutes > 0 else np.nan,
            "per_min_per_channel": float(n / minutes / x.shape[0])
            if minutes > 0 and x.shape[0] else np.nan}


# =========================================================================
# Bad channels
# =========================================================================
def find_bad_channels(x: np.ndarray, fs: float, flat_uv: float = 0.05,
                      corr_threshold: float = 0.20,
                      amp_z: float = 5.0) -> dict:
    """Flag flat, railed and outlier channels.

    Three independent criteria, because each misses a different failure mode:
      flat      -> disconnected electrode
      amplitude -> gel bridge / very high impedance (robust z of log SD)
      corr      -> channel that shares no signal with any neighbour
    """
    n_ch = x.shape[0]
    sd = np.nanstd(x, axis=1)
    flat = sd < flat_uv

    with np.errstate(divide="ignore"):
        log_sd = np.log10(np.where(sd > 0, sd, np.nan))
    med = np.nanmedian(log_sd)
    mad = np.nanmedian(np.abs(log_sd - med))
    rz = 0.6745 * (log_sd - med) / mad if mad and np.isfinite(mad) else np.zeros(n_ch)
    outlier = np.abs(np.nan_to_num(rz)) > amp_z

    # correlation criterion on a decimated, band-limited copy (cheap + robust)
    step = max(int(fs // 100), 1)
    xf = np.nan_to_num(x[:, ::step])
    if xf.shape[1] > 10 and n_ch > 1:
        c = np.corrcoef(xf)
        np.fill_diagonal(c, np.nan)
        med_corr = np.nanmedian(np.abs(c), axis=1)
    else:
        med_corr = np.full(n_ch, np.nan)
    uncorr = np.nan_to_num(med_corr, nan=1.0) < corr_threshold

    bad = flat | outlier | uncorr
    return {
        "bad_idx": np.flatnonzero(bad).tolist(),
        "n_bad": int(bad.sum()),
        "fraction": float(bad.mean()),
        "flat_idx": np.flatnonzero(flat).tolist(),
        "outlier_idx": np.flatnonzero(outlier).tolist(),
        "uncorrelated_idx": np.flatnonzero(uncorr).tolist(),
        "median_corr": float(np.nanmedian(med_corr)) if n_ch > 1 else np.nan,
    }


# =========================================================================
# Event-locked analysis
# =========================================================================
def epoch(x: np.ndarray, fs: float, onsets_s: np.ndarray,
          tmin: float, tmax: float) -> np.ndarray:
    """Cut (n_ch, time) into (n_trials, n_ch, n_samples). Out-of-range dropped."""
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    out = []
    for o in np.atleast_1d(onsets_s):
        s = int(round(o * fs))
        if s + a < 0 or s + b > x.shape[1]:
            continue
        out.append(x[:, s + a:s + b])
    if not out:
        return np.empty((0, x.shape[0], b - a))
    n = min(seg.shape[1] for seg in out)
    return np.stack([seg[:, :n] for seg in out])


def mean_ci(a: np.ndarray, axis: int = 0, ci: float = 1.96) -> tuple:
    """Mean and normal-approximation confidence interval half-width."""
    m = np.nanmean(a, axis=axis)
    n = np.sum(np.isfinite(a), axis=axis)
    se = np.nanstd(a, axis=axis) / np.sqrt(np.maximum(n, 1))
    return m, ci * se


def erd_index(x: np.ndarray, fs: float, onsets_s: np.ndarray,
              band: tuple[float, float], baseline: tuple[float, float],
              active: tuple[float, float]) -> float:
    """Event-related desynchronisation: (active - baseline) / baseline, in %.

    Motor imagery / execution should produce a NEGATIVE mu-band value over
    sensorimotor cortex. This is the task-decoding-free sanity check that the
    EEG in a given recording still contains the effect it is supposed to.
    """
    xf = bandpass(x, fs, band[0], band[1])
    power = xf ** 2
    ep = epoch(power, fs, onsets_s, baseline[0], active[1])
    if ep.shape[0] == 0:
        return np.nan
    t = np.arange(ep.shape[2]) / fs + baseline[0]
    bm = (t >= baseline[0]) & (t < baseline[1])
    am = (t >= active[0]) & (t < active[1])
    if not bm.any() or not am.any():
        return np.nan
    b = np.nanmean(ep[:, :, bm], axis=(0, 2))
    a = np.nanmean(ep[:, :, am], axis=(0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        erd = 100.0 * (a - b) / b
    return float(np.nanmean(erd))


# =========================================================================
# fNIRS-specific
# =========================================================================
def scalp_coupling_index(w1: np.ndarray, w2: np.ndarray, fs: float,
                         band: tuple[float, float] = (0.5, 2.5)) -> np.ndarray:
    """Correlation of the two wavelengths in the cardiac band, per channel.

    Only meaningful on raw wavelength intensities; datasets shipped as
    HbO/HbR concentrations cannot support it (the conversion mixes wavelengths).
    """
    if fs / 2.0 <= band[0]:
        return np.full(w1.shape[0], np.nan)
    hi = min(band[1], fs / 2.0 * 0.95)
    a = bandpass(w1, fs, band[0], hi)
    b = bandpass(w2, fs, band[0], hi)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        sa, sb = a[i], b[i]
        if np.std(sa) > 0 and np.std(sb) > 0:
            out[i] = np.corrcoef(sa, sb)[0, 1]
    return out


def cardiac_band_ratio(x: np.ndarray, fs: float,
                       band: tuple[float, float] = (0.5, 2.5)) -> np.ndarray:
    """Fraction of 0.01-Nyquist power that sits in the cardiac band.

    A concentration-domain stand-in for SCI: a well-coupled optode picks up the
    heartbeat, a detached one does not.
    """
    if fs / 2.0 <= band[0]:
        return np.full(x.shape[0], np.nan)
    f, p = welch_psd(x, fs, nperseg_s=min(20.0, x.shape[1] / fs / 2))
    hi = min(band[1], fs / 2.0 * 0.95)
    num = band_power(f, p, band[0], hi)
    den = band_power(f, p, 0.01, fs / 2.0 * 0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def motion_artifact_rate(x: np.ndarray, fs: float, sd_threshold: float = 5.0) -> dict:
    """Spike/baseline-shift rate per minute from the derivative of each channel."""
    d = np.diff(np.nan_to_num(x), axis=-1)
    sd = np.nanstd(d, axis=-1, keepdims=True)
    sd[sd == 0] = np.nan
    z = np.abs(d / sd)
    over = z > sd_threshold
    edges = np.diff(over.astype(np.int8), axis=-1) == 1
    n = int(edges.sum())
    minutes = x.shape[-1] / fs / 60.0
    n_ch = max(x.shape[0], 1)
    return {"count": n,
            "per_min": float(n / minutes) if minutes > 0 else np.nan,
            "per_min_per_channel": float(n / minutes / n_ch) if minutes > 0 else np.nan}


def global_signal_regress(x: np.ndarray) -> np.ndarray:
    """Remove the across-channel mean from every channel (the GSR ablation arm)."""
    g = np.nanmean(x, axis=0, keepdims=True)
    gc = g - np.nanmean(g)
    denom = float(np.sum(gc ** 2))
    if denom <= 0:
        return x.copy()
    beta = (np.nan_to_num(x) @ gc.T) / denom
    return x - beta @ gc
