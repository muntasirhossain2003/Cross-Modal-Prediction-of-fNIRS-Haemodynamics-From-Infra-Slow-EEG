"""Is the motor track salvageable? Reprocess Multi-joint fNIRS from RAW.

HEFMI -- the current primary fNIRS training set -- has HbO/HbR correlation
+0.90, which is physiologically impossible for a real haemodynamic response and
caps target reliability at ~0.13. Two explanations are possible:

  (a) the tissue signal really is systemic-dominated, or
  (b) HEFMI's shipped Beer-Lambert conversion is wrong.

Multi-joint ships RAW dual-wavelength intensities (NIRx .wl1/.wl2), so we can do
our own conversion with known constants and find out. If a controlled conversion
yields proper HbO/HbR ANTI-correlation, the motor track is salvageable and the
problem was the vendor conversion. If it still comes out positive, the tissue
signal is genuinely systemic-dominated and the motor track should be abandoned.

Also computes the true Scalp Coupling Index -- impossible on every dataset
audited so far, because they all shipped pre-converted concentrations.

Usage:  python qc/multijoint_check.py [n_subjects]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

ROOT = Path("dataset/Multi-joint Upper-Limb EEG+FNIR/FineMI/FineMI")

# Molar extinction coefficients, cm^-1 / (mol/L)  (Prahl / Wray tables)
EXT = {760: {"HbO": 1486.59, "HbR": 3843.71},
       850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}          # differential pathlength factors
SD_CM = 3.0                          # source-detector separation
CARDIAC = (0.5, 2.5)
HRF_WIN = (-2.0, 20.0)


def read_hdr(p: Path) -> dict:
    txt = p.read_text(errors="ignore")
    out = {}
    for key in ("Sources", "Detectors", "SamplingRate", "ShortBundles"):
        for ln in txt.splitlines():
            if ln.startswith(key + "="):
                out[key] = ln.split("=", 1)[1].strip().strip('"')
    return out


def bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def beer_lambert(wl1: np.ndarray, wl2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modified Beer-Lambert: raw intensities (ch, T) -> HbO, HbR in micromolar."""
    def od(x):
        base = np.median(x, axis=-1, keepdims=True)
        base = np.where(base <= 0, 1e-12, base)
        return -np.log(np.clip(x, 1e-12, None) / base)

    a1, a2 = od(wl1), od(wl2)
    L1, L2 = SD_CM * DPF[760], SD_CM * DPF[850]
    E = np.array([[EXT[760]["HbO"] * L1, EXT[760]["HbR"] * L1],
                  [EXT[850]["HbO"] * L2, EXT[850]["HbR"] * L2]])
    Einv = np.linalg.inv(E)
    hbo = Einv[0, 0] * a1 + Einv[0, 1] * a2
    hbr = Einv[1, 0] * a1 + Einv[1, 1] * a2
    return hbo * 1e6, hbr * 1e6          # mol/L -> micromolar


def sci(wl1, wl2, fs):
    """True Scalp Coupling Index: cardiac-band correlation of the two wavelengths."""
    a = bandpass(wl1, fs, *CARDIAC)
    b = bandpass(wl2, fs, *CARDIAC)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        if a[i].std() > 0 and b[i].std() > 0:
            out[i] = np.corrcoef(a[i], b[i])[0, 1]
    return out


def epoch(x, fs, onsets, tmin, tmax):
    a, b = int(tmin * fs), int(tmax * fs)
    segs = [x[:, s + a:s + b] for s in onsets if s + a >= 0 and s + b <= x.shape[1]]
    if not segs:
        return np.empty((0, x.shape[0], b - a))
    n = min(s.shape[1] for s in segs)
    return np.stack([s[:, :n] for s in segs])


def split_half(ep):
    if ep.shape[0] < 6:
        return np.nan
    rng = np.random.default_rng(0)
    p = rng.permutation(ep.shape[0])
    m1 = ep[p[:len(p) // 2]].mean(0).ravel()
    m2 = ep[p[len(p) // 2:]].mean(0).ravel()
    if m1.std() == 0 or m2.std() == 0:
        return np.nan
    return float(np.corrcoef(m1, m2)[0, 1])


def process_block(bdir: Path) -> dict | None:
    hdr_f = next(bdir.glob("*.hdr"), None)
    if hdr_f is None:
        return None
    hdr = read_hdr(hdr_f)
    fs = float(hdr.get("SamplingRate", 7.8125))

    w1 = np.loadtxt(next(bdir.glob("*.wl1"))).T      # (ch, T)
    w2 = np.loadtxt(next(bdir.glob("*.wl2"))).T
    n = min(w1.shape[1], w2.shape[1])
    w1, w2 = w1[:, :n], w2[:, :n]

    s = sci(w1, w2, fs)
    hbo, hbr = beer_lambert(w1, w2)

    # keep hemodynamic band before assessing the response
    hbo_f = bandpass(hbo, fs, 0.01, 0.2)
    hbr_f = bandpass(hbr, fs, 0.01, 0.2)

    cc = [np.corrcoef(hbo_f[i], hbr_f[i])[0, 1] for i in range(hbo_f.shape[0])
          if hbo_f[i].std() > 0 and hbr_f[i].std() > 0]

    evt_f = next(bdir.glob("*.evt"), None)
    onsets = np.array([], dtype=int)
    if evt_f is not None:
        try:
            e = np.loadtxt(evt_f)
            onsets = np.atleast_2d(e)[:, 0].astype(int)
        except Exception:
            pass

    ep = epoch(hbo_f, fs, onsets, *HRF_WIN)
    rel = split_half(ep)
    if ep.shape[0] > 1:
        m = ep.mean(0)
        se = ep.std(0) / np.sqrt(ep.shape[0])
        t = np.arange(m.shape[1]) / fs + HRF_WIN[0]
        pk = np.argmin(np.abs(t - 6.0))
        peak = float(np.nanmean(m[:, pk]))
        ci = float(np.nanmean(1.96 * se[:, pk]))
        snr = abs(peak) / ci if ci > 0 else np.nan
    else:
        peak = snr = np.nan

    return {
        "block": bdir.name, "fs_hz": fs,
        "n_channels": w1.shape[0], "n_samples": n,
        "duration_s": n / fs,
        "short_bundles": hdr.get("ShortBundles", "?"),
        "sci_median": float(np.nanmedian(s)),
        "sci_frac_above_0.5": float(np.nanmean(np.nan_to_num(s) > 0.5)),
        "hbo_hbr_corr_median": float(np.median(cc)) if cc else np.nan,
        "hbo_robust_amp_umol": float(np.percentile(np.abs(hbo_f), 99.5)),
        "n_events": int(onsets.size), "n_epochs": int(ep.shape[0]),
        "evoked_peak_umol": peak, "evoked_snr": snr,
        "split_half_reliability": rel,
    }


def main():
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    subs = sorted([d for d in ROOT.glob("subject*") if d.is_dir()],
                  key=lambda p: int("".join(c for c in p.name if c.isdigit())))[:n_sub]

    rows = []
    for sd in subs:
        for bdir in sorted((sd / "fNIRS").glob("block*")):
            try:
                r = process_block(bdir)
            except Exception as exc:
                print(f"{sd.name}/{bdir.name}: ERROR {type(exc).__name__}: {exc}")
                continue
            if r is None:
                continue
            r["subject"] = sd.name
            rows.append(r)
            print(f"{sd.name:10s} {r['block']:9s} SCI {r['sci_median']:+.3f} "
                  f"({r['sci_frac_above_0.5']*100:3.0f}% good)  "
                  f"HbO/HbR {r['hbo_hbr_corr_median']:+.3f}  "
                  f"evokedSNR {r['evoked_snr']:5.2f}  "
                  f"rel {r['split_half_reliability']:+.3f}")

    if not rows:
        raise SystemExit("nothing processed")

    df = pd.DataFrame(rows)
    out = TABLE_DIR / "multijoint_raw_check.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 74)
    print(f"blocks processed         : {len(df)}  ({df.subject.nunique()} subjects)")
    print(f"true SCI median          : {df.sci_median.median():+.3f}")
    print(f"HbO/HbR corr median      : {df.hbo_hbr_corr_median.median():+.3f}"
          "      (HEFMI: +0.898,  TU Berlin: -0.333)")
    print(f"evoked SNR median        : {df.evoked_snr.median():.2f}"
          "            (HEFMI: 0.144)")
    print(f"split-half reliability   : {df.split_half_reliability.median():+.3f}"
          "      (HEFMI raw: +0.257, after pipeline: +0.127)")
    print("=" * 74)

    corr = df.hbo_hbr_corr_median.median()
    rel = df.split_half_reliability.median()
    if corr < -0.1 and rel > 0.3:
        print("VERDICT: controlled conversion gives PHYSIOLOGICALLY CORRECT signal.")
        print("         HEFMI's +0.90 was a conversion/vendor problem.")
        print("         -> the MOTOR track is salvageable; rebuild from Multi-joint raw.")
    elif corr < -0.1:
        print("VERDICT: HbO/HbR sign is correct but the evoked response is weak.")
        print("         -> motor track is usable but the target stays low-reliability.")
    else:
        print("VERDICT: still positively correlated after a controlled conversion.")
        print("         -> the tissue signal is genuinely systemic-dominated;")
        print("            abandon the motor track and commit to the cognitive track.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
