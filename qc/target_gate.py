"""Target-quality gate applied to TU Berlin hBCI -- the last candidate motor fNIRS set.

Three physiology checks, applied BEFORE any modelling. Cheap, and had they been
run at the start they would have saved the project months:

    1. HbO-HbR correlation    must be NEGATIVE  (real response: HbO up, HbR down)
    2. evoked SNR             > 1.0             (peak larger than its own 95% CI)
    3. split-half reliability > 0.3             (target reproduces against itself)

Reference values already measured elsewhere in this audit:
    HEFMI        (motor)     HbO/HbR +0.898   SNR 0.144   rel +0.257   FAIL (sign)
    TU Berlin WL (cognitive) HbO/HbR -0.333   SNR 0.247   rel   --     best so far
    Multi-joint  (motor)     HbO/HbR -0.768   SNR 0.04    rel -0.10    FAIL (coupling)

TU Berlin hBCI ships RAW wavelength intensities (760/850 nm, 36 source-detector
pairs x 2 wavelengths = 72 series), so we do our own Beer-Lambert and can compute
true Scalp Coupling Index. Only the MI (motor imagery) runs are assessed; the MA
(mental arithmetic) runs are reported separately for contrast.

Usage:  python qc/target_gate.py [n_subjects]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

warnings.filterwarnings("ignore")

ROOT = Path("dataset/TU Berlin hBCI EEG+FNRI/NRI data/NIRS_01-29")
GATE = {"hbo_hbr_corr_max": 0.0, "evoked_snr_min": 1.0, "reliability_min": 0.3}

EXT = {760: {"HbO": 1486.59, "HbR": 3843.71},
       850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}
SD_CM = 3.0
CARDIAC = (0.5, 2.5)


def bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def beer_lambert(lo_wl, hi_wl):
    """(ch, T) raw intensities at 760/850 nm -> HbO, HbR in micromolar."""
    def od(x):
        base = np.median(x, axis=-1, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        return -np.log(np.clip(x / base, 1e-9, None))

    a1, a2 = od(lo_wl), od(hi_wl)
    L1, L2 = SD_CM * DPF[760], SD_CM * DPF[850]
    E = np.array([[EXT[760]["HbO"] * L1, EXT[760]["HbR"] * L1],
                  [EXT[850]["HbO"] * L2, EXT[850]["HbR"] * L2]])
    Ei = np.linalg.inv(E)
    return (Ei[0, 0] * a1 + Ei[0, 1] * a2) * 1e6, (Ei[1, 0] * a1 + Ei[1, 1] * a2) * 1e6


def sci(lo_wl, hi_wl, fs):
    a, b = bandpass(lo_wl, fs, *CARDIAC), bandpass(hi_wl, fs, *CARDIAC)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        if a[i].std() > 0 and b[i].std() > 0:
            out[i] = np.corrcoef(a[i], b[i])[0, 1]
    return out


def epoch(x, fs, onsets_s, tmin=-2.0, tmax=20.0):
    a, b = int(tmin * fs), int(tmax * fs)
    segs = [x[:, int(round(o * fs)) + a: int(round(o * fs)) + b]
            for o in np.atleast_1d(onsets_s)
            if int(round(o * fs)) + a >= 0 and int(round(o * fs)) + b <= x.shape[1]]
    if not segs:
        return np.empty((0, x.shape[0], b - a))
    n = min(s.shape[1] for s in segs)
    return np.stack([s[:, :n] for s in segs])


def split_half(ep):
    if ep.shape[0] < 6:
        return np.nan
    rng = np.random.default_rng(0)
    p = rng.permutation(ep.shape[0])
    m1 = ep[p[: len(p) // 2]].mean(0).ravel()
    m2 = ep[p[len(p) // 2:]].mean(0).ravel()
    if m1.std() == 0 or m2.std() == 0:
        return np.nan
    return float(np.corrcoef(m1, m2)[0, 1])


def evoked_snr(ep, fs, peak_s=6.0):
    if ep.shape[0] < 2:
        return np.nan, np.nan
    m, se = ep.mean(0), ep.std(0) / np.sqrt(ep.shape[0])
    t = np.arange(m.shape[1]) / fs - 2.0
    pk = np.argmin(np.abs(t - peak_s))
    peak = float(np.nanmean(m[:, pk]))
    ci = float(np.nanmean(1.96 * se[:, pk]))
    return (abs(peak) / ci if ci > 0 else np.nan), peak


def split_wavelengths(x, clab):
    """(T, 72) + channel labels -> (lo, hi) each (36, T), matched by pair name."""
    names = [str(c) for c in np.atleast_1d(clab)]
    lo_idx = {n.replace("lowWL", ""): i for i, n in enumerate(names) if "lowWL" in n}
    hi_idx = {n.replace("highWL", ""): i for i, n in enumerate(names) if "highWL" in n}
    pairs = sorted(set(lo_idx) & set(hi_idx))
    lo = np.stack([x[:, lo_idx[p]] for p in pairs])
    hi = np.stack([x[:, hi_idx[p]] for p in pairs])
    return lo, hi, pairs


def assess(subj_dir: Path) -> list[dict]:
    cf, mf = subj_dir / "cnt.mat", subj_dir / "mrk.mat"
    if not cf.exists() or not mf.exists():
        return []
    cnt = sio.loadmat(cf, squeeze_me=True, struct_as_record=False)["cnt"]
    mrk = sio.loadmat(mf, squeeze_me=True, struct_as_record=False)["mrk"]
    cnt, mrk = np.atleast_1d(cnt), np.atleast_1d(mrk)

    out = []
    for i in range(cnt.size):
        e = cnt.ravel()[i]
        title = str(getattr(e, "title", f"run{i}"))
        fs = float(np.atleast_1d(e.fs).ravel()[0])
        lo, hi, pairs = split_wavelengths(np.asarray(e.x, dtype=np.float64), e.clab)

        s = sci(lo, hi, fs)
        hbo, hbr = beer_lambert(lo, hi)
        hbo_f, hbr_f = bandpass(hbo, fs, 0.01, 0.2), bandpass(hbr, fs, 0.01, 0.2)
        cc = [np.corrcoef(hbo_f[j], hbr_f[j])[0, 1] for j in range(hbo_f.shape[0])
              if hbo_f[j].std() > 0 and hbr_f[j].std() > 0]

        onsets, labels = np.array([]), np.array([])
        if i < mrk.size:
            me = mrk.ravel()[i]
            onsets = np.atleast_1d(np.asarray(me.time, dtype=float)) / 1000.0
            y = np.asarray(me.y)
            labels = np.argmax(y, axis=0) if y.ndim == 2 else y

        ep = epoch(hbo_f, fs, onsets)
        snr, peak = evoked_snr(ep, fs)
        rel = split_half(ep)
        # best single class, in case the two classes cancel when pooled
        for c in (np.unique(labels) if labels.size else []):
            e2 = epoch(hbo_f, fs, onsets[labels == c])
            if e2.shape[0] < 6:
                continue
            s2, _ = evoked_snr(e2, fs)
            r2 = split_half(e2)
            if np.isfinite(s2) and (not np.isfinite(snr) or s2 > snr):
                snr = s2
            if np.isfinite(r2) and (not np.isfinite(rel) or r2 > rel):
                rel = r2

        corr = float(np.median(cc)) if cc else np.nan
        out.append({
            "subject": subj_dir.name, "run": i, "task": title, "fs_hz": fs,
            "n_pairs": len(pairs), "duration_s": lo.shape[1] / fs,
            "n_events": int(onsets.size), "n_epochs": int(ep.shape[0]),
            "sci_median": float(np.nanmedian(s)),
            "sci_frac_good": float(np.nanmean(np.nan_to_num(s) > 0.5)),
            "hbo_hbr_corr": corr, "evoked_peak_umol": peak,
            "evoked_snr": snr, "split_half_reliability": rel,
            "passes_gate": bool(np.isfinite(corr) and corr < GATE["hbo_hbr_corr_max"]
                                and np.isfinite(snr) and snr > GATE["evoked_snr_min"]
                                and np.isfinite(rel) and rel > GATE["reliability_min"]),
        })
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    subs = sorted([d for d in ROOT.iterdir() if d.is_dir()])[:n]
    rows = []
    for sd in subs:
        try:
            got = assess(sd)
        except Exception as exc:
            print(f"{sd.name}: ERROR {type(exc).__name__}: {str(exc)[:90]}")
            continue
        rows.extend(got)
        mi = [g for g in got if g["task"].upper().startswith("MI")]
        if mi:
            print(f"{sd.name:12s} MI runs={len(mi)}  "
                  f"SCI {np.median([g['sci_median'] for g in mi]):+.3f} "
                  f"({np.mean([g['sci_frac_good'] for g in mi])*100:3.0f}% good)  "
                  f"HbO/HbR {np.median([g['hbo_hbr_corr'] for g in mi]):+.3f}  "
                  f"SNR {np.nanmedian([g['evoked_snr'] for g in mi]):5.2f}  "
                  f"rel {np.nanmedian([g['split_half_reliability'] for g in mi]):+.3f}")

    if not rows:
        raise SystemExit("nothing assessed")
    df = pd.DataFrame(rows)
    out = TABLE_DIR / "target_gate_tuberlin_hbci.csv"
    df.to_csv(out, index=False)

    mi = df[df.task.str.upper().str.startswith("MI")]
    ma = df[df.task.str.upper().str.startswith("MA")]

    print("\n" + "=" * 76)
    print(f"subjects {df.subject.nunique()}   MI runs {len(mi)}   MA runs {len(ma)}")
    print(f"{'':26s}{'MI (motor)':>14s}{'MA (arith)':>14s}   reference")
    print(f"{'true SCI median':26s}{mi.sci_median.median():>14.3f}"
          f"{ma.sci_median.median():>14.3f}   Multi-joint 0.023")
    print(f"{'SCI frac > 0.5':26s}{mi.sci_frac_good.median():>14.3f}"
          f"{ma.sci_frac_good.median():>14.3f}")
    print(f"{'HbO/HbR corr':26s}{mi.hbo_hbr_corr.median():>14.3f}"
          f"{ma.hbo_hbr_corr.median():>14.3f}   HEFMI +0.898")
    print(f"{'evoked SNR':26s}{mi.evoked_snr.median():>14.3f}"
          f"{ma.evoked_snr.median():>14.3f}   TUB-WL 0.247")
    print(f"{'split-half reliability':26s}{mi.split_half_reliability.median():>14.3f}"
          f"{ma.split_half_reliability.median():>14.3f}   HEFMI +0.257")
    print(f"{'runs passing gate':26s}{int(mi.passes_gate.sum()):>10d}/{len(mi):<3d}"
          f"{int(ma.passes_gate.sum()):>10d}/{len(ma):<3d}")
    print("=" * 76)

    c, s, r = (mi.hbo_hbr_corr.median(), mi.evoked_snr.median(),
               mi.split_half_reliability.median())
    if mi.passes_gate.mean() >= 0.5:
        print("VERDICT: TU Berlin hBCI MI passes the gate -> a viable MOTOR track EXISTS.")
    elif c < 0 and s > 0.25 and r > 0.25:
        print("VERDICT: best motor fNIRS in the project (correct physiology, usable")
        print("         reliability) but still below the evoked-SNR bar. Usable with")
        print("         trial averaging / amplitude targets, not for single-trial waveform.")
    elif c < 0:
        print("VERDICT: physiology correct but response too weak -> motor track marginal.")
    else:
        print("VERDICT: fails -> no viable motor fNIRS track. Commit to the cognitive track.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
