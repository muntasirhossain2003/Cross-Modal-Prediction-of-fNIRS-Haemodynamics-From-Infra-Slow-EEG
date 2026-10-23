"""EEG quality control, including the simultaneous EEG-fMRI artifact checks.

Produces one row per recording with every quantity the audit brief asks for,
saves the PSD vector and a diagnostic figure, and (for in-scanner data)
compares raw against provider-corrected traces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from qc import metrics as M
from qc.config import (ARRAY_DIR, BETA_BAND, CARDIAC_BAND, DATASETS,
                       EEG_BAND_FILTER, FIG_DIR, MU_BAND)
from qc.rules import EEG_FMRI_RULES, EEG_RULES, RuleHit, band, evaluate


def _slug(row) -> str:
    return (f"{row['dataset']}_{row['subject'].replace('/', '-')}"
            f"_{row['session']}_{row['task']}_{row['modality']}_{row['data_role']}")


def _flatten_epoched(d: dict) -> np.ndarray:
    """(ch, time, trials) -> (ch, time*trials) for continuous-style statistics."""
    x = d["data"]
    return x.reshape(x.shape[0], -1) if x.ndim == 3 else x


# =========================================================================
def qc_eeg_recording(row: dict, data: dict, save_arrays=True, save_figs=True) -> dict:
    """Compute the full EEG QC row for one recording."""
    ds = row["dataset"]
    fs = float(data["fs"])
    x = _flatten_epoched(data)
    slug = _slug(row)

    r: dict = {
        "dataset": ds, "subject": row["subject"], "session": row["session"],
        "task": row["task"], "data_role": row["data_role"],
        "path": row["primary_path"],
    }

    # ---- units, duration, shape -----------------------------------------
    r["units"] = data.get("units", "uV")
    r["fs_hz"] = fs
    r["n_channels"] = int(data.get("n_channels", x.shape[0]))
    r["n_samples"] = int(x.shape[1])
    r["duration_s"] = float(x.shape[1] / fs) if fs > 0 else np.nan
    r["n_trials"] = int(data.get("n_trials", 0)) if data.get("epoched") else 0

    doc_fs = DATASETS[ds].get("documented_eeg_fs")
    r["documented_fs_hz"] = doc_fs
    r["fs_matches_doc"] = (abs(fs - doc_fs) / doc_fs <= EEG_RULES["fs_tolerance_frac"]
                           if doc_fs else None)
    doc_ch = DATASETS[ds].get("documented_eeg_channels")
    r["documented_channels"] = doc_ch
    r["channels_match_doc"] = (r["n_channels"] == doc_ch) if doc_ch else None

    # ---- amplitude -------------------------------------------------------
    amp = M.amplitude_range(x)
    r.update({f"amp_{k}_uv": v for k, v in amp.items()})
    r["robust_amplitude_uv"] = M.robust_amplitude(x)
    r["max_abs_uv"] = float(np.nanmax(np.abs(x))) if x.size else np.nan

    # ---- integrity -------------------------------------------------------
    r["nan_fraction"] = M.nan_fraction(x)
    r["clipped_fraction"] = M.clipped_fraction(x)
    r["constant_fraction"] = M.constant_fraction(x, fs, EEG_RULES["constant_run_ms"])

    # ---- spectrum --------------------------------------------------------
    f, psd = M.welch_psd(x, fs, nperseg_s=4.0, fmax=min(fs / 2, 250.0))
    line_f, line_ratio = M.detect_line_freq(f, psd)
    r["line_freq_hz"] = line_f
    r["line_noise_ratio"] = line_ratio
    r["bp_delta"] = float(np.nanmean(M.band_power(f, psd, 1, 4)))
    r["bp_theta"] = float(np.nanmean(M.band_power(f, psd, 4, 8)))
    r["bp_mu"] = float(np.nanmean(M.band_power(f, psd, *MU_BAND)))
    r["bp_beta"] = float(np.nanmean(M.band_power(f, psd, *BETA_BAND)))
    r["bp_gamma"] = float(np.nanmean(M.band_power(f, psd, 30, min(80, fs / 2 - 1))))
    r["total_power"] = float(np.nanmean(M.band_power(f, psd, 0.5, min(100, fs / 2 - 1))))

    # ---- bad channels ----------------------------------------------------
    bad = M.find_bad_channels(x, fs, EEG_RULES["bad_channel_flat_uv"],
                              EEG_RULES["bad_channel_corr_threshold"],
                              EEG_RULES["bad_channel_amp_zscore"])
    r["n_bad_channels"] = bad["n_bad"]
    r["bad_channel_fraction"] = bad["fraction"]
    r["bad_channel_idx"] = ";".join(map(str, bad["bad_idx"]))
    r["flat_channel_idx"] = ";".join(map(str, bad["flat_idx"]))
    r["uncorrelated_channel_idx"] = ";".join(map(str, bad["uncorrelated_idx"]))
    r["median_interchannel_corr"] = bad["median_corr"]
    # channels the provider already marked as rejected
    prov = data.get("excluded_channels", [])
    r["provider_excluded_channels"] = ";".join(map(str, np.atleast_1d(prov).ravel().tolist()))

    # ---- post-filter amplitude distribution + transients ------------------
    xf = M.bandpass(x, fs, *EEG_BAND_FILTER)
    ampf = M.amplitude_range(xf)
    r.update({f"filt_amp_{k}_uv": v for k, v in ampf.items()})
    r["filt_robust_amplitude_uv"] = M.robust_amplitude(xf)
    tr = M.transient_count(xf, fs, EEG_RULES["transient_threshold_uv"])
    r["transient_count"] = tr["count"]
    r["transients_per_min"] = tr["per_min"]
    r["transients_per_min_per_ch"] = tr["per_min_per_channel"]

    # ---- events ----------------------------------------------------------
    ev = data.get("events", [])
    onsets = data.get("event_onsets_s", np.array([o for o, _ in ev] if ev else []))
    r["n_events"] = int(np.size(onsets))
    r["event_coverage_frac"] = (float((np.max(onsets) - np.min(onsets)) / r["duration_s"])
                                if np.size(onsets) > 1 and r["duration_s"] > 0 else np.nan)
    r["n_trials_from_labels"] = int(np.size(data.get("labels", [])))

    # =====================================================================
    # Simultaneous EEG-fMRI specific
    # =====================================================================
    is_mr = DATASETS[ds]["simultaneous_fmri"]
    r["in_scanner"] = is_mr
    if is_mr:
        tr_s = DATASETS[ds].get("documented_tr", 2.0)
        n_slices = row.get("n_slices", 32)
        slice_f = n_slices / tr_s          # gradient fundamental
        vol_f = 1.0 / tr_s
        gh = M.harmonic_residual(f, psd, slice_f, n_harmonics=5)
        vh = M.harmonic_residual(f, psd, vol_f, n_harmonics=8)
        r["gradient_slice_freq_hz"] = slice_f
        r["gradient_harmonic_ratio_max"] = gh["max"]
        r["gradient_harmonic_ratio_mean"] = gh["mean"]
        r["volume_harmonic_ratio_max"] = vh["max"]
        # BCG: cardiac-band peak vs neighbourhood
        fc, pc = M.welch_psd(x, fs, nperseg_s=20.0, fmax=20.0)
        cb = M.band_power(fc, pc, *CARDIAC_BAND)
        nb = M.band_power(fc, pc, 2.5, 8.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r["bcg_residual_ratio"] = float(np.nanmedian(np.where(nb > 0, cb / nb, np.nan)))
        r["cardiac_band_power"] = float(np.nanmean(cb))
    else:
        for k in ("gradient_slice_freq_hz", "gradient_harmonic_ratio_max",
                  "gradient_harmonic_ratio_mean", "volume_harmonic_ratio_max",
                  "bcg_residual_ratio", "cardiac_band_power"):
            r[k] = np.nan

    # ---- persist ---------------------------------------------------------
    if save_arrays:
        out = ARRAY_DIR / "eeg_psd"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{slug}.npz", freqs=f,
                            psd_mean=np.nanmean(psd, axis=0),
                            psd_per_channel=psd.astype(np.float32))
    if save_figs:
        _figure_eeg(row, x, xf, f, psd, fs, slug, r)

    # ---- decision --------------------------------------------------------
    r.update(_decide_eeg(r))
    return r


# =========================================================================
def _decide_eeg(r: dict) -> dict:
    """Apply the pre-registered rules. No thresholds live here."""
    hits: list[RuleHit] = []
    R, RF = EEG_RULES, EEG_FMRI_RULES

    # the audit brief's explicit hard rule for in-scanner raw data
    if r.get("in_scanner") and r["data_role"] == "raw":
        if r["robust_amplitude_uv"] > RF["raw_amplitude_hard_limit_uv"]:
            hits.append(RuleHit(
                "raw_amplitude_hard_limit_uv", "FAIL",
                round(r["robust_amplitude_uv"], 1), RF["raw_amplitude_hard_limit_uv"],
                "uncorrected in-scanner EEG above the 5 mV limit must not enter training"))

    hits.append(band(r["robust_amplitude_uv"], R["warn_robust_amplitude_uv"],
                     R["max_robust_amplitude_uv"], "robust_amplitude_uv",
                     "implausible amplitude for scalp EEG in uV"))
    hits.append(band(r["robust_amplitude_uv"], None, R["min_robust_amplitude_uv"],
                     "robust_amplitude_uv_low", "signal is effectively flat",
                     higher_is_worse=False))
    hits.append(band(r["nan_fraction"], R["warn_nan_fraction"], R["max_nan_fraction"],
                     "nan_fraction", "missing samples in the recording"))
    hits.append(band(r["clipped_fraction"], None, R["max_clipped_fraction"],
                     "clipped_fraction", "samples pinned at the ADC rail"))
    hits.append(band(r["constant_fraction"], None, R["max_constant_fraction"],
                     "constant_fraction", "extended constant runs indicate dropout"))
    hits.append(band(r["line_noise_ratio"], R["warn_line_ratio"], R["max_line_ratio"],
                     "line_noise_ratio", "mains contamination relative to neighbourhood"))
    hits.append(band(r["bad_channel_fraction"], R["warn_bad_channel_fraction"],
                     R["max_bad_channel_fraction"], "bad_channel_fraction",
                     "too many flat/railed/uncorrelated channels to interpolate"))
    hits.append(band(r["transients_per_min_per_ch"], R["warn_transients_per_min"],
                     R["max_transients_per_min"], "transients_per_min_per_ch",
                     "high-amplitude transient rate indicates movement contamination"))
    hits.append(band(r["duration_s"], None, R["min_duration_s"], "duration_s",
                     "too short to support 30 s analysis windows", higher_is_worse=False))

    if r["data_role"] == "corrected" and r.get("in_scanner"):
        hits.append(band(r["robust_amplitude_uv"],
                         RF["corrected_warn_robust_amplitude_uv"],
                         RF["corrected_max_robust_amplitude_uv"],
                         "corrected_robust_amplitude_uv",
                         "corrected trace still does not look like scalp EEG"))
        hits.append(band(r["gradient_harmonic_ratio_max"],
                         RF["warn_gradient_residual_ratio"],
                         RF["max_gradient_residual_ratio"],
                         "gradient_harmonic_ratio_max",
                         "residual scanner-gradient harmonics after correction"))
        hits.append(band(r["bcg_residual_ratio"], RF["warn_bcg_residual_ratio"],
                         RF["max_bcg_residual_ratio"], "bcg_residual_ratio",
                         "residual ballistocardiogram after correction"))

    if r["fs_matches_doc"] is False:
        hits.append(RuleHit("fs_matches_doc", "WARN", r["fs_hz"], r["documented_fs_hz"],
                            "sampling rate disagrees with the documented value"))
    if r["channels_match_doc"] is False:
        hits.append(RuleHit("channels_match_doc", "WARN", r["n_channels"],
                            r["documented_channels"],
                            "channel count disagrees with the documented value"))

    hits = [h for h in hits if h is not None]
    decision, reason = evaluate(hits)
    return {"qc_decision": decision, "qc_reason": reason,
            "n_rule_hits": len(hits)}


# =========================================================================
def _figure_eeg(row, x, xf, f, psd, fs, slug, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = FIG_DIR / row["dataset"] / "eeg"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle(f"{slug}   [{r['qc_decision'] if 'qc_decision' in r else 'QC'}]",
                 fontsize=10)

    # time series (first 10 s, up to 8 channels)
    n = min(int(10 * fs), x.shape[1])
    t = np.arange(n) / fs
    for i in range(min(8, x.shape[0])):
        ax[0, 0].plot(t, x[i, :n] + i * 4 * np.nanstd(x[i, :n] + 1e-9), lw=0.4)
    ax[0, 0].set_title(f"raw trace, first 10 s (robust amp "
                       f"{r['robust_amplitude_uv']:.1f} uV)", fontsize=9)
    ax[0, 0].set_xlabel("s")

    # PSD
    ax[0, 1].semilogy(f, np.nanmean(psd, axis=0), lw=0.8)
    ax[0, 1].set_title(f"mean PSD (line {r['line_freq_hz']} Hz, "
                       f"ratio {r['line_noise_ratio']:.1f})", fontsize=9)
    ax[0, 1].set_xlabel("Hz")
    ax[0, 1].set_xlim(0, min(120, fs / 2))

    # post-filter amplitude distribution
    v = xf[np.isfinite(xf)]
    if v.size > 200000:
        v = np.random.default_rng(0).choice(v, 200000, replace=False)
    ax[1, 0].hist(v, bins=200, log=True)
    ax[1, 0].set_title(f"post-filter ({EEG_BAND_FILTER[0]}-{EEG_BAND_FILTER[1]} Hz) "
                       f"amplitude distribution", fontsize=9)
    ax[1, 0].set_xlabel("uV")

    # per-channel SD, bad channels marked
    sd = np.nanstd(x, axis=1)
    bad = [int(i) for i in str(r["bad_channel_idx"]).split(";") if i != ""]
    ax[1, 1].bar(range(len(sd)), sd, color=["red" if i in bad else "steelblue"
                                            for i in range(len(sd))])
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title(f"per-channel SD ({r['n_bad_channels']} bad, red)", fontsize=9)
    ax[1, 1].set_xlabel("channel")

    fig.tight_layout()
    fig.savefig(out / f"{slug}.png", dpi=85)
    plt.close(fig)


# =========================================================================
def compare_raw_vs_corrected(raw_row, raw_data, corr_row, corr_data,
                             save_figs=True) -> dict:
    """Raw-vs-corrected comparison for simultaneous EEG-fMRI.

    Saves the paired time series and spectra the audit brief asks for and
    reports how much power the correction actually removed.
    """
    fs_r, fs_c = float(raw_data["fs"]), float(corr_data["fs"])
    xr, xc = _flatten_epoched(raw_data), _flatten_epoched(corr_data)
    ds = raw_row["dataset"]
    tr_s = DATASETS[ds].get("documented_tr", 2.0)
    slice_f = 32 / tr_s

    fr, pr = M.welch_psd(xr, fs_r, nperseg_s=4.0, fmax=min(fs_r / 2, 250))
    fc, pc = M.welch_psd(xc, fs_c, nperseg_s=4.0, fmax=min(fs_c / 2, 250))

    tot_r = float(np.nanmean(M.band_power(fr, pr, 0.5, min(100, fs_r / 2 - 1))))
    tot_c = float(np.nanmean(M.band_power(fc, pc, 0.5, min(100, fs_c / 2 - 1))))
    with np.errstate(divide="ignore", invalid="ignore"):
        red_db = 10 * np.log10(tot_r / tot_c) if tot_c > 0 and tot_r > 0 else np.nan

    gr = M.harmonic_residual(fr, pr, slice_f, 5)
    gc = M.harmonic_residual(fc, pc, slice_f, 5)

    out = {
        "dataset": ds, "subject": raw_row["subject"], "task": raw_row["task"],
        "raw_robust_amplitude_uv": M.robust_amplitude(xr),
        "corrected_robust_amplitude_uv": M.robust_amplitude(xc),
        "raw_total_power": tot_r, "corrected_total_power": tot_c,
        "power_reduction_db": float(red_db),
        "raw_gradient_harmonic_ratio_max": gr["max"],
        "corrected_gradient_harmonic_ratio_max": gc["max"],
        "raw_fs_hz": fs_r, "corrected_fs_hz": fs_c,
        "raw_n_channels": xr.shape[0], "corrected_n_channels": xc.shape[0],
        "correction_effective": bool(np.isfinite(red_db) and
                                     red_db >= EEG_FMRI_RULES["min_gradient_power_reduction_db"]),
    }

    if save_figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = FIG_DIR / ds / "raw_vs_corrected"
        d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(13, 7))
        slug = f"{raw_row['subject']}_{raw_row['task']}"
        fig.suptitle(f"{ds} {slug}: raw vs gradient/BCG-corrected", fontsize=10)

        n_r, n_c = min(int(5 * fs_r), xr.shape[1]), min(int(5 * fs_c), xc.shape[1])
        ax[0, 0].plot(np.arange(n_r) / fs_r, xr[0, :n_r], lw=0.4)
        ax[0, 0].set_title(f"RAW ch0, 5 s (robust {out['raw_robust_amplitude_uv']:.0f} uV)",
                           fontsize=9)
        ax[0, 1].plot(np.arange(n_c) / fs_c, xc[0, :n_c], lw=0.4, color="green")
        ax[0, 1].set_title(f"CORRECTED ch0, 5 s "
                           f"(robust {out['corrected_robust_amplitude_uv']:.1f} uV)",
                           fontsize=9)
        ax[1, 0].semilogy(fr, np.nanmean(pr, axis=0), lw=0.7, label="raw")
        ax[1, 0].semilogy(fc, np.nanmean(pc, axis=0), lw=0.7, color="green",
                          label="corrected")
        for k in range(1, 6):
            if slice_f * k < min(fr[-1], 120):
                ax[1, 0].axvline(slice_f * k, color="red", ls=":", lw=0.6)
        ax[1, 0].set_xlim(0, min(120, fr[-1]))
        ax[1, 0].legend(fontsize=8)
        ax[1, 0].set_title(f"spectra (red = slice-freq harmonics, {slice_f:.0f} Hz)",
                           fontsize=9)
        ax[1, 0].set_xlabel("Hz")
        ax[1, 1].bar(["raw", "corrected"],
                     [out["raw_gradient_harmonic_ratio_max"],
                      out["corrected_gradient_harmonic_ratio_max"]],
                     color=["firebrick", "green"])
        ax[1, 1].axhline(EEG_FMRI_RULES["max_gradient_residual_ratio"], color="k", ls="--",
                         lw=0.8, label="FAIL limit")
        ax[1, 1].set_title(f"gradient harmonic ratio "
                           f"({out['power_reduction_db']:.1f} dB total reduction)",
                           fontsize=9)
        ax[1, 1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / f"{slug}.png", dpi=85)
        plt.close(fig)

    return out
