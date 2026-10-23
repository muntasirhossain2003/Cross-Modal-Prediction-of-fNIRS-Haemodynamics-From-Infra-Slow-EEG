"""fNIRS quality control.

Covers channel availability, coupling quality, saturation/flat-line, motion,
concentration amplitude, baseline stability, event-locked response with CI, and
usable trial counts -- plus the explicit with/without global-signal-regression
ablation, because GSR can remove genuine broad motor responses and must not be
applied silently.
"""
from __future__ import annotations

import numpy as np

from qc import metrics as M
from qc.config import ARRAY_DIR, DATASETS, FIG_DIR, FNIRS_CARDIAC_BAND, HRF_PEAK_S
from qc.rules import FNIRS_RULES, RuleHit, band, evaluate


def _slug(row) -> str:
    return (f"{row['dataset']}_{row['subject'].replace('/', '-')}"
            f"_{row['session']}_{row['task']}_fnirs")


def _continuous(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1) if x.ndim == 3 else x


# =========================================================================
def qc_fnirs_recording(row: dict, data: dict, events=None,
                       save_arrays=True, save_figs=True) -> dict:
    ds = row["dataset"]
    fs = float(data["fs"]) if np.isfinite(data.get("fs", np.nan)) else np.nan
    chrom = data.get("chromophores", {})
    hbo = chrom.get("oxy")
    hbr = chrom.get("doxy")
    slug = _slug(row)

    # HEFMI ships fs as an empty field on the NIRS struct; fall back to the
    # documented rate rather than silently producing NaN metrics.
    if not np.isfinite(fs):
        fs = float(DATASETS[ds].get("documented_fnirs_fs", np.nan))
        fs_source = "documented (missing in file)"
    else:
        fs_source = "file"

    r: dict = {
        "dataset": ds, "subject": row["subject"], "session": row["session"],
        "task": row["task"], "data_role": row["data_role"],
        "path": row["primary_path"], "fs_hz": fs, "fs_source": fs_source,
        "units": data.get("units", "umol/L"),
    }

    ho = _continuous(hbo) if hbo is not None else None
    hr_ = _continuous(hbr) if hbr is not None else None
    if ho is None:
        r["qc_decision"], r["qc_reason"] = "FAIL", "no HbO data in file"
        return r

    n_ch, n_t = ho.shape
    r["n_channels"] = int(n_ch)
    r["n_samples"] = int(n_t)
    r["duration_s"] = float(n_t / fs) if fs > 0 else np.nan
    r["n_trials"] = int(data.get("n_trials", 0)) if data.get("epoched") else 0
    doc_ch = DATASETS[ds].get("documented_fnirs_channels")
    r["documented_channels"] = doc_ch
    r["channels_match_doc"] = (n_ch == doc_ch) if doc_ch else None

    # ---- channel / source-detector availability --------------------------
    finite = np.isfinite(ho)
    per_ch_finite = finite.mean(axis=1)
    ch_sd = np.nanstd(np.where(finite, ho, np.nan), axis=1)
    available = (per_ch_finite > 0.5) & (ch_sd > 0)
    r["n_channels_available"] = int(available.sum())
    r["channel_availability"] = float(available.mean())
    r["dead_channel_idx"] = ";".join(map(str, np.flatnonzero(~available).tolist()))

    # ---- coupling quality ------------------------------------------------
    # True SCI needs the two raw wavelengths. These datasets ship Beer-Lambert
    # concentrations, so we report the cardiac-band power fraction instead and
    # record why SCI itself is unavailable.
    if data.get("has_raw_wavelengths"):
        sci = M.scalp_coupling_index(chrom["wl1"], chrom["wl2"], fs, FNIRS_CARDIAC_BAND)
        r["coupling_metric"] = "SCI (raw wavelengths)"
    else:
        sci = M.cardiac_band_ratio(ho, fs, FNIRS_CARDIAC_BAND)
        r["coupling_metric"] = ("cardiac-band power fraction "
                                "(SCI unavailable: data shipped as concentrations)")
    r["coupling_median"] = float(np.nanmedian(sci)) if np.size(sci) else np.nan
    poor = np.nan_to_num(sci, nan=0.0) < FNIRS_RULES["sci_threshold"]
    # for the power-fraction variant a much lower bar is appropriate; scale it
    if not data.get("has_raw_wavelengths"):
        poor = np.nan_to_num(sci, nan=0.0) < 0.02
    r["poor_coupling_fraction"] = float(poor.mean())
    r["poor_coupling_idx"] = ";".join(map(str, np.flatnonzero(poor).tolist()))

    # ---- saturation / flat-line -----------------------------------------
    r["nan_fraction"] = M.nan_fraction(ho)
    r["saturated_fraction"] = M.clipped_fraction(ho)
    r["flatline_fraction"] = M.constant_fraction(
        ho, fs, run_ms=FNIRS_RULES["flatline_run_s"] * 1000.0)

    # ---- amplitude -------------------------------------------------------
    for name, arr in (("hbo", ho), ("hbr", hr_)):
        if arr is None:
            continue
        a = M.amplitude_range(arr)
        r.update({f"{name}_{k}_umol": v for k, v in a.items()})
        r[f"{name}_robust_amplitude_umol"] = M.robust_amplitude(arr)
    r["max_abs_conc_umol"] = float(np.nanmax(np.abs(ho)))

    # HbO and HbR should be anti-correlated in a real haemodynamic response
    if hr_ is not None and hr_.shape == ho.shape:
        with np.errstate(invalid="ignore"):
            cc = [np.corrcoef(ho[i], hr_[i])[0, 1]
                  for i in range(n_ch)
                  if np.std(ho[i]) > 0 and np.std(hr_[i]) > 0]
        r["hbo_hbr_corr_median"] = float(np.nanmedian(cc)) if cc else np.nan
    else:
        r["hbo_hbr_corr_median"] = np.nan

    # ---- motion ----------------------------------------------------------
    mo = M.motion_artifact_rate(ho, fs, FNIRS_RULES["motion_threshold_sd"])
    r["motion_count"] = mo["count"]
    r["motion_per_min"] = mo["per_min"]
    r["motion_per_min_per_ch"] = mo["per_min_per_channel"]

    # ---- baseline stability + event-locked response ----------------------
    ep_hbo, ep_hbr, onsets = _epochs_for(row, data, ho, hr_, fs, events)
    r["n_epochs"] = int(ep_hbo.shape[0]) if ep_hbo is not None else 0

    if ep_hbo is not None and ep_hbo.shape[0] > 0:
        n_base = max(int(round(2.0 * fs)), 1)
        base = ep_hbo[:, :, :n_base]
        r["baseline_sd_umol"] = float(np.nanmean(np.nanstd(base, axis=2)))
        r["baseline_drift_umol"] = float(np.nanmean(
            np.abs(np.nanmean(base[:, :, -n_base // 2:], axis=2) -
                   np.nanmean(base[:, :, :n_base // 2], axis=2))))
        # trial-level usability: reject trials with any implausible value
        bad_trial = (np.nanmax(np.abs(ep_hbo), axis=(1, 2)) >
                     FNIRS_RULES["max_abs_conc_umol"])
        r["n_usable_trials"] = int((~bad_trial).sum())
        r["n_rejected_trials"] = int(bad_trial.sum())

        # event-locked mean + CI at the canonical HRF peak
        m, ci = M.mean_ci(ep_hbo, axis=0)
        t = np.arange(m.shape[1]) / fs - 2.0
        pk = np.argmin(np.abs(t - HRF_PEAK_S))
        r["evoked_hbo_peak_umol"] = float(np.nanmean(m[:, pk]))
        r["evoked_hbo_peak_ci"] = float(np.nanmean(ci[:, pk]))
        r["evoked_hbo_snr"] = (float(abs(r["evoked_hbo_peak_umol"]) /
                                     r["evoked_hbo_peak_ci"])
                               if r["evoked_hbo_peak_ci"] > 0 else np.nan)
    else:
        for k in ("baseline_sd_umol", "baseline_drift_umol", "evoked_hbo_peak_umol",
                  "evoked_hbo_peak_ci", "evoked_hbo_snr", "baseline_drift_umol"):
            r[k] = np.nan
        r["n_usable_trials"] = r["n_trials"]
        r["n_rejected_trials"] = 0

    # =====================================================================
    # GSR ABLATION -- explicit, never applied silently
    # =====================================================================
    gsr = M.global_signal_regress(ho)
    r["gsr_variance_removed_frac"] = float(
        1.0 - np.nansum(gsr ** 2) / max(np.nansum(ho ** 2), 1e-12))
    r["gsr_robust_amplitude_umol"] = M.robust_amplitude(gsr)
    if ep_hbo is not None and ep_hbo.shape[0] > 0 and np.size(onsets):
        ep_g = M.epoch(gsr, fs, onsets, -2.0, 15.0)
        if ep_g.shape[0] > 0:
            mg, cig = M.mean_ci(ep_g, axis=0)
            tg = np.arange(mg.shape[1]) / fs - 2.0
            pkg = np.argmin(np.abs(tg - HRF_PEAK_S))
            r["gsr_evoked_hbo_peak_umol"] = float(np.nanmean(mg[:, pkg]))
            r["gsr_evoked_hbo_peak_ci"] = float(np.nanmean(cig[:, pkg]))
            base_amp = abs(r.get("evoked_hbo_peak_umol", np.nan))
            r["gsr_evoked_amplitude_retained_frac"] = (
                float(abs(r["gsr_evoked_hbo_peak_umol"]) / base_amp)
                if base_amp and np.isfinite(base_amp) and base_amp > 0 else np.nan)
        else:
            r["gsr_evoked_hbo_peak_umol"] = np.nan
            r["gsr_evoked_hbo_peak_ci"] = np.nan
            r["gsr_evoked_amplitude_retained_frac"] = np.nan
    else:
        r["gsr_evoked_hbo_peak_umol"] = np.nan
        r["gsr_evoked_hbo_peak_ci"] = np.nan
        r["gsr_evoked_amplitude_retained_frac"] = np.nan

    # ---- persist ---------------------------------------------------------
    if save_arrays and ep_hbo is not None and ep_hbo.shape[0] > 0:
        out = ARRAY_DIR / "fnirs_evoked"
        out.mkdir(parents=True, exist_ok=True)
        m, ci = M.mean_ci(ep_hbo, axis=0)
        payload = {"t": np.arange(m.shape[1]) / fs - 2.0,
                   "hbo_mean": m.astype(np.float32),
                   "hbo_ci": ci.astype(np.float32)}
        if ep_hbr is not None and ep_hbr.shape[0] > 0:
            mr, cir = M.mean_ci(ep_hbr, axis=0)
            payload["hbr_mean"] = mr.astype(np.float32)
            payload["hbr_ci"] = cir.astype(np.float32)
        np.savez_compressed(out / f"{slug}.npz", **payload)

    if save_figs:
        _figure_fnirs(row, ho, hr_, ep_hbo, ep_hbr, gsr, fs, slug, r, onsets)

    r.update(_decide_fnirs(r))
    return r


# =========================================================================
def _epochs_for(row, data, ho, hr_, fs, events):
    """Return (epochs_hbo, epochs_hbr, onsets_s) for either epoched or continuous data."""
    if data.get("epoched") and data["chromophores"]["oxy"].ndim == 3:
        # already trial-structured: (ch, time, trials) -> (trials, ch, time)
        eo = np.transpose(data["chromophores"]["oxy"], (2, 0, 1))
        er = (np.transpose(data["chromophores"]["doxy"], (2, 0, 1))
              if "doxy" in data["chromophores"] else None)
        return eo, er, np.array([])
    onsets = np.asarray(events.get("onsets_s", [])) if events else np.array([])
    if onsets.size == 0:
        return None, None, onsets
    eo = M.epoch(ho, fs, onsets, -2.0, 15.0)
    er = M.epoch(hr_, fs, onsets, -2.0, 15.0) if hr_ is not None else None
    return eo, er, onsets


# =========================================================================
def _decide_fnirs(r: dict) -> dict:
    hits: list[RuleHit] = []
    R = FNIRS_RULES

    hits.append(band(r.get("channel_availability"), R["warn_channel_availability"],
                     R["min_channel_availability"], "channel_availability",
                     "too much of the montage carries no data",
                     higher_is_worse=False))
    hits.append(band(r.get("poor_coupling_fraction"), R["warn_poor_coupling_fraction"],
                     R["max_poor_coupling_fraction"], "poor_coupling_fraction",
                     "too many optodes show no cardiac pulsation (poor scalp coupling)"))
    hits.append(band(r.get("saturated_fraction"), None, R["max_saturated_fraction"],
                     "saturated_fraction", "detector saturation"))
    hits.append(band(r.get("flatline_fraction"), None, R["max_flatline_fraction"],
                     "flatline_fraction", "flat-lined signal"))
    hits.append(band(r.get("motion_per_min_per_ch"), R["warn_motion_per_min"],
                     R["max_motion_per_min"], "motion_per_min_per_ch",
                     "motion artifact rate too high"))
    hits.append(band(r.get("max_abs_conc_umol"), R["warn_abs_conc_umol"],
                     R["max_abs_conc_umol"], "max_abs_conc_umol",
                     "concentration outside physiological range (unit/conversion problem)"))
    hits.append(band(r.get("baseline_sd_umol"), R["warn_baseline_sd_umol"],
                     R["max_baseline_sd_umol"], "baseline_sd_umol",
                     "pre-event baseline too unstable to define an amplitude"))
    hits.append(band(r.get("n_usable_trials"), R["warn_usable_trials"],
                     R["min_usable_trials"], "n_usable_trials",
                     "too few usable trials", higher_is_worse=False))
    if r.get("channels_match_doc") is False:
        hits.append(RuleHit("channels_match_doc", "WARN", r.get("n_channels"),
                            r.get("documented_channels"),
                            "channel count disagrees with the documented montage"))

    hits = [h for h in hits if h is not None]
    decision, reason = evaluate(hits)
    return {"qc_decision": decision, "qc_reason": reason, "n_rule_hits": len(hits)}


# =========================================================================
def _figure_fnirs(row, ho, hr_, ep_hbo, ep_hbr, gsr, fs, slug, r, onsets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = FIG_DIR / row["dataset"] / "fnirs"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle(f"{slug}   [{r.get('qc_decision', 'QC')}]", fontsize=10)

    n = min(int(120 * fs), ho.shape[1]) if fs > 0 else min(2000, ho.shape[1])
    t = np.arange(n) / fs
    for i in range(min(6, ho.shape[0])):
        ax[0, 0].plot(t, ho[i, :n], lw=0.5)
    ax[0, 0].set_title(f"HbO, first {n/fs:.0f} s "
                       f"(avail {r['channel_availability']*100:.0f}%, "
                       f"motion {r['motion_per_min_per_ch']:.1f}/min/ch)", fontsize=9)
    ax[0, 0].set_xlabel("s"); ax[0, 0].set_ylabel("uM")

    # event-locked mean +/- CI
    if ep_hbo is not None and ep_hbo.shape[0] > 0:
        m, ci = M.mean_ci(ep_hbo, axis=0)
        tt = np.arange(m.shape[1]) / fs - (2.0 if np.size(onsets) else 0.0)
        gm, gci = np.nanmean(m, axis=0), np.nanmean(ci, axis=0)
        ax[0, 1].plot(tt, gm, color="crimson", label="HbO")
        ax[0, 1].fill_between(tt, gm - gci, gm + gci, color="crimson", alpha=0.25)
        if ep_hbr is not None and ep_hbr.shape[0] > 0:
            mr, cir = M.mean_ci(ep_hbr, axis=0)
            gmr, gcir = np.nanmean(mr, axis=0), np.nanmean(cir, axis=0)
            ax[0, 1].plot(tt, gmr, color="royalblue", label="HbR")
            ax[0, 1].fill_between(tt, gmr - gcir, gmr + gcir, color="royalblue", alpha=0.25)
        ax[0, 1].axvline(0, color="k", lw=0.6, ls=":")
        ax[0, 1].legend(fontsize=8)
        ax[0, 1].set_title(f"event-locked mean +/- 95% CI "
                           f"(n={ep_hbo.shape[0]} trials)", fontsize=9)
        ax[0, 1].set_xlabel("s from onset"); ax[0, 1].set_ylabel("uM")
    else:
        ax[0, 1].text(0.5, 0.5, "no epochs available", ha="center")

    # per-channel coupling
    sci = M.cardiac_band_ratio(ho, fs, FNIRS_CARDIAC_BAND)
    poor = [int(i) for i in str(r.get("poor_coupling_idx", "")).split(";") if i != ""]
    ax[1, 0].bar(range(len(sci)), np.nan_to_num(sci),
                 color=["red" if i in poor else "seagreen" for i in range(len(sci))])
    ax[1, 0].set_title(f"cardiac-band power fraction per channel "
                       f"({len(poor)} poor, red)", fontsize=9)
    ax[1, 0].set_xlabel("channel")

    # GSR ablation
    ax[1, 1].plot(t, np.nanmean(ho[:, :n], axis=0), label="raw (channel mean)", lw=0.8)
    ax[1, 1].plot(t, np.nanmean(gsr[:, :n], axis=0), label="after GSR", lw=0.8)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title(f"GSR ablation: {r['gsr_variance_removed_frac']*100:.0f}% of "
                       f"variance removed", fontsize=9)
    ax[1, 1].set_xlabel("s")

    fig.tight_layout()
    fig.savefig(out / f"{slug}.png", dpi=85)
    plt.close(fig)
