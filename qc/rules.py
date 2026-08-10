"""PRE-REGISTERED exclusion rules for the NeuroVascNet data-quality audit.

=============================================================================
FROZEN 2026-07-25, BEFORE ANY QC METRIC WAS COMPUTED OR ANY MODEL RE-RUN.
=============================================================================

Every threshold below is chosen from standard practice in the EEG / fNIRS /
fMRI methods literature, NOT from looking at this project's distributions and
NOT from looking at model performance. Changing a number in this file after
results have been seen invalidates the audit; if a threshold genuinely has to
change, add a new named rule set and report both.

Rationale is given inline for every threshold so a reviewer can judge it
independently of the outcome it produces.

Severity model
--------------
  FAIL  -> recording is excluded from model training/evaluation.
  WARN  -> recording is retained but the reason is carried into the report.
Each rule returns (severity, reason) so the report can list *why* every
excluded recording was excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

RULESET_VERSION = "1.0.0-frozen-2026-07-25"

Severity = Literal["PASS", "WARN", "FAIL"]


# =============================================================================
# EEG
# =============================================================================
EEG_RULES = {
    # Amplitude / units -------------------------------------------------------
    # Scalp EEG in microvolts sits well inside +/-500 uV once artifacts are
    # excluded. A robust (99.5th pct) amplitude above 5 mV means the trace is
    # either not in uV or is dominated by an uncorrected artifact (e.g. MR
    # gradient). This is the rule the audit brief calls out explicitly for
    # ds002336 raw.
    "max_robust_amplitude_uv": 5000.0,
    # Below this the channel is almost certainly dead/disconnected rather than
    # quiet brain activity.
    "min_robust_amplitude_uv": 0.05,
    # A plausible-but-suspicious band: flag, do not exclude.
    "warn_robust_amplitude_uv": 500.0,

    # Data integrity ----------------------------------------------------------
    # Any NaN at all is a loader/interpolation bug worth surfacing; >1% of
    # samples missing makes the recording unusable.
    "max_nan_fraction": 0.01,
    "warn_nan_fraction": 0.0,
    # Samples pinned at the ADC rail. 1% is already severe for a continuous
    # recording.
    "max_clipped_fraction": 0.01,
    # Constant runs (>= 100 ms of identical values) indicate dropout.
    "max_constant_fraction": 0.05,
    "constant_run_ms": 100.0,

    # Noise -------------------------------------------------------------------
    # Line noise expressed as the ratio of power at the line frequency to the
    # local spectral neighbourhood. 10x over neighbourhood is a strong but
    # correctable contamination; 100x means the recording is line-dominated.
    "warn_line_ratio": 10.0,
    "max_line_ratio": 100.0,

    # Bad channels ------------------------------------------------------------
    # A channel is "bad" if it is flat, railed, or poorly correlated with its
    # peers. Recordings where more than a third of channels are bad cannot be
    # rescued by interpolation.
    "bad_channel_flat_uv": 0.05,
    "bad_channel_corr_threshold": 0.20,   # median |r| with all other channels
    "bad_channel_amp_zscore": 5.0,        # robust z of log channel SD
    "warn_bad_channel_fraction": 0.15,
    "max_bad_channel_fraction": 0.33,

    # Transients --------------------------------------------------------------
    # Count of excursions beyond 100 uV after 0.5-40 Hz filtering, per minute.
    # >60/min (i.e. one per second) indicates a recording dominated by movement.
    "transient_threshold_uv": 100.0,
    "warn_transients_per_min": 20.0,
    "max_transients_per_min": 60.0,

    # Duration / events -------------------------------------------------------
    # Anything under 60 s cannot support 30 s analysis windows with any margin.
    "min_duration_s": 60.0,
    # A task recording with no usable events cannot be epoched.
    "min_events": 1,
    "warn_min_events": 10,
    # Sampling rate must match documentation within 1%.
    "fs_tolerance_frac": 0.01,
}


# =============================================================================
# EEG recorded inside the scanner (simultaneous EEG-fMRI)
# =============================================================================
EEG_FMRI_RULES = {
    # The audit brief's explicit rule: raw signals above 5 mV must not enter
    # training. Applied to the *robust* amplitude so a handful of spikes does
    # not by itself condemn a recording.
    "raw_amplitude_hard_limit_uv": 5000.0,

    # After gradient correction the trace must actually look like EEG.
    "corrected_max_robust_amplitude_uv": 500.0,
    "corrected_warn_robust_amplitude_uv": 250.0,

    # Residual gradient: power at harmonics of the slice/volume frequency
    # relative to the local spectral neighbourhood. 3x = visible residual,
    # 10x = correction failed.
    "warn_gradient_residual_ratio": 3.0,
    "max_gradient_residual_ratio": 10.0,

    # Residual ballistocardiogram: power in the cardiac band relative to
    # neighbouring bands.
    "warn_bcg_residual_ratio": 3.0,
    "max_bcg_residual_ratio": 10.0,

    # Correction must demonstrably reduce power, otherwise it did nothing.
    "min_gradient_power_reduction_db": 6.0,
}


# =============================================================================
# fNIRS
# =============================================================================
FNIRS_RULES = {
    # Channel availability ----------------------------------------------------
    # Fraction of the montage that carries real (non-zero, non-NaN) data.
    "min_channel_availability": 0.50,
    "warn_channel_availability": 0.80,

    # Signal quality ----------------------------------------------------------
    # Scalp Coupling Index: correlation of the two wavelengths in the cardiac
    # band. Pollonini et al. (2014) use 0.75; 0.5 is the common permissive
    # threshold. Channels below threshold are counted as poorly coupled.
    "sci_threshold": 0.50,
    "warn_poor_coupling_fraction": 0.25,
    "max_poor_coupling_fraction": 0.50,

    # Saturation / flat-line --------------------------------------------------
    "max_saturated_fraction": 0.05,
    "max_flatline_fraction": 0.05,
    "flatline_run_s": 2.0,

    # Motion ------------------------------------------------------------------
    # Spike/shift detection on the concentration traces; rate per minute.
    "motion_threshold_sd": 5.0,
    "warn_motion_per_min": 5.0,
    "max_motion_per_min": 20.0,

    # Amplitude plausibility --------------------------------------------------
    # Task-evoked HbO/HbR in adults is of order 0.05-1 uM (here: micromolar).
    # Values beyond 100 uM indicate a unit or conversion problem.
    "max_abs_conc_umol": 100.0,
    "warn_abs_conc_umol": 10.0,

    # Baseline stability ------------------------------------------------------
    # SD of the pre-event baseline, in uM. Large drift makes the epoch's
    # amplitude estimate meaningless.
    "warn_baseline_sd_umol": 2.0,
    "max_baseline_sd_umol": 10.0,

    # Trials ------------------------------------------------------------------
    "min_usable_trials": 10,
    "warn_usable_trials": 20,
}


# =============================================================================
# fMRI
# =============================================================================
FMRI_RULES = {
    # Motion ------------------------------------------------------------------
    # Power et al. (2012)/(2014) framewise displacement conventions.
    "fd_censor_mm": 0.5,
    "warn_mean_fd_mm": 0.2,
    "max_mean_fd_mm": 0.5,
    "warn_censored_fraction": 0.20,
    "max_censored_fraction": 0.40,

    # Temporal SNR ------------------------------------------------------------
    # Whole-brain median tSNR. <20 is poor for a standard EPI acquisition;
    # <10 is unusable. In-scanner EEG caps lower tSNR somewhat, hence the WARN
    # band being generous.
    "warn_median_tsnr": 30.0,
    "min_median_tsnr": 10.0,

    # Volumes -----------------------------------------------------------------
    # 30 s windows at TR=2 s need 15 volumes; require several windows' worth.
    "min_usable_volumes": 60,
    "warn_usable_volumes": 100,

    # ROI coverage ------------------------------------------------------------
    # Fraction of atlas ROIs with a non-empty, non-constant time series.
    "min_roi_coverage": 0.75,
    "warn_roi_coverage": 0.90,
    # An individual ROI needs this many in-brain voxels to be trusted.
    "min_voxels_per_roi": 10,
}


# =============================================================================
# Pairing rules (a subject is only usable if its modalities line up)
# =============================================================================
PAIRING_RULES = {
    # Paired EEG and hemodynamic recordings must agree on duration within this
    # fraction, otherwise the alignment is not trustworthy.
    "max_duration_mismatch_frac": 0.05,
    # Trial labels must match exactly between the paired modalities.
    "require_label_match": True,
}


@dataclass
class RuleHit:
    """One rule firing on one recording."""
    rule: str
    severity: Severity
    value: float | int | str | None
    threshold: float | int | str | None
    reason: str


def evaluate(checks: list[RuleHit]) -> tuple[Severity, str]:
    """Collapse a list of rule hits into a single decision + documented reason."""
    fails = [c for c in checks if c.severity == "FAIL"]
    warns = [c for c in checks if c.severity == "WARN"]
    if fails:
        return "FAIL", "; ".join(f"{c.rule}={c.value} (limit {c.threshold}): {c.reason}" for c in fails)
    if warns:
        return "WARN", "; ".join(f"{c.rule}={c.value} (limit {c.threshold}): {c.reason}" for c in warns)
    return "PASS", "all pre-registered checks passed"


def band(value, warn_at, fail_at, rule, reason, higher_is_worse=True):
    """Helper: turn a value + WARN/FAIL thresholds into a RuleHit (or None)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return RuleHit(rule, "WARN", "NaN", fail_at, f"{reason} (metric could not be computed)")
    if higher_is_worse:
        if fail_at is not None and v > fail_at:
            return RuleHit(rule, "FAIL", round(v, 4), fail_at, reason)
        if warn_at is not None and v > warn_at:
            return RuleHit(rule, "WARN", round(v, 4), warn_at, reason)
    else:
        if fail_at is not None and v < fail_at:
            return RuleHit(rule, "FAIL", round(v, 4), fail_at, reason)
        if warn_at is not None and v < warn_at:
            return RuleHit(rule, "WARN", round(v, 4), warn_at, reason)
    return None
