"""Roll per-recording QC rows up to per-subject decisions and write the report.

Acceptance criterion for the audit: every included subject has a saved QC row,
an explicit pass/fail decision, and a documented reason for every exclusion.
This module produces exactly that, plus the dataset-level summary.

Subject-level rule: a subject is EXCLUDED if it has no usable recording in a
modality the dataset is supposed to supply. Individual recordings can fail
without condemning the subject, as long as something usable remains -- the
report states which recordings survived.

Usage:  python qc/report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import DATASETS, MANIFEST_CSV, QC_ROOT, TABLE_DIR
from qc.rules import RULESET_VERSION

TABLES = {"eeg": TABLE_DIR / "qc_eeg.csv",
          "fnirs": TABLE_DIR / "qc_fnirs.csv",
          "fmri": TABLE_DIR / "qc_fmri.csv"}
SUBJECT_CSV = TABLE_DIR / "subject_decisions.csv"
REPORT_MD = QC_ROOT / "QC_REPORT.md"


def load_all() -> pd.DataFrame:
    frames, broken = [], []
    for mod, p in TABLES.items():
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except pd.errors.ParserError as exc:
            broken.append(f"  {p.name}: {exc}")
            continue
        df["modality"] = mod
        frames.append(df)
    if broken:
        raise SystemExit(
            "These QC tables have inconsistent field counts per line:\n"
            + "\n".join(broken)
            + "\n\nThey were written by the older schema-unstable appender.\n"
              "Salvage them (keeps every parseable row, backs up the original):\n"
              "    python qc/repair_tables.py\n"
              "then recompute only the dropped rows:\n"
              "    python qc/run_qc.py\n")
    if not frames:
        raise SystemExit("No QC tables found. Run qc/run_qc.py first.")
    return pd.concat(frames, ignore_index=True, sort=False)


# =========================================================================
def subject_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, subject) with a decision and a documented reason."""
    rows = []
    for (ds, subj), g in df.groupby(["dataset", "subject"], sort=True):
        rec = {"dataset": ds, "subject": subj,
               "n_recordings": len(g),
               "n_pass": int((g.qc_decision == "PASS").sum()),
               "n_warn": int((g.qc_decision == "WARN").sum()),
               "n_fail": int((g.qc_decision == "FAIL").sum()),
               "n_error": int((g.qc_decision == "ERROR").sum())}

        required = DATASETS[ds]["modalities"]
        missing, weak, detail = [], [], []
        for mod in required:
            gm = g[g.modality == mod]
            rec[f"n_{mod}_recordings"] = len(gm)
            if len(gm) == 0:
                missing.append(f"no {mod} recording present")
                rec[f"{mod}_usable"] = 0
                continue
            # for datasets with a corrected variant, only corrected data is
            # eligible for training -- raw in-scanner EEG is diagnostic only
            elig = gm
            if mod == "eeg" and DATASETS[ds]["simultaneous_fmri"]:
                corrected = gm[gm.data_role == "corrected"]
                if len(corrected):
                    elig = corrected
            usable = elig[elig.qc_decision.isin(["PASS", "WARN"])]
            rec[f"{mod}_usable"] = len(usable)
            if len(usable) == 0:
                missing.append(f"all {mod} recordings failed QC")
            elif len(usable) < len(elig):
                weak.append(f"{len(elig)-len(usable)}/{len(elig)} {mod} "
                            f"recordings excluded")
            bad = elig[elig.qc_decision.isin(["FAIL", "ERROR"])]
            for b in bad.itertuples():
                detail.append(f"[{mod}/{getattr(b,'task','?')}/"
                              f"{getattr(b,'data_role','?')}] "
                              f"{str(getattr(b,'qc_reason',''))[:200]}")

        if missing:
            rec["decision"] = "EXCLUDE"
            rec["reason"] = "; ".join(missing)
        elif weak:
            rec["decision"] = "INCLUDE_PARTIAL"
            rec["reason"] = "; ".join(weak)
        else:
            rec["decision"] = "INCLUDE"
            rec["reason"] = "all required modalities have usable recordings"
        rec["failed_recording_reasons"] = " || ".join(detail)[:2000]
        rows.append(rec)
    return pd.DataFrame(rows)


# =========================================================================
def _md_table(df: pd.DataFrame, cols=None, max_rows=None) -> str:
    d = df[cols] if cols else df
    if max_rows:
        d = d.head(max_rows)
    if d.empty:
        return "_(none)_\n"
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join(
        ("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, (int, float, np.floating))
                                else str(v)[:180]))
        for v in r) + " |" for r in d.itertuples(index=False))
    return f"{head}\n{sep}\n{body}\n"


def write_report(df: pd.DataFrame, subj: pd.DataFrame):
    man = pd.read_csv(MANIFEST_CSV) if MANIFEST_CSV.exists() else pd.DataFrame()
    L = []
    A = L.append

    A("# NeuroVascNet — Full Data-Quality Audit\n")
    A(f"**Ruleset:** `{RULESET_VERSION}` (frozen before any metric was computed)  ")
    A(f"**Recordings in manifest:** {len(man)}  ")
    A(f"**Recordings with a QC row:** {len(df)}  ")
    A(f"**Subjects assessed:** {len(subj)}\n")
    A("Every threshold applied here is declared in `qc/rules.py` with its rationale. "
      "No threshold was chosen after seeing a result, and no model performance was "
      "consulted at any point in this audit.\n")

    # ---- coverage check -------------------------------------------------
    A("## 1. Manifest coverage\n")
    if len(man):
        cov = (man.groupby(["dataset", "modality", "data_role"]).size()
               .rename("in_manifest").reset_index())
        got = (df.groupby(["dataset", "modality", "data_role"]).size()
               .rename("qc_rows").reset_index())
        m = cov.merge(got, on=["dataset", "modality", "data_role"], how="left")
        m["qc_rows"] = m["qc_rows"].fillna(0).astype(int)
        m["complete"] = np.where(m.qc_rows >= m.in_manifest, "yes", "NO")
        A(_md_table(m))
        gaps = m[m.complete == "NO"]
        if len(gaps):
            A(f"\n> **{int((gaps.in_manifest - gaps.qc_rows).sum())} manifest "
              f"recordings have no QC row.** The audit is not complete until this "
              f"table shows `yes` on every line.\n")

    # ---- decisions ------------------------------------------------------
    A("\n## 2. Recording-level decisions\n")
    A(_md_table(df.groupby(["dataset", "modality", "qc_decision"]).size()
                .rename("n").reset_index()))

    A("\n## 3. Subject-level decisions\n")
    A(_md_table(subj.groupby(["dataset", "decision"]).size().rename("n").reset_index()))
    A("\n### Included subjects\n")
    inc = subj[subj.decision.str.startswith("INCLUDE")]
    A(_md_table(inc, ["dataset", "subject", "decision", "n_recordings",
                      "n_pass", "n_warn", "n_fail", "reason"]))
    A("\n### Excluded subjects, with documented reason\n")
    exc = subj[subj.decision == "EXCLUDE"]
    A(_md_table(exc, ["dataset", "subject", "reason", "failed_recording_reasons"]))

    # ---- every failed recording ----------------------------------------
    A("\n## 4. Every failed / errored recording\n")
    bad = df[df.qc_decision.isin(["FAIL", "ERROR"])]
    cols = [c for c in ["dataset", "subject", "session", "task", "modality",
                        "data_role", "qc_decision", "qc_reason"] if c in bad.columns]
    A(_md_table(bad[cols]))

    # ---- modality detail ------------------------------------------------
    A("\n## 5. Modality summaries\n")
    for mod, keys in (
        ("eeg", ["fs_hz", "n_channels", "duration_s", "robust_amplitude_uv",
                 "nan_fraction", "clipped_fraction", "line_noise_ratio",
                 "bad_channel_fraction", "transients_per_min_per_ch",
                 "gradient_harmonic_ratio_max", "bcg_residual_ratio"]),
        ("fnirs", ["fs_hz", "n_channels", "channel_availability",
                   "poor_coupling_fraction", "saturated_fraction",
                   "flatline_fraction", "motion_per_min_per_ch",
                   "max_abs_conc_umol", "baseline_sd_umol", "n_usable_trials",
                   "gsr_variance_removed_frac", "gsr_evoked_amplitude_retained_frac"]),
        ("fmri", ["n_volumes", "tr_s", "median_tsnr", "mean_fd_mm",
                  "censored_fraction", "n_usable_volumes", "roi_coverage",
                  "n_events", "motor_roi_evoked_peak"]),
    ):
        g = df[df.modality == mod]
        if g.empty:
            continue
        A(f"\n### {mod.upper()}\n")
        present = [k for k in keys if k in g.columns]
        s = g.groupby("dataset")[present].agg(["median", "min", "max"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        A(_md_table(s.reset_index(), max_rows=20))

    # ---- raw vs corrected ----------------------------------------------
    rvc = TABLE_DIR / "qc_raw_vs_corrected.csv"
    if rvc.exists():
        A("\n## 6. Simultaneous EEG-fMRI: raw vs corrected\n")
        r = pd.read_csv(rvc)
        A(_md_table(r[["subject", "task", "raw_robust_amplitude_uv",
                       "corrected_robust_amplitude_uv", "power_reduction_db",
                       "raw_gradient_harmonic_ratio_max",
                       "corrected_gradient_harmonic_ratio_max",
                       "correction_effective"]]))
        n_over = int((r.raw_robust_amplitude_uv > 5000).sum())
        A(f"\n**{n_over}/{len(r)} raw in-scanner recordings exceed the 5 mV limit** "
          f"and are barred from training by the pre-registered rule. Units after "
          f"correction are confirmed by the `corrected_robust_amplitude_uv` column "
          f"rather than assumed from array magnitude.\n")

    # ---- GSR ablation ---------------------------------------------------
    g = df[(df.modality == "fnirs") & df.get("gsr_variance_removed_frac").notna()] \
        if "gsr_variance_removed_frac" in df.columns else pd.DataFrame()
    if len(g):
        A("\n## 7. Global-signal-regression ablation\n")
        A("GSR is reported as an explicit arm, never applied silently, because it can "
          "remove genuine broad motor responses.\n")
        cols = [c for c in ["gsr_variance_removed_frac",
                            "gsr_evoked_amplitude_retained_frac"] if c in g.columns]
        A(_md_table(g.groupby("dataset")[cols].agg(["median", "min", "max"])
                    .pipe(lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1))
                    .reset_index()))

    A("\n## 8. Artifacts saved\n")
    A("- `qc_outputs/manifest.csv` — every recording on disk\n"
      "- `qc_outputs/tables/qc_{eeg,fnirs,fmri}.csv` — one row per recording, all metrics\n"
      "- `qc_outputs/tables/qc_raw_vs_corrected.csv` — in-scanner correction comparison\n"
      "- `qc_outputs/tables/subject_decisions.csv` — pass/fail + reason per subject\n"
      "- `qc_outputs/arrays/eeg_psd/` — PSD per recording (mean + per-channel)\n"
      "- `qc_outputs/arrays/fnirs_evoked/` — event-locked HbO/HbR mean + CI\n"
      "- `qc_outputs/arrays/fmri_roi/` — ROI time series + design matrix\n"
      "- `qc_outputs/arrays/fmri_roi_masks/` — the ROI mask used for each subject\n"
      "- `qc_outputs/figures/<dataset>/<modality>/` — per-recording diagnostic figures\n")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT_MD}")


# =========================================================================
if __name__ == "__main__":
    df = load_all()
    if "qc_decision" not in df.columns:
        raise SystemExit("QC tables have no qc_decision column.")
    subj = subject_decisions(df)
    subj.to_csv(SUBJECT_CSV, index=False)
    print(f"wrote {SUBJECT_CSV}  ({len(subj)} subjects)")
    write_report(df, subj)

    print("\nsubject decisions:")
    print(subj.groupby(["dataset", "decision"]).size().to_string())
    print("\nrecording decisions:")
    print(df.groupby(["dataset", "modality", "qc_decision"]).size().to_string())
