# NeuroVascNet full data-quality audit

Replaces the `quick_mode` CSV inspection with a full QC pass over every recording
on disk. Nothing here reads model performance, and every threshold is frozen in
`rules.py` before the first metric was computed.

## Run order

```bash
python qc/build_manifest.py          # 1. enumerate everything on disk
python qc/run_qc.py                  # 2. full QC pass (resumable)
python qc/report.py                  # 3. subject decisions + QC_REPORT.md
```

Step 2 is the long one. Useful flags:

```bash
python qc/run_qc.py --limit 2                    # smoke test, ~2 files per group
python qc/run_qc.py --dataset HEFMI              # one dataset at a time
python qc/run_qc.py --modality fmri              # one modality
python qc/run_qc.py --no-figs                    # metrics only, much faster
python qc/run_qc.py --force                      # discard existing rows and redo
```

Re-running without `--force` skips recordings that already have a row, so an
interrupted run picks up where it stopped. A file that cannot be read produces a
row with `qc_decision=ERROR` instead of aborting the pass.

**Suggested first command:** `python qc/run_qc.py --limit 2` — confirms every
reader works against the real files before committing to the full run.

## Expected cost

610 recordings, 63.2 GB. ds002739 dominates: 48 EEG `.mat` files at ~780 MB each
plus 48 BOLD series. Run that dataset last, or on its own.

## Files

| file | role |
|---|---|
| `config.py` | paths, dataset registry, documented-vs-actual constants |
| `rules.py` | **pre-registered thresholds** with per-rule rationale |
| `build_manifest.py` | enumerates every recording → `manifest.csv` |
| `readers.py` | one reader per on-disk format, all returning the same schema |
| `metrics.py` | pure signal-quality maths, no thresholds |
| `qc_eeg.py` | EEG QC + in-scanner gradient/BCG checks + raw-vs-corrected |
| `qc_fnirs.py` | fNIRS QC + the explicit GSR ablation |
| `qc_fmri.py` | motion, tSNR, ROI coverage, design matrix, event-locked ROI |
| `run_qc.py` | orchestrator (resumable) |
| `report.py` | subject-level pass/fail + `QC_REPORT.md` |

## What gets produced

```
qc_outputs/
  manifest.csv
  QC_REPORT.md
  tables/qc_eeg.csv  qc_fnirs.csv  qc_fmri.csv
  tables/qc_raw_vs_corrected.csv
  tables/subject_decisions.csv
  arrays/eeg_psd/            PSD per recording (mean + per channel)
  arrays/fnirs_evoked/       event-locked HbO/HbR mean + CI
  arrays/fmri_roi/           ROI time series + design matrix
  arrays/fmri_roi_masks/     the ROI mask used for each subject
  figures/<dataset>/<modality>/*.png
```

## The rules that decide inclusion

All in `rules.py`, versioned `1.0.0-frozen-2026-07-25`. The load-bearing ones:

- **In-scanner raw EEG above 5 mV robust amplitude is a hard FAIL** and cannot
  enter training. Units after correction are confirmed by measuring the
  corrected trace, not inferred from the array looking smaller.
- Corrected in-scanner EEG must be under 500 µV robust amplitude, with gradient
  harmonic and BCG residual ratios under 10× the local spectral background.
- fNIRS needs ≥50% of the montage live, <50% poorly coupled optodes, and ≥10
  usable trials.
- fMRI needs mean FD ≤0.5 mm, ≤40% censored volumes, median tSNR ≥10, ≥60
  usable volumes, and ≥75% ROI coverage.

Changing a number in `rules.py` after seeing results invalidates the audit. If a
threshold genuinely has to move, add a second named ruleset and report both.

## Known limitations, stated up front

- **True SCI is not computable for HEFMI or TU Berlin.** Both ship
  Beer-Lambert-converted HbO/HbR, and the conversion mixes the two wavelengths.
  The audit substitutes cardiac-band power fraction and labels the column
  accordingly (`coupling_metric`) rather than reporting a fake SCI.
- **ds002739 supplies no head-motion file.** Those recordings get a DVARS-based
  censoring proxy, flagged in `motion_source`, so they still receive a decision
  instead of passing by default.
- **HEFMI fNIRS carries no sampling rate in the file.** The documented 11 Hz is
  substituted and recorded as `fs_source="documented (missing in file)"`.

## Two discrepancies found while building the manifest

Both are structural, both were found before any metric ran, and both need your
decision rather than a code change:

1. **`dataset/TU Berlin EEG + NIRS/` is not the dataset the docs describe.** It
   contains `nback` / `dsr` / `wg` (EEG) and `nback` / `gonogo` / `vf` (NIRS) —
   the Shin et al. **mental workload** dataset, with **26 subjects**.
   `study_reference_facts.csv` and `NeuroVascNet_Findings_Summary.md` describe
   the external fNIRS validation set as **29 subjects, motor imagery + mental
   arithmetic** — which is the *separate* `dataset/TU Berlin hBCI EEG+FNRI/`
   directory, currently unused by `neurovascnet_loaders_v3.py`.
   If the cached `tuberlin_ext_*.npz` was built from the mental-workload
   directory, the external fNIRS validation is a **cross-task** test on a
   different dataset than documented.

2. **3 of 10 ds002336 subjects have no MIpre/MIpost recordings** (sub-xp101,
   sub-xp104, sub-xp109). Any analysis relying on the real Rest/Task-MI labels
   is limited to the remaining 7 before QC exclusions are applied at all.
