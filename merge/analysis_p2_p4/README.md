# Analysis scripts for FINAL_PLAN P2–P4

These are copies of the scripts that produced the results in `merge/FINAL_PLAN.md` and
`merge/RESULTS_SUMMARY.md`. The raw result files are in `results/`. The scripts read the notebook caches in
`merge/neurovascnet_merge_v1_cache/`, and several of them execute cells of `Neurovascnet_marge_v1.ipynb` to
reuse its loaders and statistics. A few scripts expect sibling scripts to sit in the same folder, because
they `exec` part of another script (for example `trf_delay.py` reads `continuous_lag.py`).

| Plan section | Script | Notes |
|---|---|---|
| Step 7 reproduction, T1/T2 bands | `run_step7.py`, `run_bands.py` | |
| T5 person-specificity | `person_specific.py` | argument `trials` \| `runs` \| `runs_detrend` |
| T3/T4 topography, EOG | `topo_eog.py` | |
| T6 epoch lag | `lag_test.py` | |
| R replication (mental workload) | `replicate_mw.py` | builds `replication_tuberlin_mw` |
| P2.1 full notebook, slow band kept | `run_p2_full.py`, `dump_outputs.py` | |
| P2.3 continuous lag (+ sanity) | `continuous_lag.py`, `lag_sanity.py` | |
| P2.5 / P2.5b calibration | `calibration.py` | env `TARGET=raw\|local\|global`; cohort names as arguments |
| P2.6 evoked timing | `evoked_timing.py` | |
| P3.1 / P3.1b / P3.1c Claim 1 | `claim1_linear.py`, `claim1_shared.py`, `claim1_within.py` | |
| P4.1 Claim 1 with 27 fMRI people | `build_ds002338.py`, `claim1_shared27.py`, `claim1_within27.py` | env `FMRI_SET=pooled\|ds002336\|ds002338`, `RANK_OVERRIDE` (exploratory) |
| P3.2 Claim 2 | `external_claim2.py`, `external_claim2_class.py` | |
| P3.3 / P4.3 Claim 3 | `trf_delay.py` | `shard K N` then `merge`; env `COHORT=finemi`, `EEG_HP=0.01`, `REREF=lm` |
| P3.4 / P4.4 Claim 4 | `downstream_claim4.py`, `downstream_finemi.py` | |
| P3.5 Claim 5 | `reliability_claim5.py` | |
| P4 FineMI | `build_finemi.py`, `replicate_finemi.py`, `external_finemi.py` | env `REREF=lm`, `DETREND=1` |
| P5.1 / P5.1b / P5.5 Claim 1, rank 32, fresh fNIRS cohorts | `claim1_fresh.py` | env `FNIRS_COHORT=mental_workload\|finemi\|hbci`, `SCARCE=5`, `SEED` |
| P5.2 / P5.2c FineMI hardware emulation | `build_finemi.py` (env `HWHP=1`), `external_finemi.py`, `trf_delay.py` | env `FINEMI_TAG=_hw`; `EA=1` = Euclidean alignment (P5.2b) |
| P5.3 / P5.3b Claim 4 data-scarce and representation forms | `downstream_scarce.py`, `downstream_repr.py` | env `N_LAB` (0 = all labelled people) |
| P5.4 / P5.5 Claim 5 person-level calibrated form | `calib_reliability.py` | env `TARGET=local`, `GAIN=1`, `SEED` |
| P5.6 Claim 2, fMRI output across studies | `external_fmri.py` | |
| P5.7 Claim 5 trial-level calibrated form | `trial_reliability.py` | |
| P6.1 Claim 2 harmonised window + montage | `harmonised_transfer.py` | env `FINEMI_TAG=_hw` (hardware-emulated cache) |
| P6.2 Claim 2 calibrated-transfer stacking | `calibrated_transfer.py` | env `FINEMI_TAG` |
| P6.3 Claim 1 on the local target | `claim1_fresh.py` | env `TARGET=local` (plus the P5.1 env vars) |
| P6.4 Claim 3 classical band-power coupling | `nvc_bandpower.py` | `shard K N` then `merge`; env `NSURR`, `SMOKE`, `COHORTS`; needs `continuous_lag.py` |
| P6.4b coupling beyond the task | `nvc_residual.py` | `shard K 6` then `merge`; needs `nvc_bandpower.py` |
| P7.1 Claim 3 kernel reproducibility | `nvc_kernel.py` | `shard K N` then `merge`; env `RESID=1` (task-residual data), `NK` (surrogates, default 20) |
| P7.2 Claim 2 band-power coupling transfer | `nvc_transfer.py` | single run; needs `nvc_bandpower.py`, `continuous_lag.py` |
| P8.1 Claim 3 smoothed personal delay | `nvc_delay_smooth.py` | reads the P7.1 kernel shard files |
| P8.2 / P8.3 / P8.3b Claim 2 CBSI target, calibrated transfer, seeds | `nvc_transfer2.py` | env `CAL_SEEDS=1,2,3,4,5` runs P8.3b only |
| P9.1 Claim 3 recording-length diagnostic | `nvc_power.py` | `durations`, then `shard K N`, then `merge`; needs `nvc_bandpower.py`, `nvc_delay_smooth.py` |
| P9.2 / P9.3 Claim 2 band diagnostic, FineMI LOPO, alpha + beta kernel | `nvc_bands.py` | single run; needs `nvc_transfer2.py` |
| P9.4 Claim 2 lab-difference diagnostic | `nvc_labdiff.py` | single run; needs `nvc_bands.py` |
| P10.1 Claim 2 broadband term | `nvc_broadband.py` | single run; needs `nvc_bands.py` |
| P10.2 Claim 3 coupling-strength diagnostic | `nvc_snr.py` | `calibrate K N`, `pick`, `shard K N`, `merge`; needs `nvc_bandpower.py`, `nvc_delay_smooth.py` |
| P11 Claim 3 template-alignment delay | `nvc_woody.py` | single run; reads the kernel and P10.2 shard files |
| Paper figures | `../paper/make_figures.py` | reads the result JSONs and shard files here |

New scripts write their JSON next to themselves; the copies in `results/` are the reported ones.
