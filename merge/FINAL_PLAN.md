# NeuroVascNet — final analysis plan (pre-declared)

Written 2026-11-04, before any of the runs below were executed. Every rule here is fixed;
results are reported whatever they show. Nothing in this file is changed after a result is seen;
additions get a new dated section.

## Status at the time of writing

| Run | hBCI cross-subject r_subject (subject x task demeaned) |
|---|---|
| Merged v1, EEG 0.5–45 Hz band-pass + 50 Hz notch (blueprint preprocessing), seed 1337 | +0.019 [+0.008, +0.030] |
| Merged v1, EEG unfiltered (README Step 7), seed 1337 | +0.158 [+0.122, +0.191] |
| v6 FINAL (unfiltered) | +0.158 |

The merge reproduces v6 exactly once the filter is off. The cross-subject signal therefore lives in
EEG content below 0.5 Hz or above 45 Hz. The tests below locate it and ask whether it is neural or
systemic. hBCI fNIRS targets are band-passed 0.01–0.2 Hz.

## Fixed settings for all tests

- Merged v1 notebook, `ENCODER = "trunk"`, `EEG_INPUT = "raw"`, joint training on hBCI + ds002336, seed 1337 folds.
- The band under test is applied to the EEG of both training datasets, then per-trial z-scored
  (same procedure as the notebook's `filter_eeg_inplace`, 4th-order Butterworth, zero phase).
- Metric: `r_subject` on hBCI from `subject_level_test`, subject x task demeaned; practical size 0.05.

## T1 — where does the signal live? (band split)

Bands: unfiltered; < 0.5 Hz; < 0.1 Hz; 0.1–0.5 Hz; > 45 Hz (with 50 Hz notch). The 0.5–45 Hz value is the run above.

Rule: the signal is **located** in a band when that band alone gives r_subject >= 0.10 with the CI above 0.
If no single band does, the signal needs a combination of bands, and that is reported as such.

## T2 — systemic or local? (global vs local target)

For each chromophore, global = mean over the 36 channels; local = each channel minus that mean.
The out-of-fold predictions of each T1 run are scored separately against both parts (prediction split the same way).

Rules:
- **systemic-dominated**: r_global has its CI above 0 and r_local < 0.05.
- **local component present**: r_local >= 0.05 with its CI above 0.
- Anything else: inconclusive, reported as such.

A local component is necessary but not sufficient for a neural reading: motion and scalp blood flow
can also be spatially non-uniform. T3 and T4 address that.

## T3 — ocular/frontal contribution (planned next)

Retrain the best T1 configuration with the frontal-pole channels (Fp*, AF*) removed.
Rule: frontal-driven if r_subject drops by more than half.

## T4 — topography (planned next)

Channel-group occlusion on the T1 models (frontal, central/motor, parietal, temporal, occipital).
Reported descriptively; motor-dominant topography supports, frontal/edge-dominant topography argues against, a neural source.

## T5 — is the coupling person-specific? (added 2026-11-04, before running)

hBCI only, CPU ridge regression (not the deep model), three EEG feature sets fixed in advance:
slow EEG (< 0.5 Hz, 1 s bins), band power (delta/theta/alpha/beta/low-gamma, 3 s bins), high band (> 45 Hz power, 3 s bins).
Features are z-scored per person (EEG only, no target information). Target = HbO/HbR, 1 s bins.

For each person s: 5-fold split of s's own trials; a ridge model (alpha by RidgeCV on s's training trials)
predicts (a) s's held-out trials — the diagonal — and (b) every trial of every other person — the off-diagonal.
Score = the notebook's per-trial correlation after subject x task demeaning.

Rules:
- **person-specific coupling**: mean(diagonal − that person's mean off-diagonal) > 0.03 with the subject-bootstrap CI above 0.
- **identification**: fraction of people whose own model scores best on their trials, against chance 1/29.
- Reported per feature set; no feature set is dropped after the fact.

### T5 amendment (2026-11-04, after the first T5 run)

The first T5 run split each person's trials into 5 shuffled folds. hBCI trials are ~25–27 s apart and
the analysis window is 30 s, so neighbouring trials overlap in time and share slow drifts: a test trial's
neighbour sat in the training set. That inflates the diagonal (and identification); the off-diagonal is
unaffected. T5 is re-run with **leave-one-run-out** within each person (6 runs: 3 MI + 3 MA), with the
same rules. Both runs are reported; the leave-one-run-out run is the one the rules are applied to.

### T5b — linear-drift control (added 2026-11-04, before running)

Leave-one-run-out T5 again, with a linear trend removed within every trial from the filtered EEG
(before binning) and from every fNIRS channel. Rule: if the slow-band diagonal falls below half its
T5 value, the slow coupling is mostly shared linear drift (non-neural) and is reported that way.

### T3/T4 in ridge form (added 2026-11-04, before running; replaces the deep-model T3/T4 above)

Slow-band features (< 0.5 Hz, 1 s bins) from one channel group at a time. hBCI is re-loaded with the
notebook's own loader but with VEOG/HEOG kept; trials must match the cache exactly.
Groups: frontal (AFp1/2, AFF1h/2h/5h/6h, F3/4/7/8), central/motor (Cz, FCC3h/4h/5h/6h, CCP3h/4h/5h/6h),
parietal-occipital (Pz, P3/4/7/8, PPO1h/2h, POO1/2), temporal (T7/8), EOG (VEOG, HEOG), all 30 EEG.
Scored two ways: within-person leave-one-run-out (as T5) and cross-person leave-one-person-out (one pooled ridge).

Rules:
- **ocular/systemic**: EOG alone reaches >= 70% of the all-EEG score (either scoring).
- **frontal-driven**: frontal reaches >= 70% and central/motor < 50% of the all-EEG score.
- **motor-dominant**: central/motor is the highest of the four EEG groups, and EOG < 50%.
- Otherwise: spatially diffuse, reported as such.

## T6 — does slow EEG lead the fNIRS? (added 2026-11-04, before running)

Ridge on slow EEG (< 0.5 Hz, 1 s bins) predicting a fixed fNIRS segment (window seconds 8–29) from EEG
shifted by L = −8 … +8 s (positive L = EEG earlier than fNIRS). Within-person leave-one-run-out and
across-person leave-one-person-out, as in T3/T4.

Rules:
- **neural-compatible timing**: the peak is at L = +2 … +8 s and beats L = 0 by >= 0.02 (subject-bootstrap CI above 0).
- **simultaneous (systemic-compatible)**: the peak is at |L| <= 1 s.
- Otherwise: reported as is.

## R — independent replication on TU Berlin mental workload (added 2026-11-04, before loading it)

HEFMI cannot replicate: its raw EEG was recorded with a 0.5–30 Hz hardware filter (no infra-slow content);
ds002739 EEG has none either. The TU Berlin mental-workload set (`dataset/TU Berlin EEG + NIRS`, 26 people,
not the hBCI people; n-back, word generation/verbal fluency, DSR/go-no-go; BrainAmp time constant 10 s,
so content down to ~0.016 Hz) is the blueprint's own external fNIRS set and is used as the replication.

Loader: same window, resampling, EEG z-scoring, 0.01–0.2 Hz HbO/HbR band-pass and baseline as `load_hbci`;
EEG and NIRS cut by their own markers (label sequences must match exactly); runs recovered from the
EEG-NIRS clock offset; VEOG/HEOG kept separately if present. Task ids = paradigm.

Tests (ridge, as T3–T5; subject x paradigm demeaning):
- **R1 cross-person** (leave-one-person-out, pooled ridge) on slow < 0.5 Hz, < 0.1 Hz, 0.1–0.5 Hz, band power 1–45 Hz, EOG slow.
  Replicates if slow < 0.5 Hz has CI above 0 AND beats band power (paired over people, CI above 0).
- **R2 person-specificity** (leave-one-run-out within person): same rules as T5.
- **R3 global/local** of the R1 slow predictions: same rules as T2.

## P2 — infra-slow protocol (declared 2026-11-04, before running)

Motivation: T1/R show the cross-person EEG->fNIRS information is in < 0.1–0.5 Hz EEG, which the blueprint
preprocessing (0.5–45 Hz band-pass) removes. P2 re-tests the blueprint claims with that band kept.
The original-protocol (filtered) results stay in the paper; P2 is reported as a second, motivated protocol.

- P2.1: the merged notebook exactly as is, except `EEG_FILTER = False`; every verdict rule in the notebook
  unchanged. Seed 1337 first, then the five-seed protocol.
- P2.2: Claim 2 external fNIRS = TU Berlin mental workload (the blueprint's own external set), because HEFMI's
  hardware 0.5 Hz high-pass removes the band. HEFMI is still reported, with that limitation.
- P2.3: Claim 3 lag on continuous recordings (hBCI + mental workload).
- P2.4: Claim 4 with a two-timescale encoder (slow stream + band-power stream) vs the same baselines.
- P2.5: personalised deployment: per-person calibration from k fNIRS runs, reported against the no-EEG
  personal mean (the earlier personalisation lesson).
Each of P2.2–P2.5 gets its exact rule written here before it is run.

### P2.3 rule (written 2026-11-04, before running)

Continuous recordings, every run of every person (hBCI 29, mental workload 26), EEG and NIRS aligned by the
run's marker offset. EEG: < 0.1 Hz, first principal component over channels (per run). NIRS: HbO channel mean,
0.01–0.1 Hz. Both at 1 Hz. Cross-correlation at lags −20 … +20 s (positive = EEG earlier), |r| averaged over a
person's runs; the person's lag = argmax |r|. Null: the same peak |r| after circularly shifting the NIRS by
≥ 60 s (200 surrogates per person).

- **coupled**: a majority of people beat their own null at p < 0.05.
- **neural-compatible timing**: median person lag in +2 … +10 s and Wilcoxon p < 0.05 against 0.
- **simultaneous (systemic-compatible)**: median |lag| <= 1 s.
- Otherwise reported as is. Each cohort is judged separately.

### P2.5 rule (written 2026-11-05, before running)

Deployment question: a new person gives k runs of simultaneous EEG + fNIRS for calibration, then EEG only.
Does EEG predict their hemodynamics better than simply replaying their own average response?
Cohorts: hBCI (29 people, 6 runs) and mental workload (26 people, 3 n-back runs), judged separately.
Features: slow EEG < 0.5 Hz, 1 s bins, all EEG channels (no EOG), z-scored with the calibration trials' stats.
Target: all 72 hemo channels, 1 s bins. Every run is held out once; calibration = every subset of k of the
remaining runs (at most 10 subsets, seed 1337). Models, all fitted on calibration trials only:
- baseline (no EEG): the person's calibration mean response for the held-out run's task (all calibration
  trials if that task is absent from the calibration runs);
- personal: baseline + RidgeCV(alphas 1e0…1e6) on the calibration trials' residuals from that baseline;
- hybrid (descriptive): baseline + mean of the personal ridge and a population ridge fitted on the other people's
  residuals from their own task means.
Score: per-trial correlation (notebook mean_corr) on the raw target, averaged per person; subject-level mean,
bootstrap 95% CI, sign-flip p. Secondary: the same after removing the held-out run's mean from prediction and
target (trial-specific part).
- **EEG adds deployment value**: personal − baseline at k = 2 ≥ 0.03 with CI > 0 (primary, raw score).
- Calibration curve k = 1 … 5 (hBCI) / 1 … 2 (mental workload) and the k = 0 population model are descriptive.

### P2.6 rule — task-evoked timing (written 2026-11-05, before running)

Resolves the P2.3 note (evoked EEG-first vs continuous HbO-first). Epoch cache (−2 … +28 s around the cue),
hBCI and mental workload. Per person and task: average the trials; slow EEG < 0.1 Hz, 10 Hz, baseline
(−2 … 0 s) removed; evoked strength = global field power (GFP). HbO = mean of the HbO channels, baseline removed.
Peak latency of each inside 0 … 25 s; person delay = HbO peak − GFP peak, averaged over tasks.
Evoked-EEG presence per person: peak GFP above the 95th percentile of 200 averages with random trial sign flips.
- **evoked EEG-first**: median person delay > 0 with Wilcoxon p < 0.05, counted over people with evoked EEG
  present (that count reported).
- The P2.3 continuous result and the P2.6 result are then reported together as two timescales, without
  choosing one.

### P2.5b rule — local target (written 2026-11-05, before running)

Answers the systemic objection to P2.5. Everything as in P2.5 except the target: for each chromophore, the
local part from T2 (each channel minus the 36-channel mean at each time point), 72 channels. Every model,
including the no-EEG baseline, is refitted on the local target. Global part (2 channel means) run the same way,
descriptive only.
- **local deployment value**: personal − baseline at k = 2 ≥ 0.03 with CI > 0, per cohort (primary).
- If the local gain fails while the raw gain passed, P2.5 is reported as systemic-driven.

## P3 — best route to each blueprint claim (declared 2026-11-05, before running)

User instruction: improve results toward the five blueprint claims. Each P3 test names the claim it serves.
Model family fixed from hBCI work: ridge on slow EEG (< 0.5 Hz, 1 s bins), features z-scored per person (EEG only).

### P3.2 — Claim 2, external validation (replaces P2.2)

Train once on all 29 hBCI people; freeze; test on the 26 mental-workload people (never used for any choice).
EEG: mental-workload channels mapped onto hBCI positions with the notebook's `interpolation_matrix`
(spherical splines, 28 -> 30). Target: pooled mean HbO and mean HbR (2 × 30 s at 1 Hz), because the NIRS
montages differ — the notebook's own Claim 2 comparison for HEFMI. Score: notebook r_subject (per-trial
correlation after subject × task demeaning, subject-level bootstrap CI, sign-flip p).
- **Claim 2 passes on this set**: the notebook rule — CI > 0, p < 0.05 and r_subject >= 0.05 (practical_r).
- Reverse direction (train mental workload, test hBCI) reported, same rule, secondary.
- Descriptive: the same with k = 2 personal calibration runs on top of the frozen model (P2.5 hybrid).

### P3.3 — Claim 3, delay from a multichannel lagged model (continuous recordings)

Runs and alignment as P2.3. EEG < 0.1 Hz, all EEG channels (no EOG), 1 Hz; HbO 0.01–0.1 Hz, 1 Hz.
Primary target: local HbO (each channel minus the channel mean); global HbO descriptive.
For each lag L = −10 … +20 s (positive = EEG earlier): ridge from EEG(t − L) to all target channels,
leave-one-run-out; score r(L) = mean over channels of the held-out correlation. Person delay = argmax r(L).
- exists: a majority of people beat their own null (peak r over lags after circularly shifting the NIRS by
  ≥ 60 s, 100 surrogates, p < 0.05);
- reproducible: delay re-estimated on the first and on the second temporal half of every run; across people
  Pearson r between halves > 0 with permutation p < 0.05;
- **Claim 3a passes**: exists AND reproducible (per cohort; hBCI primary).
- **physiological (3b-type)**: median delay in +2 … +10 s with Wilcoxon p < 0.05.

### P3.5 — Claim 5, calibrated reliability on the ridge pipeline (written 2026-11-05, before running)

The notebook's Claim 5 machinery unchanged (`trial_error`, `fit_calibrator`, `apply_calibrator`,
`reliability_report`, coverage 0.75), applied to the P3.2 model (slow-EEG ridge, pooled HbO/HbR target).
Units harmonised by z-scoring each cohort's target with that cohort's overall SD (the audit's unit mismatch).
Reliability features per trial (3, like the notebook): (1) spread of 10 ridge models trained on bootstrap
resamples of the training people, relative to the mean |prediction|; (2) mean squared z of the trial's EEG
features against the training distribution (out-of-distribution); (3) log slow-EEG energy.
Internal: hBCI, 5 person folds; features and errors out-of-fold; the calibrator is cross-fitted (fitted on four
folds' out-of-fold rows, applied to the fifth). External: models and calibrator fitted on all hBCI, applied to
mental workload.
- **Claim 5 HOLDS**: the notebook rule — (internal rank_trial > 0.2 or rank_subject > 0.2) AND internal
  ece_iso < ece_raw AND external rank_trial > 0.2. PARTIAL if internal only.

### P3.4 — Claim 4, downstream EEG-only task with "virtual hemodynamics" (written 2026-11-05, before running)

Notebook protocol: cross-person, 3 person folds (seed 1337), hBCI motor imagery left vs right (primary);
hBCI mental arithmetic vs rest and mental-workload 0/2/3-back (secondary). Classifier: standardise +
logistic regression (C = 1), identical for every feature set. Features, all from EEG only:
- A band power: log power 8–13 and 13–30 Hz per channel, 0–10 s after the cue (the field baseline);
- B = A + virtual hemodynamics: a slow-EEG ridge trained on the training-fold people only (all their trials,
  target local HbO/HbR, subject × task demeaned) predicts the 72 local channels; each channel's mean over
  5–15 s after the cue;
- S = A + the same pipeline with the ridge trained on targets shuffled across trials within each person
  (shuffled-target control);
- R = A + a random projection of the same slow-EEG features to 72 dimensions (random-weights control).
Score: per-person accuracy on the held-out folds; subject-level bootstrap CI of the differences.
- **Claim 4 passes (4a type)**: B − A ≥ 0.02 with CI > 0 AND B − S > 0 with CI > 0 AND B − R > 0 with CI > 0,
  on the primary task. Secondary tasks reported with the same numbers.

### P3.1 — Claim 1, cross-modal supervision in the linear shared-latent form (written 2026-11-05, before running)

EEG: slow features (< 0.5 Hz, 1 s bins), ds002336 EEG mapped onto hBCI positions (spherical splines 63 -> 30),
z-scored per person. Targets demeaned by subject × task (notebook Claim 1 scoring): hBCI 72 channels × 30 s;
ds002336 5 ROIs × 15 samples. Shared latent of fixed rank 8: per dataset a RidgeCV map B_d (features -> targets);
single = top-8 left singular vectors of B_fnirs / ||B_fnirs||; joint = top-8 of [B_fnirs / ||B_fnirs||,
B_fmri / ||B_fmri||]; joint-shuffled = joint with the fMRI targets shuffled across trials within person.
The fNIRS head is then refitted (RidgeCV) on the 8 latent features. Evaluation: hBCI, 5 person folds (seed 1337),
all ds002336 people in every fold; notebook r_subject on held-out people.
- **Claim 1 confirmed (primary, all training people)**: joint − single > 0 with subject-level CI > 0 AND
  joint − joint-shuffled > 0 with CI > 0.
- **Data-scarce benefit (secondary, declared now)**: the same two conditions when each fold trains on only
  5 hBCI people (fixed random choice, seed 1337).

### P3.1b — Claim 1 as "one shared space serves both outputs" (written 2026-11-05, after P3.1, before running)

User framing: a single EEG input -> one common latent space -> an fNIRS head and an fMRI head. Question: does
the one shared space give both outputs on unseen people, at no cost versus modality-specific spaces?
(P3.1's "fMRI improves fNIRS" question stays answered: no.)
Same features/targets as P3.1. Rank r chosen on fNIRS-only results alone: the smallest r in {4, 8, 16, 32, 64}
whose single-modality latent reaches >= 95% of the full fNIRS ridge r_subject (hBCI 5 person folds).
fNIRS side: hBCI 5 person folds, joint space built with all ds002336 people. fMRI side: ds002336 leave-one-
person-out (10), joint space built with all hBCI people. Comparators: modality-specific latent of the same
rank, and the full ridge of that modality.
- **both outputs work**: from the joint space, fNIRS r_subject detectable with r >= 0.05, AND fMRI r_subject
  CI > 0 with p_signflip < 0.05;
- **no cost from sharing**: joint − modality-specific (same rank) 90% CI lower bound > −0.02, for each modality;
- **Claim 1 (shared-space form) holds**: both conditions.

### P3.1c — Claim 1, fMRI arm within person and fNIRS -> fMRI transfer (written 2026-11-06, before running)

ds002739 cannot serve: its published EEG has no infra-slow power (slow/alpha power 0.04 vs 36.7 in ds002336).
ds002336 (10 people, EEG and fMRI simultaneous in every run, 4–6 runs, ~38 trials each). Slow-EEG features as
P3.1 (mapped onto hBCI positions, z-scored per person). Target BOLD 5 ROIs × 15 samples. Every model personal,
leave-one-run-out within person; predictions then demeaned by subject × task and scored with the notebook's
subject_level_test (pairing null within person).
Personal heads (RidgeCV) on: raw = the 900 features; fnirs_latent = projection onto the rank-16 space of the
ridge map fitted on all 29 hBCI people (fNIRS supervision only); fmri_pop_latent = rank-16 space of the ridge
map fitted on the other 9 ds002336 people; random = a random orthonormal rank-16 space (seed 1337).
- **(a) fMRI arm works within person**: raw r_subject CI > 0 and p_signflip < 0.05.
- **(b) Claim 1, cross-modal transfer form**: fnirs_latent − raw > 0 with subject-level CI > 0 AND
  fnirs_latent − random > 0 with CI > 0. fmri_pop_latent reported alongside.

## P4 — third cohort: FineMI multi-joint upper-limb motor imagery (declared 2026-11-07, before loading)

On disk (18 people; Neuroscan EEG 1000 Hz, no hardware high-pass; NIRx NIRScout 24 channels, 7.8 Hz; 8 imagined
joint movements, ~19 s apart; blocks 1–4 in one file, blocks 5–8 separate). Loader = the notebook helpers as for
mental workload (−2 … +28 s, EEG unfiltered 200 Hz z-scored per trial, HbO/HbR 0.01–0.2 Hz, baseline-corrected,
MNE Beer–Lambert with ppf 6). EEG: 60 scalp channels (mastoid/cerebellar/EOG/EKG/EMG excluded); HEO/VEO kept
apart. Trials kept only where EEG and NIRS event-code sequences match; runs = blocks (block 1–4 file split at
inter-trial gaps > 40 s). No FineMI result is used to change any setting below; every rule is copied from
the earlier cohorts.
- **P4.R1 (replication)**: R1 rule on FineMI — slow-EEG cross-person ridge minus band-power, CI > 0.
- **P4.2 (Claim 2)**: P3.2 unchanged — model frozen on hBCI, tested on FineMI (pooled HbO/HbR, spherical
  splines 60 -> hBCI 30). Primary hBCI -> FineMI; mental workload -> FineMI secondary. Notebook rule.
- **P4.3 (Claim 3)**: P3.3 unchanged on FineMI continuous blocks (lagged ridge, local HbO, 100 surrogates,
  split halves). Claim 3a rule; physiological flag reported.
- **P4.4 (Claim 4)**: P3.4 unchanged; task = 8-class joint imagery (chance 0.125), primary for this cohort.
- **P4.5 (calibration)**: P2.5 and P2.5b unchanged (k = 2 runs; raw and local targets).

### P4.1 — Claim 1 retest with 27 fMRI people (declared 2026-11-07, before the download finished)

ds002338 (XP2 of the same study: 17 new people, same EEG/fMRI hardware and preprocessing, tasks 1dNF, 2dNF,
MIpre, MIpost) built with the notebook's own `load_ds002336` (root and task list changed only; derivatives
eeg_pp EEG, motor-ROI BOLD, R128 alignment; EEG unfiltered). fMRI arm = ds002336 + ds002338 pooled (27 people;
task ids kept distinct). P3.1b and P3.1c rerun unchanged on the pooled arm (P3.1b fMRI side: leave-one-person-
out over 27; P3.1b rank rule re-applied — fNIRS-only, so unchanged at 8). Same verdict rules. Also reported
descriptively: P3.1b with ds002338 alone.
Amendment (2026-11-07, after download, before any ds002338 result): ds002338 names its preprocessed EEG
`d_<sub>_task-<task>_run-<NN>_eeg_pp.vhdr` and keeps one events file per task; the loader's two path lines are
changed to that naming (each run becomes its own task entry with its own BOLD run). Nothing else changes.

### P4.A — post-hoc amendment: matching the EEG band to the HbO band (written 2026-11-07, AFTER seeing the
first FineMI Claim 2 / Claim 3 numbers; reported as post hoc, next to the declared results)

Reason: FineMI EEG is DC-coupled (slow/alpha power ~10^4–10^5, measured before any test) while hBCI and mental
workload EEG pass a hardware high-pass (BrainAmp, time constant 10 s ≈ 0.016 Hz) and the HbO target is
band-passed 0.01–0.1/0.2 Hz. Electrode drift below 0.01 Hz therefore dominates FineMI's slow-EEG features in
continuous data and in cross-dataset transfer. Amendment, applied identically to every cohort:
- P3.3/P4.3 lag test: EEG additionally high-passed at 0.01 Hz (so EEG and HbO share the 0.01–0.1 Hz band).
- P3.2/P4.2 transfer: each EEG trial linearly detrended before the slow-EEG features (T5b showed detrending does
  not remove the hBCI signal).
Same verdict rules. Declared results stay the primary record; the amended ones are labelled post hoc.

### P4.B — post-hoc amendment: FineMI reference (written 2026-11-07, after P4.2 and P4.A transfer failed)

P4.A (detrend) left FineMI transfer at −0.02 … −0.06 (hBCI <-> mental workload unaffected: 0.195 / 0.180).
FineMI's M1 channel has the lowest variance (4.9 µV vs scalp median 15.8 µV), i.e. the recording reference sits
at/near the left mastoid, while slow-potential topographies are reference-dependent. Amendment: FineMI EEG and
EOG re-referenced to linked mastoids ((M1 + M2)/2) in the loader, then P4.2 (transfer, with and without the
P4.A detrend) and P4.3 (lag, with P4.A high-pass) rerun. Post hoc; same verdict rules.

## P5 — further routes to the failing claims (declared 2026-11-07, before running any P5 test)

User instruction: keep trying every route to the five claims. Every P5 test below is written before it is run.
P5 tests come after the P3/P4 results were seen, so they are a second confirmatory round: each one either uses
data never used for that claim (fresh cohort) or fixes its settings from earlier, already-reported results.
Exploratory follow-ups are labelled as such when added.

### P5.1 — Claim 1 confirmatory test on fresh fNIRS cohorts (rank fixed at 32)

Rank 32 is fixed from the hBCI exploratory sweep (P4.1). The fNIRS side is a cohort never used for Claim 1:
mental workload (26 people, primary) and FineMI (18, secondary). fMRI arm = ds002336 + ds002338 pooled (27).
Slow EEG as P3.1 (< 0.5 Hz, 1 s bins, z-scored per person). The fMRI arm's EEG is mapped onto the fNIRS
cohort's own montage (spherical splines). Targets: every NIRS channel × 30 s (MW 72 × 30; FineMI 48 × 30) and
BOLD 5 ROIs × 15, demeaned by subject × task. fNIRS side: 5 person folds (seed 1337), joint space built with all
27 fMRI people; fMRI side: leave-one-person-out over 27, joint space built with all fNIRS-cohort people.
- **P3.1b form (shared space serves both outputs)**: both outputs work (fNIRS r_subject CI > 0, p < 0.05,
  r >= 0.05; fMRI CI > 0, p < 0.05) AND no cost (shared − specific 90% CI lower bound > −0.02, each modality).
  **Claim 1 (shared-space form) confirmed** if it holds in mental workload; FineMI reported as a replication.
- **P3.1 form (fMRI supervision improves fNIRS)**, same rank 32: joint − single > 0 with 95% subject-level CI > 0
  AND joint − joint-shuffled (fMRI targets shuffled within person) > 0 with CI > 0. Reported for MW, FineMI,
  and hBCI (hBCI is not fresh at rank 32 — descriptive).
- **P5.1b data-scarce P3.1 form** (added 2026-11-07 after the P5.1 results, before running): same rank 32 and
  cohorts, but each fNIRS fold trains on only 5 people (fixed random choice of the training fold, seed 1337);
  the fMRI map uses all 27. Same rule (joint − single CI > 0 AND joint − joint-shuffled CI > 0), MW primary.

### P5.2 — Claim 2 into FineMI by hardware emulation

The TU Berlin EEG passed a BrainAmp hardware high-pass (first order, time constant 10 s); FineMI is DC-coupled.
P4.A detrending is not the same operation (it removes a line, not the causal filter's phase and shape).
FineMI continuous EEG and EOG are passed through a first-order causal high-pass with time constant 10 s
(`scipy.signal.lfilter`, bilinear) before epoching; the rest of the P4 loader is unchanged (native reference).
P4.2 is rerun unchanged on this cache (hBCI -> FineMI primary; MW -> FineMI, FineMI -> hBCI secondary).
- **Claim 2 into FineMI passes**: the notebook rule on hBCI -> FineMI.
- Secondary, same rule: (a) the same plus linked-mastoid reference; (b) Euclidean alignment — each person's
  slow-EEG features whitened by that person's own spatial covariance (unsupervised; no target used), applied to
  every cohort and reported for hBCI <-> MW as well.

### P5.3 — Claim 4 data-scarce form

Same as P3.4/P4.4 (features A, B, S, R; same classifier; 3 person folds), but the classifier is trained on the
labelled trials of only 5 training people per fold (fixed random choice, seed 1337), while the virtual-
hemodynamics ridge uses the hemodynamics of all training-fold people (hemodynamics need no task labels).
This is the semi-supervised setting the blueprint's downstream claim implies. Primary: hBCI MI left/right.
- **Claim 4 (data-scarce form) passes**: B − A >= 0.02 with CI > 0 AND B − S > 0 with CI > 0 AND B − R > 0 with
  CI > 0. Secondary: hBCI MA, MW n-back, FineMI 8-class, same numbers.

### P5.3b — Claim 4 representation form (written 2026-11-07 after P3.4/P4.4/P5.3 results, before running)

Post-hoc-motivated: the hemo-only accuracies (virtual hemodynamics alone) were visible in the P3.4/P4.4/P5.3
outputs; this form was not declared before them and is labelled as such. Question: is the EEG representation
learned under haemodynamic supervision a better compact code for the downstream task than unsupervised codes of
the same size from the same slow EEG? Same folds, classifier and features as P3.4, but band power is NOT added:
V = virtual hemodynamics alone (one value per NIRS channel, 5–15 s); Rp = random projection of the slow-EEG
features to the same dimension; PC = principal components of the slow-EEG features (fitted on training people)
to the same dimension; Full = all slow-EEG features (reference). Both regimes: all labelled training people
(primary) and 5 labelled people (P5.3 regime).
- **Claim 4 (representation form) passes**: V − Rp > 0 with CI > 0 AND V − PC > 0 with CI > 0 on hBCI MI,
  primary regime. Other tasks/cohorts reported with the same numbers.

### P5.4 — Claim 5 at the person level, in the calibrated-deployment setting

P3.5 passed only at person level internally and failed externally. Deployment (P2.5) gives each new person two
calibration runs, so their reliability can be measured, not guessed: reliability score = the person's own
leave-one-calibration-run-out r of the k = 2 hybrid model (computed on the two calibration runs only).
Error = 1 − mean test-run r of the hybrid model on the remaining runs. Cohorts: hBCI (internal), MW and FineMI
(external — nothing fitted across cohorts; the score is computed the same way).
- **Claim 5 (person-level, calibrated form) holds**: Spearman(score, −error) > 0.2 with 95% bootstrap CI > 0 in
  hBCI AND in at least one external cohort.
- Robustness (added 2026-11-07 after the raw-target P5.4 result, before running it): the same test on the local
  target (P2.5b: each channel minus the channel mean), to check that the ranking is not only the systemic signal.
- EEG-specific check (added 2026-11-07 after the raw and local P5.4 results, before running it): score and outcome
  both taken as the hybrid's gain over the no-EEG personal mean (calibration-run gain vs test-run gain), so the
  ranking must come from the EEG part, not from how stereotyped a person's haemodynamics are. Same rule.

### Claim 3

No new hBCI / mental-workload variant is declared: P3.3/P4.A already used the full data with both band choices;
a third variant chosen now would be fishing. The group-level delay is already reported.
- **P5.2c** (written 2026-11-07, before running): the P4.3 lag test (declared settings, no extra 0.01 Hz high-pass)
  rerun on the P5.2 hardware-emulated FineMI continuous data (native reference; linked mastoids secondary).
  Claim 3a rule unchanged. Reason: same as P5.2 — make FineMI's EEG chain match the cohorts where 3a passed.

## Results so far (2026-11-04, seed 1337, hBCI)

| Test | Result | Rule outcome |
|---|---|---|
| T1 | unfiltered 0.158; < 0.5 Hz 0.155; < 0.1 Hz 0.139; 0.1–0.5 Hz 0.053; 0.5–45 Hz 0.019; > 45 Hz 0.021 | located in < 0.5 Hz, and in < 0.1 Hz alone |
| T2 | HbO global 0.19–0.23, HbO local 0.07–0.08 (CI > 0) in every band carrying signal | local component present (systemic part larger) |
| T5 (leave-one-run-out) | slow: same person 0.360 vs others 0.074, identification 93% (chance 3.4%); band power 0.074 vs 0.014 | person-specific in all three feature sets |
| T5b | slow same-person 0.404 after linear detrend | not linear drift |
| T3/T4 | slow EEG, % of all-EEG score (within / across): parietal-occipital 91/92, central 88/65, frontal 56/61, temporal 49/51, EOG 14/14 | not ocular, not frontal; spatially diffuse |
| Claim 1 TOST | fNIRS joint − single +0.002, 90% CI [−0.008, +0.013], equivalent within ±0.02 (p 0.006) | equivalence shown (filtered EEG) |

T6 (lag): curve nearly flat — within 0.286 (L −8) … 0.315 (L 0) … 0.320 (L +8); across 0.134 … 0.156 … 0.160.
Peak at L = +8 / +7 s but only +0.005 above L = 0 (CIs include 0): **neither rule met**. Infra-slow content
barely changes over ±8 s inside a 30 s window, so epoched data cannot time it; a continuous-recording
analysis is needed.

### Replication (R), TU Berlin mental workload, 26 people — 2026-11-04

Loader audit: n-back usable in all 26 people (27 blocks, 3 runs each); word generation usable only for VP001
(EEG/NIRS label sequences differ in the other 25 — dropped by rule); DSR markers unreadable — dropped.
762 trials (702 n-back + 60 WG from VP001).

| Test | Result | Rule outcome |
|---|---|---|
| R1 cross-person | slow < 0.5 Hz 0.136 [0.089, 0.183]; < 0.1 Hz 0.131; 0.1–0.5 Hz 0.032; band power 1–45 Hz 0.006 [−0.015, 0.026]; EOG slow 0.001 | **replicates**: slow − band power +0.129 [0.080, 0.182] |
| R2 person-specificity | slow: same 0.182 vs other 0.024, diff +0.158 [0.102, 0.212], identification 62% (chance 3.8%); band power: diff +0.012 [−0.013, 0.040] | slow person-specific; band power not |
| R3 global/local | HbO global 0.215, HbO local 0.096 [0.063, 0.130]; HbR global 0.061, local 0.026 | local component present |

### P2.3 continuous lag — 2026-11-04

| Cohort | beat own null | median lag (positive = EEG earlier) | IQR | Wilcoxon vs 0 |
|---|---|---|---|---|
| hBCI (29) | 72% | −6.0 s | [−7, −2] | 0.0008 |
| mental workload (26) | 73% | −2.5 s | [−7.8, +3.5] | 0.18 |

Rule outcome: **coupled** in both cohorts; timing neither neural-compatible nor simultaneous — in hBCI the
HbO leads the infra-slow EEG by ~6 s (same direction, not significant, in mental workload).
Sanity checks (hBCI, descriptive): an injected +5 s NIRS delay moves every person's lag by exactly +5 s
(sign convention correct); flipping the alignment offset lowers peak |r| in 5/5 people (alignment correct).
Marker-locked averages (descriptive, not a declared test): the task-evoked slow-EEG GFP peaks at +3.5 s (MI) /
+4.5 s (MA) after the cue, HbO at +15.5 s / +16.5 s — so the task-evoked part runs EEG-first while the
continuous fluctuations run HbO-first. Not resolved by the declared tests; needs its own declared test.

### P2.1 full notebook, EEG_FILTER = False, seed 1337 — 2026-11-05

Run tag trunk_raw_0d875ccf39; executed copy in the session scratchpad (p2_full_executed.ipynb). Notebook verdict
rules unchanged.

| Claim | Result | Notebook verdict |
|---|---|---|
| 1 (hBCI) | single 0.162, joint 0.158, joint-shuffled 0.154; joint − single −0.004 [−0.017, +0.008], p 0.52 | NOT CONFIRMED |
| 1 (ds002336 fMRI) | single 0.040, joint 0.034; joint − single −0.005 [−0.035, +0.026] | — |
| 2 | HEFMI 0.005 [−0.005, +0.016] (n.s.; hardware 0.5 Hz high-pass, expected); ds002739 0.010 [0.002, 0.018] (detectable, negligible) | FAILS |
| 3a | data-driven lag hBCI 5.98 s, ICC 0.20 (shuffle p 0.03); HEFMI 5.99 s, ICC 0.54 (p 0.01) | FAILS |
| 3b | learned delay sticks to the window edge (2.005 s in [2, 8] s; 0.10 s in [0, 20] s) | FAILS |
| 4a | frozen supervised 0.563 vs band power 0.568, random init 0.600, shuffled-target 0.563 | FAILS |
| 4b | fine-tuned from supervised 0.735 vs from scratch 0.754 | FAILS |
| 5 | rank internal +0.017, external +0.032; ECE 0.070 -> 0.072 after isotonic | FAILS |

Reading: keeping the infra-slow band restores cross-person prediction (0.158), but the signal is carried
equally well by fNIRS-only training — the cross-modal (joint, fMRI) supervision adds nothing (Claim 1 CI lies
inside ±0.02). The deep pipeline's other claims fail under both protocols. The deep model is beaten by a
pooled ridge on the same band (0.197).

### P2.5 personalised calibration — 2026-11-05

Raw-target score (mean per-trial r), subject level, 95% CI. Baseline = the person's own calibration mean response
(no EEG). Personal = baseline + ridge on slow EEG fitted on the same calibration runs.

| Cohort | k | baseline (no EEG) | personal | hybrid | personal − baseline |
|---|---|---|---|---|---|
| hBCI (29) | 0 (population model, no calibration) | — | 0.204 [0.170, 0.238] | — | — |
| hBCI | 1 | 0.075 | 0.193 | 0.235 | +0.118 [0.088, 0.153] |
| hBCI | **2 (primary)** | 0.093 | 0.269 | 0.284 | **+0.176 [0.135, 0.218], p 0.0001, 29/29** |
| hBCI | 5 | 0.118 | 0.376 | 0.365 | +0.258 [0.206, 0.308] |
| mental workload (26) | 0 (population) | — | 0.189 (25 people; VP001 has a task no one else has) | — | — |
| mental workload | 1 | 0.256 | 0.282 | 0.301 | +0.027 [0.004, 0.048] |
| mental workload | **2 (primary)** | 0.293 | 0.352 | 0.362 | **+0.059 [0.025, 0.099], p 0.002, 17/26** |

Trial-specific score (held-out run mean removed; the baseline scores 0 by construction), k = 2: hBCI personal 0.262,
mental workload 0.187.
Rule outcome: **EEG adds deployment value in both cohorts** (gain ≥ 0.03 with CI > 0 at k = 2). The gain keeps
rising with more calibration runs (hBCI k = 1 … 5). Mental-workload k ≥ 3 rows exist only for VP001 (n = 1) and are
not interpreted. Open caveat: the target includes the systemic (global) component; a local-target version of P2.5
is not yet run.

### P2.6 task-evoked timing — 2026-11-05

| Cohort | evoked slow EEG present | median delay (HbO peak − EEG GFP peak) | IQR | Wilcoxon |
|---|---|---|---|---|
| hBCI | 22/29 | +1.8 s | [−2.5, +5.1] | 0.24 |
| mental workload | 17/26 | −2.1 s | [−9.6, +1.8] | 0.24 |

Rule outcome: **evoked EEG-first not shown** in either cohort. The earlier pooled marker-locked observation
(EEG +3.5 s, HbO +15.5 s) does not hold at person level. A task-evoked slow-EEG response exists in most people,
but its timing relative to HbO is not consistent. Timing claim stays limited to P2.3 (continuous coupling, HbO
leading by ~6 s in hBCI only).

### P2.5b local vs global target — 2026-11-05

k = 2 calibration runs, every model refitted on that target. Gain = personal − no-EEG personal mean.

| Cohort | target | baseline | personal | gain [95% CI], people positive |
|---|---|---|---|---|
| hBCI | raw (P2.5) | 0.093 | 0.269 | +0.176 [0.135, 0.218], 29/29 |
| hBCI | **local** | 0.092 | 0.195 | **+0.103 [0.074, 0.136], p 0.0001, 29/29** |
| hBCI | global (descriptive) | 0.094 | 0.339 | +0.246 [0.193, 0.296], 29/29 |
| mental workload | raw (P2.5) | 0.293 | 0.352 | +0.059 [0.025, 0.099], 17/26 |
| mental workload | **local** | 0.302 | 0.333 | **+0.031 [0.013, 0.052], p 0.003, 19/26** |
| mental workload | global (descriptive) | 0.241 | 0.357 | +0.116 [0.058, 0.178], 18/26 |

hBCI local gain grows with calibration: k = 1 +0.069, k = 2 +0.103, k = 5 +0.162. Local population model
(k = 0): hBCI 0.097, mental workload 0.176.
Rule outcome: **local deployment value in both cohorts** — clear in hBCI, at the threshold in mental workload
(+0.031 against 0.03). The systemic (global) part carries the larger share of the gain, but the gain does not
reduce to it.

### P3.2 Claim 2 external validation — 2026-11-05

Slow-EEG ridge frozen on all of one cohort, tested on the other (different people, task, EEG and NIRS montage;
EEG mapped with the notebook's spherical-spline matrix). Target pooled mean HbO + mean HbR.

| Direction | r_subject [95% CI] | null | p_signflip | people positive | HbO / HbR | notebook verdict |
|---|---|---|---|---|---|---|
| **hBCI -> mental workload (primary)** | **+0.205 [0.128, 0.285]** | −0.001 | 0.0002 | 20/26 | 0.239 / 0.065 | **detectable — Claim 2 PASS** |
| mental workload -> hBCI | +0.177 [0.143, 0.211] | +0.000 | 0.0001 | 28/29 | 0.200 / 0.035 | detectable — PASS |
| robustness: subject × task × class demeaning, hBCI -> MW | +0.209 [0.135, 0.281] | | 0.0002 | 21/26 | | PASS |
| robustness: same, MW -> hBCI | +0.189 [0.155, 0.224] | | 0.0001 | 29/29 | | PASS |

For comparison, the deep notebook (P2.1) on HEFMI gave 0.005 and on ds002739 0.010. Caveats: (1) the target is the
pooled (global) signal, which T2/P2.5b showed is systemic-heavy — a local target cannot transfer across different
NIRS montages; (2) both cohorts were recorded by the same TU Berlin group, so some people may appear in both
(not identifiable from the public files); (3) the k = 2 hybrid descriptive with the frozen external model is not
interpretable on the raw score because the two cohorts' hemo units differ (the audit's unit mismatch); the
correlation-based Claim 2 score is unit-free.

### P3.5 Claim 5 reliability, ridge pipeline — 2026-11-05

| | rank_trial | rank_subject | ECE raw -> isotonic | coverage | error falls with abstention |
|---|---|---|---|---|---|
| internal (hBCI, person folds, cross-fitted) | +0.110 | **+0.219** | 0.008 -> 0.021 | 0.75 | no |
| external (mental workload) | −0.014 | −0.268 | 0.100 -> 0.101 | 0.77 | no |

Rule outcome: **PARTIAL (internal only)** — the person-level ranking passes 0.2 (0.219), trial ranking does not,
isotonic does not improve ECE (raw ECE is already 0.008), and nothing transfers externally. Same verdict class
as the deep notebook (which FAILED). No further feature search will be run on this claim without a new declared
rule — picking reliability features after seeing these numbers would be fishing.

### P3.4 Claim 4 downstream — 2026-11-05

Cross-person accuracy (3 person folds). A band power; B + virtual hemodynamics; S + shuffled-target virtual
hemodynamics; R + random projection of the same slow EEG.

| Task | A | B | S | R | B − A | B − S | B − R | rule |
|---|---|---|---|---|---|---|---|---|
| **hBCI MI left/right (primary)** | 0.606 | 0.701 | 0.533 | 0.716 | +0.094 [0.066, 0.124] | +0.168 [0.133, 0.203] | −0.015 [−0.040, +0.009] | **FAIL** (B − R) |
| hBCI MA vs rest | 0.622 | 0.690 | 0.555 | 0.653 | +0.068 [0.039, 0.097] | +0.135 [0.106, 0.167] | +0.037 [0.010, 0.066] | pass (secondary) |
| MW n-back 0/2/3 (chance 0.33) | 0.402 | 0.405 | 0.373 | 0.430 | +0.003 [−0.027, +0.033] | +0.031 [0.001, 0.064] | −0.026 [−0.068, +0.014] | fail |

Rule outcome: **Claim 4 FAILS on the primary task**: virtual hemodynamics add +9.4 points over band power, but
a random projection of the same slow EEG adds as much — the gain comes from the slow-EEG band, not from the
hemodynamic supervision. Mental arithmetic (secondary) passes all three controls. Side observation (not a
declared claim): slow EEG lifts cross-person MI accuracy from 0.61 to 0.70–0.72. The shuffled-target control
falls below band power alone (it adds noise features), so it is a weak control; R is the informative one.

### P3.1 Claim 1 linear shared latent (rank 8) — 2026-11-05

| Regime | full ridge | single (fNIRS latent) | joint (fNIRS + fMRI latent) | joint-shuffled | joint − single | joint − shuffled | rule |
|---|---|---|---|---|---|---|---|
| all training people | 0.182 | 0.174 | 0.148 | 0.146 | −0.026 [−0.036, −0.016] | +0.002 [−0.003, +0.008] | NOT CONFIRMED |
| 5 training people | 0.124 | 0.123 | 0.101 | 0.097 | −0.021 [−0.031, −0.013] | +0.004 [−0.002, +0.010] | NOT CONFIRMED |

Rule outcome: **Claim 1 not confirmed** — at a fixed rank 8, sharing the latent with fMRI costs fNIRS accuracy
(the fMRI directions displace fNIRS ones), and real fMRI is no better than shuffled fMRI. Same direction as
the deep notebook (fMRI joint −0.030). Data scarcity does not change it.

### P3.1b Claim 1 shared-space form — 2026-11-05

fNIRS r_subject by latent rank (hBCI, 5 person folds): full ridge 0.182; fNIRS-specific 4: 0.151, 8: 0.174,
16: 0.181, 32: 0.174, 64: 0.173; joint 4: 0.115, 8: 0.148, 16: 0.174, 32: 0.177, 64: 0.168. Rule chose rank 8
(0.174 >= 0.95 × 0.182).

| Output (rank 8) | full ridge | modality-specific space | shared space | shared − specific, 90% CI |
|---|---|---|---|---|
| fNIRS (hBCI, 29) | 0.182 [0.144, 0.219] | 0.174 [0.133, 0.213] | 0.148 [0.111, 0.182] | −0.026 [−0.034, −0.018] |
| fMRI (ds002336, 10) | 0.030 [−0.009, 0.066], p 0.19 | 0.025, p 0.27 | 0.023 [−0.023, 0.061], p 0.30 | −0.002 [−0.017, +0.012] |

Rule outcome: **does not hold**. Both conditions fail: (1) the fMRI output is not predictable from EEG on unseen
people by any model, shared or not (10 people, in-scanner EEG); (2) at the chosen rank the shared space costs
fNIRS 0.026. Descriptive only: at rank 16–32 the shared space costs fNIRS nothing (0.174 vs 0.181; 0.177 vs 0.174),
so the cost is a capacity effect, not a conflict between modalities. The fMRI arm is the bottleneck for Claim 1.

### P3.3 Claim 3 delay, multichannel lagged ridge — 2026-11-05

(First launch stopped after 2 people and rerun in 6 parallel shards; per-person seeds fixed; a bug that zeroed
the descriptive global curve — RidgeCV returns 1-D predictions for a single target — fixed before the rerun.
The local, primary curve was unaffected.)

| Cohort | beat own null | median delay (positive = EEG earlier) | IQR | Wilcoxon | split-half r (perm p) | peak r local / global | 3a | physiological |
|---|---|---|---|---|---|---|---|---|
| hBCI (29) | **29/29** | **−2 s** | [−2, −1] | 0.006 | **+0.555 (0.015)** | 0.37 / 0.59 | **PASS** | no |
| mental workload (26) | **26/26** | **−2 s** | [−2, −1] | 0.004 | **+0.467 (0.026)** | 0.30 / 0.56 | **PASS** | no |

Rule outcome: **Claim 3a passes in both cohorts** — the delay exists in every person, is reproducible between
the two halves of the recordings and is subject-varying. Its direction is the opposite of the classic
neurovascular one: the infra-slow EEG follows HbO by ~1–2 s, identically in two independent cohorts (local
and global targets alike). This fits the known non-neuronal (vascular / blood–brain-barrier) generator of
infra-slow scalp potentials. Caveat: the BrainAmp hardware high-pass (time constant 10 s) advances EEG phase at
these frequencies (≈ 1 s at 0.05 Hz), so the true EEG lag is, if anything, larger. The 30 s-epoch and PC1
analyses (T6, P2.3) were cruder versions of the same question; P3.3 supersedes them.

### P3.1c Claim 1 fMRI arm within person — 2026-11-06

| Personal head on | r_subject [95% CI] | p_signflip | people positive |
|---|---|---|---|
| raw slow-EEG features | +0.015 [−0.030, +0.059] | 0.54 | 7/10 |
| fNIRS-learned space (rank 16) | +0.028 [−0.003, +0.059] | 0.15 | 7/10 |
| other-people fMRI space (rank 16) | +0.032 [−0.005, +0.071] | 0.18 | 7/10 |
| random space (rank 16) | −0.038 [−0.083, +0.009] | 0.16 | 3/10 |

fNIRS space − raw +0.013 [−0.043, +0.066]; fNIRS space − random +0.066 [+0.006, +0.122].
Rule outcome: **(a) fMRI arm not shown, (b) Claim 1 transfer FAILS.** BOLD is not predictable from EEG in
ds002336 even within person. The fNIRS-learned space is closer to BOLD-relevant directions than a random one,
but no better than raw features. **Conclusion for Claim 1 with the data available: not achievable.** The
fMRI arm needs a simultaneous EEG–fMRI dataset that keeps the infra-slow EEG (DC-coupled or unfiltered) and
has more than 10 people; ds002739 removed that band before publication, HEFMI's fNIRS-side EEG likewise.

### P4.1 Claim 1 retest, 27 fMRI people — 2026-11-07

ds002338 built: 17 people, 648 trials, every run aligned (slow/alpha power 50.7 — infra-slow band present).

| fMRI arm | fMRI full ridge | fMRI specific (rank 8) | fMRI shared (rank 8) | fNIRS shared − specific (90% CI) | fMRI shared − specific (90% CI) | both work | no cost | verdict |
|---|---|---|---|---|---|---|---|---|
| **pooled 27 (primary)** | 0.070 [0.045, 0.097] | 0.074 [0.048, 0.102] | **0.074 [0.046, 0.104], p 0.0001** | −0.021 [−0.028, −0.014] | −0.000 [−0.006, +0.006] | **yes** | no (fNIRS) | **does not hold** |
| ds002338 alone (descriptive) | 0.085 [0.045, 0.124] | 0.081 | 0.088 [0.041, 0.132], p 0.003 | −0.010 [−0.015, −0.005] | +0.007 [−0.001, +0.015] | yes | yes | (holds — descriptive only) |

P3.1c within person, pooled 27: raw 0.073 [0.040, 0.109] p 0.0003 (23/27) -> **(a) fMRI arm works**; fNIRS space
0.071, other-people fMRI space 0.067, random 0.033; fNIRS space − raw −0.002 [−0.031, +0.026] -> **(b) transfer FAILS**.

Rule outcome: with 27 people **the fMRI output becomes predictable from EEG** (cross-person and within person;
the 10-person null was a sample-size problem), and one shared space carries both outputs with no cost to fMRI.
The primary Claim 1 verdict still fails on the declared rank 8 because sharing costs fNIRS 0.021 (lower 90%
bound −0.028 < −0.02). P3.1b showed this cost disappears at rank 16–32 (descriptive); that has to be confirmed
as a newly declared test on fresh data, not adopted here.

Exploratory rank sweep (pooled 27; run after seeing the rank-8 verdict — descriptive, NOT a verdict):

| rank | fNIRS specific / shared | fNIRS shared − specific (90% CI) | fMRI specific / shared | fMRI shared − specific (90% CI) |
|---|---|---|---|---|
| 8 (declared) | 0.174 / 0.153 | −0.021 [−0.028, −0.014] | 0.074 / 0.074 | −0.000 [−0.006, +0.006] |
| 16 | 0.181 / 0.175 | −0.006 [−0.009, −0.002] | 0.070 / 0.070 | +0.000 [−0.004, +0.005] |
| 32 | 0.174 / 0.178 | +0.004 [+0.002, +0.007] | 0.067 / 0.068 | +0.001 [−0.003, +0.005] |

At ranks 16 and 32 the shared-space conditions are met (at 32 the shared space is slightly better for fNIRS).
For the paper: Claim 1 (shared-space form) is supported only as an exploratory result with a capacity
dependence; the confirmatory, pre-declared rank-8 test failed.

### P4 FineMI results (declared tests) — 2026-11-07

Loader audit: 18 people, 5294 trials; 9 blocks in 5 people dropped by rule (EEG/NIRS event sequences differ).

| Test | Result | Rule outcome |
|---|---|---|
| P4.R1 cross-person | slow < 0.5 Hz **0.203 [0.164, 0.248]**; < 0.1 Hz 0.217; 0.1–0.5 Hz 0.084; band power 0.046; EOG slow −0.008; slow − band power **+0.157 [0.115, 0.204]** | **replicates (third cohort)** |
| R2 person-specificity | slow same 0.252 vs other 0.093, diff +0.160 [0.111, 0.208], identification 83% (chance 5.6%); band power diff +0.071, identification 83% | person-specific |
| R3 global/local | HbO global 0.353, **HbO local 0.159 [0.123, 0.199]**; HbR global 0.096, local 0.024 | local component present |
| P4.2 Claim 2 hBCI -> FineMI | −0.023 [−0.050, +0.004], 6/18 | **FAIL** |
| P4.2 mental workload -> FineMI | +0.001 [−0.026, +0.030] | fail |
| P4.2 FineMI -> hBCI | −0.059 [−0.119, −0.005] | fail (negative) |
| P4.4 Claim 4, 8-class (chance 0.125) | A 0.130, B 0.146, S 0.119, R 0.138; B − A +0.016 [0.005, 0.026]; B − R +0.008 [−0.001, +0.017] | **FAIL** (B − A < 0.02) |

| P4.3 Claim 3 lag (declared) | 8/18 beat own null (44%); median delay −4 s (HbO first) IQR [−8.5, −0.25], Wilcoxon 0.079; split-half r +0.32 (perm p 0.12); global median +1 s | **3a FAIL** |
| P4.5 calibration, raw target | k = 2: baseline 0.173, personal 0.240, hybrid 0.288; gain **+0.067 [0.033, 0.112], 18/18**; k = 7 personal 0.308 vs baseline 0.182; population (k = 0) 0.233 | **EEG adds deployment value** |
| P4.5b calibration, local target | k = 2: baseline 0.156, personal 0.197; gain **+0.041 [0.014, 0.079], 16/18** | **local deployment value** |

P4.A detrend (post hoc): hBCI -> FineMI −0.024, MW -> FineMI +0.004, FineMI -> hBCI −0.053 — still fail; hBCI ->
MW 0.195 and MW -> hBCI 0.180 unchanged (pass). So the within-cohort signal replicates strongly in FineMI, but
a model from the TU Berlin cohorts does not transfer to it (different lab, amplifier, reference, motor-only
NIRS montage, overlapping 19 s trials).

### P4.A / P4.B post-hoc amendment results — 2026-11-07

Transfer (Claim 2), FineMI re-referenced to linked mastoids: hBCI -> FineMI −0.025 [−0.055, +0.004];
MW -> FineMI +0.005; FineMI -> hBCI −0.052 [−0.111, +0.001]. With detrend as well: −0.026 / +0.006 / −0.047.
**FineMI transfer fails under every harmonisation tried**; hBCI <-> MW is unaffected by the same steps
(0.195 / 0.180). Accepted as a negative result: transfer holds between the two TU Berlin cohorts only.

Lag (Claim 3) with EEG high-passed at 0.01 Hz (band-matched to HbO):

| Cohort | beat own null | median delay | IQR | Wilcoxon | split-half r (perm p) | 3a |
|---|---|---|---|---|---|---|
| hBCI | 29/29 | −2.0 s | [−2, −1] | 0.006 | +0.578 (0.003) | PASS (unchanged) |
| mental workload | 26/26 | −1.5 s | [−2, −1] | 0.001 | −0.125 (0.75) | FAIL (was PASS) |
| FineMI (linked mastoids) | 9/18 | 0.0 s | [−4, +4] | 0.84 | +0.535 (0.035) | FAIL (50% is not a majority) |

Reading: the HbO-leads-slow-EEG delay of 1–2 s is stable in both TU Berlin cohorts under both band choices,
and almost identical across people (IQR 1 s). The between-person *differences* in delay reproduce in hBCI
under both versions but not in mental workload once the band is matched — so Claim 3a ("reproducible,
subject-varying delay") is robust only in hBCI; the group-level delay (−1 to −2 s) is robust in both. FineMI
(DC-coupled, motor-only NIRS) shows no reliable continuous coupling.

Also observed: a pooled cross-person ridge on slow EEG (0.197) beats the deep model on the same band (0.155).

### P5.5 — seed robustness of the P5 passes (written 2026-11-07 after P5.1/P5.4 passed, before running)

P5.1 (MW, shared-space form) rerun with person-fold seeds 1338–1341; P5.4 (raw target) rerun with the
calibration-run choice drawn with seeds 1338–1341. Reported: how many of the 5 seeds (1337 included) pass the
same rule. Robust = at least 4 of 5.

### P5.6 — Claim 2 on the fMRI output (written 2026-11-07, before running)

ds002336 (XP1, 10 people) and ds002338 (XP2, 17 other people) were recorded separately. A slow-EEG ridge
(features as P3.1, mapped onto hBCI positions, z-scored per person; target BOLD 5 ROIs × 15, subject × task
demeaned) is trained on all of one study and tested frozen on the other. Primary: ds002336 -> ds002338 (larger
test set). Also the shared-space model of P5.1 (rank 32 space from all hBCI fNIRS + the training study's fMRI;
fMRI head on the training study) frozen and tested the same way. Notebook rule (CI > 0, p < 0.05, r >= 0.05).
- **Claim 2 holds for the fMRI output**: notebook rule on ds002336 -> ds002338 for the plain ridge.

### P5.7 — Claim 5 trial-level form in the calibrated deployment (written 2026-11-07, before running)

Same deployment as P5.4 (two calibration runs per person drawn with seed 1337, k = 2 hybrid, raw target). For each
test trial: error = 1 − trial correlation of the hybrid prediction with the measured haemodynamics.
Reliability scores fixed now: (primary) −OOD = minus the mean squared z of the trial's slow-EEG features against
the person's own calibration-run distribution; (secondary) predicted-signal energy = norm of the hybrid's
predicted deviation from the personal mean. Score: Spearman(score, −error) within each person, then the mean
over people with a subject-level bootstrap CI.
- **Claim 5 trial-level form holds**: primary mean within-person Spearman > 0.2 with CI > 0 in hBCI AND in at least
  one external cohort (MW, FineMI). Secondary reported with the same numbers.

### P5 results — 2026-11-07 (seed 1337)

**P5.1 Claim 1, rank 32, fresh fNIRS cohorts (fMRI arm pooled 27)**

| fNIRS cohort | fNIRS full / specific / shared | fNIRS shared − specific (90% CI) | fMRI specific / shared | fMRI shared − specific (90% CI) | both work | no cost | shared-space form |
|---|---|---|---|---|---|---|---|
| **mental workload (primary, fresh)** | 0.132 / 0.130 / **0.123** [0.080, 0.170] | −0.006 [−0.014, +0.001] | 0.067 / **0.068** [0.043, 0.095] p 0.0001 | +0.001 [−0.003, +0.005] | yes | yes | **HOLDS** |
| FineMI (replication, fresh) | 0.196 / 0.176 / **0.180** [0.141, 0.226] | +0.004 [+0.002, +0.005] | 0.064 / 0.063 | −0.000 [−0.004, +0.004] | yes | yes | **HOLDS** |
| hBCI (not fresh at rank 32, descriptive) | 0.182 / 0.174 / 0.178 | +0.004 [+0.001, +0.007] | 0.067 / 0.068 | +0.001 [−0.003, +0.005] | yes | yes | (holds) |

P3.1 form at rank 32 (fMRI supervision improves fNIRS): joint − single MW −0.006 [−0.015, +0.002], FineMI
+0.004 [+0.002, +0.006], hBCI +0.004 [+0.001, +0.008]; joint − joint-shuffled MW −0.001, FineMI +0.003
[−0.001, +0.007], hBCI +0.001 [−0.001, +0.003] -> **FAILS in every cohort** (real fMRI no better than shuffled).

Rule outcome: **Claim 1, shared-space form, CONFIRMED** on a fresh cohort with a rank fixed in advance (and
replicated in FineMI): one EEG-derived space of rank 32 predicts fNIRS and fMRI on unseen people at no cost to
either. The stronger form — fMRI supervision *improves* fNIRS — is not supported: the small joint − single gain
in FineMI/hBCI is matched by shuffled fMRI targets, i.e. it is extra capacity, not fMRI information. Caveat: the fMRI arm (27) is the one used in P4.1; only the fNIRS
side of this confirmation is fresh.
P5.1b data-scarce (5 fNIRS training people per fold): joint − single MW −0.000 [−0.003, +0.003], FineMI +0.005
[+0.003, +0.007]; joint − shuffled MW −0.001, FineMI −0.000 [−0.003, +0.002] -> **FAILS**; scarcity does not
make the fMRI information useful.

**P5.2b Claim 2, Euclidean alignment (secondary)**: hBCI -> FineMI −0.027 [−0.054, +0.001]; MW -> FineMI −0.005;
FineMI -> hBCI −0.046; hBCI -> MW 0.192 [0.126, 0.260] (pass); MW -> hBCI 0.143 [0.111, 0.174] (pass).
Alignment does not rescue FineMI transfer.

**P5.2 Claim 2, FineMI with emulated BrainAmp hardware high-pass (primary of P5.2)**: hBCI -> FineMI
**−0.064 [−0.094, −0.037]**, p 0.0006, 3/18 positive -> **FAIL**; MW -> FineMI −0.028; FineMI -> hBCI −0.114
[−0.171, −0.058]. Linked mastoids + emulation: −0.070 / −0.026 / −0.113. hBCI <-> MW unchanged (0.205 / 0.177).
Matching the hardware filter makes the transfer *reliably negative* rather than null. Check run afterwards (not a
claim test): blink polarity is normal in every cohort (positive skew of 1–10 Hz EEG at frontal-pole channels in
29/29 hBCI, 26/26 MW, 18/18 FineMI people), so the sign reversal is not a recording-polarity error; the
EEG-to-haemodynamics mapping itself differs between the labs' set-ups (motor-only 24-channel NIRS, 19 s trials,
different cap and amplifier). **Claim 2 holds within one lab's hardware only.**
Descriptive side result (exploratory): Euclidean alignment raises the *within-person* k = 2 personal model
(raw score) from 0.337 to 0.407 in hBCI and from 0.119 to 0.167 in FineMI.

**P5.3 Claim 4, data-scarce form (5 labelled people)**

| Task | A | B | S | R | B − A | B − S | B − R | rule |
|---|---|---|---|---|---|---|---|---|
| hBCI MI (primary) | 0.587 | 0.615 | 0.529 | 0.657 | +0.028 [−0.012, +0.067] | +0.086 [+0.043, +0.129] | −0.042 [−0.076, −0.011] | **FAIL** |
| hBCI MA | 0.595 | 0.630 | 0.539 | 0.597 | +0.035 [+0.012, +0.059] | +0.091 [+0.053, +0.130] | +0.033 [−0.008, +0.072] | fail |
| MW n-back | 0.370 | 0.409 | 0.326 | 0.427 | +0.039 [−0.003, +0.080] | +0.083 [+0.048, +0.121] | −0.019 [−0.051, +0.014] | fail |
| FineMI 8-class | 0.131 | 0.129 | 0.128 | 0.122 | −0.002 | +0.001 | +0.007 [−0.002, +0.015] | fail |

Fewer labels do not make the haemodynamic supervision useful: the random projection still matches or beats it.

**P5.3b Claim 4, representation form (post-hoc-motivated)** — classifier on a compact code alone, no band power:

| Task | regime | V (virtual hemo) | Rp (random) | PC (PCA) | Full slow EEG | V − Rp | V − PC | rule |
|---|---|---|---|---|---|---|---|---|
| hBCI MI (primary) | all labels | 0.697 | 0.707 | **0.760** | 0.722 | −0.010 [−0.034, +0.013] | −0.064 [−0.089, −0.039] | **FAIL** |
| hBCI MI | 5 labelled | 0.612 | 0.654 | 0.692 | 0.679 | −0.043 | −0.081 | fail |
| hBCI MA | all labels | 0.675 | 0.636 | 0.668 | 0.672 | +0.039 [+0.006, +0.067] | +0.007 [−0.021, +0.036] | fail |
| MW n-back | all labels | 0.413 | 0.423 | 0.453 | 0.467 | −0.010 | −0.040 | fail |
| FineMI 8-class | all labels | 0.153 | 0.143 | 0.161 | 0.156 | +0.010 [−0.003, +0.023] | −0.009 | fail |

**Claim 4 is refuted in every form tried**: an unsupervised PCA code of the same slow EEG is as good or better
than the haemodynamically supervised code. The downstream benefit is the slow band itself (A 0.61 -> PC 0.76 on
hBCI MI, cross-person).

**P5.4 Claim 5, person-level calibrated form (raw target)**

| Cohort | Spearman(calibration-run score, test r) [95% CI] | rule |
|---|---|---|
| hBCI (29) | **+0.680 [0.370, 0.865]** | pass |
| mental workload (26, external) | **+0.558 [0.221, 0.783]** | pass |
| FineMI (18, external) | **+0.899 [0.689, 0.971]** | pass |

Rule outcome: **Claim 5, person-level calibrated form, HOLDS**: two calibration runs tell in advance how well the
EEG model will work for that person, in every cohort. This is a person-level reliability statement; the
trial-level form (P3.5) still fails.
Robustness, local target: hBCI +0.575 [0.262, 0.807], MW +0.605 [0.249, 0.815], FineMI +0.837 [0.574, 0.946] -> holds.
EEG-specific check (score and outcome = hybrid gain over the no-EEG personal mean): hBCI +0.712 [0.380, 0.914],
MW +0.833 [0.664, 0.915], FineMI +0.868 [0.627, 0.956] -> holds. The calibration runs predict how much the EEG
model will add for that person, not only how stereotyped their haemodynamics are.

**P5.6 Claim 2 on the fMRI output**: ds002336 -> ds002338 ridge +0.017 [−0.009, +0.042], p 0.22 (11/17) -> **FAIL**;
shared-space model +0.026 [−0.006, +0.056]. Reverse ds002338 -> ds002336: ridge +0.039 [+0.018, +0.058], p 0.011
(9/10), shared space +0.041 [+0.021, +0.061] -> detectable but below practical_r 0.05 ("negligible size").
Training on 10 people is too little for the fMRI output; training on 17 transfers, weakly.

**P5.2c Claim 3 on hardware-emulated FineMI (native reference)**: 8/18 beat their own null (44%); median delay
+1 s, IQR [−5, +10], Wilcoxon p 0.42; split-half r +0.69 (perm p 0.006), but 19 of 36 half-run estimates sit
at the −10 s edge of the lag range, so the split-half agreement reflects edge-pinned noise. -> **3a FAIL**
("exists" needs a majority). Peak local r per person 0.006–0.05 (hBCI 0.37). FineMI shows no continuous
EEG–HbO coupling under any EEG chain tried.
Secondary, linked mastoids + emulation: 8/18 beat null, median −1 s, IQR [−4.5, +4.5], split-half r +0.41
(perm p 0.073), 17/36 half estimates at the edge -> 3a FAIL.

**P5.7 Claim 5 trial-level calibrated form**: primary −OOD within-person Spearman hBCI −0.007 [−0.063, +0.048],
MW +0.234 [+0.086, +0.369], FineMI +0.042 [−0.018, +0.101] -> **FAIL** (hBCI). Secondary predicted-signal energy:
hBCI +0.216 [0.152, 0.278], MW +0.060, FineMI +0.164 [0.093, 0.236] — positive in two cohorts but at or below
0.2. Reliability is solid per person, weak per trial. No further trial-level score will be tried.

**P5.5 seed robustness**
- P5.1 (MW shared-space form), person-fold seeds 1337–1341: **holds 5/5**. fNIRS shared r 0.114–0.123; fNIRS
  shared − specific −0.002 … −0.010, 90% lower bounds −0.008 … −0.016 (all > −0.02); fMRI shared 0.068.
- P5.4 (raw target), calibration-run seeds 1337–1341: **holds 5/5** (rule: hBCI and at least one external).
  hBCI passes 5/5 (0.60–0.79), FineMI 5/5 (0.84–0.90), mental workload 4/5 (0.28–0.66; seed 1340 CI crosses 0).

## P6 — third round on the claims still failing (declared 2026-11-07, before running any P6 test)

User instruction: try to solve the remaining failures too. Each P6 test is written here before it is run. All are
motivated by P5 results already seen, so they are a third confirmatory round; the declared earlier forms remain the
primary record. No new test is declared for Claim 4 (refuted in four forms; a fifth would be fishing) or for the
trial-level form of Claim 5 (P5.7 already committed to "no further trial-level score").

### P6.1 — Claim 2 into FineMI with paradigm and montage harmonisation

Two differences between the labs were not yet matched. (1) Trial timing: FineMI cues come every ~19 s, TU Berlin
every ~25 s, so the 30 s FineMI epoch (−2..28 s) contains the next trial's cue and response, while the TU Berlin
epoch does not. (2) Target montage: FineMI NIRS covers only the left fronto-central motor area (24 channels, nearest
10-05 sites FFC1h..C5h); the P4.2 target is the mean over all hBCI channels (frontal + occipital + both motor areas).
Harmonised version: features and target restricted to the first 20 bins (−2..18 s, before the next FineMI cue) in
both cohorts; hBCI target = mean of its 12 left-motor channels (C5CP5 … C1CP1, file order 12–23) per chromophore;
FineMI target = mean of all its channels (unchanged). Model, features, z-scoring, demeaning, alpha grid: as P4.2.
- **Primary: hBCI -> FineMI, both harmonisations, P4 native-reference cache. Claim 2 into FineMI passes**: notebook
  rule (CI > 0, p < 0.05, r >= 0.05).
- Secondary, same rule (reported, not used for the verdict): the same on the P5.2 hardware-emulated cache; window
  only; montage only; MW -> FineMI (window only — MW has no matching left-motor layout); FineMI -> hBCI (both).

### P6.2 — Claim 2, calibrated-transfer form

Question: when a person in the new lab gives the two calibration runs of the deployment protocol, does the model
frozen in the other lab add anything to their own personal model? Per FineMI person: two calibration runs drawn
once (seed 1337), all other runs are test runs. Personal model = ridge on the calibration runs (as P5.4). The
frozen external model (P4.2, hBCI-trained) and the personal model are stacked: one gain per model per person,
fitted by least squares on the calibration runs, the personal model's calibration predictions cross-fitted (train
on one calibration run, predict the other). The gains absorb the unit difference and any per-person sign. The
comparison is the same stacking with the personal model only. Score: test-run trial correlation (raw target).
- **Claim 2 (calibrated-transfer form) passes**: mean over people of r(personal + external) − r(personal) > 0 with
  95% subject-level bootstrap CI > 0, hBCI -> FineMI (P4 native cache). Secondary: hardware-emulated cache;
  MW -> FineMI; hBCI -> MW (within lab, reference).

### P6.3 — Claim 1, fMRI supervision on the local fNIRS target

The shared space is dominated by the global (systemic) fNIRS signal, which fMRI ROI targets do not share. On the
local target (each NIRS channel minus the mean over that chromophore's channels, per trial and bin; P2.5b) the
remaining signal is cortical, where fMRI could carry information. P5.1 rerun unchanged (rank 32, same folds, same
27-person fMRI arm) with the local fNIRS target.
- **Claim 1 (P3.1 form, local target) passes**: joint − single > 0 with 95% CI > 0 AND joint − joint-shuffled > 0
  with 95% CI > 0, mental workload (primary). FineMI and hBCI reported with the same numbers. The shared-space
  (P3.1b) numbers are reported too.

### P6.4 — Claim 3, classical neurovascular coupling from EEG band power (written 2026-11-07, before running)

User instruction (2026-11-07): the target is the coupling. Every Claim 3 test so far used infra-slow EEG, whose
coupling runs the "wrong" way (slow EEG follows HbO by 1–2 s). The classical neurovascular coupling — EEG power
changes leading the haemodynamic response by seconds — was never tested on the continuous recordings. Band power is
also unaffected by the hardware high-pass that made FineMI's slow band incomparable.
Features: per EEG channel, log power in theta 4–8, alpha 8–13, beta 13–30 and gamma 30–45 Hz (band-pass at the
native rate, squared, 1 s means, log), then band-passed 0.01–0.1 Hz at 1 Hz to match the HbO band (P4.A lesson),
z-scored per run. Target: local HbO (channels minus channel mean), as P3.3. Model, lags (−10..+20 s, positive =
EEG earlier), leave-one-run-out, 100 circular-shift surrogates (min shift 60 s), split-half and all rules: as P3.3.
Cohorts: hBCI (primary), MW, FineMI (raw CNT, native reference, same runs as P4).
- **Claim 3a (classical form) passes** in a cohort: more than half the people beat their own null (p < 0.05) AND
  split-half delay r > 0 with permutation p < 0.05. **Physiological**: median delay 2–10 s with Wilcoxon p < 0.05.
  Coupling claim (classical form) supported if 3a passes in hBCI and in at least one other cohort.
- Descriptive: the same with the global HbO target; the per-band share of the ridge weights.

### P6.4b — coupling beyond the task (written 2026-11-07 after the first 8 hBCI people of P6.4, before the rest)

A cue makes EEG power drop at once and HbO rise ~5–7 s later, so a +5..+7 s delay can arise from the shared task
response alone, without ongoing coupling. Control: in each continuous run, the cue-locked average response (finite
impulse response regression, one regressor per second 0–29 s after every cue, all cue types pooled, plus an
intercept) is removed from every EEG band-power feature and every HbO channel; P6.4 is rerun unchanged on the
residuals (same lags, null, split-half, rules). Cohorts as P6.4.
- **Coupling beyond the task**: Claim 3a (P6.4 rule) passes on the residuals in hBCI and at least one other cohort.
  If P6.4 passes but P6.4b fails, the coupling is reported as task-evoked only.

### P6 results — 2026-11-07 (seed 1337)

**P6.1 Claim 2 into FineMI, harmonised trial window (−2..18 s) and NIRS montage (hBCI left motor)**

| Direction | setting | P4 native cache | hardware-emulated cache |
|---|---|---|---|
| **hBCI -> FineMI (primary)** | window + montage | **−0.010 [−0.041, +0.021]**, p 0.53 (8/18) -> FAIL | −0.058 [−0.091, −0.025] (4/18) |
| hBCI -> FineMI | window only | −0.019 | −0.066 |
| hBCI -> FineMI | montage only | −0.013 | −0.056 |
| MW -> FineMI | window only | +0.002 | −0.033 |
| FineMI -> hBCI | window + montage | −0.055 | −0.124 |
| hBCI -> MW (within-lab reference) | window + montage | **+0.230 [0.156, 0.309]** pass | — |

Harmonising the trial window and the NIRS montage does not rescue cross-lab transfer; it only moves the native-cache
result from slightly negative to null. The within-lab transfer stays strong under the same harmonisation.

**P6.2 Claim 2, calibrated-transfer form** (frozen external model stacked on the person's own k = 2 model)

| Direction | baseline | personal | personal + external | difference [95% CI] | rule |
|---|---|---|---|---|---|
| **hBCI -> FineMI (primary)** | 0.067 | 0.103 | 0.103 | **−0.0001 [−0.003, +0.003]** (10/18) | **FAIL** |
| MW -> FineMI | 0.067 | 0.105 | 0.105 | +0.0000 | fail |
| hBCI -> FineMI, hardware-emulated | 0.067 | 0.103 | 0.105 | +0.0026 [−0.0007, +0.0067] | fail |
| hBCI -> MW (within lab) | 0.243 | 0.319 | **0.378** | **+0.059 [0.023, 0.095]** (18/26) | pass |

Within one lab the frozen population model adds +0.059 to a person's own calibration; across labs it adds nothing,
even with a per-person gain and sign. **Claim 2 holds within a lab's hardware only, in every form tried.**

**P6.3 Claim 1 on the local fNIRS target (rank 32)**

| fNIRS cohort | single / joint / joint-shuffled | joint − single [95% CI] | joint − shuffled [95% CI] | P3.1 form | shared-space form |
|---|---|---|---|---|---|
| **mental workload (primary)** | 0.074 / 0.078 / 0.075 | +0.003 [−0.001, +0.007] | +0.002 [−0.003, +0.008] | **FAIL** | holds (fMRI 0.071) |
| FineMI | 0.116 / 0.119 / 0.119 | +0.003 [+0.001, +0.005] | +0.000 [−0.003, +0.004] | fail | holds |
| hBCI | 0.072 / 0.070 / 0.071 | −0.002 | −0.000 | fail | holds |

fMRI supervision does not improve the local (cortical) fNIRS target either. The shared-space form of Claim 1
holds on the local target in all three cohorts (new robustness result: the shared space is not only the systemic signal).

**P6.4 Claim 3, classical neurovascular coupling (EEG band power -> local HbO, continuous runs)**

| Cohort | beat own null | median delay, EEG first [IQR] | Wilcoxon p | split-half delay r (perm p) | median peak r | global-HbO median delay | 3a | physiological |
|---|---|---|---|---|---|---|---|---|
| **hBCI (primary)** | **29/29 (100%)** | **+6 s [2, 8]** | < 0.0001 | **+0.39 (0.020)** | 0.191 | +9 s | **PASS** | **YES** |
| mental workload | 21/26 (81%) | +4 s [1, 8.75] | 0.0014 | −0.15 (0.77) | 0.135 | +7 s | fail (not reproducible) | yes |
| FineMI | 16/18 (89%) | +3 s [−2.25, 5.5] | 0.31 | +0.12 (0.32) | 0.053 | +7.5 s | fail (not reproducible) | no |

Band shares of the ridge weights are flat (theta 0.29–0.33, alpha 0.27–0.29, beta 0.21–0.23, gamma 0.17–0.20).
Rule outcome: **the classical coupling claim is NOT SUPPORTED as declared** (3a passes in hBCI only). What holds:
(1) in hBCI, EEG band power predicts local HbO 6 s later in every person, with a reproducible personal delay —
the classical neurovascular order, opposite to the infra-slow result (slow EEG follows HbO by 1–2 s);
(2) the coupling **exists** (most people beat their own null) in all three cohorts, including FineMI, where the
infra-slow EEG showed none (P4.3, P5.2c); (3) the group delay is EEG-first in MW (+4 s, p 0.001). What fails:
outside hBCI, a person's delay estimate does not repeat between the two halves of their recording.

**P6.4b coupling beyond the task (pooled cue-locked FIR response, 0–29 s, removed from EEG features and HbO)**

| Cohort | beat own null | median delay, EEG first [IQR] | Wilcoxon p | split-half delay r (perm p) | median peak r | global-HbO median delay | 3a | physiological |
|---|---|---|---|---|---|---|---|---|
| **hBCI (primary)** | **29/29 (100%)** | **+6 s [2, 8]** | < 0.0001 | **+0.33 (0.041)** | 0.164 | +8 s | **PASS** | **YES** |
| mental workload | 22/26 (85%) | +2 s [1, 7.75] | 0.0016 | −0.10 (0.70) | 0.094 | +4.5 s | fail (not reproducible) | yes |
| FineMI | 15/18 (83%) | +1.5 s [−4.75, 6] | 0.52 | +0.22 (0.18; 4 half-estimates at the lag edge) | 0.061 | +2 s | fail (not reproducible) | no |

Rule outcome: **coupling beyond the task is NOT SUPPORTED as declared** (passes in hBCI only, same as P6.4).
In hBCI the coupling survives removal of the task response almost unchanged: every person still beats their
own null, the delay stays at +6 s and remains reproducible within person, and peak r falls only from 0.191 to
0.164. So the hBCI result is ongoing (spontaneous) neurovascular coupling, not two signals that both follow
the cues. In mental workload and FineMI, most people still beat their null after the task is removed, but the
personal delay does not repeat between halves, as in P6.4.

## P7 — fourth round on Claims 3 and 2 (declared 2026-11-08, before running any P7 test)

User instruction: try the failing Claims 3 and 2 again. These are new tests with new rules. They do not replace
P6.4 / P6.4b / P6.1 / P6.2, which stay the primary record of those declared tests.

### P7.1 — Claim 3, within-person reproducibility of the coupling kernel

Why a new statistic: the P6.4 reproducibility check is the across-person Pearson r between first-half and
second-half delays. That measures whether people *differ* reliably in delay. If everyone has a similar delay, that
r is near 0 even when each person's coupling repeats perfectly. P7.1 asks the direct question: does a person's
lag curve in the first half of their recording look like their lag curve in the second half, more than chance?

- Data, features, target, model and lags exactly as P6.4: band power theta–gamma -> local HbO, lags −10..+20 s,
  leave-one-run-out ridge. Each run is split into a first and second half as in P6.4.
- Per person: curves c_A (first halves) and c_B (second halves); k = Pearson r of c_A and c_B over the 31 lags.
  Null: 20 surrogates, each recomputing c_B after a circular shift of HbO in every second-half run (shift ≥ 60 s);
  k_null = r(c_A, c_B surrogate). Person score d = k − mean(k_null).
- Cohort rule: **kernel reproducible** if median d > 0 and one-sided Wilcoxon signed-rank p < 0.05 on d.
  **Claim 3a′ passes** in a cohort if the coupling exists there (P6.4: more than half the people beat their own
  null, already measured) and the kernel is reproducible. **Claim 3 (kernel form) supported** if 3a′ passes in
  hBCI and at least one other cohort.
- Secondary, same rule, on the P6.4b task-residual data (env `RESID=1`): coupling beyond the task.
- Descriptive: fraction of people with |delay_A − delay_B| ≤ 2 s, against chance from pairing each first half
  with other people's second halves.

### P7.2 — Claim 2, transfer of a frozen band-power coupling kernel across labs

Why: every Claim 2 failure into FineMI used infra-slow EEG. That band depends on the amplifier's DC / high-pass
behaviour, which differs between labs (P5.2). Band power at 4–45 Hz does not. P6.4 found the global HbO response
following EEG band power by 7–9 s in all three cohorts.

- Source: all 29 hBCI people, continuous runs. Features are the P6.4 band power (4 bands × 30 hBCI electrodes),
  z-scored per run. Target cohorts' raw EEG is first mapped onto the hBCI electrodes with the notebook's
  spherical-spline `interpolation_matrix`, then turned into band power the same way.
- Target: global HbO (mean over all NIRS channels, 0.01–0.1 Hz), z-scored per run. It is montage-independent, so
  no NIRS channel matching is needed. It includes systemic haemodynamics, and is reported as such.
- Model: one ridge (RidgeCV, alphas 10^0..10^6) with EEG lags 0, 2, 4, …, 14 s (EEG earlier), fitted once on all
  hBCI people and frozen. No refitting, rescaling or sign choice on the target cohort.
- Score: per run, Pearson r between prediction and global HbO; person r = mean over runs. Person null: 200
  circular shifts of HbO per run (≥ 60 s), with the mean over runs.
- Rule: **transfer passes into a cohort** if mean person r > 0 with bootstrap (5000) 95% CI lower bound > 0, and
  more than half of the people beat their own null (p < 0.05). **Primary: hBCI -> FineMI (cross-lab).**
  Reference: hBCI -> mental workload (same lab). Descriptive: MW -> FineMI, and a leave-one-person-out
  within-hBCI score.

### P7 results — 2026-11-08 (seed 1337)

**P7.2 Claim 2, frozen band-power -> global HbO kernel (hBCI, EEG lags 0–14 s)**

| Direction | mean person r [95% CI] | people r > 0 | beat own null | verdict |
|---|---|---|---|---|
| within hBCI, leave one person out (descriptive) | +0.148 | — | 76% | — |
| hBCI -> mental workload (same lab, reference) | +0.098 [+0.071, +0.124] | 23/26 | 54% | **PASS** |
| **hBCI -> FineMI (other lab, primary)** | **−0.053 [−0.071, −0.034]** | 2/18 | 0% | **FAIL** |
| MW -> FineMI (descriptive) | −0.008 [−0.027, +0.011] | 8/18 | 0% | fail |

The frozen model's alpha was 1, the lowest value on the declared grid, for both source fits. Kernel weight is
spread over EEG lags 2–12 s (6% at 0 s, 7% at 14 s).

Rule outcome: **Claim 2 cross-lab transfer FAILS again**, and is reliably negative (7/18 FineMI people below
their null's 5th percentile). Transfer within the TU Berlin lab passes with a different EEG band and a different
target. So the failure into FineMI does not depend on the infra-slow band or on the amplifier's high-pass filter.
Post-hoc check (not a declared test): FineMI's HbO sign is normal. The trial-average HbO peaks positive 9–12 s
after the cue, as in hBCI, so a sign flip is ruled out. One difference: HbO and HbR are almost uncorrelated in
FineMI (r −0.02 vs −0.23 in both TU Berlin cohorts), which points to a different mix of the global haemodynamic
signal in that lab.

**P7.1 Claim 3, within-person reproducibility of the band-power -> local HbO lag curve (20 shifted nulls per person)**

| Cohort | data | median k (half A vs half B) | median k, shifted null | median d | Wilcoxon p (d > 0) | people p < 0.05 | delays within 2 s (chance) | 3a′ |
|---|---|---|---|---|---|---|---|---|
| **hBCI** | raw (primary) | +0.753 | −0.015 | +0.776 | < 0.0001 | 45% | 72% (29%) | **PASS** |
| mental workload | raw (primary) | +0.179 | +0.042 | +0.101 | 0.091 | 8% | 15% (21%) | fail |
| **FineMI** | raw (primary) | +0.205 | −0.031 | +0.191 | **0.033** | 22% | 17% (15%) | **PASS** |
| **hBCI** | task removed (secondary) | +0.740 | −0.034 | +0.744 | < 0.0001 | 48% | 72% (28%) | **PASS** |
| **mental workload** | task removed (secondary) | +0.227 | +0.003 | +0.224 | **0.006** | 12% | 35% (23%) | **PASS** |
| **FineMI** | task removed (secondary) | +0.277 | −0.012 | +0.238 | **0.001** | 22% | 17% (16%) | **PASS** |

Existence (P6.4 / P6.4b, more than half the people beat their own null) holds in every cohort.

Rule outcome: **Claim 3 (kernel form) SUPPORTED**: 3a′ passes in hBCI and FineMI. **Beyond the task, also
SUPPORTED, in all three cohorts.** What it means: in every cohort, a person's EEG-band-power -> local HbO
coupling curve in the first half of the recording predicts its shape in the second half, beyond what a
time-shifted recording gives. Removing the task response makes the kernel *more* reproducible in MW and FineMI,
so the task-evoked part was adding noise, not creating the coupling.

Limits, to be reported with the result:
1. This statistic was declared after the P6.4 split-half delay test failed outside hBCI. P6.4 stays the primary record of that test; P7.1 is a fourth-round reformulation with its own pre-declared rule.
2. Outside hBCI, the curve *shape* repeats but the *peak delay* does not: delay agreement within 2 s is at chance in MW and FineMI. A precise personal delay is only established in hBCI.
3. Individual-level evidence outside hBCI is weak (8–22% of people p < 0.05, with 20 surrogates). The result is a group-level effect. The FineMI primary p (0.033) is modest.

## P8 — improving Claims 3 and 2 (declared 2026-11-08, before running any P8 test)

User instruction: try to improve these results. No new reproducibility statistic is introduced for Claim 3.
P7.1 stays as declared. P8 addresses the open caveat of P7.1 and tests the one lead left from P7.2.

### P8.1 — Claim 3, a smoothed personal delay (computed from the P7.1 half curves already saved)

The 1 s lag-curve argmax is noisy when the curve is broad. Delay estimator: argmax of the half curve after a 5-lag
centred moving average (edge lags averaged over the available points). For each cohort and both P7.1 data
versions, report (a) the fraction of people with |delay_A − delay_B| ≤ 2 s against the pairing-with-others
chance level (one-sided permutation p over 10 000 shuffles of the second halves), and (b) the P6.4 across-person
split-half r with its permutation p. **Personal delay reproducible** in a cohort if (a) is above chance with p < 0.05.
This is a delay-estimator change only; the reported delays are descriptive outside this rule.

### P8.2 — Claim 2, harmonised haemodynamic target (CBSI)

P7.2 showed a different HbO–HbR relation in FineMI (r −0.02 vs −0.23). The correlation-based signal improvement
(CBSI; Cui et al. 2010) removes the HbO component that moves with HbR in the same direction, mostly systemic,
and applies the same definition in every lab. Per channel and run, on the 0.01–0.1 Hz 1 Hz signals:
CBSI = (HbO − α·HbR)/2 with α = std(HbO)/std(HbR); target = mean over channels, z-scored per run. Everything else
as P7.2 (features, interpolation, lags 0–14 s, frozen RidgeCV on all hBCI people, score, null, rule).
**Primary: hBCI -> FineMI.** Reference: hBCI -> MW. The P7.2 global-HbO result is recomputed in the same run as a check.

### P8.3 — Claim 2, calibrated transfer of the band-power kernel (P6.2 logic)

For each target person with ≥ 3 runs, 2 calibration runs are drawn (seed 1337). Personal model: the same
band-power FIR ridge fitted on the two calibration runs, cross-fitted between them for the stacking gains.
External: the frozen hBCI kernel. Stacking gains are fitted by least squares on the calibration runs, for
personal only and for personal + external. Score: mean test-run r. Target: global HbO (as P7.2).
**Rule: the external model adds** if mean [r(personal + external) − r(personal)] > 0 with bootstrap 95% CI > 0.
**Primary: FineMI.** Reference: MW. Descriptive: the calibrated external model alone (gain fitted on the
calibration runs).

### P8 results — 2026-11-08

**P8.1 smoothed personal delay (5-lag moving average of the P7.1 half curves)**

| Cohort | data | delays within 2 s (chance, perm p) | split-half r (perm p) | median delay [IQR] | personal delay |
|---|---|---|---|---|---|
| **hBCI** | raw | **66% (29%, p 0.0003)** | +0.35 (0.033) | +5 s [2, 8] | **reproducible** |
| mental workload | raw | 23% (21%, p 0.48) | −0.16 (0.78) | +5 s [1, 9.25] | not |
| FineMI | raw | 17% (15%, p 0.53) | +0.25 (0.16) | −2 s [−5.5, 4.5] | not |
| **hBCI** | task removed | **69% (30%, p 0.0001)** | +0.19 (0.16) | +5.5 s [2, 8] | **reproducible** |
| mental workload | task removed | 15% (20%, p 0.80) | −0.17 (0.79) | +4 s [1, 10.25] | not |
| FineMI | task removed | 11% (15%, p 0.79) | +0.19 (0.23) | −1.5 s [−5.75, 4.25] | not |

Rule outcome: smoothing does not rescue the personal delay outside hBCI. The P7.1 caveat stands: in mental
workload and FineMI a person's coupling *kernel shape* repeats (P7.1), but its peak delay does not. In hBCI the
personal delay of about 5–6 s (EEG first) repeats under both estimators.

**P8.2 harmonised CBSI target (frozen hBCI band-power kernel)**

| Direction | global HbO (P7.2 check) | CBSI | CBSI verdict |
|---|---|---|---|
| hBCI -> mental workload (reference) | +0.098 [+0.071, +0.124], beat null 54% | +0.062 [+0.029, +0.096], 19/26 > 0, beat null 31% | fail (null criterion) |
| **hBCI -> FineMI (primary)** | −0.053 [−0.071, −0.034] | **−0.055 [−0.081, −0.032]**, 2/18 > 0, beat null 0% | **FAIL** |

The global-HbO numbers reproduce P7.2. CBSI does not remove the negative transfer into FineMI, so the
different HbO–HbR relation in FineMI does not explain the failure. With CBSI the transfer is weaker even within
the TU Berlin lab, so part of what transfers there is the systemic component.

**P8.3 calibrated transfer of the band-power kernel (global HbO, 2 calibration runs, seed 1337)**

| Cohort | personal (2 runs) | personal + frozen hBCI | frozen hBCI alone, calibrated gain | diff [95% CI] | better | negative external gain | verdict |
|---|---|---|---|---|---|---|---|
| mental workload (reference) | +0.107 | +0.173 | +0.116 | +0.066 [+0.028, +0.110] | 19/26 | 12% | **PASS** |
| **FineMI (primary)** | +0.027 | +0.046 | +0.044 | **+0.020 [+0.004, +0.039]** | 11/18 | **78%** | **PASS** |

Rule outcome: **Claim 2 calibrated-transfer form PASSES across labs** (first cross-lab pass, after P3–P8). What it
means: the frozen hBCI kernel carries information about FineMI haemodynamics that a person's own two-run model
lacks, once one gain per person is set on that person's calibration runs. Limits, to be reported with it:
1. In 78% of FineMI people the fitted external gain is negative, so the kernel transfers *up to its sign*. The sign of the EEG band power -> global HbO relation is inverted in FineMI. That matches P7.2's reliably negative frozen transfer.
2. The effect is small (+0.020) and the CI lower bound is close to 0 (+0.004).
3. One seed was used for the calibration-run choice. Robustness: P8.3b below.

### P8.3b — seed robustness of P8.3 (declared 2026-11-08, after P8.3 and before running it)

P8.3 is repeated unchanged with calibration-run seeds 1–5 (seed 1337 was the primary). Rule: the P8.3 cross-lab
pass is called **robust** if FineMI passes the P8.3 rule for at least 4 of the 5 seeds. MW is reported as reference.

**P8.3b results**

| Seed | FineMI diff [95% CI] | better | negative gain | FineMI | MW diff [95% CI] | MW |
|---|---|---|---|---|---|---|
| 1 | +0.031 [+0.001, +0.071] | 13/18 | 72% | pass | +0.045 [+0.008, +0.088] | pass |
| 2 | +0.015 [+0.004, +0.025] | 15/18 | 78% | pass | +0.027 [−0.013, +0.074] | fail |
| 3 | +0.023 [+0.010, +0.039] | 14/18 | 78% | pass | +0.036 [+0.011, +0.064] | pass |
| 4 | +0.024 [+0.012, +0.040] | 16/18 | 83% | pass | +0.035 [+0.006, +0.065] | pass |
| 5 | +0.020 [+0.004, +0.036] | 16/18 | 89% | pass | +0.026 [−0.004, +0.057] | fail |

Rule outcome: **FineMI passes 5/5 seeds -> the P8.3 cross-lab pass is ROBUST.** The diff is positive in every seed
for both cohorts. The MW reference fails the CI criterion in 2 seeds; MW has only 3 runs per person, so only one
test run. The negative external gain in FineMI (72–89%) is stable across seeds. The sign inversion is a property
of the FineMI cohort, not of the draw.

## P9 — fifth round: why do Claims 3 and 2 fail outside TU Berlin? (declared 2026-11-08, before running any P9 test)

P9 is the fifth round on these claims. P9.1 and P9.2 are **diagnostics**: they say why the earlier tests failed, and
none of them can turn a failed claim into a pass. P9.3 is one confirmatory test with a restriction justified from the
literature. All three are labelled fifth-round in the scorecard. The P6–P8 tests remain the primary record.

Known before P9 (from the P6.4 shards): the local band-power coupling is much weaker outside hBCI. Median peak local
r is hBCI 0.191, mental workload 0.135 and FineMI 0.053.

### P9.1 — Claim 3 diagnostic: is the personal delay failure a recording-length effect?

The hBCI recordings are cut down to the per-person length of the other cohorts, and the P7.1 / P8.1 analysis is
repeated on hBCI alone. Each hBCI person keeps their first `n` runs, with `n` = min(6, median run count of the
target cohort). Each run is cut to (target cohort median total seconds per person) / `n` seconds. There are two
levels, matched to mental workload and to FineMI. The pipeline, lags, nulls (NK = 20) and statistics are the same
as in P7.1 (kernel k vs shifted null, Wilcoxon) and P8.1 (smoothed peak, agreement within 2 s vs permutation).

Interpretation rule:
- If the hBCI personal delay stays reproducible under the P8.1 rule at a matched length, recording length does not
  explain the failure in that cohort. The cause is then weaker coupling, as the peak r above shows, or a delay that
  is really unstable.
- If it does not stay reproducible, recording length alone can explain the failure. The MW / FineMI delay results
  are then **inconclusive (underpowered)** rather than negative.

### P9.2 — Claim 2 diagnostic: which EEG band carries the sign inversion into FineMI?

Each cohort gets its own pooled kernel: band power -> global HbO, the same model as P7.2 and the same hBCI
electrode space. For each kernel, the coefficients are summed per band over lags and electrodes, and the signs are
compared across cohorts. Two transfer scores are also computed:
- the frozen hBCI kernel restricted to each single band, scored hBCI -> FineMI and hBCI -> MW;
- a FineMI leave-one-person-out kernel, scored on held-out FineMI people with the P7.2 score.

The LOPO score tests whether FineMI people share a cross-person relation of their own.

### P9.3 — Claim 2 confirmatory: classical alpha + beta kernel

In the literature, alpha and beta power are the bands that consistently relate inversely to cortical
haemodynamics. Gamma is the band most open to muscle artefact, and theta to ocular artefact. P9.3 therefore
refits the frozen hBCI kernel on **alpha + beta features only**. The restriction is fixed here, before P9.2 is
run, so the P9.2 result cannot choose it.

The primary rule is the P7.2 rule for hBCI -> FineMI: mean r > 0, 95% CI > 0 and more than half of the people beating their null. The secondary rule is the P8.3 calibrated rule for FineMI. MW is reported as the within-lab reference.

### P9 results — 2026-11-08

**P9.2 Claim 2 diagnostic: single-band frozen hBCI kernels (global HbO, P7.2 score)**

| Kernel | hBCI -> mental workload | hBCI -> FineMI |
|---|---|---|
| all bands (P7.2 check) | +0.098 [+0.071, +0.124] | −0.053 [−0.071, −0.034] |
| theta only | +0.074 [+0.039, +0.112] | −0.034 [−0.050, −0.018] |
| alpha only | +0.072 [+0.040, +0.103] | −0.019 [−0.035, −0.002] |
| beta only | +0.074 [+0.037, +0.110] | −0.043 [−0.065, −0.020] |
| gamma only | +0.100 [+0.070, +0.129] | −0.048 [−0.073, −0.023] |
| FineMI leave-one-person-out (FineMI's own kernel) | — | **+0.074 [+0.047, +0.105], 16/18 > 0, 67% beat null** |

The sum of pooled-kernel coefficients per band (theta / alpha / beta / gamma) is:
- hBCI −0.136 / +0.002 / −0.045 / −0.002;
- MW +0.082 / +0.003 / −0.011 / −0.015;
- FineMI +0.052 / −0.096 / +0.024 / +0.212.

The features are strongly correlated, so these sums are only descriptive.

Diagnostic outcome:
1. The inversion into FineMI is **broadband**: every single band transfers with a negative sign into FineMI and with a positive sign into MW. It is not caused by one artefact-prone band. Something shared by all bands differs between the labs, for example broadband EEG power, amplifier or reference, or its relation to systemic haemodynamics.
2. FineMI people **share a cross-person kernel of their own** (LOPO passes the P7.2 rule). Band-power coupling therefore generalises across people within each lab: hBCI LOPO +0.148 (P7.2), FineMI LOPO +0.074. The *kernel itself is lab-specific*.

**P9.3 Claim 2 confirmatory: frozen hBCI alpha + beta kernel**

| Direction | frozen (P7.2 rule) | calibrated (P8.3 rule) | negative external gain |
|---|---|---|---|
| hBCI -> mental workload (reference) | +0.089 [+0.051, +0.128], beat null 54%: pass | +0.065 [+0.024, +0.114]: pass | 23% |
| **hBCI -> FineMI (primary)** | **−0.050 [−0.070, −0.030], beat null 0%: FAIL** | +0.015 [−0.003, +0.035]: **FAIL** | 72% |

Rule outcome: the classical alpha + beta restriction does not rescue frozen cross-lab transfer, and its calibrated
form is weaker than the all-band P8.3 pass. **P9.3 FAILS.** The P8.3 all-band calibrated pass stays the only cross-lab pass.

**P9.1 Claim 3 diagnostic: hBCI cut to the other cohorts' recording length**

Per person, the median recorded length is:
- hBCI 3604 s in 6 runs;
- mental workload 1878 s in 3 runs;
- FineMI 6111 s in 8 runs.

FineMI therefore has *more* data per person than hBCI, and the FineMI-matched level leaves hBCI almost uncut
(6 runs × 1018 s is longer than an hBCI run).

| hBCI cut to | kernel (P7.1): median d, Wilcoxon p | smoothed delay within 2 s (P8.1): agree, chance, p | delay |
|---|---|---|---|
| mental workload length (3 runs × 626 s) | +0.142, p 0.019 | 34%, 19%, p 0.028 | **reproducible** |
| FineMI length (6 runs × 1018 s, about uncut) | +0.743, p < 0.0001 | 66%, 29%, p 0.0001 | **reproducible** |

Rule outcome: the hBCI personal delay stays reproducible at both matched lengths, although it is weaker at the MW length. **Recording length does not explain the delay failures** in MW or FineMI. Under the declared rule, the remaining causes are:
- **weaker coupling.** Median peak local r is hBCI 0.191, MW 0.135 and FineMI 0.053. This is the most likely cause in FineMI, where the coupling is under a third of hBCI's.
- **a personal delay that really is less stable** in those cohorts.

P9.1 cannot tell these two causes apart. The MW and FineMI delay results stand as negative, not underpowered.

### P9.4 — Claim 2 diagnostic: what differs between the labs? (declared 2026-11-08, after P9.1–P9.3, before running it)

This is a diagnostic only; it cannot change any verdict. It has two parts.

(a) **Reverse direction.** The pooled FineMI kernel (same model as P9.2, hBCI electrode space) is scored on hBCI
and on mental workload with the P7.2 score. If it is negative too, the inversion is symmetric between the labs and
not a defect of the hBCI kernel.

(b) **Broadband power.** For each person, the broadband EEG power at lag L is the mean of all z-scored band-power
features at that lag. It is correlated with global HbO for L = 0, 2, …, 14 s (EEG first), and the mean over lags
and runs gives one value per person. The cohort mean has a bootstrap 95% CI.

Prediction declared now: the sign is opposite between TU Berlin (hBCI, MW) and FineMI. If it is, the broadband
power–haemodynamics relation, which is plausibly systemic or tied to the amplifier and reference, explains the
P7.2 / P9.2 inversion.

**P9.4 lab-difference diagnostic**

| | hBCI | mental workload | FineMI |
|---|---|---|---|
| (a) FineMI pooled kernel scored on | −0.029 [−0.056, −0.000], 10/29 > 0 | −0.021 [−0.060, +0.019], 10/26 > 0 | — |
| (b) broadband EEG power vs global HbO, mean r over lags 0–14 s | **−0.038 [−0.055, −0.022]**, 5/29 > 0 | −0.035 [−0.077, +0.008], 9/26 > 0 | **+0.044 [+0.023, +0.067]**, 15/18 > 0 |

Diagnostic outcome: **the declared prediction holds.**
- (a) The inversion is **symmetric**: FineMI's kernel is negative on TU Berlin data, just as the TU Berlin kernel is negative on FineMI data. Neither kernel is defective; the labs differ.
- (b) The overall EEG power level relates to global HbO with **opposite sign** in the two labs: negative at TU Berlin (clearly so in hBCI) and positive in FineMI (CI > 0). That broadband term is the common factor behind the inversion seen in every band (P9.2).

Plausible sources are differences in amplifier and reference, in electrode impedance or motion, or in how systemic physiology (arousal, blood pressure) drives both signals in the two paradigms. These data cannot separate them. Practical consequence: band-power -> haemodynamics kernels must be calibrated per site (P8.3) and cannot be frozen across labs.

## What will NOT be done

- No change of band edges, targets or metric after seeing T1/T2.
- No external-dataset tuning. HEFMI and ds002739 are evaluated only with the configuration chosen here.
