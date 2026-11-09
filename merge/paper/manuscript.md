# Infra-slow EEG is a person-specific, calibratable proxy of cortical haemodynamics that standard preprocessing discards

*Draft v0.1, 2026-11-08. Placeholders are marked [TODO] and citations not yet verified are marked [CITE].*

## Abstract

Predicting the haemodynamic signal from EEG would allow low-cost, portable monitoring of brain metabolism. Its
reported accuracy across people has been close to zero. We show that this failure is largely caused by
preprocessing. In three simultaneous EEG–fNIRS cohorts from two laboratories (29, 26 and 18 people), the EEG
information about oxyhaemoglobin (HbO) lies almost entirely below 0.5 Hz, a band the conventional 0.5–45 Hz
filter removes.

Cross-person prediction of HbO from infra-slow EEG reaches r = 0.14–0.20 in each cohort. The same pipeline with
band power gives r = 0.006–0.045, and with the conventional filter r = 0.019. The signal is not ocular, it is
strongly person-specific, and it keeps a local component after the systemic (global) haemodynamic signal is
removed.

In a further cohort from a third laboratory (15 people, rules declared in advance), slow EEG was no better than band power (r = 0.057 vs 0.065), but that test does not recover the advantage in two of the three established cohorts, so it is uninformative. An exploratory window analysis, which does recover it in those three cohorts, found nothing in the third laboratory. Whether the infra-slow advantage extends to that laboratory is not established. Coupling was present there, but the frozen and calibrated models did not transfer.

We find four results for practical use and for physiology:
- One EEG-derived linear space predicts both fNIRS and fMRI (27 people) in people not seen in training, at no cost to either output.
- Two calibration runs of simultaneous EEG–fNIRS are enough to monitor a person with EEG alone, beating that person's own average response in all three cohorts. The calibration-run score predicts how well this will work later (Spearman ρ 0.56–0.90).
- Classical neurovascular coupling is reproducible within person: EEG band power predicts local HbO about 6 s later in all 29 people of the largest cohort, and the coupling persists after the task response is removed.
- The coupling kernel generalises across people within a laboratory but not between laboratories, where its sign is inverted in all frequency bands.

Pass rules were written into a dated internal analysis plan before each test was run; this was not a public preregistration, and tests amended after a first look are labelled. We report all attempts, including
negative results: fMRI supervision does not improve fNIRS prediction, the learned haemodynamic code gives no
benefit for downstream decoding, and models frozen in one laboratory do not transfer to another. We apply
correction for multiplicity across all attempts.

## 1. Introduction

[TODO: 4 paragraphs]
1. Multimodal EEG–haemodynamic fusion; the promise of "virtual haemodynamics" from EEG alone [CITE].
2. Prior cross-modal prediction (EEG -> fNIRS / BOLD) reports small effects that are mostly within person [CITE]. Standard EEG preprocessing applies a 0.5–1 Hz high-pass filter [CITE].
3. Infra-slow (< 0.1–0.5 Hz) scalp potentials relate to BOLD, to vascular and blood–brain-barrier potentials, and to arousal [CITE]. They are rarely used, because DC-coupled amplifiers are needed.
4. Contribution: a multi-cohort test of five claims under a dated internal analysis plan; what holds, what does not, and why.

## 2. Methods

### 2.1 Data

| Cohort | People | Paradigm | Recording | Source |
|---|---|---|---|---|
| hBCI (TU Berlin) | 29 | motor imagery + mental arithmetic, 6 runs | EEG 30 ch + fNIRS 36 ch | Shin et al. 2017 [CITE] |
| Mental workload (TU Berlin) | 26 | n-back, 3 runs | EEG + fNIRS | Shin et al. 2018 [CITE] |
| FineMI (independent lab) | 18 | multi-joint upper-limb motor imagery, 8 runs | EEG (Neuroscan) + fNIRS (NIRx) | [CITE] |
| ds002336 / ds002338 | 10 + 17 | motor-imagery neurofeedback | EEG + fMRI | Lioi et al. 2020 [CITE] |

The two TU Berlin cohorts may share participants. The public files do not allow a check, so we never treat them as fully independent samples.

Two further EEG–haemodynamic datasets (HEFMI, ds002739) were excluded before analysis. Their published EEG contains no power below 0.5 Hz, because of a hardware filter and pre-publication filtering respectively.

### 2.2 Signals

**EEG.**
- *Infra-slow features:* EEG without a high-pass filter, low-passed below 0.5 Hz. [TODO: exact filter and resampling from `run_p2_full.py`.]
- *Band-power features:* log power in theta (4–8 Hz), alpha (8–13), beta (13–30) and gamma (30–45), in 1 s bins, band-passed 0.01–0.1 Hz.
- *Electrode space:* for analyses across cohorts, EEG is mapped onto the hBCI montage by spherical-spline interpolation.

**fNIRS.** HbO and HbR are obtained by the modified Beer–Lambert law and band-passed 0.01–0.1 Hz.
- *Global target:* the channel mean.
- *Local target:* channels minus the channel mean. It removes the systemic component and was tested separately for every claim that says "local".

### 2.3 Models and validation

All main analyses use ridge regression; regularisation is chosen by internal cross-validation. Validation follows the claim being tested:
- *cross-person prediction:* leave-one-person-out;
- *within-person prediction:* leave-one-run-out;
- *transfer between cohorts:* a model trained on all people of one cohort and frozen.

**Nulls.** Chance is estimated by circular time shifts of the haemodynamic signal, with shifts of at least 60 s. A person beats their null when their observed score exceeds the 95th percentile of their own surrogates.

**Coupling and delay.** We use a lagged ridge model with lags from −10 to +20 s, where a positive lag means EEG leads. Every lag gets its own model, scored on held-out runs. A person's delay is the peak of this lag curve.

**Kernel reproducibility (P7.1).** For each person we split each recording into halves and correlate the half-A and half-B lag curves. This value is compared with 20 surrogates in which half B is circularly shifted.

**Task-response removal.** The cue-locked average response of each condition is subtracted from both signals before the analysis.

### 2.4 Analysis plan and multiplicity

Each test's pass rule was written into a dated analysis plan before the test was run. Tests rerun after a first look are labelled as amendments.

The five claims were attacked over up to five rounds. We therefore apply a Bonferroni correction within each claim family over **all** attempts at that claim (8, 16, 14, 5 and 4 attempts). Results that pass only nominally are reported as such.

## 3. Results

### 3.1 The EEG information about haemodynamics is infra-slow (Fig. 1)

In hBCI, the conventional 0.5–45 Hz preprocessing gave a cross-person prediction of HbO of r = 0.019. Without it,
the same deep pipeline gave r = 0.155, and ridge regression on infra-slow features gave r = 0.197. The result
replicated in the two other cohorts:
- mental workload: r = 0.135 [0.089, 0.183];
- FineMI, an independent lab: r = 0.203 [0.164, 0.248].

Band power from 1–45 Hz carried little: r = 0.02, 0.006 (n.s.) and 0.045.

Three properties replicated in all cohorts:
1. EOG channels carried little or no information.
2. A person's own model identified their data far better than other people's models did (identification accuracy 93%, 62% and 83%).
3. A local component survived removal of the global signal (local HbO r = 0.07–0.16).

### 3.2 One EEG space for fNIRS and fMRI (Claim 1)

We fixed a linear shared latent of rank 32 in advance, using hBCI only. We then tested it on fNIRS cohorts never
used for this claim, together with 27 fMRI participants.
- The shared space predicted fNIRS in unseen people: mental workload r = 0.123, FineMI r = 0.180.
- It predicted fMRI in unseen people: r = 0.063–0.068, p = 0.0001.
- Neither output paid a cost. The lower bounds of the 90% CI were −0.014 and −0.003, both above the preset margin of −0.02.
- The result held for 5/5 person-fold seeds and on the local target.

Supervision by fMRI did **not** improve fNIRS prediction. A joint model was no better than a joint model trained
on shuffled fMRI targets, in all cohorts and even with only five training people.

### 3.3 Calibrated EEG-only monitoring and reliability known in advance (Claims 2 and 5; Fig. 5)

After two calibration runs of simultaneous recording, a person's EEG-only model beat replaying that person's own
mean response:
- hBCI: +0.176 [0.135, 0.218], 29/29 people;
- mental workload: +0.059 [0.025, 0.099];
- FineMI: +0.067 [0.033, 0.112], 18/18 people.

The gain persisted on the local target (+0.103, +0.031, +0.041). It grew with more calibration runs (hBCI: +0.118 at k = 1, +0.258 at k = 5).

The score on the two calibration runs predicted how well the model would work on the person's later runs:
- Spearman ρ = 0.68 [0.37, 0.86] in hBCI;
- ρ = 0.56 [0.22, 0.78] in mental workload;
- ρ = 0.90 [0.69, 0.97] in FineMI.

This held for 5/5 seeds of the calibration-run choice and survived correction. Reliability at the level of single trials did not hold.

### 3.4 Within-lab external validation (Claim 2)

Models frozen on hBCI predicted mental-workload participants, a different task in the same lab:
- infra-slow EEG: r = 0.205 [0.128, 0.285], p = 0.0002;
- reverse direction: r = 0.177, p = 0.0001;
- band-power coupling kernel: r = 0.098 [0.071, 0.124].

All three survive correction.

Transfer to the independent lab (FineMI) failed under every harmonisation we tried:
- detrending;
- re-referencing;
- Euclidean alignment;
- emulation of the TU Berlin hardware filter;
- paradigm-window and montage matching;
- a CBSI target.

A calibrated variant gave a small gain (+0.020 [+0.004, +0.039]; 5/5 seeds positive). It is **nominal**: p = 0.017, which does not survive correction for 16 cross-lab attempts.

### 3.5 Classical neurovascular coupling, reproducible within person (Claim 3; Figs. 2–3)

EEG band power predicted local HbO with EEG leading (Fig. 2). In hBCI, all 29 people beat their own null, with
a median delay of +6 s, the textbook neurovascular order.

The personal delay was reproducible between the two halves of a recording:
- smoothed peaks agreed within 2 s in 66% of people, against 29% by chance (p = 0.0003);
- with the cue-locked task response removed: 69%, p = 0.0001.

The coupling is therefore ongoing, not a shared task response.

The *shape* of a person's coupling kernel was reproducible beyond the task in all three cohorts (Fig. 3):
- hBCI: p < 0.0001;
- FineMI: p = 0.001;
- mental workload: p = 0.006, nominal after correction.

Outside hBCI the peak delay itself did not repeat. This is not explained by recording length. Cut to the length
of a mental-workload recording, hBCI still showed a reproducible delay (34% vs 19% chance, p = 0.03), and FineMI
has more data per person than hBCI. The coupling was weaker outside hBCI: median peak r was 0.19 in hBCI, 0.14 in
mental workload and 0.05 in FineMI. When hBCI recordings were degraded with added noise to FineMI's coupling strength, their delay no longer repeated (10% vs 16% chance). The FineMI delay result is therefore inconclusive rather than negative.

Infra-slow EEG itself followed HbO by 1–2 s in both TU Berlin cohorts (29/29 and 26/26 people). That is the
reverse of the neurovascular order and consistent with a vascular or blood–brain-barrier origin of the
infra-slow potential [CITE]. The group-level direction is nominal after correction.

### 3.6 The coupling kernel is lab-specific (Fig. 4)

Within each lab, a band-power kernel learned from other people predicted a new person:
- hBCI: r = 0.148;
- FineMI: r = 0.074 [0.047, 0.105], p = 0.00006.

Between labs, the frozen hBCI kernel was reliably *negative* in FineMI (r = −0.053 [−0.071, −0.034]). The
inversion was broadband: each of theta, alpha, beta and gamma alone transferred negatively into FineMI and
positively into the second TU Berlin cohort. It is therefore not an artefact of one band.

The inversion was symmetric. A kernel learned on FineMI was negative on hBCI (r = −0.029 [−0.056, −0.000]).
A candidate common factor is the overall EEG power level: averaged over all bands and electrodes, EEG power correlated
negatively with global HbO at TU Berlin (hBCI −0.038 [−0.055, −0.022]) and positively in FineMI
(+0.044 [+0.023, +0.067], 15/18 people). These correlations are small, the mental-workload estimate (−0.035 [−0.077, +0.008]) includes zero, and the prediction was written after the inverted kernels had been seen, so we tested it directly. Removing the broadband term reduced the inversion (paired difference +0.017 [+0.007, +0.027]) but did not remove it (r = −0.036 [−0.053, −0.021]). The broadband term therefore explains only part of the inversion. The
person-specific coupling shape is shared within each lab. The sign of the broadband term, which is plausibly
systemic, tied to the amplifier or reference, or tied to the paradigm, differs between labs.

### 3.7 No downstream benefit (Claim 4)

The learned "virtual haemodynamics" did not improve EEG-only decoding beyond controls. This held in the
standard, data-scarce and representation-only forms. A PCA code of the same infra-slow EEG did as well or better
(hBCI motor imagery: PCA 0.760 vs virtual haemodynamics 0.697). The benefit comes from the infra-slow band itself:
band power gives 0.61, infra-slow EEG features 0.70–0.76.

### 3.8 Confirmation in a third laboratory

All rules were fixed in advance on this cohort, using no slow-EEG result from it; an early band-power baseline on these data (r about 0.00) did not inform them (implicit-learning task, Nihon Kohden EEG, Hitachi fNIRS; 15 of 30 people passed the sync and length rules). One of six tests passed (α = 0.0083). Band-power coupling was present (9/15 people beat their null; median delay +7 s, EEG first). Slow EEG predicted global HbO weakly (r = 0.057 [0.014, 0.103], p = 0.013, above α) and was no better than band power (r = 0.065; difference −0.008 [−0.075, +0.052]). Kernel reproducibility (p = 0.10) and delay reproducibility (33% vs 19% chance, p = 0.15) failed, as did frozen (r = −0.026) and calibrated (−0.037) transfer of the hBCI kernel.

We then asked, in a separately declared test, whether the continuous method used for the slow-EEG comparison can recover the advantage where it is established. It did so in hBCI (difference +0.152 [0.086, 0.220]) but not in mental workload (+0.061 [−0.018, 0.137]) or FineMI (−0.026 [−0.060, 0.008]), so the third-laboratory comparison is uninformative. The window-based analysis of the founding results, applied to consecutive 30 s windows not aligned to any cue, did recover the advantage in all three cohorts (+0.133 [0.099, 0.170], +0.084 [0.043, 0.128], +0.124 [0.082, 0.174]; α = 0.0071). In the third laboratory it found no prediction from either feature set (slow r = −0.003 [−0.015, 0.009]; band power r = +0.004); this analysis is exploratory because the cohort had already been seen.

## 4. Discussion

[TODO: expand]
- **Main message.** The haemodynamic information in scalp EEG sits in a band that standard preprocessing removes. Cross-modal work that high-pass filters EEG at 0.5–1 Hz cannot find it.
- **Two signals, two orders.** Infra-slow potentials *follow* HbO by 1–2 s; band power *leads* local HbO by about 6 s. This fits two mechanisms: a vascular or BBB potential that tracks blood volume, and classical neurovascular coupling.
- **Personalisation is the practical route.** Cross-person and cross-lab models are weak or fail. Two calibration runs give EEG-only monitoring whose reliability is known in advance.
- **Lab specificity.** The inversion of the coupling kernel is broadband and symmetric. An opposite-signed relation between overall EEG power and global HbO in the two labs accounts for only part of it; the remaining band-power pattern is inverted as well. It points to differences in hardware, reference or systemic physiology, not to one frequency band. Kernels should be calibrated per site.
- **Limitations:**
  - the infra-slow advantage held in two labs; in a third (15 people, different task and hardware) it is not established;
  - a possible participant overlap between the TU Berlin cohorts;
  - the fMRI arm comes from one study family;
  - most results use one seed (ridge analyses are deterministic up to bootstrap resampling);
  - five rounds of tests (handled by correction across attempts);
  - DC-coupled EEG is required.

## 5. Data and code availability

All datasets are public [CITE each]. The analysis scripts, the dated analysis plan (`FINAL_PLAN.md`), the
multiplicity ledger (`TEST_LEDGER.md`) and the raw result files are in the project repository [TODO: archive DOI].

## Figures

1. `figures/fig1_slow_band` — cross-person prediction of HbO from infra-slow EEG vs band power, three cohorts.
2. `figures/fig2_coupling_lag` — band power -> local HbO lag curves (mean ± SEM); positive lag = EEG leads.
3. `figures/fig3_kernel_repro` — within-person kernel reproducibility, raw and with the task response removed.
4. `figures/fig4_kernel_transfer` — where the coupling kernel generalises: within person-pool, within lab, across labs.
5. `figures/fig5_deployment_reliability` — calibrated EEG-only monitoring gain and person-level reliability.
