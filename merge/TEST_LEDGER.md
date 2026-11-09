# NeuroVascNet test ledger and multiplicity check

Written 2026-11-08, updated 2026-11-09 after round P10. Every confirmatory test was declared in `FINAL_PLAN.md` before it was run.
This ledger asks which positive results survive when one counts **every attempt** made at the same claim, across
all rounds. Diagnostics (T-series, P9.1, P9.2, P9.4) are not counted as attempts, because they cannot pass a claim.

The correction is a **Bonferroni correction within each claim family**: α = 0.05 / (number of attempts at that claim). Attempts reuse the same data, which makes
this conservative. But an attempt often contains several cohort or condition tests, and counting attempts rather than
individual p-values makes it **lenient** in that respect. It is not a guarantee against forking paths. Results that pass only
nominally are reported as nominal or exploratory.

**Sensitivity, counting cohorts as well as attempts** (attempts × 3 cohorts): Claim 3 α ≈ 0.0012, Claim 5 α ≈ 0.0042. The FineMI
task-removed kernel (p 0.00097) and the mental-workload reliability (p ≈ 0.003) then survive only narrowly and should be
described as borderline. hBCI results (p ≤ 0.0003) and FineMI reliability are unaffected.

## Claim 1 — cross-modal shared space (8 attempts: P2.1, P3.1, P3.1b, P3.1c, P4.1, P5.1, P5.1b, P6.3; α = 0.0063)

| Result | Evidence | Survives? |
|---|---|---|
| Shared space predicts fMRI on unseen people (P5.1, 27 people) | p 0.0001 | **yes** |
| Shared space predicts fNIRS on fresh cohorts at no cost (P5.1) | MW 0.123, FineMI 0.180; 90% CI lower bounds of cost −0.014 / −0.003 > −0.02 | **yes** (non-inferiority margin fixed in advance; CI far from the margin) |
| fMRI supervision improves fNIRS | fails in every cohort, every seed, both targets | — (negative) |

## Claim 2 — external validation

Within lab: 3 attempts (P3.2, P7.2-MW, P8.3-MW reference), α = 0.017. Across labs or studies: 16 attempts
(P2.1, P4.2, P4.A, P4.B, P5.2, P5.2b, P5.2c, P5.6, P6.1, P6.2, P7.2, P8.2, P8.3, P9.3, P10.1 frozen, P10.1 calibrated), α = 0.0031.

| Result | Evidence | Survives? |
|---|---|---|
| hBCI -> mental workload, slow EEG (P3.2) | +0.205 [0.128, 0.285], p 0.0002; reverse p 0.0001 | **yes** |
| hBCI -> MW, band-power kernel (P7.2) | +0.098 [0.071, 0.124] | **yes** (CI far from 0) |
| hBCI -> FineMI, calibrated band-power kernel (P8.3) | +0.020, sign-flip p 0.017 (seed 1337); seeds 1–5: p 0.047 / 0.007 / 0.0014 / 0.0004 / 0.012 | **no — nominal only.** 2 of 6 seed draws reach α; 99.5% bootstrap CI of the primary seed includes 0 (−0.002, +0.047) |
| Every frozen cross-lab transfer | negative or null | — (negative) |
| FineMI people share their own kernel (P9.2, diagnostic) | LOPO +0.074, sign-flip p 0.00006 | diagnostic, not an attempt; would survive |

## Claim 3 — physiological delay / coupling (14 attempts: P2.1-3a, P2.1-3b, P2.3, P2.6, P3.3, P4.3, P4.A, P4.B, P5.2c-lag, P6.4, P6.4b, P7.1, P8.1, P11; α = 0.0036)

| Result | Evidence | Survives? |
|---|---|---|
| Band-power coupling exists, hBCI (P6.4) | 29/29 beat null, group p < 0.0001 | **yes** |
| Coupling kernel reproducible within person, hBCI (P7.1, raw and task removed) | Wilcoxon p < 0.0001 | **yes** |
| Personal delay (EEG first, about 6 s) reproducible, hBCI (P8.1) | p 0.0003 (raw), 0.0001 (task removed) | **yes** |
| Coupling kernel reproducible beyond the task, FineMI (P7.1) | p 0.001 | **yes** |
| Coupling kernel reproducible beyond the task, MW (P7.1) | p 0.006 | **no — nominal** |
| Coupling kernel reproducible, FineMI raw (P7.1) | p 0.033 | **no — nominal** |
| Personal delay split-half, hBCI (P6.4 / P6.4b) | p 0.020 / 0.041 | no on its own; superseded by P8.1 (p 0.0003) |
| Slow EEG follows HbO by 1–2 s, TU Berlin (P3.3) | 29/29 and 26/26; group p 0.006 / 0.004 | group delay direction: **no — nominal** (0.006, 0.004 > 0.0036), and lost in MW under the P4.A amendment; per-person existence (29/29, 26/26) is not in doubt |
| Personal delay outside hBCI | not shown in any form | — (inconclusive: P10.2 shows hBCI also fails at FineMI's coupling strength) |

## Claim 4 — downstream EEG-only task (5 attempts) — refuted in every form; nothing to correct.

## Claim 5 — calibrated reliability (4 attempts: P2.1, P3.5, P5.4, P5.7; α = 0.0125)

| Result | Evidence | Survives? |
|---|---|---|
| Person-level calibrated reliability (P5.4) | Spearman hBCI 0.68 (p ≈ 5e-5), MW 0.56 (p ≈ 0.003), FineMI 0.90 (p < 1e-6) | **yes, all three cohorts** |
| Trial-level reliability | fails | — (negative) |

## P12 — third-lab confirmation (own family: 6 tests, α = 0.0083; n = 15; not used in the main analysis, but an early band-power baseline had been run on it)

| Result | Evidence | Survives? |
|---|---|---|
| Band-power coupling exists | 9/15 beat their null | **yes** (declared rule) |
| Slow EEG predicts HbO and beats band power | slow p 0.013; difference −0.008 | **no — fails; uninformative, see P13** |
| Kernel repeats | p 0.10 | **no** |
| Personal delay repeats | p 0.15 | **no** |
| Frozen hBCI kernel transfers | −0.026 | **no — fails** |
| Calibrated hBCI kernel transfers | −0.037 | **no — fails** |

## P13 — method check (own family: 7 tests, α = 0.0071)

| Result | Evidence | Survives? |
|---|---|---|
| P12.1 method recovers the advantage, hBCI | difference +0.152 [0.086, 0.220] | **yes** |
| P12.1 method, mental workload | +0.061 [−0.018, 0.137] | **no** |
| P12.1 method, FineMI | −0.026 [−0.060, 0.008] | **no** |
| Uncued windows, hBCI | +0.133 [0.099, 0.170], slow p ≈ 5e-6 | **yes** |
| Uncued windows, mental workload | +0.084 [0.043, 0.128], slow p ≈ 5e-6 | **yes** |
| Uncued windows, FineMI | +0.124 [0.082, 0.174], slow p ≈ 5e-6 | **yes** |
| Uncued windows, third lab (exploratory) | slow −0.003, difference −0.007 [−0.023, 0.007] | **no** |

## P14 — stress tests of the claims that hold (own family: 12 tests, α = 0.0042)

| Result | Evidence | Survives? |
|---|---|---|
| Claim 2, hBCI -> MW without the closest pairs | +0.217, p 0.0005 | **yes** |
| Claim 2, no cues | +0.201, p 0.0002 | **yes** |
| Claim 5, no cues (raw and gain, 3 cohorts, 6 tests) | ρ 0.79–0.87, p ≤ 1.4e-5 for raw and ≤ 6.7e-6 for gain | **yes, all 6** (raw and gain are not independent here) |
| Claim 5, third lab (raw and gain) | ρ 0.29 and 0.30, n = 15 | **no** |
| Claim 1, no cues (MW, FineMI) | fNIRS 0.134, 0.175; fMRI 0.065; p 0.0001 | **yes, both** |

## Bottom line for the paper

Results that survive the correction across all attempts:
1. Infra-slow EEG predicts fNIRS across people in three cohorts. These are the founding results (T1/T2, R, P4); they were tested once each and have p ≪ 0.001. The advantage also holds on windows not aligned to cues (P13.2). **In the third lab it is not established: the declared P12.1 method cannot detect it (P13.1), and an exploratory window analysis found nothing (P13.2d).**
2. One EEG-derived shared space serves fNIRS and fMRI (Claim 1, shared-space form).
3. Within-lab external validation (Claim 2, TU Berlin cohorts).
4. Classical band-power -> HbO coupling in hBCI, with a reproducible personal kernel and a reproducible personal delay of about 6 s (EEG first). The kernel is also reproducible beyond the task in FineMI (Claim 3).
5. Person-level reliability from two calibration runs (Claim 5).

Results that must be labelled **nominal** in the paper:
- the cross-lab calibrated transfer (P8.3), which was also not reproduced in the third lab (P12.6);
- MW kernel reproducibility;
- the FineMI raw-data kernel.

Negative results:
- Claim 4;
- fMRI supervision improving fNIRS;
- frozen cross-lab transfer;
- the personal delay outside hBCI is not shown, and is inconclusive rather than negative (P10.2).
