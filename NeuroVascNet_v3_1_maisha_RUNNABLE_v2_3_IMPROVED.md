# NeuroVascNet v3.1-v2.3 (Maisha's pipeline): full results, five seeds

**Notebook:** `NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED.ipynb`
**Model version:** `v3.1-v2.3-personalized-physio-fixed-delay`
**Seeds run:** 1337, 20260726, 7, 42, 2718 (all five complete, each in a fresh kernel)
**Sources for this document:** the notebook's own outputs (the saved notebook shows the last run, seed 2718), the five per-seed summary files `output/summary_..._seed<seed>.json`, the executed copy of every seed in `output/executed_seed_notebooks/`, and the five-seed aggregate `output/multi_seed_summary_v3.1-v2.3-personalized-physio-fixed-delay.json`. Comparisons with Muntasir's final pipeline use `NeuroVascNet_Research_Report_v2.pdf` and `neurovascnet_v6_outputs/`.

---

## How to read the numbers

- **r (correlation)** goes from −1 to +1. **0 means no link at all.** 0.1 is a weak link, 0.3 moderate, 0.5 strong. In this project almost every useful number is between 0 and 0.2.
- **Null** is the score you get by chance, when the predictions are randomly re-paired with the wrong trials. A result only counts if it is clearly above its null.
- **p-value** is how likely the result is to be chance. Below 0.05 is usually called "significant". A tiny effect can be significant and still useless, so the notebook also labels every effect by size ("practically negligible", "very small"…).
- **Subjects, not trials, are the real sample size.** 540 trials from 7 people count as 7 independent observations.
- **Five seeds** means the whole training was repeated five times with different random starting points. If a result changes a lot between seeds, it is not stable.

---

## 1. The short version

**What works (positive):**
1. **EEG and fNIRS are linked inside each person.** When a model is trained and tested on the same person, it predicts that person's fNIRS above chance (r = 0.094 vs chance 0.021, p = 0.005, 30 people).
2. **The data checks and safety rules work.** The notebook caught a broken data split (only patients in validation), turned off a harmful cleaning step (global signal regression), checked units, checked timing, and blocked claims the evidence could not support.
3. **The reliability score ranks predictions within the same dataset** (correlation 0.42 on average over five seeds, range 0.25–0.50).
4. **Personalisation raises accuracy** for a new person (r from 0.035 to 0.191 using 20% of that person's data).

**What does not work (negative):**
1. **Predicting new people on HEFMI does not beat a simple shortcut.** The model (r = 0.026) is no better than just predicting each person's own average (r = 0.029).
2. **Adding fMRI training does not help fNIRS prediction.** A model trained on fNIRS alone was better in 4 of 5 seeds.
3. **Nothing transfers to other datasets.** TU Berlin: r = −0.002; ds002739: r = 0.006 (five-seed averages).
4. **Most of the personalisation gain is not from EEG.** Using only the person's own average response, with no EEG, already gives r = 0.160; EEG adds about +0.03, which is not statistically significant.
5. **The model cannot fit real EEG–fNIRS pairs better than shuffled ones** at small scale.
6. **The delay output is not a real measurement.** It is fixed at 5 s by design and does not follow each person's measured delay.
7. **The abstention rule is unstable.** On the external dataset it refused almost everything in two seeds and accepted everything in another, with 97% of those answers wrong.

**What is unclear:**
- Timing between EEG and fNIRS could not be confirmed for three of the six data splits, so timing-based claims are blocked.
- fMRI results rest on 7 training and 3 validation people, too few for firm conclusions.

---

## 2. What the notebook does

### 2.1 Data and roles

| Role | Dataset | Signal | People | Trials (30 s windows) |
|---|---|---|---|---|
| fNIRS training | HEFMI-ICH | EEG + fNIRS (90 channels, HbO + HbR) | 30 | 2,400 |
| fNIRS validation | HEFMI-ICH | same | 7 | 540 |
| External fNIRS test | TU Berlin hBCI | EEG + fNIRS (36 real channels) | 29 | 3,480 |
| fMRI training | ds002336 | EEG + fMRI (48 brain regions) | 7 | 305 |
| fMRI validation | ds002336 | same | 3 | 124 |
| External fMRI test | ds002739 (perceptual task, not motor) | EEG + fMRI | 24 | 2,448 |

- The HEFMI validation people were chosen **within each clinical group** (3 of 17 controls, 4 of 20 haemorrhage patients) with a fixed random seed (20260807). The old alphabetical split had put only patients into validation; that run is marked invalid.
- The notebook verified there is no overlap of people between training and validation.
- All datasets use the same 32 EEG electrodes, checked by name; the motor electrodes (C3, C4, Cz and neighbours) are present.
- ds002336: 52 EEG–fMRI pairs kept; 1 EEG-only and 3 fMRI-only recordings dropped because their partner file is missing. ds002739: 48 pairs.

### 2.2 Preprocessing choices

- **Target = response amplitude**, kept in physical units (no per-trial z-scoring of the target).
- **Baseline** = first 20% of each window, before the cue.
- **Global signal regression (GSR): OFF.** The notebook's own A/B test (on training data only) found GSR removed 40–50% of the fNIRS signal (five seeds: 40%, 42%, 46%, 50%, 40%), and the removed part was only moderately uniform across channels (r ≈ 0.46). It was removing real, spatially structured signal, not just body noise.
- **Filters:** fNIRS 0.01–0.2 Hz, fMRI 0.008–0.15 Hz, EEG 0.5–45 Hz with a 50 Hz notch.
- **EEG input:** band power, 4 frequency bands × 32 electrodes = 128 channels at 20 samples per second (600 per window).
- **EEG normalisation:** statistics from training people only, applied unchanged to everyone else (no peeking at test data).
- **Units:** TU Berlin converted from millimolar to molar; scale check passed (external/train ratio 9.57, inside the allowed range).

### 2.3 Model and training

- Shared EEG encoder (EEGNet front-end + temporal convolution network), **142,922 parameters**, separate normalisation statistics for the fNIRS and fMRI datasets, equal training steps for both (19 per epoch), 15 epochs.
- Two output heads: fNIRS (180 signals × 300 time points) and fMRI (48 regions × 15 time points), each predicting a value and its uncertainty.
- **Turned off in this version:** stage-2 cross-modal alignment (no benefit), the learned delay (replaced by a fixed 5 s), downstream transfer test and ablations (no benefit shown earlier).
- Reliability: learned gate → temperature scaling → isotonic calibration, fitted on 4 "calibration" validation people and tested on the other 3.

---

## 3. Checks before any model (data quality)

### 3.1 Timing check (does the blood response peak at a sensible delay after EEG?)

| Split | People | Median peak delay | % people inside 2–8 s | Status |
|---|---|---|---|---|
| HEFMI fNIRS (train) | 30 | 5.2 s | 50% | PASS |
| HEFMI fNIRS (val) | 7 | 6.3 s | 14% | **WARN** |
| TU Berlin fNIRS (external) | 29 | 9.7 s | 24% | **WARN** |
| ds002336 fMRI (train) | 7 | 8.0 s | 71% | PASS |
| ds002336 fMRI (val) | 3 | 10.0 s | 33% | **WARN** |
| ds002739 fMRI (external) | 24 | 6.0 s | 62% | PASS |

**Meaning:** in three splits most people's best-matching delay falls outside the normal 2–8 s range. That can happen if the event clocks are slightly offset. Until trigger timing is checked against the raw marker files, the notebook blocks any claim that depends on timing. **(Negative / unresolved.)**

### 3.2 Claim guard

Every seed passed the automatic claim guard with one restriction: *"HEFMI timing QA did not pass in both train and validation; alignment-based claims blocked."*

---

## 4. Results in detail

### 4.1 Main endpoint: compact response vs simple baselines (HEFMI validation, 7 people)

The main target in this version is four numbers per trial: mean HbO, mean HbR, HbO peak and HbR trough, inside a 5.2–15.2 s window chosen on training people only.

| Predictor | r | 95% range | Error (MAE) |
|---|---|---|---|
| Training average (no EEG) | −0.063 | −0.18 to +0.04 | 0.686 |
| Average for each task condition (no EEG) | **+0.058** | −0.06 to +0.19 | **0.685** |
| Ridge regression on EEG band power | +0.045 | −0.10 to +0.18 | 0.757 |

**Meaning:** the EEG model does **not** beat the simple "average for this task" guess; its error is actually 0.07 units worse (p = 0.06). **(Negative.)** These numbers are the same in every seed because this analysis does not use the neural network.

### 4.2 Personalisation: using part of a new person's own data

| Share of the new person's data used to calibrate | General model (no personalisation) | Person's own average, **no EEG** | Personalised model **with EEG** | p (EEG model vs general) |
|---|---|---|---|---|
| 10% | 0.028 | 0.167 | 0.195 | 0.15 |
| 20% | 0.035 | 0.160 | 0.191 | 0.22 |
| 40% | 0.038 | 0.180 | 0.206 | 0.24 |

**Meaning:** personalising looks like a big jump (0.03 → 0.19), but almost all of it comes from simply knowing the person's average level. EEG adds only about **+0.03**, and with 7 people none of the differences is statistically significant. **(Weak.)**

### 4.3 Predicting the full blood-signal time course (five seeds)

| Test | Seed 1337 | 20260726 | 7 | 42 | 2718 | **Mean ± SD** | Chance level (mean) | Verdict |
|---|---|---|---|---|---|---|---|---|
| HEFMI validation (7 people) | 0.024 | 0.034 | 0.040 | 0.020 | 0.026 | **0.029 ± 0.008** | 0.021 | barely above chance; "practically negligible" |
| ds002336 fMRI validation (3 people) | 0.099 | 0.076 | 0.049 | 0.097 | 0.100 | **0.084 ± 0.022** | 0.088 | at chance |
| TU Berlin (external, 29 people) | −0.002 | −0.004 | −0.004 | −0.005 | +0.003 | **−0.002 ± 0.003** | — | zero |
| ds002739 (external, 24 people) | 0.009 | 0.000 | 0.010 | 0.007 | 0.007 | **0.006 ± 0.004** | — | statistically above zero in 4 of 5 seeds, but practically zero |

**Meaning:** across all five repeats, the model's predictions of new people's signals are essentially chance. **(Negative, stable across seeds.)**

### 4.4 Size of the response (amplitude) instead of the full shape — HEFMI validation

Three questions are separated: Q1 = does the model rank *people* by their average response? Q2 = within a person, does it track which trials are bigger? Q3 = does Q2 hold for people it has never seen?

| | 1337 | 20260726 | 7 | 42 | 2718 | Mean |
|---|---|---|---|---|---|---|
| Q2 within-person (p) | 0.029 (0.16) | −0.028 (0.33) | 0.048 (0.10) | **0.116 (0.02)** | **0.078 (0.05)** | 0.049 |
| Q3 new people, own mean removed | 0.042 | −0.003 | 0.046 | 0.118 | 0.104 | 0.062 |
| Q1 ranking people | −0.35 | −0.78 | −0.19 | −0.13 | −0.76 | −0.44 |

**Meaning:** a small amplitude signal appears in some seeds (significant in 2 of 5) but disappears in others. With only 7 validation people this is **not stable enough to claim.** **(Unclear.)** On the fMRI *training* data the model ranks people well (Q1 between 0.46 and 0.98) but tracks trials only weakly (Q2 ≈ 0.20–0.25): it mostly learned **who** the person is, not **what** their brain is doing.

### 4.5 Can the network learn real EEG–fNIRS pairs at all? (capacity test)

The network is trained on a few windows with the real pairing, and again with the pairing shuffled. If EEG really carries information, real pairs should be learned better.

| Seed | 1337 | 20260726 | 7 | 42 | 2718 |
|---|---|---|---|---|---|
| Real minus shuffled (at 128 windows) | +0.004 | −0.004 | +0.014 | −0.006 | −0.003 |

**Meaning:** real and shuffled pairs are learned equally well. At this scale, the network finds no usable EEG→fNIRS structure. **(Negative.)**

### 4.6 Inside one person (within-subject coupling)

| Dataset | r | Chance | p | People | Verdict |
|---|---|---|---|---|---|
| HEFMI fNIRS | **0.094** (95% range 0.067–0.127) | 0.021 | **0.005** | 30 | **real coupling** |
| ds002336 fMRI | 0.130 (0.021–0.273) | 0.118 | 0.30 | 7 | not distinguishable from chance |

**Meaning:** EEG and fNIRS **are** linked inside each person. The link exists but is personal: it does not carry over to other people. **(Positive.)**

### 4.7 Leave-one-person-out (strict test on new people)

| Dataset | People | Model r | Shortcut: predict each person's own average | Difference | Verdict |
|---|---|---|---|---|---|
| HEFMI fNIRS | 30 | 0.026 | 0.029 | −0.003 (p = 0.67–0.70) | **no better than the shortcut**; enough people to trust this |
| ds002336 fMRI | 7 | 0.055 | 0.189 | −0.134 (p = 0.013–0.015) | **worse than the shortcut**; too few people for a firm claim |

**(Negative.)**

### 4.8 Does adding fMRI help? (cross-modal model vs fNIRS-only model, same training budget)

| Seed | Cross-modal r | fNIRS-only r | Better | Cross-modal timing excess | fNIRS-only timing excess | Task decoding (cross / fNIRS-only) |
|---|---|---|---|---|---|---|
| 1337 | 0.024 | 0.071 | fNIRS-only | 0.004 | 0.044 | 0.456 / 0.502 |
| 20260726 | 0.034 | 0.023 | cross-modal | 0.003 | 0.017 | 0.507 / 0.522 |
| 7 | 0.040 | 0.051 | fNIRS-only | 0.012 | 0.031 | 0.507 / 0.476 |
| 42 | 0.020 | 0.063 | fNIRS-only | 0.009 | 0.031 | 0.513 / 0.483 |
| 2718 | 0.026 | 0.053 | fNIRS-only | 0.006 | 0.025 | 0.496 / 0.476 |
| **Mean** | **0.029** | **0.052** | fNIRS-only in 4/5 | **0.007** | **0.030** | 0.496 / 0.492 |

**Meaning:** adding fMRI to training makes fNIRS prediction **worse**, not better, in 4 of 5 seeds, and worse on the timing measure in all 5. Task decoding is at chance (≈ 0.50) for both. **(Negative, replicated across seeds.)**

### 4.9 Reliability score, calibration and abstention

| Seed | Calibration people | Ranking: internal test people | Ranking: external | Calibration error, internal (raw → calibrated) | Calibration error, external (raw → calibrated) | Locked threshold | External windows answered |
|---|---|---|---|---|---|---|---|
| 1337 | 10, 18, 20, 24 | 0.46 | −0.20 | 0.058 → 0.081 | 0.235 → 0.177 | 0.89 | 0.9% |
| 20260726 | 6, 10, 18, 35 | 0.50 | −0.22 | 0.083 → 0.073 | 0.180 → 0.085 | 0.73 | 0% |
| 7 | 5, 10, 24, 35 | 0.25 | −0.07 | 0.168 → 0.138 | 0.211 → 0.192 | 0.80 | 9.1% |
| 42 | 10, 18, 20, 35 | 0.43 | −0.23 | 0.080 → 0.076 | 0.024 → 0.028 | 0.54 | 0% |
| 2718 | 5, 10, 20, 35 | 0.46 | −0.05 | 0.104 → 0.101 | 0.858 → 0.886 | 0.62 | **100%** (97% of answers wrong) |
| **Mean** | | **0.42 ± 0.10** | **−0.15 ± 0.08** | | | | |

**Meaning:**
- **Positive:** inside HEFMI, the score does put better predictions ahead of worse ones (0.42 on average, positive in every seed).
- **Negative:** on the external dataset the ranking is reversed in every seed (the score is most confident when it is most wrong).
- **Unstable:** which 4 people are used to fit the calibration changes with the seed, and so does everything downstream. On TU Berlin the locked rule answered 0% of windows in two seeds and 100% in another, where 97% of answers were outside the error tolerance and calibration error rose to 0.89. The abstention layer cannot be trusted on new datasets.

### 4.10 Delay

The delay output is fixed at **5.00 s** for everyone in this version (the learned delay was switched off). Compared with each person's delay measured directly from the data, it does no better than a constant guess:

| Group | People | Measured delay (mean) | Model delay | Verdict |
|---|---|---|---|---|
| HEFMI validation | 7 | 5.9 s | 5.00 s | not validated |
| TU Berlin | 29 | 8.1 s | 5.00 s | not validated |
| ds002336 validation | 3 | 10.0 s | 5.00 s | too few people |
| HEFMI training | 30 | 5.2 s | 5.00 s | not validated |

**Meaning:** report it only as a "bounded model timing parameter", never as a neurovascular delay. **(Negative.)**

### 4.11 Not tested in this version

- **Downstream transfer** (is the learned EEG representation useful for other tasks?) — disabled: earlier versions showed no benefit over a fairly trained comparison.
- **Ablations** (removing parts of the model one at a time) — disabled for the same reason.

---

## 5. Five-seed summary table (from the notebook's last cell)

| Metric (plain words) | Mean | SD across seeds | Min | Max |
|---|---|---|---|---|
| HEFMI validation prediction | +0.029 | 0.008 | +0.020 | +0.040 |
| fMRI validation prediction | +0.084 | 0.022 | +0.049 | +0.100 |
| TU Berlin external prediction | −0.002 | 0.003 | −0.005 | +0.003 |
| ds002739 external prediction | +0.006 | 0.004 | +0.000 | +0.010 |
| HEFMI leave-one-person-out | +0.026 | 0 | | |
| …its shortcut (person's own average) | +0.029 | 0 | | |
| fMRI leave-one-person-out | +0.055 | 0 | | |
| Ridge on compact endpoint | +0.045 | 0 | | |
| Personalised with EEG (20%) | +0.191 | 0 | | |
| Person's own average, no EEG (20%) | +0.160 | 0 | | |
| General model (20%) | +0.035 | 0 | | |
| Within-person HEFMI | +0.094 | 0 | | |
| Within-person fMRI | +0.130 | 0 | | |
| Reliability ranking, internal | +0.42 | 0.10 | +0.25 | +0.50 |
| Reliability ranking, external | −0.15 | 0.08 | −0.23 | −0.05 |

**Why some rows have SD = 0:** leave-one-person-out, ridge, personalisation and within-person analyses do not use the neural network's random seed, so they give identical results in every run. The five seeds measure run-to-run variation only for the neural-network rows.

---

## 6. All findings sorted

### Positive
- Real EEG–fNIRS coupling inside each person (r = 0.094, p = 0.005, 30 people).
- Reliability score ranks predictions within the training dataset (0.42 average, positive in all 5 seeds).
- Personalisation improves prediction (0.035 → 0.191), mostly by learning the person's own average.
- Strong protocol safeguards: clinically balanced split, subject-level statistics, no test-data leakage in normalisation, unit checks, timing checks, a claim guard, per-prediction audit files, five-seed repetition.
- GSR correctly identified as harmful and switched off (removes 40–50% of the signal, part of it real).
- The ds002336 and ds002739 loaders, pairing and channel alignment pass their checks.

### Negative
- Prediction for new people on HEFMI is no better than knowing each person's average (0.026 vs 0.029).
- fMRI supervision does not help; fNIRS-only training is better in 4 of 5 seeds.
- No transfer to TU Berlin (−0.002) or ds002739 (0.006).
- Compact endpoint: EEG ridge does not beat the task-average guess.
- The network cannot learn real pairs better than shuffled pairs.
- The delay output does not follow measured delays.
- The reliability score reverses on new datasets, and the abstention rule behaves unpredictably there.

### Unclear or unstable
- Amplitude tracking within a person on new people (significant in 2 of 5 seeds only).
- fMRI results (3–7 people).
- EEG–fNIRS timing in three splits (WARN).

---

## 7. Maisha's pipeline vs Muntasir's pipeline

| | **Muntasir (final v6)** | **Maisha (v3.1-v2.3)** |
|---|---|---|
| fNIRS training data | TU Berlin hBCI (29 people) | HEFMI-ICH (30 train + 7 validation) |
| fNIRS test data | HEFMI (external) | TU Berlin (external) |
| fMRI training data | ds002336 | ds002336 |
| Quality of the fNIRS training target | good (reliability 0.50, neural signature) | poor (reliability 0.08, mostly body signal) |
| EEG input | raw EEG, 200 Hz | band power, 20 Hz |
| Main target | full time course | four compact numbers per trial |
| Evaluation | 5-fold by person, person-average removed | fixed validation people + leave-one-person-out |
| Seeds | 3 (Claim 1), several elsewhere | 5, all analyses |

The two pipelines differ in data, code and model, so where they agree the result is stronger than either alone.

### Table 3 (rebuilt, easy language) — Do the two pipelines agree?

| Question | Muntasir's answer | Maisha's answer (five seeds) | What it means |
|---|---|---|---|
| **Does training with fMRI as well as fNIRS make fNIRS prediction better?** | **No.** Slightly worse (−0.018 on 5 people; −0.007 on 29 people). Shuffled fMRI works just as well as real fMRI. | **No.** fNIRS-only was better in 4 of 5 seeds (0.052 vs 0.029 on average). | **Both say no.** Adding fMRI does not help. |
| **Can the model predict the blood signal of a person it has never seen?** | **Yes, a little:** r = 0.158, all 5 test groups positive. | **No:** r = 0.026, the same as just guessing each person's average (0.029). | **They differ because the data differ.** Muntasir trained on clean, repeatable data; Maisha on noisy data that is mostly body signal. Clean data is needed. |
| **Are EEG and blood signals linked inside one person?** | **Yes:** r = 0.098, positive in 27 of 29 people. | **Yes:** r = 0.094 vs chance 0.021 (p = 0.005). | **Both say yes.** The link is real but personal. |
| **Does it work on a completely different dataset?** | **No:** 0.012 (HEFMI) and 0.000 (ds002739). | **No:** −0.002 (TU Berlin) and 0.006 (ds002739). | **Both say no.** Nothing transfers between datasets. |
| **Does the network learn the brain–blood delay?** | **No:** its answer just follows the limits it is given (2.0 s, then 0.07 s). | **No:** fixed at 5.00 s for everyone; does not match measured delays. | **Both say no.** Measure the delay directly instead (Muntasir: 6.4 s, stable within a person). |
| **Can the confidence score tell good predictions from bad ones?** | **No:** link between confidence and error is only 0.02. | **Within HEFMI, yes** (0.42); **on new datasets, no** (−0.15; abstention unstable). | **Partly disagree.** Maisha's score works inside its own dataset; Muntasir's does not. Both fail on new datasets. |
| **Should global signal regression (GSR) be used?** | **No:** it removes about 83% of the real response. | **No:** it removes 40–50% of the signal, part of it real. | **Both say turn it off.** |
| **Does using some of a new person's data help?** | Not tested in this way. | **Yes, but mostly without EEG:** 0.191 with EEG vs 0.160 using only the person's own average. | EEG adds only about +0.03. |

### Table 4 (rebuilt, easy language) — What can we claim?

| Claim in plain words | Answer | Why |
|---|---|---|
| EEG can predict fNIRS in new people, **if the training data are clean** | **Yes** | Muntasir: r = 0.158 on clean data, all 5 test groups positive, p = 0.001. Maisha's noisy data shows the opposite, which is why clean data matters. |
| EEG can predict fMRI in new people | **Weak** | Muntasir: r = 0.064. Maisha: at chance (0.084 vs 0.088) and worse than the person-average shortcut. Only 7–10 people. |
| EEG and fNIRS are linked inside one person | **Yes** | Both pipelines find r ≈ 0.09–0.10, clearly above chance. |
| Training with fMRI makes predictions better | **No** | Both pipelines: no gain. Muntasir's shuffled test and Maisha's five seeds agree. |
| The model works on other datasets | **No** | Both pipelines ≈ 0 on every outside dataset. |
| The brain–blood delay is a stable personal measure | **Yes (measured directly)** | Muntasir: 6.4 s average, consistent between sessions (ICC 0.53). |
| The network itself learns the delay | **No** | Muntasir: follows its limits. Maisha: fixed 5 s, does not match measured delays. |
| Training with blood signals gives a better EEG representation | **No** | Muntasir: no better than a random network. Maisha: not tested (switched off after earlier tests showed no benefit). |
| The confidence numbers are honest (calibrated) | **Within the same dataset only** | Muntasir: calibration error 0.27 → 0.02. Maisha: internal 0.08–0.14, but external errors are large and unstable (up to 0.89). |
| The confidence score can spot bad predictions | **Mixed** | Maisha: yes inside HEFMI (0.42, all seeds); both pipelines: no on new datasets. |
| Brain haemorrhage patients differ from healthy people | **Possible, not proven** | Muntasir: HbO–HbR link 0.81 vs 0.64 (corrected p = 0.003), but light-travel differences in injured heads could explain it. |
| EEG adds useful information when personalising | **Very little** | Maisha: +0.03 over the person's own average, 7 people, not significant. |

---

## 8. Limitations of this notebook

- **Noisy training target.** HEFMI fNIRS is mostly body signal (reliability 0.08); this limits every prediction result here.
- **Few validation people.** 7 for HEFMI and 3 for fMRI; several effects swing between seeds because of this.
- **Unverified timing** in three splits.
- **The calibration people change with the seed**, which makes the abstention rule unstable.
- **External tests change several things at once** (dataset, device, electrode layout, task), so the cause of failure cannot be pinned down.
- **Some analyses are deterministic** (SD = 0 across seeds), so the five seeds say nothing about their stability; that would need different data splits, not different seeds.

## 9. Suggested next steps

1. Repeat the personalisation test on TU Berlin (29 people, clean target), always against the "person's own average, no EEG" shortcut.
2. Fix the calibration people (not seed-dependent) or use leave-one-person-out calibration, then re-test abstention.
3. Check the event timing against raw marker files for the three WARN splits.
4. Report fMRI analyses as exploratory unless more people are added.
