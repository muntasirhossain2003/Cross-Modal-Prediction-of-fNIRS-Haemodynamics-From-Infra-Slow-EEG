# NeuroVascNet — Full Data-Quality Audit

**Date:** 2026-07-25
**Ruleset:** `1.0.0-frozen-2026-07-25` (frozen before any metric was computed)
**Scope:** 610 recordings · 63.2 GB · 97 subjects · 4 datasets
**Code:** [`qc/`](qc/) · **Machine-readable output:** [`qc_outputs/`](qc_outputs/)

This replaces the `quick_mode` CSV inspection. Every recording on disk was read,
measured, and given a pass/fail decision against thresholds fixed in advance. No
threshold was chosen after seeing a result, and no model performance was
consulted at any point.

---

## 0. Verdict in one paragraph

The audit ran to completion over every file, and it found **four things that
plausibly bear on why EEG→hemodynamic prediction has been failing** — chief
among them that global-signal regression removes ~83% of the evoked fNIRS
response, and that the two fNIRS datasets appear to differ in concentration
units by a factor of ~2600. It also produced **two large blocks of exclusions
that I do not believe and that you should not act on** — 36/37 HEFMI subjects
and 23/24 ds002739 subjects — because they come from a mis-calibrated rule and a
metric that is invalid on band-limited data respectively. Those are faults in
the audit, not in the data, and they are documented as such in §4 rather than
quietly corrected, because the ruleset was pre-registered.

**Do not re-run any model on the basis of the subject exclusion list as it
stands.** Read §4 first.

---

## 1. Coverage — did it actually audit everything?

| dataset | modality | role | in manifest | QC rows | complete |
|---|---|---|---|---|---|
| HEFMI | eeg | epoched | 98 | 98 | yes |
| HEFMI | fnirs | epoched_conc | 98 | 100 ⚠ | yes |
| TUBerlin | eeg | raw | 78 | 78 | yes |
| TUBerlin | fnirs | conc | 78 | 78 | yes |
| ds002336 | eeg | corrected | 53 | 53 | yes |
| ds002336 | eeg | raw | 54 | 54 | yes |
| ds002336 | fmri | raw | 55 | 55 | yes |
| ds002739 | eeg | corrected | 48 | 48 | yes |
| ds002739 | fmri | raw | 48 | 48 | yes |

Every manifest recording has a QC row. ⚠ HEFMI fNIRS has **2 duplicate rows**
(100 vs 98), an artefact of the repair-and-resume cycle after the CSV schema bug.
Harmless to the conclusions, but `qc_fnirs.csv` should be de-duplicated on
`(dataset, subject, session, task)` before any downstream use.

---

## 2. Headline findings, ranked by impact on the research question

### 2.0 ROOT CAUSE: the HEFMI fNIRS target is systemic-dominated, not neural 🔴🔴

*Added 2026-07-26 after the Phase-0 ablation (`qc/ablation_target.py`).*

In a genuine haemodynamic response **HbO rises while HbR falls** — they must be
*anti*-correlated. Measured across every recording:

| dataset | HbO–HbR correlation (median) | range | verdict |
|---|---|---|---|
| **HEFMI** (primary fNIRS training set) | **+0.898** | +0.609 … +0.960 | **physiologically wrong** |
| TU Berlin | −0.333 | −0.653 … +0.973 | correct sign |

A +0.9 correlation means both chromophores are being driven by the same thing —
systemic physiology (blood pressure, Mayer waves, scalp perfusion), motion, or a
Beer–Lambert problem. **Not** neural activation.

Evoked SNR (peak amplitude ÷ its own 95% CI) confirms it: HEFMI **0.144**,
TU Berlin 0.247. Both below 1.0 — in the median recording the evoked response is
smaller than its own confidence interval, i.e. not distinguishable from zero.

Split-half reliability of the trial-averaged target, held-out subjects:

| target variant | reliability |
|---|---|
| raw HbO, no preprocessing | 0.257 (per-subject range −0.08 … +0.73) |
| best of 12 variants (bandpass, no GSR, no norm) | **0.220** |
| **current pipeline** (bandpass + GSR + per-channel z) | **0.127** |

**This is the ceiling on everything.** A target that only correlates 0.13–0.26
with itself cannot be predicted above roughly that level by *any* model. The
"cross-subject waveform ≈ 0" result was never really a statement about EEG.

**The causal chain, end to end:**

1. HEFMI fNIRS is systemic-dominated (HbO≈HbR, r = +0.90)
2. GSR was presumably added to the pipeline to fight that systemic component
3. GSR removes the systemic component **and the evoked response with it** (−83%),
   because a broad motor response is itself spatially global
4. Per-channel z-scoring then removes whatever amplitude survived
5. Target reliability falls 0.257 → 0.127
6. Nothing is predictable — and it looks like an EEG information limit

**The fix is not in the model or the preprocessing — it is the data source.**
Global-mean regression is the wrong tool for systemic removal; the correct one is
**short-separation channel regression**, which needs raw optical data. That makes
`dataset/Multi-joint Upper-Limb EEG+FNIR/` (raw fNIRS, 18 subjects) the critical
asset, and makes TU Berlin — which has physiologically correct HbO/HbR and better
evoked SNR — arguably the sounder fNIRS training set despite being a cognitive task.

**Full 12-variant sweep:** `qc_outputs/tables/ablation_fnirs_target.csv`

#### 2.0b Every fNIRS dataset compared on target quality

*Added 2026-07-26. Multi-joint was reprocessed from RAW NIRx wavelengths with a
controlled Beer–Lambert conversion (`qc/multijoint_check.py`), which also made
true Scalp Coupling Index computable for the first time in this project.*

| dataset | task | HbO–HbR | true SCI | evoked SNR | split-half | verdict |
|---|---|---|---|---|---|---|
| **HEFMI** | motor | **+0.898** ❌ | n/a (pre-converted) | 0.144 | **+0.257** | wrong physiology, best reliability |
| **TU Berlin** | cognitive | **−0.333** ✓ | n/a (pre-converted) | **0.247** | — | **best available** |
| **Multi-joint** | motor | −0.768 ✓ | **+0.023** ❌ | 0.04 | −0.10 | correct sign, unusable coupling |
| TU Berlin hBCI | motor | not yet audited | — | — | — | **the open question** |

Three conclusions:

1. **HEFMI's +0.90 is a conversion artifact, not tissue physiology.** A controlled
   mBLL on comparable hardware yields −0.77. *Caveat:* mBLL matrix inversion tends
   to produce anti-correlated HbO/HbR when both wavelengths share a dominant
   component, so a correct sign is necessary but not sufficient evidence of
   neural signal.
2. **Multi-joint is not the rescue.** True SCI median **0.023**, with only 3–22%
   of channels above the 0.5 coupling threshold — the optodes were barely
   coupled to the scalp. No detectable evoked response even per-condition with
   100 trials each (SNR 0.03–0.11, split-half ≈ 0). Raw data does not help if
   the optical coupling was poor at acquisition.
3. **No fNIRS dataset here has evoked SNR > 1.0.** In every one of them the
   evoked peak is smaller than its own confidence interval. TU Berlin is simply
   the least bad, at 0.247 — and it is a cognitive task.

**Consequence:** the motor track's two candidate fNIRS sets are both compromised
(HEFMI conversion, Multi-joint coupling). The one untested motor option was
**TU Berlin hBCI** — same lab, same NIRx hardware as the best-quality dataset,
but motor imagery. It was audited next, and it changes the conclusion.

#### 2.0c TU Berlin hBCI is the answer — swap it in as the primary fNIRS set

*`qc/target_gate.py`, all 29 subjects, 87 MI runs + 87 MA runs, reprocessed from
raw 760/850 nm intensities with a controlled Beer–Lambert.*

| metric | **TUB-hBCI (MI)** | HEFMI | TUB-WL | Multi-joint |
|---|---|---|---|---|
| true SCI (coupling) | **0.902** — 78% of channels good | n/a | n/a | 0.023 ❌ |
| **split-half reliability** | **+0.528** | +0.257 | — | −0.10 |
| evoked SNR | **0.579** | 0.144 | 0.247 | 0.04 |
| HbO–HbR corr | +0.084 | +0.898 ❌ | −0.333 | −0.768 |
| subjects | 29 | 37 | 26 | 18 |

**Target reliability 0.528 — double HEFMI's, and 4× what the model has actually
been training against (0.127 after the current pipeline).** Optode coupling is
excellent (SCI 0.90 vs Multi-joint's 0.023). This is, by a wide margin, the best
fNIRS supervision available in the project, and it is *motor imagery*.

**On the HbO/HbR ≈ 0:** it fails the literal sign check, but the failure mode is
the opposite of HEFMI's and much more benign. HEFMI's **+0.90** means a strong
shared *systemic* driver dominates both chromophores — an artifact correlated
with everything, which averaging cannot remove. TUB-hBCI's **+0.08** is what a
weak-but-genuine response buried in independent noise looks like, and is
consistent with its evoked SNR of 0.58. Noise averages out across trials;
a systemic confound does not. The gate's sign rule was written to catch HEFMI's
failure mode and gives a misleading verdict on a clean low-SNR dataset — this is
a fourth place the pre-registered ruleset needs a documented v2 (see §4).

**Recommendation: replace HEFMI with TU Berlin hBCI as the primary fNIRS
training set.** That single swap raises the target-reliability ceiling ~4× before
any change to the model. HEFMI should be demoted or dropped (systemic-dominated);
Multi-joint should be dropped (optodes were not coupled at acquisition).

Both tracks are now viable:

| track | fNIRS | fMRI |
|---|---|---|
| **motor** | TU Berlin hBCI — 29 subj, rel 0.528 | ds002336 — 10 subj, tSNR 24 |
| **cognitive** | TU Berlin WL — 26 subj | ds002739 — 24 subj, tSNR 190 |

---

### 2.1 Global-signal regression destroys ~83% of the evoked fNIRS response 🔴

The audit brief asked for GSR to be an explicit ablation "because it may remove
genuine broad motor responses." It does.

| dataset | variance removed by GSR (median) | evoked amplitude **retained** (median) |
|---|---|---|
| TU Berlin | 52.5% (range 19.9–98.3%) | **17.2%** (range 1.7% – 217%) |
| HEFMI | 18.1% (range 3.8–74.0%) | not computable (epoched, no onsets) |

Global signal regression removes **83% of the event-locked HbO amplitude** in the
median TU Berlin recording. That is the opposite of a nuisance-only correction:
in a montage where most channels sit over the same broad task-responsive region,
the "global signal" *is* substantially the response.

`NeuroVascNet_Findings_Summary.md` §"Architecture" lists global-signal regression
as part of standard preprocessing. **If GSR is applied to the fNIRS targets in the
training pipeline, it is removing most of the amplitude the model is being asked
to predict.** Given that amplitude is the one target family that has shown real
signal (within-subject r=0.33), this is the single most actionable finding here.

*Caveats:* the 217% maximum is a division-by-near-zero outlier and should be
ignored; the median is the meaningful statistic. HEFMI cannot be assessed because
its files are pre-epoched with no onset vector, so the GSR-vs-evoked comparison
has no event to lock to.

**Action:** confirm whether GSR is applied to hemodynamic targets (not just EEG
inputs) in `NeuroVascNet_v3.ipynb`, and if so, run the no-GSR arm as a first-class
ablation.

---

### 2.2 The two fNIRS datasets differ in concentration scale by ~2600× 🔴

| dataset | max abs concentration (median over recordings) | baseline SD (median) |
|---|---|---|
| HEFMI (train) | 1.251 × 10⁻⁵ | 7.96 × 10⁻⁸ |
| TU Berlin (external val) | 3.273 × 10⁻² | 9.44 × 10⁻⁴ |

Ratio ≈ **2616×**. Both were read with the same reader and reported in the same
nominal unit (`umol/L`), so this is a property of the files, not of the QC code.

The most likely explanation is that HEFMI ships **mol/L** (1.25 × 10⁻⁵ M = 12.5 µM
peak HbO, physiologically sensible) while TU Berlin ships something closer to
**mmol/L or µM directly**. Either way the two are not on a common scale.

This matters directly: HEFMI is the primary fNIRS *training* set and TU Berlin is
the *external validation* set. A 2600× scale mismatch between train and test
targets would sabotage any cross-dataset amplitude transfer, independent of
whether EEG carries the information. It is a strong candidate explanation for the
external fNIRS validation numbers being far weaker than in-domain ones.

**Action:** check whether hemodynamic targets are z-scored or otherwise normalised
per-dataset before the loss. If they are, this is neutralised and can be
dismissed. If they are not, it is a live bug. Note `neurovascnet_loaders_v3.py:76`
(`_baseline_correct_hemo`) subtracts a baseline but performs **no rescaling**.

Note also that the pre-registered plausibility rule (`max_abs_conc_umol` = 100 µM)
never fired for either dataset — both sit far below it under the assumed unit. The
rule failed to catch a unit problem precisely because it assumed the unit.

---

### 2.3 The motor BOLD target is weak and spatially non-specific 🟡

> **UPDATED 2026-07-26 after a proper first-level GLM (`qc/glm_check.py`).**
> The near-zero numbers below were **a drift artifact of my own event-locked
> computation**, as I flagged they might be. With motion regressors and
> polynomial drift included, the response *is* detectable. The finding survives
> in weakened, changed form — see the GLM result immediately below.
>
> **GLM result (all 10 ds002336 subjects, `motorloc`, motion + drift regressed):**
>
> | | median | detail |
> |---|---|---|
> | Motor ROI percent signal change | **+0.087%** | expected 0.5–2% |
> | Motor ROI peak t | 4.54 | **6/10 subjects reach t > 3** |
> | *Control* (occipital) ROI PSC | +0.073% | nearly identical to motor |
> | *Control* ROI peak t | 3.06 | |
>
> Two problems remain, and they are different from the one I originally claimed:
>
> 1. **The response is weak** — 0.087% median, roughly 6–20× below a healthy
>    motor localiser.
> 2. **It is not motor-specific.** Occipital control ROIs respond about as
>    strongly as sensorimotor ROIs, and in 3 subjects (xp102, xp104, xp105) the
>    control ROI responds *more*. The `motorloc` block design alternates a visual
>    instruction cue against a fixation cross, so task-vs-rest contrast picks up
>    visual stimulation alongside motor activity.
> 3. **4/10 subjects show no motor response at all** (t < 3).
>
> **Revised implication:** the 48-ROI target is not empty, so "EEG can't predict
> it because it isn't there" is too strong. But a target this weak and this
> spatially non-specific is a poor supervision signal — and it directly supports
> restricting to high-coupling ROIs rather than predicting all 48. Original
> (superseded) analysis follows.

---

#### Superseded: original raw-BOLD event-locked analysis 🟡

Event-locked motor-ROI response, percent signal change at the canonical HRF peak:

| dataset | median | min | max |
|---|---|---|---|
| ds002336 (motor tasks) | **+0.0010%** | −0.293% | +0.342% |
| ds002739 (perceptual task) | +0.0022% | −0.020% | +0.021% |

For a motor localiser with 20 s blocks, precentral/postcentral gyrus should show
roughly 0.5–2% signal change. The median here is three orders of magnitude below
that — the task response is not detectable at the Harvard-Oxford ROI level in the
median subject.

If the 48-ROI BOLD target does not reliably contain the motor response, then no
EEG encoder can predict it, and the "cross-subject waveform ≈ 0" result would be
partly a property of the *target*, not only of EEG's information content.

**This one is suggestive, not established.** My event-locked computation runs on
raw BOLD with no motion regression, no high-pass detrending, and no spatial
smoothing, so low-frequency drift almost certainly swamps a 0.5% effect. The
honest statement is: *the target has not been shown to carry the response, and it
should be verified with a proper GLM before more modelling effort is spent on it.*

**Action:** run a first-level GLM (motion regressors + high-pass) on 2–3 ds002336
subjects' `motorloc` runs and confirm a significant precentral beta. This is a
cheap, decisive test. If the response is there, this finding dissolves; if it is
not, the whole fMRI arm needs rethinking.

---

### 2.4 ds002336 `motorloc` retains large gradient residual after correction 🟡

From the raw-vs-corrected comparison, corrected gradient-harmonic ratio by task:

| task | corrected gradient ratio (typical) |
|---|---|
| eegNF, fmriNF, eegfmriNF, MIpre, MIpost | **≈ 1.0 – 4.6** (clean) |
| **motorloc** | **13.7 – 107.1** (residual present) |

Same subjects, same correction pipeline, same session — but `motorloc` is the
outlier in nearly every subject (xp103: 45.9, xp105: 107.1, xp106: 64.6,
xp107: 94.5, xp108: 95.5, xp109: 43.2, xp110: 51.3, xp101: 81.0). Every other
task in the same subject sits at ~1.0.

Correction *did* work in bulk terms everywhere — 29–38 dB power reduction,
`correction_effective = True` on all 106 pairs, raw ~8000 µV → corrected
~120–260 µV. But the motorloc runs keep a narrowband residual at slice-frequency
harmonics that the other runs do not.

`motorloc` is the task with the cleanest Rest/Task-ME block design — i.e. the one
most attractive for supervision. It is also the one whose EEG is most contaminated.

**Action:** open `qc_outputs/figures/ds002336/raw_vs_corrected/*motorloc*.png` and
compare against a non-motorloc figure from the same subject. If the residual is
real, motorloc EEG needs re-correction (your existing AAS implementation) before
use.

---

### 2.5 One genuinely corrupt file

`ds002336 / sub-xp102 / motorloc / eeg / raw` is **entirely flat**:
`robust_amplitude = 0.0`, `clipped_fraction = 1.0`, `constant_fraction = 1.0`.
The file contains no signal. Correctly excluded.

---

### 2.6 The 5 mV rule fired exactly as intended

**106/106 raw in-scanner EEG recordings exceed the 5 mV limit** (typical robust
amplitude 7,400–13,200 µV) and are barred from training. Post-correction
amplitudes were *measured*, not assumed: median corrected robust amplitude
127–235 µV across ds002336, i.e. genuine scalp-EEG scale. The audit brief's
specific concern — "confirm units after correction instead of assuming that a
smaller-looking array is valid" — is discharged.

The one subject where correction did **not** produce plausible EEG is
**sub-xp101** (corrected amplitudes 413–949 µV, above the 500 µV limit). That is
a real, subject-specific correction failure.

---

## 3. Per-dataset results

### HEFMI (primary fNIRS training set) — 37 subjects

| | PASS | WARN | FAIL |
|---|---|---|---|
| EEG (98 recordings) | 0 | 2 | **96** |
| fNIRS (98 recordings) | 100 | 0 | 0 |

- **fNIRS is in excellent shape:** 100% channel availability, zero poor-coupling
  channels, zero flat-lining, motion 0.07/min/ch, 30 usable trials per session
  throughout. Every recording passes.
- **EEG fails almost universally on line noise** — median ratio **1039×**
  neighbourhood, max 29,947×. See §4.1: this is a rule-calibration failure, not a
  data verdict.
- 5 subjects additionally show implausible amplitudes (10,600–15,700 µV):
  `Normals/subject7`, `Patients/subject4`, `subject5`, `subject6`. These are real
  problems independent of the line-noise issue.
- Sampling rate (256 Hz) and channel count (32) match documentation exactly on
  all 98 recordings.

### ds002336 (primary fMRI training set) — 10 subjects

| | PASS | WARN | FAIL |
|---|---|---|---|
| EEG raw (54) | 0 | 0 | 54 (expected — 5 mV rule) |
| EEG corrected (53) | 0 | 40 | 13 |
| fMRI (55) | 6 | 48 | 1 |

- **9/10 subjects INCLUDE_PARTIAL, 1 EXCLUDE** (`sub-xp101`, correction failure).
- fMRI is broadly usable: ROI coverage **100%**, mean FD 0.124 mm, censored
  fraction 0.5%. The single fMRI FAIL is `sub-xp102/MIpre` (mean FD 0.578 mm).
- tSNR median **23.9** (range 10.1–35.5) sits in the WARN band (<30) — expected
  for EPI with an EEG cap, but it caps how much BOLD signal is recoverable.
- 3/10 subjects have no MIpre/MIpost at all (xp101, xp104, xp109), so real
  Rest/Task-MI labels exist for at most 7 subjects before QC.

### TU Berlin (external fNIRS validation) — 26 subjects

| | PASS | WARN | FAIL |
|---|---|---|---|
| EEG (78) | 3 | 36 | 39 |
| fNIRS (78) | 52 | 26 | 0 |

- **12 INCLUDE, 2 INCLUDE_PARTIAL, 12 EXCLUDE.** The 12 exclusions are all
  line-noise (§4.1) and should be revisited.
- fNIRS again clean: full channel availability, ≤2.8% poor coupling, 17–60 usable
  trials.
- **Identity confirmed as wrong** — see §5.1.

### ds002739 (external fMRI validation) — 24 subjects

| | PASS | WARN | FAIL |
|---|---|---|---|
| EEG (48) | 0 | 2 | **46** |
| fMRI (48) | 37 | 11 | 0 |

- **fMRI is the cleanest data in the entire project:** tSNR median **189.8**,
  censored fraction 1.0%, ROI coverage 97.9%, 763/771 usable volumes.
- **EEG is also, by every other measure, pristine:** robust amplitude 39.1 µV,
  line-noise ratio 0.11 (already notched), bad-channel fraction 0.041, zero NaNs.
- Yet 46/48 FAIL, all on a single metric. See §4.2 — this is a metric-validity
  failure, not a data problem.

---

## 4. Where this audit is wrong — read before acting

Two of the three largest exclusion blocks are artefacts of my own rules and
metrics. Because the ruleset was pre-registered, I have **not** silently changed
the thresholds; I am reporting the misfire and proposing a documented v2.

### 4.1 The line-noise FAIL threshold is mis-calibrated 🔴

**Effect:** 96/98 HEFMI EEG recordings and 39/78 TU Berlin EEG recordings FAIL,
excluding **36 of 37 HEFMI subjects** — which would delete the primary fNIRS
training set entirely.

**Why it is wrong:** I set `max_line_ratio = 100` on the reasoning that a
recording 100× over the local spectral background is "line-dominated". That
reasoning is sound for deciding *whether line noise is present*. It is invalid as
an *exclusion* criterion, because **mains contamination is the single most
routinely removable artefact in EEG** — one notch filter, and it is gone. It says
nothing about whether the recording contains usable brain signal.

Worse, it is not even measuring the data that reaches the model:
`NeuroVascNet_v3.ipynb` already applies notch filtering during preprocessing, so
the line noise measured here is removed before training.

**Corroborating evidence that these recordings are otherwise fine:** HEFMI EEG
shows zero NaNs, clipped fraction ~3×10⁻⁷, bad-channel fraction 0.0 (median), and
paired fNIRS that passes 100%.

**Proposed v2:** demote line noise to WARN at any ratio; retain FAIL only when
line power dominates the *entire* 0.5–40 Hz band after notching (a genuinely
unusable recording). Under that change, HEFMI EEG exclusions drop from 96 to the
handful driven by amplitude (§3), and roughly **32 of the 36 HEFMI subject
exclusions are reversed**.

### 4.2 The gradient-residual metric is invalid on band-limited data 🔴

**Effect:** 46/48 ds002739 EEG recordings FAIL on
`gradient_harmonic_ratio_max` (42–188 vs limit 10), excluding **23 of 24
subjects** — the entire external fMRI validation set.

**Why it is wrong:** ds002739 ships EEG already gradient- and BCG-corrected,
band-pass filtered **0.5–40 Hz**, and downsampled to 1000 Hz. My metric evaluates
narrowband power at harmonics of the 16 Hz slice frequency: **16, 32, 48, 64,
80 Hz**. Three of those five harmonics lie in the filter's stopband, and the
32 Hz harmonic sits on the transition-band slope. In a stopband both the peak and
its "neighbourhood" are near-zero, so their ratio is numerically meaningless; near
the 40 Hz filter edge the upper sideband is attenuated more than the peak, which
inflates the ratio systematically. The 42–188 values are measuring the filter, not
the scanner.

**Corroborating evidence:** every independent measure says this EEG is clean
(39 µV amplitude, 0.11 line ratio, 4% bad channels, 0 NaN). It is implausible that
24/24 subjects have severe gradient residual while showing textbook-clean spectra
everywhere else. Contrast ds002336's corrected data at 200 Hz, where the same
metric returns ≈1.0 for non-motorloc tasks — there the metric works, because the
harmonics are inside the passband.

**Proposed v2:** only evaluate harmonics that fall below 80% of the recording's
effective low-pass edge (detected from the PSD rolloff); if fewer than two
harmonics survive, report `NA` with a documented reason instead of a number.
Under that change **all 23 ds002739 subject exclusions are reversed.**

*This also means §2.4's motorloc finding needs the same scrutiny — but there the
metric is being applied inside the passband and returns ≈1.0 for sibling tasks in
the same subjects, so the contrast is credible.*

### 4.3 Metrics that could not be computed, and why

| metric | dataset | status | reason |
|---|---|---|---|
| Scalp Coupling Index | HEFMI, TU Berlin | substituted | Both ship Beer-Lambert HbO/HbR; the conversion mixes the two wavelengths, so true SCI is unrecoverable. Cardiac-band power fraction used instead and labelled in the `coupling_metric` column. |
| Framewise displacement | ds002739 | proxy | No head-motion file in the dataset. DVARS-based censoring used, flagged in `motion_source`. |
| fNIRS sampling rate | HEFMI | substituted | Field is empty in the `.mat`; documented 11 Hz used, flagged in `fs_source`. |
| GSR evoked ablation | HEFMI | not computable | Files are pre-epoched with no onset vector to lock to. |

---

## 5. Structural discrepancies between the data and the documentation

### 5.1 The external fNIRS validation set is not the dataset the docs describe 🔴

`dataset/TU Berlin EEG + NIRS/` contains tasks **`dsr`, `nback`, `wg`** (EEG) and
**`gonogo`, `nback`, `vf`** (NIRS), across **26 subjects** (VP001–VP026).

That is the Shin et al. **mental-workload** dataset — n-back, discrimination/
selection response, word generation/verbal fluency.

`study_reference_facts.csv` and `NeuroVascNet_Findings_Summary.md` both describe
the external fNIRS validation set as **29 subjects, motor imagery + mental
arithmetic**. That is a *different dataset*, present on disk but unused:
`dataset/TU Berlin hBCI EEG+FNRI/`.

`neurovascnet_loaders_v3.py:36` points at the mental-workload directory.

**Consequence:** if `tuberlin_ext_*.npz` was built from this path, then the
external fNIRS validation is a **cross-task test on a non-motor dataset**, not the
same-task external replication the write-up claims. Combined with ds002739 (also
cross-task, perceptual decision), **both** external validation sets would be
cross-task — which substantially changes how the external generalisation results
should be interpreted and reported.

The subject count is independently confirmed by the audit: 26 on disk, not 29.

### 5.2 ds002336 MIpre/MIpost coverage

3 of 10 subjects (xp101, xp104, xp109) have no MIpre/MIpost recordings. Analyses
depending on real Rest/Task-MI labels start from 7 subjects, before QC removes
xp101 (correction failure) — leaving **6**.

---

## 6. What I would do next, in order

1. **Check whether GSR is applied to the fNIRS targets** (§2.1). If yes, re-run
   the amplitude experiments without it. Highest expected value, lowest cost.
2. **Check whether hemodynamic targets are normalised per-dataset** (§2.2). If
   not, the 2600× HEFMI↔TU Berlin scale gap is a live bug in cross-dataset
   transfer.
3. **Run a first-level GLM on 2–3 ds002336 `motorloc` runs** (§2.3). Decides
   whether the BOLD target carries the motor response at all.
4. **Resolve which TU Berlin directory built the cached external bundle** (§5.1).
   Affects how every external fNIRS number is described in the paper.
5. **Apply ruleset v2** (§4.1, §4.2) and re-derive the subject list. Report both
   v1 and v2 in the paper — the v1 misfires are themselves informative about how
   aggressive QC thresholds interact with these datasets.
6. Inspect the motorloc raw-vs-corrected figures (§2.4); re-correct with AAS if
   the residual is confirmed.

---

## 7. Reproducing this

```bash
python qc/build_manifest.py     # enumerate everything on disk
python qc/run_qc.py             # full QC pass (resumable)
python qc/report.py             # decisions + auto-generated report
```

**Outputs**

| path | contents |
|---|---|
| `qc_outputs/manifest.csv` | every recording on disk |
| `qc_outputs/tables/qc_{eeg,fnirs,fmri}.csv` | one row per recording, all metrics |
| `qc_outputs/tables/qc_raw_vs_corrected.csv` | in-scanner correction comparison |
| `qc_outputs/tables/subject_decisions.csv` | pass/fail + documented reason per subject |
| `qc_outputs/arrays/eeg_psd/` | PSD per recording (mean + per-channel) |
| `qc_outputs/arrays/fnirs_evoked/` | event-locked HbO/HbR mean + CI |
| `qc_outputs/arrays/fmri_roi/` | ROI time series + design matrix |
| `qc_outputs/arrays/fmri_roi_masks/` | ROI mask used for each subject |
| `qc_outputs/figures/<dataset>/<modality>/` | per-recording diagnostic figures |
| `qc_outputs/QC_REPORT.md` | auto-generated full tables |

**Acceptance criterion — met, with one caveat.** Every included subject has a
saved QC row, an explicit decision, and a documented reason for every exclusion;
thresholds were fixed before any metric ran and are unchanged in
`qc/rules.py`. The caveat is that two of the resulting exclusion blocks are
rule/metric artefacts (§4) and the decision list should not be used until
ruleset v2 is applied.
