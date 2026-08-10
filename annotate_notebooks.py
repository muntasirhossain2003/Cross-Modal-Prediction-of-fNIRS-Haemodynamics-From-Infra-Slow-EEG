"""Insert a standard orientation header into every NeuroVascNet notebook.

Each header states: purpose, what was done, what was achieved, what was NOT
achieved, and current status. Re-runnable -- an existing header is replaced
rather than duplicated.

The critical retrospective note carried by most headers: these experiments were
run against the HEFMI fNIRS target, which the 2026-07 data-quality audit showed
to be systemic-dominated (HbO/HbR = +0.898) with a split-half reliability of
0.127 after preprocessing. Their negative results are therefore CONFOUNDED --
they measured the pipeline, not EEG.
"""
from pathlib import Path

import nbformat as nbf

MARKER = "<!-- NVN-HEADER -->"

AUDIT = (
    "> **Retrospective (data-quality audit, 2026-07):** this notebook trained against the "
    "**HEFMI** fNIRS target, later shown to be **systemic-dominated** (HbO/HbR = **+0.898**, "
    "should be negative) with split-half reliability of only **0.127** after the pipeline's "
    "bandpass -> global-signal regression -> z-score chain. GSR alone removes ~83% of the "
    "evoked response. **The negative results below are confounded: they measure the "
    "preprocessing, not EEG.** See `NeuroVascNet_Data_Quality_Audit.md` sections 2.0-2.0c."
)

H = {}

H["NeuroVascNet_v3.ipynb"] = f"""
# NeuroVascNet v3 — canonical pipeline

**Status:** was the canonical pipeline; **superseded by `NeuroVascNet_v4_hBCI.ipynb`** for the
fNIRS arm.

## Purpose
Full end-to-end system: EEG-only in, hemodynamics out. One shared EEG encoder trained with
paired EEG-fNIRS *and* EEG-fMRI supervision, emitting a hemodynamic-aware latent, HbO/HbR and
BOLD predictions, a neurovascular delay, and a calibrated reliability score with abstention.

## What we did
- Real loaders for all 4 datasets (HEFMI, ds002336, TU Berlin, ds002739).
- Preprocessing: bandpass / notch / z-score, plus **global-signal regression on the fNIRS
  target** (`CLEAN_FNIRS_TARGET = True`).
- Band-power transform -> SharedEncoder (EEGNet + TCN) -> two heads + delay + reliability.
- Two-stage training with the capped-epoch fix; rigorous evaluation (concat-Pearson,
  bootstrap CI, shuffled null).

## What we achieved
- A complete, runnable, well-instrumented pipeline with honest evaluation machinery.
- EEG-only task decoding from the latent at ~0.56-0.61 (well above chance) — proving the
  encoder and the EEG signal are sound.
- A reliability/abstention layer that works in-domain (reliability-error r ~ 0.55).

## What we failed to achieve
- Cross-subject **waveform** prediction ~= 0 for both modalities.
- Cross-modal supervision did not beat single-modality baselines.
- The reliability layer degrades badly under domain shift (external 0.09-0.24).

{AUDIT}
"""

H["NeuroVascNet_v3.2.ipynb"] = f"""
# NeuroVascNet v3.2 — windows-gap experiment

**Status:** experiment complete; its winning fix was folded into v3.

## Purpose
Test whether the fNIRS/fMRI **data-size imbalance** (1230 vs 376 windows, so fMRI was shown
~81x vs fNIRS's ~25x over training) was suppressing held-out performance.

## What we did
Compared three balancing strategies — baseline (recycle), **capped**, and reweighted —
measuring fMRI train/val correlation and the overfitting gap.

## What we achieved
- "Capped" reduced the overfit gap by **24%** (0.404 -> 0.306) and was **adopted as the v3
  default**.
- Established that the imbalance is real and measurable.

## What we failed to achieve
- **None of the three unlocked held-out signal** — all validation correlations stayed ~= 0
  (0.007 / -0.017 / -0.002). The imbalance was not the bottleneck.

{AUDIT}
"""

H["NeuroVascNet_v3.3.ipynb"] = f"""
# NeuroVascNet v3.3 — CORAL domain adaptation on the shared latent

**Status:** experiment complete; **not adopted**.

## Purpose
Test whether Deep CORAL — aligning held-out validation subjects to the training distribution
on the shared 64-dim latent — could close the cross-subject gap for the **waveform** target.

## What we did
Added a CORAL loss on the latent and re-ran the waveform evaluation for both modalities.

## What we achieved
- A clean, well-controlled A/B, and the **third independent confirming negative** (alongside
  alignment-removal and the windows-gap fix).

## What we failed to achieve
- CORAL did **not** help: fNIRS val +0.0637 -> +0.0652 (no change), fMRI -0.0172 -> -0.0332
  (worse). Not adopted.

{AUDIT}
"""

H["NeuroVascNet_CoarseCrossSubject_V4.ipynb"] = f"""
# Coarse cross-subject probe — is *amplitude* recoverable?

**Status:** superseded by `NeuroVascNet_CrossSubj_TrialLevel_Verification.ipynb`.

## Purpose
First test of predicting response **amplitude** (one number per window) instead of the full
waveform, using simple Ridge regression rather than the deep model.

## What we did
Single 30% hold-out split, Ridge on EEG features -> fNIRS/fMRI amplitude.

## What we achieved
- Established amplitude as a target worth pursuing, and produced the first non-zero
  cross-subject number (fNIRS r = 0.21).

## What we failed to achieve
- **The 0.21 was fragile.** Proper leave-one-subject-out later dropped it to ~0.01 — the
  single-split protocol was too optimistic. Read the TrialLevel notebook instead.

{AUDIT}
"""

H["NeuroVascNet_CrossSubj_TrialLevel_Verification.ipynb"] = f"""
# Trial-level verification — is the amplitude signal genuine?

**Status:** the methodologically correct version of the amplitude result.

## Purpose
Determine whether the cross-subject amplitude signal is genuine **trial-level** prediction, or
merely **subject-ranking** (some people simply respond more strongly on average).

## What we did
Removed the subject mean before correlating, and ran a proper leave-one-subject-out protocol
alongside the original single split.

## What we achieved
- **fNIRS trial-level r = 0.33 (p < 0.001)** — genuine, and the project's strongest positive.
- fMRI trial-level r = 0.08 — real but weak.
- Showed the naive single-split 0.21 collapses to ~0.01 under LOSO — an important
  methodological correction.

## What we failed to achieve
- Did not establish cross-subject **waveform** recovery (not its aim).
- The fMRI amplitude signal remained weak.

{AUDIT}
"""

H["NeuroVascNet_CrossSubj_CORAL_BothModalities.ipynb"] = f"""
# CORAL A/B on the amplitude target (Ridge)

**Status:** experiment complete; per-modality conclusion, not a universal fix.

## Purpose
Test domain adaptation (CORAL) on the **amplitude** target for both modalities — the target
family that, unlike waveform, had shown real signal.

## What we did
Baseline vs CORAL, Ridge, both fNIRS and fMRI, cross-subject.

## What we achieved
- **fNIRS amplitude: 0.012 -> 0.253 with CORAL** — a large, real gain.

## What we failed to achieve
- **fMRI amplitude got worse** with CORAL (0.148 -> 0.068), attributed to small-sample
  covariance instability. No single setting works for both modalities.

{AUDIT}
"""

H["NeuroVascNet_SharedLatent_Amplitude.ipynb"] = f"""
# Shared-latent architecture, tested on the amplitude target

**Status:** the thesis architecture's most successful demonstration to date.

## Purpose
Test the actual thesis structure — **one shared encoder, two heads, jointly trained on two
disjoint datasets** — on the target family that is recoverable (amplitude).

## What we did
Built a shared MLP encoder (not the full EEGNet+TCN) with separate fNIRS and fMRI amplitude
heads, trained jointly on both datasets despite zero subject overlap.

## What we achieved
- **All heads beat null**: fNIRS r = 0.10 (0.15 with CORAL), fMRI r = 0.14.
- Demonstrated the shared-latent thesis is viable *for amplitude* — the core architectural
  claim of the project.

## What we failed to achieve
- Effect sizes are modest and sensitive to evaluation protocol.
- Used a simplified encoder, so this does not validate the full deep architecture.

{AUDIT}
"""

H["NeuroVascNet_Waveform_Reconstruction.ipynb"] = f"""
# Waveform reconstruction from predicted amplitude

**Status:** honest workaround, not a waveform-prediction result.

## Purpose
Given that amplitude is recoverable but the waveform is not, reconstruct a plausible
hemodynamic waveform as `predicted_amplitude x canonical_HRF_shape`.

## What we did
Combined the predicted per-window amplitude with a fixed physiological HRF template.

## What we achieved
- A deployable waveform output whose **magnitude is genuinely predicted** from EEG.
- A transparent framing that does not overclaim.

## What we failed to achieve
- **The shape is assumed, not learned** — it comes from physiology, not from the EEG. This is
  not waveform prediction and must never be presented as such.

{AUDIT}
"""

H["NeuroVascNet_E2E.ipynb"] = f"""
# NeuroVascNet E2E — original end-to-end pipeline

**Status:** **superseded by `NeuroVascNet_v3.ipynb`.** Kept for provenance.

## Purpose
The project's first complete implementation of the thesis: shared EEG encoder, dual
hemodynamic heads, delay module, reliability layer, trained and evaluated end to end.

## What we did
Established the architecture, the data schema (`eeg [N,C,T]`, `hemo [N,C,T]`, labels,
subject_ids) and the training/evaluation loop that every later notebook inherited.

## What we achieved
- A working baseline system and the reusable schema + component library — the foundation all
  subsequent work was built on.

## What we failed to achieve
- Predates the bug fixes, the capped-epoch training fix, the channel verification, and the
  rigorous null/bootstrap evaluation added in v3. **Its numbers should not be quoted.**

{AUDIT}
"""

H["NeuroVascNet_E2E_v2.ipynb"] = f"""
# NeuroVascNet E2E v2 — second iteration

**Status:** **superseded by `NeuroVascNet_v3.ipynb`.** Kept for provenance.

## Purpose
Iteration on the original E2E pipeline, refining loaders, preprocessing and training before
the v3 consolidation.

## What we did
Carried the E2E architecture forward with incremental fixes to data handling and training.

## What we achieved
- Intermediate improvements that were folded into v3.

## What we failed to achieve
- Still predates v3's rigorous evaluation (concat-Pearson + bootstrap CI + shuffled null).
  **Its numbers should not be quoted.**

{AUDIT}
"""

H["NeuroVascNet_E2E_trialscale.ipynb"] = f"""
# NeuroVascNet E2E — trial-scale variant

**Status:** exploratory; **superseded**.

## Purpose
Explore operating at trial scale rather than the sliding-window scale used by the main
pipeline.

## What we did
Reworked the data path to trial-level units and re-ran the E2E pipeline.

## What we achieved
- Informed the later, cleaner separation between window-level (waveform) and trial-level
  (amplitude) framings that the amplitude notebooks adopted.

## What we failed to achieve
- Did not produce a result that changed the project's conclusions. Not carried forward.

{AUDIT}
"""

H["NeuroVascNet_v4_hBCI.ipynb"] = """
# NeuroVascNet v4 — TU Berlin hBCI as primary fNIRS supervision

**Status:** current. **The first uncontaminated test of the thesis.**

## Purpose
The data-quality audit showed every previous fNIRS result was confounded by a broken target.
This notebook re-runs the evaluation with two changes and nothing else: **the data source**
(TU Berlin hBCI instead of HEFMI) and **GSR off**. Features, model class and protocol are
unchanged, so any difference is attributable to the target.

## What we did
- Built a new paired loader (`neurovascnet_loaders_v4.py`) reading TU Berlin hBCI **raw
  760/850 nm intensities**, with our own Beer-Lambert conversion — 29 subjects, MI runs,
  1380 train / 360 val trials across 23 / 6 subjects.
- Measured target quality (split-half reliability, HbO/HbR sign) for HEFMI and hBCI
  side by side.
- Ridge on EEG band-power -> waveform and amplitude, held-out subjects, against shuffled nulls.

## What we achieved
- **Target reliability -0.116 -> +0.506** on held-out subjects; HbO/HbR **+0.588 -> -0.147**
  (physiologically correct). The data swap works, decisively.
- **Waveform prediction above null for the first time**: r = 0.0723 vs null 0.0056
  (HEFMI sat at its null: 0.0138 vs 0.0109).
- Because the target is now demonstrably intact, the remaining shortfall is **for the first
  time legitimately attributable to the EEG->hemodynamic mapping** rather than to
  preprocessing. Every earlier negative was uninterpretable on this point.
- **Both arms are now measured side by side under one protocol** (reliability gate,
  permutation null, Ridge and deep encoder):

  | target | reliability | ceiling | ridge r (p) | deep r (p) |
  |---|---|---|---|---|
  | fNIRS HEFMI (v3) | -0.116 | — | 0.0138 (0.005) | 0.0263 (0.005) |
  | **fNIRS hBCI (v4)** | **+0.506** | 0.711 | **0.0723 (0.005)** | 0.0543 (0.035) |
  | fMRI v3 (broken) | **-0.793** | — | 0.0706 (0.005) | -0.0017 (0.527) |
  | **fMRI v4 (fixed)** | **+0.518** | 0.720 | 0.0161 (0.473) | 0.0385 (0.403) |

- **Both targets are now repaired.** The fMRI rebuild (`neurovascnet_loaders_v4_fmri.py`) —
  event-locked on stimulus onsets via the R128 scanner triggers, motion + drift regressed,
  restricted to sensorimotor ROIs — moved reliability from **-0.793 to +0.518**, on a par
  with the fNIRS arm.
- **The spurious fMRI positive disappeared exactly as predicted.** The broken target gave
  r = 0.0706 at **p = 0.005**; the repaired target gives r = 0.0161 at **p = 0.473**. A
  false positive replaced by an honest null is the strongest single validation of the
  reliability-gate method in this project.

## What we failed to achieve
- **The fMRI arm still has no positive result** — and it is *underpowered*, not disproven.
  The rebuild yields only **83 training / 40 validation trials across 3 ROIs**, versus
  1380 / 360 for fNIRS. At that size only a large effect could reach significance, so
  p = 0.47 is inconclusive rather than negative.
- The trial-averaged fMRI row (7 train / 5 val averages) is too small to interpret at all
  and reports NaN for amplitude. Ignore it.

- **Waveform is statistically significant** under a 200-fold permutation null:
  r = 0.0723 vs null 0.0220 +- 0.0160, **p = 0.005**.
- Fixed the earlier single-shot null (which produced nulls larger than the real values) by
  using a proper permutation distribution plus PCA before the averaged regression.

## What we failed to achieve
- **Prediction is still weak.** Reliability 0.506 implies a ceiling near sqrt(0.506) ~ 0.71;
  Ridge reaches 0.072 — about **10% of the ceiling**.
- **The deep EEGNet+TCN encoder did not beat Ridge**: 0.0543 vs 0.0723 (p = 0.035, still
  above null, but lower). A nonlinear encoder did not claim the remaining headroom. Caveat:
  1380 training trials is small, training loss was still falling at 60 epochs, and no
  hyperparameter search was run — this is one data point, not a settled result.
- **Amplitude is not significant** on the new data in any configuration
  (single-trial p = 0.075, deep p = 0.378, trial-averaged p = 0.299) — the project's
  previously strongest target did not survive the move.
- **Cautionary result:** HEFMI's trial-averaged *amplitude* comes out at r = 0.306,
  "p = 0.040" — against a target whose split-half reliability is **-0.051**. A target that
  does not reproduce against itself cannot be genuinely predicted; this is what a spurious
  positive looks like, and it is a reason to re-examine earlier amplitude claims under the
  reliability gate.
"""


def apply(path: Path, header: str) -> str:
    if not path.exists():
        return "missing"
    nb = nbf.read(path, as_version=4)
    cell = nbf.v4.new_markdown_cell(MARKER + "\n" + header.strip())
    if nb.cells and nb.cells[0].cell_type == "markdown" \
            and MARKER in nb.cells[0].source:
        nb.cells[0] = cell
        action = "updated"
    else:
        nb.cells.insert(0, cell)
        action = "inserted"
    nbf.write(nb, path)
    return action


if __name__ == "__main__":
    root = Path(".")
    for name, header in H.items():
        print(f"{name:52s} {apply(root / name, header)}")
    missing = sorted(p.name for p in root.glob("NeuroVascNet*.ipynb")
                     if p.name not in H)
    if missing:
        print("\nno header defined for:", ", ".join(missing))
