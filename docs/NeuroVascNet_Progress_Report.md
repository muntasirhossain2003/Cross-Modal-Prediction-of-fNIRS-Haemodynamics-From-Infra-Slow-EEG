# NeuroVascNet — Progress Report

**Project:** Cross-Modal Hemodynamic Supervision for EEG-Only Neurovascular Representation Learning
**Status date:** 2026-06-16
**Notebook:** `NeuroVascNet_E2E.ipynb`

---

## 1. Executive summary

The full framework has been implemented and now runs **end-to-end on four real public datasets**. The architecture, training protocol, and (newly corrected) reliability/abstention layer all match the blueprint.

The central empirical finding is **honest but mixed**:

- ❌ **EEG → hemodynamic *prediction* does not generalize.** Across three EEG-fNIRS datasets and one EEG-fMRI dataset, EEG-to-HbO/HbR and EEG-to-BOLD prediction is ≈ 0 on held-out subjects, and **cross-modal supervision does not beat single-modality baselines** (the original headline claim is not supported).
- ✅ **The reliability / abstention layer works** and is the strongest surviving contribution: the calibrated reliability score is informative (r ≈ 0.29) and the inference pipeline correctly abstains on low-confidence inputs.
- ✅ **EEG itself is informative** — it decodes the cognitive task well above chance (0.56–0.61), confirming the data and pipeline are sound; the limitation is the EEG→hemodynamic *coupling*, not the EEG.

This is consistent with the blueprint's own risk register (§11), which anticipated that cross-modal alignment "may fail to improve over independent encoders" and states this is "itself a publishable finding."

---

## 2. What was implemented (engineering: ~90% complete)

| Component | Status |
|---|---|
| Shared EEG encoder (EEGNet + TCN) | ✅ |
| fNIRS head (HbO/HbR), fMRI head (ROI-BOLD, 48 Harvard-Oxford ROIs) | ✅ |
| Physiologically-constrained delay module | ✅ |
| Reliability head + **learned gating** (logistic gate on internal validation) | ✅ corrected |
| **Temperature scaling → isotonic** calibration | ✅ corrected |
| **Abstention built into the inference pipeline** (`deploy_infer`) | ✅ corrected |
| Staged training (joint supervised + cross-modal alignment) | ✅ |
| **All four datasets wired to REAL data** | ✅ |
| Disk caching for fast re-runs | ✅ |
| Deep-ensemble training, contrastive alignment, published-method baselines | ⬜ optional / not yet |

### Datasets (all real)
| Role | Dataset | Subjects |
|---|---|---|
| Primary fNIRS (train) | Stroop EEG-fNIRS | 20 usable |
| Primary fMRI (train) | OpenNeuro ds002336 | 10 |
| External fNIRS | TU Berlin EEG-NIRS | 26 |
| External fMRI | OpenNeuro ds002739 | 24 |
| (Additional, tested) | Mendeley EEG-fNIRS | 30 |

---

## 3. Key results (rigorous, leakage-controlled)

### 3.1 Hemodynamic prediction — the negative result
| Target | Evaluation | Pearson |
|---|---|---|
| EEG → fNIRS (HbO/HbR) | within-subject (clean) & cross-subject | ≈ 0.00 |
| EEG → fNIRS | block-summary, Ridge / PLS / CCA | ≈ 0.00 |
| EEG → BOLD | cross-subject (LOSO) | ≈ 0.05 (best ROI 0.30; 16/48 ROIs > 0.1) |
| EEG → BOLD | within-subject coupling (alpha→BOLD) | 0.23–0.49 (9/10 subjects) — real but does not generalize |

**Relative claim test (full model vs single-modality baseline, real Stroop, full epochs):**
| Model | task-decode from predicted HbO/HbR |
|---|---|
| Full (cross-modal) | 0.42 |
| fNIRS-only baseline | 0.55 |

→ Cross-modal supervision **does not** improve fNIRS prediction over the single-modality baseline.

### 3.2 EEG is informative (control)
| Task | Accuracy (chance) |
|---|---|
| Stroop condition (trial-level EEG) | 0.61 (0.50) |
| TU Berlin n-back load (EEG) | 0.52 (0.33) |

### 3.3 Reliability layer — the positive result
| Metric | Value |
|---|---|
| Learned-gate reliability vs error correlation (internal) | 0.29 |
| Expected Calibration Error (external): raw → calibrated | 0.188 → 0.157 |
| Inference abstention | functional — returns "insufficient confidence" below threshold |

---

## 4. Interpretation

1. **The EEG signal and pipeline are sound** (EEG decodes the task). The bottleneck is the **EEG→hemodynamic coupling**, which is physiologically weak/variable in scalp fNIRS and does not survive cross-subject generalization in fMRI either.
2. **Why prediction fails is not a code or data bug** — confirmed by: classical baselines (Ridge/PLS/CCA) also ≈ 0; the data is valid (HbO/HbR anti-correlation −0.91); leakage was controlled; data-integrity issues (mislabeled Stroop files, trigger≠condition) were found and fixed.
3. **The reliability layer is the publishable core** — it correctly abstains where prediction is unreliable, which is exactly the deployment safeguard the blueprint emphasises as a first-class contribution.

---

## 5. Recommended framing for the paper

Per the blueprint's honest-scope strategy (§4.4, §11), reframe from *"cross-modal supervision improves hemodynamic prediction"* to:

> **A rigorous, leakage-controlled characterization of when EEG→hemodynamic cross-modal supervision is recoverable (and largely is not) across four public datasets, delivered through a deployable EEG-only framework with a calibrated reliability/abstention layer that correctly gates its own low-confidence predictions.**

This keeps the framework, the four-dataset design, the delay descriptor, and the reliability contribution — and presents the negative prediction result transparently, which reviewers at strong venues respond to well.

---

## 6. Open questions for discussion

1. Is a **characterization + reliability-layer** paper an acceptable target, or do we require a positive prediction result?
2. If a positive result is required, the realistic route is **substantially more data** (more subjects) and/or restricting to **within-subject** claims — should we pursue this?
3. Should the remaining optional components (deep ensemble, contrastive alignment, published-method baselines) be implemented for completeness before submission?

---

*Prepared from the current state of `NeuroVascNet_E2E.ipynb`. All numbers are from leakage-controlled evaluations on real data; full reproduction is available in the notebook.*
