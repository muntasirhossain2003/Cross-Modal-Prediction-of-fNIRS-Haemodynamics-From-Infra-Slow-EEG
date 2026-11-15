# NeuroVascNet — Rebuild Workflow (the path)

**Thesis:** one EEG encoder, trained with paired EEG-fNIRS **and** EEG-fMRI supervision, that at deployment takes **EEG only** and emits hemodynamic-aware outputs.

```
TRAIN:   EEG ──► fNIRS HbO/HbR   (fNIRS head)
         EEG ──► fMRI ROI-BOLD   (fMRI head)
DEPLOY:  EEG only ──► HbO/HbR-like + BOLD-like prediction
                     + hemodynamic-aware latent
                     + neurovascular delay estimate
                     + calibrated reliability / abstention
```
Goal is **prediction** (not classification, not fusion — fusion would need fNIRS at deployment and breaks the EEG-only premise).

---

## Lessons baked into this plan (so we don't repeat them)
1. **Verify the signal BEFORE building the model.** Last time we built everything, then found prediction ≈ 0. This time: a **signal-feasibility gate** comes first for every dataset.
2. **Strong-signal data first.** Cognitive-task scalp fNIRS (Stroop/n-back) had almost no recoverable response. **Motor tasks** (HEFMI-ICH, Multi-joint) give far stronger, localized hemodynamics. Start there.
3. **Always evaluate leakage-controlled**, within-subject AND cross-subject, with peak-xcorr + decode-from-prediction, not raw Pearson alone.
4. **Reliability gates the weakness** — keep it; it is the deployable safeguard and worked.

---

## THE PATH — stages with go/no-go gates

### Stage 0 — Foundation (data + feasibility) ⛳ *gate before any modeling*
- **0.1** Pick datasets by signal strength:
  - fNIRS supervision: **HEFMI-ICH** (motor imagery, 17 healthy + 20 patients), **Multi-joint** (motor, 18 subj).
  - fMRI supervision: **ds002336** (train), **ds002739** (external).
- **0.2** Write loaders → common dict schema `{eeg, hemo, labels, subject_ids, true_delays}`. Cache to `.npz`.
- **0.3 SIGNAL GATE (critical):** for each dataset, *before modeling*, measure the **task-evoked hemodynamic response** and a simple **EEG-bandpower→hemo coupling** (ridge, leakage-controlled).
  - ✅ If a dataset shows real coupling (e.g., |r| clearly > null in most subjects) → keep it for prediction.
  - ❌ If ≈ 0 → it cannot support the prediction thesis; drop or use only as a stress-test.
  - **Decision:** proceed to modeling only with datasets that pass.

### Stage 1 — Encoder (start here, step by step)
- **1.1** Shared EEG encoder (EEGNet front-end → TCN bottleneck → latent `z`).
- **1.2** Verify: input shape, latent shape, parameter count; sanity-overfit a tiny batch to confirm it *can* learn.
- **Gate:** encoder produces a stable latent and can overfit a few samples.

### Stage 2 — Heads (prediction targets)
- **2.1** fNIRS head → HbO/HbR time-series.
- **2.2** fMRI head → ROI-BOLD time-series.
- **2.3** Heteroscedastic output (mean + variance) for reliability later.
- **Gate:** each head maps latent → correct output shape; trains on its own modality.

### Stage 3 — Delay module (neurovascular coupling)
- **3.1** Per-modality learnable 2–8 s delay applied before the loss.
- **3.2** Expose the delay as an interpretable output.
- **Gate:** recovered delays fall in the physiological range.

### Stage 4 — Full model + training
- **4.1** Assemble encoder + both heads + delay (+ reliability stub).
- **4.2** Stage-1 joint supervised training (fNIRS + fMRI, balanced batches).
- **4.3** Stage-2 cross-modal latent alignment.
- **Gate:** training is stable; **prediction beats a shuffled-label / mean baseline on internal validation** (the real go/no-go for the thesis).

### Stage 5 — Rigorous evaluation
- **5.1** Within-subject (clean, block-grouped) + cross-subject (LOSO).
- **5.2** Metrics: MAE/RMSE, Pearson, peak-xcorr, decode-from-prediction, delay distribution.
- **5.3** Relative test: full cross-modal **vs** single-modality & independent-encoder baselines.
- **5.4** External validation (held-out dataset, no tuning).
- **Gate:** report whichever is true — a positive prediction result, or an honest characterization.

### Stage 6 — Reliability & deployment outputs
- **6.1** Learned gating over aleatoric + epistemic + domain-shift signals.
- **6.2** Temperature scaling → isotonic calibration (internal only).
- **6.3** Inference pipeline with abstention (`deploy_infer`).
- **6.4** Selective-prediction curves on external data.

### Stage 7 — Clinical angle (HEFMI-ICH bonus)
- **7.1** Healthy→patient transfer: does the EEG-only model behave/transfer on ICH patients?
- **7.2** Report patient-vs-healthy reliability and delay differences (clinically meaningful even if prediction is modest).

---

## Visual flow
```
 Stage 0  DATA + SIGNAL GATE ──❌──► drop dataset
    │ ✅ (signal present)
 Stage 1  ENCODER ──► Stage 2 HEADS ──► Stage 3 DELAY
    │
 Stage 4  TRAIN (joint + align) ──❌ beats baseline?──► reframe honestly
    │ ✅
 Stage 5  EVALUATE (within/cross/external)
    │
 Stage 6  RELIABILITY + ABSTENTION ──► Stage 7  CLINICAL TRANSFER
    │
   4 deployment outputs from EEG alone
```

---

## How we execute
We go **one stage at a time**, top to bottom. Each stage has a **gate** — we don't advance until it passes (or we consciously decide to). Stage 0's signal gate is the most important: it tells us *early* whether a dataset can support prediction, so we never again build the whole model on data that can't.

**Next action (tomorrow):** Stage 0.3 — run the signal-feasibility gate on **HEFMI-ICH** (strong motor signal, the most promising). If it passes, proceed to Stage 1 (encoder) on that data.
