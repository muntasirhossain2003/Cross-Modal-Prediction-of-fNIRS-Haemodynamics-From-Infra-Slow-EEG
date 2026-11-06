# NeuroVascNet: results against the five blueprint claims

Status as of 2026-11-07. Every test below had its pass rule written in `merge/FINAL_PLAN.md` before it was
run. Post-hoc amendments are labelled as such. Scripts and raw result files are in `merge/analysis_p2_p4/`.

## Data used

| Cohort | People | Modality | Role |
|---|---|---|---|
| TU Berlin hBCI (motor imagery + mental arithmetic) | 29 | EEG + fNIRS | primary fNIRS cohort |
| TU Berlin mental workload (n-back) | 26 | EEG + fNIRS | second fNIRS cohort / external set |
| FineMI (multi-joint upper-limb motor imagery) | 18 | EEG + fNIRS | third fNIRS cohort (different lab) |
| ds002336 (motor imagery neurofeedback, XP1) | 10 | EEG + fMRI | fMRI arm |
| ds002338 (motor imagery neurofeedback, XP2, downloaded 2026-11-07) | 17 | EEG + fMRI | fMRI arm |

HEFMI and ds002739 cannot be used for the main analysis. Their published EEG has no power below 0.5 Hz, because of a hardware filter in HEFMI and preprocessing before publication in ds002739.

## The finding the paper is built on

The EEG information about haemodynamics sits almost entirely in the **infra-slow band (< 0.5 Hz)**, and the
standard 0.5–45 Hz preprocessing removes it. With the band-pass, the original pipeline gave r = 0.019. Without
it, the same pipeline gave r = 0.158, and a plain ridge model on slow-EEG features does better than the deep model.

| Cross-person prediction of fNIRS (ridge, slow EEG < 0.5 Hz) | slow EEG | band power 1–45 Hz |
|---|---|---|
| hBCI | 0.155 (deep) / 0.197 (ridge) | ≈ 0.02 |
| Mental workload | 0.136 [0.089, 0.183] | 0.006 (n.s.) |
| FineMI | 0.203 [0.164, 0.248] | 0.046 |

These properties replicate in all three cohorts:
- **Not ocular.** EOG channels carry little or nothing.
- **Person-specific.** A person's own model predicts their data far better than other people's do: identification 93%, 62% and 83%.
- **Local component.** A local haemodynamic part remains after the global (systemic) signal is removed: HbO local r 0.07–0.16.

## Scorecard against the blueprint

| Claim | Best test | Verdict |
|---|---|---|
| **1. Cross-modal (fNIRS + fMRI) supervision / one shared space** | Linear shared latent; fMRI arm grown from 10 to 27 people | **Not confirmed (declared test).** With 27 people the fMRI output becomes predictable from EEG: cross-person 0.074 [0.046, 0.104], within-person 0.073 [0.040, 0.109]. The declared rank-8 shared space costs fNIRS 0.021. At ranks 16 and 32 the shared space serves both outputs at no cost, but this result is exploratory. fMRI supervision does not improve fNIRS prediction. |
| **2. External validation** | Model frozen on one cohort, tested on another | **Passes between the two TU Berlin cohorts:** hBCI → mental workload 0.205 [0.128, 0.285], reverse 0.177; robust to class demeaning and detrending. **Fails into FineMI** under every harmonisation tried (detrend, linked-mastoid reference). |
| **3. Physiological delay** | Multichannel lagged ridge on continuous recordings | **3a passes in hBCI** (29/29 people beat their own null; split-half reproducibility r 0.56–0.58 under both band choices). Mental workload passes only in the declared version. The group delay is robust: **slow EEG follows HbO by 1–2 s** in both TU Berlin cohorts. That is the reverse of the classical neurovascular order, and consistent with a vascular / blood–brain-barrier origin of infra-slow scalp potentials. Task-evoked timing is not consistent across people. FineMI: no reliable continuous coupling. |
| **4. Downstream EEG-only task** | Band power + EEG-predicted ("virtual") haemodynamics vs band power, a shuffled-target control and a random-projection control | **Fails on the primary tasks** (hBCI motor imagery; FineMI 8-class). A random projection of the same slow EEG helps as much as the virtual haemodynamics, so the slow band carries the gain, not the haemodynamic supervision. Passes on hBCI mental arithmetic (secondary). Side result: slow EEG lifts cross-person motor-imagery accuracy from 0.61 to 0.70–0.72. |
| **5. Calibrated reliability** | Notebook Claim 5 machinery on the ridge pipeline | **Partial.** Person-level ranking 0.219 (> 0.2); trial-level ranking, calibration gain and external transfer fail. |

## Deployment result (all three cohorts)

A person who gives two runs of simultaneous EEG + fNIRS can afterwards be monitored with EEG alone, and the
EEG model beats replaying that person's own average response (gain at k = 2 calibration runs):

| Cohort | raw target | local target (systemic part removed) |
|---|---|---|
| hBCI | +0.176 [0.135, 0.218], 29/29 | +0.103 [0.074, 0.136], 29/29 |
| Mental workload | +0.059 [0.025, 0.099] | +0.031 [0.013, 0.052] (at the threshold) |
| FineMI | +0.067 [0.033, 0.112], 18/18 | +0.041 [0.014, 0.079], 16/18 |

The gain grows with more calibration runs (hBCI: +0.118 at k = 1, +0.258 at k = 5).

## Suggested paper framing

**Infra-slow EEG is a person-specific, calibratable proxy of cortical haemodynamics that standard EEG
preprocessing discards.** The paper would rest on four results:
- replication in three cohorts from two labs;
- a reproducible 1–2 s delay in which slow EEG follows HbO;
- transfer between the two cohorts from the same lab;
- personal-calibration deployment that holds on the local, non-systemic signal.

The paper should report the negative results openly:
- The deep cross-modal model fails all five blueprint claims under both preprocessing protocols.
- fMRI supervision does not improve fNIRS prediction.
- Transfer to a different lab's hardware fails.
- The downstream and reliability claims are not supported.

## Main caveats

1. A large share of the predictable signal is global (systemic). Every claim above that says "local" was tested on the local target separately.
2. The two TU Berlin cohorts may share participants; the public files do not allow a check.
3. Several tests were rerun after a first look: the P2.5b local target, the P4.A band matching, the P4.B re-reference and the Claim 1 rank sweep. They are labelled as amendments or exploratory in `FINAL_PLAN.md`. The declared versions remain the primary record.
4. All results are from seed 1337. The ridge analyses are deterministic apart from bootstrap resampling. The deep-model results were not rerun over five seeds under the infra-slow protocol.
