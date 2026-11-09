# NeuroVascNet: results against the five blueprint claims

Status as of 2026-11-09 (after round P10). The multiplicity check across all attempts is in `TEST_LEDGER.md`. Every test below had its pass rule written in `merge/FINAL_PLAN.md` before it was
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
| Mental workload | 0.135 [0.089, 0.183] | 0.006 (n.s.) |
| FineMI | 0.203 [0.164, 0.248] | 0.045 |

These properties replicate in all three cohorts:
- **Not ocular.** EOG channels carry little or nothing.
- **Person-specific.** A person's own model predicts their data far better than other people's do: identification 93%, 62% and 83%.
- **Local component.** A local haemodynamic part remains after the global (systemic) signal is removed: HbO local r 0.07–0.16.

## Scorecard against the blueprint

Round P5 (2026-11-07) added second-round confirmatory tests, each declared in `FINAL_PLAN.md` before it was run.

| Claim | Best test | Verdict |
|---|---|---|
| **1. Cross-modal (fNIRS + fMRI) supervision / one shared space** | Linear shared latent at rank 32, fixed in advance from hBCI; tested on fNIRS cohorts never used for Claim 1 (mental workload primary, FineMI replication); fMRI arm 27 people | **Shared-space form confirmed.** One EEG-derived space predicts fNIRS (MW 0.123, FineMI 0.180) and fMRI (0.063–0.068, p 0.0001) on unseen people, at no cost to either output (90% CI lower bounds −0.014 and −0.003 > −0.02). **Stronger form not supported:** fMRI supervision does not improve fNIRS. Joint is no better than joint with shuffled fMRI targets, in all three cohorts and with only 5 training people. Holds for 5/5 person-fold seeds and on the local (channel-mean-removed) target in all three cohorts (P6.3); fMRI does not help the local target either. The declared rank-8 test (P4.1) failed on capacity; rank 32 was fixed from that sweep and confirmed on fresh cohorts. |
| **2. External validation** | Model frozen on one cohort, tested on another | **Passes within one lab:** hBCI -> mental workload 0.205 [0.128, 0.285], reverse 0.177; robust to class demeaning, detrending and Euclidean alignment. **Fails across labs (FineMI)** under every harmonisation: detrend, linked mastoids, Euclidean alignment and emulation of the TU Berlin hardware filter. With emulation the transfer becomes reliably negative (−0.064 [−0.094, −0.037]). Blink polarity is normal in all cohorts, so this is not a sign error. **fMRI output across studies** (ds002336 -> ds002338): fails, 0.017 [−0.009, +0.042]. The reverse direction is detectable but below the 0.05 threshold (0.039). **P6 (harmonised window −2..18 s + matched left-motor montage; calibrated-transfer stacking on a person's two calibration runs):** FineMI still fails (−0.010 [−0.041, +0.021]; stacking gain −0.0001). The within-lab reference passes under both (hBCI -> MW 0.230; stacking gain +0.059 [0.023, 0.095]). |
| **3. Physiological delay** | Multichannel lagged ridge on continuous recordings | **3a passes in hBCI** (29/29 people beat their own null; split-half r 0.56–0.58 under both band choices). Mental workload passes only in the declared version. Robust group result: **slow EEG follows HbO by 1–2 s** in both TU Berlin cohorts. That is the reverse of the classical neurovascular order and fits a vascular / blood–brain-barrier origin. FineMI: no reliable coupling under any EEG chain (declared, band-matched, linked mastoids, hardware-emulated: 44–50% of people beat their null). **Classical coupling (P6.4, EEG band power theta–gamma -> local HbO):** in hBCI, EEG predicts local HbO **6 s later** in 29/29 people, and the personal delay is reproducible (split-half r +0.39, p 0.02). That is the textbook neurovascular order. It survives removal of the cue-locked task response (P6.4b: 29/29, +6 s, split-half r +0.33, p 0.04), so it is ongoing coupling, not a shared task response. Coupling also exists in MW (81–85% beat null, group delay +2 to +4 s EEG-first, p 0.002) and FineMI (83–89%), but the personal delay is not reproducible there. The declared rule (hBCI plus one more cohort) is therefore **not met**. **Kernel form (P7.1, fourth round, own declared rule): SUPPORTED.** A person's lag curve in the first half of the recording matches its second half beyond a time-shifted null: hBCI (p < 0.0001) and FineMI (p 0.033), MW not (p 0.09). With the task response removed it passes in **all three** cohorts (MW p 0.006, FineMI p 0.001). Outside hBCI the curve shape repeats but the peak delay does not, so a precise personal delay is shown only in hBCI. Recording length is not the reason (P9.1): cut to MW length, the hBCI delay still repeats (34% vs 19% chance, p 0.03), and FineMI has more data per person than hBCI. Weaker coupling can account for it (median peak local r: hBCI 0.19, MW 0.14, FineMI 0.05): degraded with added noise to FineMI's coupling strength, the hBCI delay no longer repeats (P10.2: 10% vs 16% chance). The FineMI delay result is therefore inconclusive, not negative. **Claim 2 coupling-transfer form (P7.2):** the frozen hBCI band-power kernel transfers within lab (hBCI -> MW +0.098 [0.071, 0.124]) but is negative into FineMI (−0.053 [−0.071, −0.034]). The FineMI failure is therefore not specific to the slow band, and a CBSI target does not fix it (P8.2: −0.055). **Calibrated cross-lab form PASSES (P8.3, robust 5/5 seeds):** the frozen hBCI band-power kernel adds to a FineMI person's own two-run model, +0.020 [+0.004, +0.039] (11/18; seeds 1–5: +0.015 to +0.031, all CI > 0). Limit: in 72–89% of FineMI people the fitted external gain is negative, so the kernel transfers up to its sign. The effect is small. **P9 (fifth round, diagnostic):** the inversion is broadband (every single EEG band transfers negatively into FineMI and positively into MW), and FineMI people share a kernel of their own (FineMI leave-one-person-out +0.074 [+0.047, +0.105], 67% beat null). Coupling generalises across people within each lab; the kernel is lab-specific. A classical alpha + beta kernel does not transfer either (P9.3 FAIL: frozen −0.050, calibrated +0.015 [−0.003, +0.035]). The inversion is symmetric (FineMI kernel on hBCI −0.029) and broadband EEG power vs global HbO has opposite signs, negative in hBCI (−0.038 [−0.055, −0.022]) and positive in FineMI (+0.044 [+0.023, +0.067]) (P9.4). This is a candidate explanation only: the correlations are small, MW includes zero, and the prediction came after the inverted kernels had been seen. Tested directly (P10.1), removing the broadband term shrinks the inversion but does not remove it (−0.053 to −0.036, CI < 0), so that explanation is not supported. **Multiplicity:** the P8.3 cross-lab pass is nominal (p 0.017; α 0.0031 over 16 cross-lab attempts); see `TEST_LEDGER.md`. |
| **4. Downstream EEG-only task** | Virtual haemodynamics vs shuffled-target, random-projection and PCA controls | **Refuted in every form:** standard, data-scarce (5 labelled people) and representation-only. A PCA code of the same slow EEG is as good or better (hBCI motor imagery: PCA 0.760 vs virtual haemodynamics 0.697). The benefit is the slow band itself: band power 0.61 vs 0.70–0.76 with slow EEG. |
| **5. Calibrated reliability** | Person level, in the calibrated deployment: does a new person's score on their own two calibration runs predict how well the model works on their later runs? | **Person-level calibrated form holds in all three cohorts:** Spearman hBCI 0.68 [0.37, 0.86], MW 0.56 [0.22, 0.78], FineMI 0.90 [0.69, 0.97]. It also holds on the local target, and for the EEG-specific gain (0.71 / 0.83 / 0.87). Holds for 5/5 seeds of the calibration-run choice; mental workload alone passes for 4/5. Trial-level reliability still fails, both the notebook form (P3.5) and a calibrated form (P5.7). The predicted-signal size ranks trials only weakly (0.16–0.22). |

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
- classical neurovascular coupling: EEG band power leads local HbO by 6 s in every hBCI person, reproducibly, beyond the task response. The person's coupling kernel is reproducible beyond the task in all three cohorts (P7.1); the precise personal delay only in hBCI;
- transfer between the two cohorts from the same lab, and a calibrated cross-lab transfer of the band-power coupling kernel (small, sign-inverted in the other lab);
- band-power coupling generalises across people within each lab (hBCI and FineMI leave-one-person-out both pass), but the kernel is lab-specific and broadband sign-inverted between labs;
- personal-calibration deployment that holds on the local, non-systemic signal;
- one shared EEG space for fNIRS and fMRI, confirmed on fresh cohorts (Claim 1, shared-space form);
- per-person reliability known in advance from two calibration runs (Claim 5, person-level form).

The paper should report the negative results openly:
- The deep cross-modal model fails all five blueprint claims under both preprocessing protocols.
- fMRI supervision does not improve fNIRS prediction; only the shared-space form of Claim 1 holds.
- Transfer to a different lab's hardware fails, and becomes negative when the hardware filter is emulated.
- The downstream claim is refuted: unsupervised PCA of slow EEG does as well.
- Trial-level reliability is not supported; only person-level reliability is.

## Main caveats

1. A large share of the predictable signal is global (systemic). Every claim above that says "local" was tested on the local target separately.
2. The two TU Berlin cohorts may share participants; the public files do not allow a check.
3. Several tests were rerun after a first look: the P2.5b local target, the P4.A band matching, the P4.B re-reference, the Claim 1 rank sweep and the P5.3b representation form of Claim 4. Rank 32 for the
   confirmatory Claim 1 test was chosen from the exploratory sweep, then tested on fNIRS cohorts never used for Claim 1.
   The fMRI arm (27 people) is the same one used in P4.1, so only the fNIRS side of that confirmation is fresh. They are labelled as amendments or exploratory in `FINAL_PLAN.md`. The declared versions remain the primary record.
4. Most results are from seed 1337; the two P5 passes (Claims 1 and 5) were rerun with five seeds and hold in 5/5. The ridge analyses are deterministic apart from bootstrap resampling. The deep-model results were not rerun over five seeds under the infra-slow protocol.
