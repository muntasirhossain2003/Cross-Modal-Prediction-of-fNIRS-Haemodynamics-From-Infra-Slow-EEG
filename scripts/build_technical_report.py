"""Builds merge/NeuroVascNet_Technical_Report.pdf: technical results and the step-by-step record, status after round P15."""
import shutil
from pathlib import Path

import lecture_common as L

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "merge" / "paper" / "figures"
d = L.Doc("technical_report", "NeuroVascNet_Technical_Report.pdf", "NeuroVascNet technical report")
d.pdf = ROOT / "merge" / "NeuroVascNet_Technical_Report.pdf"
_t, _f = [0], [0]


def T(title, head, rows, widths=None, note=""):
    _t[0] += 1
    h = L.table(_t[0], title, head, rows, widths, note)
    return h.replace('<div class="tbl">', '<div class="tbl" style="break-inside:auto">', 1) if len(rows) > 12 else h


def F(name, title, body="", width=100):
    _f[0] += 1
    shutil.copy(FIGS / f"{name}.png", d.figdir / f"{name}.png")
    return f'<figure><img src="figures/{name}.png" style="width:{width}%;margin:0 auto"><figcaption><b>Figure {_f[0]}. {title}</b> {body}</figcaption></figure>'


Y, N, M_ = '<span class="yes">Holds</span>', '<span class="no">Fails</span>', '<span class="mid">Partly</span>'
B = []
B.append(L.title_block("NeuroVascNet", "Technical results and step-by-step record",
                       "Models, tests, results, every step in order, and what to do next",
                       "Status after analysis round P15 &nbsp;·&nbsp; 10 November 2026 &nbsp;·&nbsp; branch data-quality-audit, commit ece8015"))

# ------------------------------------------------------------------ 1
B.append("<h2>1. Bottom line</h2>")
B.append("<p>The project set out to predict haemodynamic signals (fNIRS HbO/HbR and fMRI BOLD) from EEG, with five claims taken from the blueprint. The deep cross-modal model fails all five. Plain ridge regression on EEG below 0.5 Hz supports Claims 1, 2, 3 and 5 in narrower forms than the blueprint stated, and refutes Claim 4.</p>")
B.append("<p>The central finding is that the EEG information about haemodynamics lies almost entirely in the infra-slow band (below 0.5 Hz), which the standard 0.5–45 Hz preprocessing removes. In hBCI, the same deep pipeline gives r = 0.019 with the band-pass and r = 0.158 without it.</p>")
B.append(T("Scorecard against the five claims. Numbers are correlations unless stated. CIs are 95%.",
           ["Claim", "Status", "Form that holds", "What does not hold"],
           [["1. One space for fNIRS and fMRI", M_, "One shared EEG space (rank 32) predicts fNIRS on unseen people (MW 0.123, FineMI 0.180) and fMRI (0.063–0.068, p 0.0001), at no cost to either output.", "fMRI supervision does not improve fNIRS. The fMRI side (27 people) is not a fresh sample."],
            ["2. External validation", M_, "Within one lab: hBCI to mental workload 0.205 [0.128, 0.285]; reverse 0.177.", "Frozen transfer across labs (FineMI, third lab). fMRI across studies (0.017). Pooled training does not help."],
            ["3. Physiological delay", M_, "EEG band power leads local HbO by about 6 s in hBCI (29/29 people), reproducibly, beyond the task response. Coupling kernel repeats beyond the task in hBCI and FineMI.", "A precise personal delay outside hBCI (inconclusive). HbR timing signature not resolved."],
            ["4. Downstream EEG-only task", N, "None.", "PCA of the same slow EEG is as good or better (0.760 against 0.697 for virtual haemodynamics)."],
            ["5. Calibrated reliability", M_, "Two calibration runs predict a person's later accuracy: Spearman 0.68, 0.56, 0.90 (cue-aligned); 0.85, 0.83, 0.84 (no cues). Not explained by signal amplitude or EEG power.", "Trial-level reliability. Ranking people in the third lab (ρ 0.29, n = 15)."]],
           ["19%", "9%", "42%", "30%"], "MW = TU Berlin mental workload. Every test had its pass rule written before it was run, unless labelled post hoc."))
B.append(L.callout("myth", "Unresolved quality flag on FineMI", "<p>The July 2026 data audit measured FineMI (listed there as Multi-joint) with a median scalp coupling index of 0.023, evoked SNR 0.04 and split-half reliability near zero, and recommended dropping it. It was later adopted as the third cohort, and no later round re-examined that flag. FineMI's prediction results (r up to 0.20) therefore need an explanation: they could reflect a slow signal shared by EEG and poorly coupled optodes rather than cortical haemodynamics. Section 10 lists this as the first thing to check.</p>"))

# ------------------------------------------------------------------ 2
B.append("<h2>2. Data</h2>")
B.append(T("Datasets used in the analysis.",
           ["Cohort", "People", "Paradigm", "Recording", "Role"],
           [["TU Berlin hBCI", "29", "Motor imagery and mental arithmetic, 6 runs", "EEG 30 ch, fNIRS 36 ch", "Primary fNIRS cohort"],
            ["TU Berlin mental workload", "26", "n-back, 3 runs", "EEG, fNIRS; BrainAmp time constant 10 s", "Second fNIRS cohort, same lab"],
            ["FineMI (Multi-joint Upper-Limb)", "18", "Upper-limb motor imagery, 8 runs", "EEG Neuroscan (DC-coupled), fNIRS NIRx 24 ch", "Third fNIRS cohort, different lab"],
            ["Mendeley Learning Capability", "30 (15 usable)", "Implicit learning", "EEG, fNIRS", "Confirmation in a third lab (P12–P15)"],
            ["OpenNeuro ds002336", "10", "Motor-imagery neurofeedback", "EEG, fMRI", "fMRI arm"],
            ["OpenNeuro ds002338", "17", "Motor-imagery neurofeedback", "EEG, fMRI", "fMRI arm, added 7 Nov"]],
           ["25%", "9%", "26%", "22%", "18%"]))
B.append("<p><b>Excluded.</b> HEFMI has a 0.5–30 Hz hardware filter and ds002739 was filtered before publication, so neither has EEG below 0.5 Hz. HEFMI's fNIRS was also found to be systemic-dominated in the audit. The Stroop set was used only for early band-power baselines in June 2026.</p>")
B.append("<p><b>Third-lab inclusion.</b> 15 of the 30 Mendeley people pass the sync and length rules that were declared before any EEG–fNIRS relation was computed. An early band-power baseline had been run on this cohort in June 2026 (r about 0.00), so it is not fully untouched.</p>")

# ------------------------------------------------------------------ 3
B.append("<h2>3. Methods</h2>")
B.append("<h3>3.1 Three analysis pipelines</h3>")
B.append(T("The analysis forms and where each is used.",
           ["", "Window analysis", "Continuous coupling analysis", "P12.1 continuous prediction"],
           [["Used for", "Founding results, Claims 1, 2, 4, 5", "Claim 3 and kernel transfer", "Third-lab confirmation only"],
            ["Unit", "30 s windows at task cues; from P13 also consecutive 30 s windows not aligned to cues", "Whole runs at 1 Hz", "Continuous segments at 1 Hz"],
            ["EEG features", "Low-pass below 0.5 Hz (4th-order Butterworth, zero phase), each window z-scored, 30 × 1 s bins per channel (900 features). Comparison: log band power, 5 bands from 1 to 45 Hz, 10 bins", "Log power of theta (4–8), alpha (8–13), beta (13–30), gamma (30–45 Hz), 1 s means, band-passed 0.01–0.1 Hz", "Slow EEG or band power, lags −12 to +12 s in 4 s steps"],
            ["Target", "HbO and HbR band-passed 0.01–0.2 Hz, 2 s baseline, 30 bins. P3.2 and P15.1 use pooled channel means of HbO and HbR. Uncued windows use HbO only", "Local HbO (channel minus channel mean), 0.01–0.1 Hz. Global target also reported", "Global HbO, 0.01–0.1 Hz"],
            ["Model", "Ridge, α grid 10<sup>0</sup> to 10<sup>6</sup> (13 values), RidgeCV", "One ridge per lag from −10 to +20 s, α grid 10<sup>0</sup> to 10<sup>4</sup>", "RidgeCV"],
            ["Validation", "Leave one person out; leave one run out within person; or frozen on one cohort", "Leave one run out within person", "Leave one person out"]],
           ["13%", "37%", "29%", "21%"]))
B.append("<h3>3.2 Scores</h3>")
B.append("<p><b>Trial correlation.</b> Prediction and target are demeaned within person × task, which removes each person's average response so that a model cannot score by replaying it. For each window, the score is the Pearson correlation between the predicted and measured channel-by-time pattern.</p>")
B.append("<p><b>r<sub>subject</sub></b> is the mean over people of each person's mean window correlation. The person is the statistical unit throughout.</p>")
B.append("<p><b>Coupling and delay.</b> Each lag gets its own ridge model scored on held-out runs. A person's lag curve is the score at each lag, and their delay is the curve's peak. A positive lag means EEG leads.</p>")
B.append("<p><b>Claim 5 score.</b> Two calibration runs are drawn with seed 1337. The model is a hybrid: the person's calibration mean plus the average of a personal ridge model and a population ridge model. The calibration score is cross-fitted between the two runs, and the test score comes from the person's remaining runs. The gain form subtracts the score of replaying the person's own mean response.</p>")
B.append("<h3>3.3 Statistics</h3>")
B.append("<ul><li><b>Verdict “detectable”</b> (from the notebook): sign-flip p &lt; 0.05 on the per-person score minus that person's within-person pairing null, CI lower bound above the null mean, and r ≥ 0.05 (the practical size).</li>"
         "<li><b>Confidence intervals:</b> bootstrap over people, seed 1337.</li>"
         "<li><b>Chance for lag curves:</b> circular shifts of the haemodynamic signal by at least 60 s. A person beats their null above the 95th percentile of their own surrogates.</li>"
         "<li><b>Non-inferiority (Claim 1):</b> joint minus modality-specific model, 90% CI lower bound above −0.02.</li>"
         "<li><b>Reliability rule (P5.4):</b> Spearman above 0.2 and 95% CI above 0.</li></ul>")
B.append("<h3>3.4 Controls on analytic freedom</h3>")
B.append("<p>Each round's rules were written into <code>merge/FINAL_PLAN.md</code> and normally committed before the tests ran. Anything decided after seeing a result is labelled post hoc or amendment, and the declared version stays the primary record. <code>merge/TEST_LEDGER.md</code> applies a Bonferroni correction across every attempt at a claim (8, 3 within-lab plus 16 cross-lab, 14, 5 and 4 attempts), and each later round has its own family (P12: 6 tests; P13: 7; P14: 12; P15: 15).</p>")

# ------------------------------------------------------------------ 4
# ------------------------------------------------------------------ 4 (pipeline)
def deep_diagram():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(L.W, 2.9))
    ax.set_xlim(-0.08, 10.08)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    L.box(ax, 0.0, 2.75, 1.45, 0.95, "hBCI EEG\n30 ch × 6000", fs=8)
    L.box(ax, 0.0, 0.7, 1.45, 0.95, "ds002336 EEG\n63 ch × 6000", fs=8)
    L.box(ax, 1.85, 2.75, 1.2, 0.95, "Adapter\n1×1 conv\n30 to 64", fs=8)
    L.box(ax, 1.85, 0.7, 1.2, 0.95, "Adapter\n1×1 conv\n63 to 64", fs=8)
    L.box(ax, 3.45, 1.45, 2.3, 1.5, "Shared trunk\n3 conv blocks\ntime 6000 to 1200, 300, 100\npool to 16, flatten 1024", fs=7, fc="#e8f0fb", ec=L.BLUE)
    L.box(ax, 3.45, 0.2, 2.3, 0.7, "Subject embedding, 8 values\n(average if unseen)", fs=7)
    L.box(ax, 6.1, 1.65, 0.85, 1.1, "z\n1032", fs=8.5, fc="#f6f2fb", ec=L.VIOLET)
    L.box(ax, 7.35, 2.75, 2.6, 0.95, "fNIRS head: 256, dropout 0.3\nmean and log-variance\n72 × 300", fs=7.6)
    L.box(ax, 7.35, 0.7, 2.6, 0.95, "fMRI head: 256, dropout 0.3\nmean and log-variance\n5 ROI × 15", fs=7.6)
    for a, b in (((1.47, 3.22), (1.83, 3.22)), ((1.47, 1.17), (1.83, 1.17)), ((3.07, 3.22), (3.43, 2.5)), ((3.07, 1.17), (3.43, 1.9)), ((5.77, 2.2), (6.08, 2.2)), ((5.77, 0.6), (6.35, 1.63)), ((6.97, 2.45), (7.33, 3.2)), ((6.97, 1.95), (7.33, 1.2))):
        L.arrow(ax, a, b)
    d.save(fig, "deep_model")


deep_diagram()
_f[0] += 1
B.append("<h2>4. Machine-learning and deep-learning pipeline, step by step</h2>")
B.append("<p>Two model families were used. The deep network came first, because the blueprint specified it. It failed all five claims, so every claim that holds rests on the linear models in section 4.2. Model settings below are read from the code in <code>merge/Neurovascnet_marge_v1.ipynb</code> and <code>merge/analysis_p2_p4/</code>. Results are the recorded values in <code>merge/FINAL_PLAN.md</code>.</p>")
B.append("<h3>4.1 Deep model</h3>")
B.append(L.fig("deep_model", _f[0], "Deep network (trunk encoder, raw EEG input).", "Both modalities share one trunk. Each dataset has its own input adapter, and each modality has its own output head."))
B.append("<ol><li><b>Prepare windows.</b> Each trial is cut from −2 to +28 s. EEG is 200 Hz (6000 samples) and z-scored per trial. fNIRS is resampled to 10 Hz (300 samples), band-passed 0.01–0.2 Hz and baseline-corrected over 2 s. fMRI is 15 volumes (TR 2 s). Global signal regression is off, because the audit showed it removes the evoked response.</li>"
         "<li><b>Choose the EEG protocol.</b> <code>EEG_FILTER = True</code> applies 0.5–45 Hz plus a 50 Hz notch (blueprint protocol). <code>False</code> keeps the infra-slow band (P2 protocol). Both were run.</li>"
         "<li><b>Choose cohorts.</b> Training: hBCI (3480 windows, motor imagery and arithmetic pooled) with ds002336 (384 windows, 5 motor ROIs). External: HEFMI and ds002739.</li>"
         "<li><b>Encode.</b> A 1×1 convolution maps each montage to 64 channels. Three convolution blocks then shrink time, and adaptive pooling gives 16 steps (next table).</li>"
         "<li><b>Add the subject embedding.</b> An 8-value vector per person, zero-initialised, is concatenated with the 1024 trunk values to give z (1032 values).</li>"
         "<li><b>Decode.</b> Each modality has a head: linear to 256, GELU, dropout 0.3, then two linear outputs for the mean and log-variance of every channel and time point. Log-variance is clamped to [−8, 8].</li>"
         "<li><b>Compute the loss.</b> See below.</li>"
         "<li><b>Train jointly.</b> AdamW, learning rate 10<sup>−3</sup>, weight decay 10<sup>−4</sup>, batch 64, 30 epochs. One epoch is as long as the largest dataset, and smaller datasets cycle. Each step adds one fNIRS batch loss and one fMRI batch loss into a single backward pass.</li>"
         "<li><b>Validate.</b> Five subject-wise folds, fixed by seed 1337. Every fold trains a new model with the held-out people removed from training.</li>"
         "<li><b>Predict.</b> Held-out people get the average subject embedding, so no identity reaches the model. Predictions and targets are demeaned per person × task and scored by r<sub>subject</sub>.</li></ol>")
B.append(T("Deep network layers and output shapes for one hBCI batch (B windows).",
           ["Layer", "Settings", "Output shape", "Parameters"],
           [["Input", "EEG, 30 channels", "B × 30 × 6000", "—"],
            ["Adapter (one per dataset)", "Conv1d, kernel 1", "B × 64 × 6000", "6,320 including the embedding"],
            ["Block 1", "Conv1d kernel 25, stride 5, BatchNorm, GELU", "B × 64 × 1200", "Trunk total 201,280"],
            ["Block 2", "Conv1d kernel 15, stride 4, BatchNorm, GELU", "B × 64 × 300", "(included)"],
            ["Block 3", "Conv1d kernel 9, stride 3, BatchNorm, GELU", "B × 64 × 100", "(included)"],
            ["Pool and flatten", "AdaptiveAvgPool1d(16)", "B × 1024", "0"],
            ["Subject embedding", "8 values per person, concatenated", "B × 1032", "(in adapter row)"],
            ["fNIRS head", "256 hidden, dropout 0.3, mean + log-variance", "B × 72 × 300 (twice)", "11,366,848"],
            ["fMRI head", "256 hidden, dropout 0.3, mean + log-variance", "B × 5 × 15 (twice)", "302,998"],
            ["Whole network", "", "", "11,877,446"]], ["24%", "38%", "19%", "19%"]))
B.append("<p class=\"formula\">L = L<sub>Pearson</sub> + 0.1 · L<sub>NLL</sub> + 0.05 · L<sub>smooth</sub> &nbsp;&nbsp;&nbsp; L<sub>total</sub> = L<sub>fNIRS</sub> + w<sub>fMRI</sub> · L<sub>fMRI</sub>, &nbsp; w<sub>fMRI</sub> = 1</p>")
B.append("<ul><li><b>L<sub>Pearson</sub></b> is 1 minus the correlation over time for each channel. It targets the shape and ignores amplitude.</li>"
         "<li><b>L<sub>NLL</sub></b> is the Gaussian loss exp(−s)(μ − y)<sup>2</sup> + s, with s the predicted log-variance. It trains the uncertainty output used by Claim 5.</li>"
         "<li><b>L<sub>smooth</sub></b> is the mean squared difference between neighbouring time points.</li></ul>")
B.append(T("How the deep model addressed each claim, and its result with the infra-slow band kept (P2.1, seed 1337).",
           ["Claim", "Deep-learning method", "Result", "Verdict"],
           [["1", "Train three arms on the same folds: fNIRS only, fNIRS + fMRI, and fNIRS + fMRI with shuffled fMRI targets", "hBCI single 0.162, joint 0.158, shuffled 0.154; joint − single −0.004 [−0.017, +0.008]. ds002336 fMRI single 0.040, joint 0.034", "Not confirmed"],
            ["2", "Train on everything, freeze, map new montages by spherical-spline interpolation, test on external cohorts", "HEFMI 0.005 [−0.005, +0.016]; ds002739 0.010 [0.002, 0.018]", "Fails"],
            ["3a", "Data-driven lag from the trained model, with a shuffled-lag null", "hBCI 5.98 s, ICC 0.20; HEFMI 5.99 s, ICC 0.54", "Fails"],
            ["3b", "Delay head on z (linear 64, GELU, linear 1), bounded or free, supervised toward the empirical lag", "Sticks to the window edge (2.005 s in [2, 8] s)", "Fails"],
            ["4a", "Freeze the encoder and fit a ridge classifier on z (motor imagery, 3 person folds; one encoder per fold)", "Supervised 0.563; band power 0.568; random init 0.600; shuffled 0.563", "Fails"],
            ["4b", "Fine-tune from the supervised encoder against training from scratch", "0.735 against 0.754", "Fails"],
            ["5", "Three uncertainty signals (20-pass MC-dropout spread, predicted variance, latent distance), logistic gate, temperature scaling, isotonic regression, abstention at 75% coverage", "Rank internal +0.017, external +0.032; ECE 0.070 to 0.072", "Fails"]], ["6%", "45%", "37%", "12%"], "Blueprint-protocol run (0.5–45 Hz filter): cross-person r = 0.019. Five seeds were declared (1337, 20260726, 7, 42, 2718); under the infra-slow protocol only seed 1337 was run."))
B.append("<p><b>Other deep-learning attempts.</b> Before the merge (July 2026), removing alignment, a windows-gap fix and CORAL domain adaptation on the latent were tried, and none moved cross-person waveform prediction. A second encoder (EEGNet front end plus a 4-layer TCN) and a fixed band-power front end exist in the merged notebook as declared secondary options. All numbers in this report come from the trunk encoder with raw EEG.</p>")
B.append("<p><b>Leakage safeguards in the deep pipeline.</b> Folds split by person. Each fold has a new model. An unseen person gets the average embedding. BatchNorm stays in evaluation mode during MC dropout. Claim 4 trains a separate encoder for each probe fold. Claim 5 calibration uses three nested inner folds.</p>")

B.append("<h3>4.2 Ridge pipeline (the models behind every claim that holds)</h3>")
B.append("<ol><li><b>Start from the same windows</b> as the deep model, with the EEG unfiltered at load time.</li>"
         "<li><b>Extract slow EEG.</b> Fourth-order Butterworth low-pass at 0.5 Hz, zero phase.</li>"
         "<li><b>Bin.</b> Average to 30 × 1 s bins per channel. In hBCI that is 30 channels × 30 bins = 900 features.</li>"
         "<li><b>Normalise features per person.</b> Each feature is z-scored over that person's windows. This uses EEG only, never the target. It does need a person's EEG before predicting.</li>"
         "<li><b>Prepare the target.</b> Average to 30 × 1 s bins, flatten channels × time, and subtract each person × task mean.</li>"
         "<li><b>Fit.</b> <code>sklearn RidgeCV</code> with 13 α values from 10<sup>0</sup> to 10<sup>6</sup>. α is chosen by efficient leave-one-out on the training set only, with one α shared by all outputs.</li>"
         "<li><b>Predict and demean.</b> Predict held-out data and subtract the person × task mean from the prediction.</li>"
         "<li><b>Score and test.</b> r<sub>subject</sub>, bootstrap CI over people, sign-flip test against the within-person pairing null, then the verdict rule of section 3.3.</li></ol>")
B.append("<h3>4.3 Model recipes used in each test</h3>")
B.append(T("Every model variant, in plain steps.",
           ["Recipe", "Used in", "Steps"],
           [["Cross-person ridge", "T1, R, P4, P13", "Train on all other people, test on the held-out person, repeat for every person."],
            ["Within-person ridge", "T5, R2", "Leave one run out within a person. Own model against every other person's model gives identification accuracy."],
            ["Frozen transfer", "P3.2, P4.2, P14.2", "Fit on all of cohort A. Map cohort B to A's montage by spherical-spline interpolation. Predict B with nothing tuned on B."],
            ["Euclidean alignment", "P5.2b", "Whiten each person's slow features by their own mean spatial covariance, with shrinkage 10<sup>−3</sup>. Unsupervised."],
            ["Shared space", "P3.1b, P5.1, P14.5", "Fit ridge maps B<sub>fNIRS</sub> and B<sub>fMRI</sub> from EEG to each target. Divide each by its Frobenius norm and stack them. Take the top 32 left singular vectors U. Fit a new ridge head for each modality on F·U. Control: fMRI targets permuted within person."],
            ["Lagged coupling", "P6.4, P7, P15.2", "Standardise band-power features. For each lag from −10 to +20 s, shift EEG against HbO, fit ridge on the other runs, score the held-out run. The peak of the curve is the delay."],
            ["Task-response removal", "P6.4b, P7.1", "Least-squares FIR model of the cue-locked response over 0–29 s plus intercept. Subtract it from both EEG features and HbO."],
            ["Hybrid calibration", "P2.5, P5.4, P14.3", "Base = person's calibration-run mean. Personal ridge fits the residual on calibration runs. Prediction = base + 0.5 × (personal + population ridge)."],
            ["Calibrated stacking", "P6.2, P8.3", "Cross-fit the personal model between two calibration runs. Least squares gives gains for the personal and frozen external predictions. Apply both gains to later runs."],
            ["Pooled training", "P15.1", "Train on cohort A plus the other people of B in A's montage, removing the held-out person's flagged partner."],
            ["Score-to-accuracy mapping", "P15.3", "Ordinary least squares from calibration score to later accuracy on other cohorts. Interval half-width = 80th percentile of absolute training residuals."],
            ["Partial rank correlation", "P15.4", "Regress ranks of score and accuracy on ranks of the two quality measures, then correlate the residuals."],
            ["Downstream task", "P3.4, P5.3, P5.3b", "Ridge predicts haemodynamics, out-of-fold for training people (3 inner person folds). Average over a response window. StandardScaler and logistic regression (C = 1), 3 person folds. Controls: PCA with the same dimension, random projection, shuffled targets, full features."]],
           ["20%", "15%", "65%"]))
B.append(L.callout("key", "Deep against linear on the same data", "<p>With the slow band kept, the deep model reaches 0.158 in hBCI and a pooled ridge reaches 0.197. The deep network has 11.9 million parameters, 96% of them in the fNIRS head, and about 3,200 training windows per fold. The ridge has 900 input features and one regularisation value. That mismatch may explain the result, but it was not tested directly.</p>"))

B.append("<h2>5. Results by claim</h2>")
B.append("<h3>5.1 Founding result: the signal is infra-slow</h3>")
B.append(T("Cross-person prediction of fNIRS (ridge, leave one person out).",
           ["Cohort", "Slow EEG below 0.5 Hz", "Band power 1–45 Hz", "Own-model identification"],
           [["hBCI", "0.197 (deep model 0.155)", "about 0.02", "93%"],
            ["Mental workload", "0.135 [0.089, 0.183]", "0.006 (not significant)", "62%"],
            ["FineMI", "0.203 [0.164, 0.248]", "0.045", "83%"]], ["25%", "30%", "25%", "20%"], "Chance identification is 1 / number of people."))
B.append("<p><b>Band split (hBCI).</b> EEG below 0.5 Hz gives 0.155 alone (98% of the unfiltered value), below 0.1 Hz 0.139, 0.1–0.5 Hz 0.053, and above 45 Hz 0.021.</p>")
B.append("<p><b>Not ocular.</b> EOG channels carry 14% of the score in hBCI and 0.001 in mental workload. The topography is diffuse, with the frontal group weakest. <b>Not linear drift.</b> Removing a linear trend raises the slow-band within-person score from 0.360 to 0.404. <b>Partly global.</b> In hBCI the HbO global score is 0.226 and local 0.083. A local component remains in all three cohorts (HbO local 0.07–0.16).</p>")
B.append(F("fig1_slow_band", "Infra-slow EEG against band power.", "Cross-person prediction of HbO in three cohorts.", 62))
B.append("<h3>5.2 Claim 1: one EEG space for fNIRS and fMRI</h3>")
B.append("<p>The declared rank-8 test (P4.1) failed on capacity, because it cost the fNIRS output 0.021. Rank 32 was then fixed from that exploratory sweep and confirmed on fNIRS cohorts never used for Claim 1 (P5.1). The fMRI arm is the same 27 people in both.</p>")
B.append(T("Claim 1, confirmatory results at rank 32.",
           ["Test", "fNIRS", "fMRI", "Cost of sharing, CI lower bounds", "Result"],
           [["P5.1 cue-aligned, MW", "0.123", "0.063–0.068", "−0.014", "Pass"],
            ["P5.1 cue-aligned, FineMI", "0.180", "0.063–0.068", "−0.003", "Pass"],
            ["P14.5 no cues, MW", "0.134 [0.094, 0.177]", "0.065", "+0.003, −0.005", "Pass, survives"],
            ["P14.5 no cues, FineMI", "0.175 [0.136, 0.222]", "0.065", "0.000, −0.003", "Pass, survives"]], ["28%", "20%", "14%", "22%", "16%"]))
B.append("<p><b>Stronger form rejected.</b> fMRI supervision does not improve fNIRS: joint equals joint with shuffled fMRI targets in all three cohorts (for example 0.134 against 0.130 in MW), also with only five training people. The form holds for 5 of 5 person-fold seeds and on the local target (P6.3).</p>")
B.append("<h3>5.3 Claim 2: external validation</h3>")
B.append(T("Claim 2, main transfer tests.",
           ["Test", "Result", "Verdict"],
           [["P3.2 hBCI to MW (frozen, slow EEG)", "0.205 [0.128, 0.285], p 0.0002", "Passes, survives"],
            ["P3.2 MW to hBCI", "0.177, p 0.0001", "Passes, survives"],
            ["P7.2 hBCI to MW, band-power kernel", "+0.098 [0.071, 0.124]", "Passes, survives"],
            ["P14.1 hBCI to MW without 5 closest cross-cohort pairs", "+0.217 [0.119, 0.305], p 0.0005", "Passes, survives"],
            ["P14.2 hBCI to MW, no cues", "+0.201 [0.128, 0.276], p 0.0002", "Passes, survives"],
            ["P5.2 hBCI to FineMI, TU Berlin hardware filter emulated", "−0.064 [−0.094, −0.037]", "Fails, reliably negative"],
            ["P6.1 harmonised window and montage into FineMI", "−0.010 [−0.041, +0.021]", "Fails"],
            ["P7.2 band-power kernel into FineMI", "−0.053 [−0.071, −0.034]", "Fails, reliably negative"],
            ["P8.3 calibrated kernel into FineMI", "+0.020 [+0.004, +0.039], p 0.017", "Nominal only (α 0.0031); not reproduced in third lab"],
            ["P5.6 fMRI output, ds002336 to ds002338", "0.017 [−0.009, +0.042]", "Fails"],
            ["P12.5 frozen hBCI kernel into third lab", "−0.026", "Fails"]], ["52%", "26%", "22%"]))
B.append("<p>Cross-lab transfer had 16 attempts, including detrending, linked-mastoid reference, Euclidean alignment, hardware emulation, harmonised window and montage, calibrated stacking, CBSI target and band-specific kernels. The inversion into FineMI is broadband and symmetric. FineMI people do share a kernel among themselves (leave one person out +0.074 [0.047, 0.105]), so coupling generalises within a lab while the kernel is lab-specific. Removing the broadband power term shrinks the inversion but leaves it (−0.053 to −0.036, CI below 0).</p>")
B.append("<h3>5.4 Claim 3: coupling and delay</h3>")
B.append(T("Claim 3, band-power coupling to local HbO (P6.4 and later).",
           ["Cohort", "Beat own null", "Median delay, EEG first", "Personal delay repeats", "Kernel repeats (beyond the task)"],
           [["hBCI", "29/29", "+6 s [3, 7]", "Yes, split-half r 0.39; P8.1 p 0.0003", "Yes, p &lt; 0.0001"],
            ["Mental workload", "81–85%", "+4 s [1.5, 7.5]", "No", "Nominal, p 0.006"],
            ["FineMI", "83–89%", "+3 s [−1.5, 5]", "No (inconclusive)", "Yes, p 0.001"],
            ["Third lab, n = 15", "9/15", "+7 s [2, 11]", "No, p 0.15", "No, p 0.10"]], ["21%", "13%", "20%", "24%", "22%"], "Median delays and their CIs are from P15.5, descriptive and not counted."))
B.append("<p><b>Beyond the task.</b> After removing the cue-locked response, hBCI still gives 29/29 people, +6 s and split-half r 0.33 (P6.4b), so the coupling is not just a shared task response.</p>")
B.append("<p><b>Why the delay does not repeat outside hBCI.</b> Recording length is not the reason (P9.1): FineMI has more data per person than hBCI. Coupling strength can account for it: degraded to FineMI's coupling strength, the hBCI delay no longer repeats (P10.2: 10% against 16% chance). The FineMI result is therefore inconclusive, not negative. A more sensitive template estimator failed its positive control (P11), which closed that line.</p>")
B.append("<p><b>Slow EEG.</b> In the two TU Berlin cohorts slow EEG follows HbO by 1–2 s, which is the reverse of classical neurovascular order and fits a vascular or blood–brain-barrier origin. The direction is nominal only (p 0.006 and 0.004 against α 0.0036).</p>")
B.append(F("fig2_coupling_lag", "Band-power to local HbO lag curves.", "Mean ± SEM. Positive lag means EEG leads.", 72))
B.append(F("fig3_kernel_repro", "Within-person kernel reproducibility.", "Raw and with the task response removed.", 82))
B.append("<h3>5.5 Claim 4: downstream task</h3>")
B.append("<p>Refuted in every form: standard, data-scarce (five labelled people) and representation-only. A PCA code of the same slow EEG matches or beats virtual haemodynamics (hBCI motor imagery 0.760 against 0.697). The benefit comes from the slow band itself: band power gives 0.61 against 0.70–0.76 with slow EEG.</p>")
B.append("<h3>5.6 Claim 5: reliability and deployment</h3>")
B.append(T("Calibrated EEG-only monitoring at two calibration runs: gain over replaying the person's own average response.",
           ["Cohort", "Raw target", "Local target"],
           [["hBCI", "+0.176 [0.135, 0.218], 29/29", "+0.103 [0.074, 0.136], 29/29"],
            ["Mental workload", "+0.059 [0.025, 0.099]", "+0.031 [0.013, 0.052]"],
            ["FineMI", "+0.067 [0.033, 0.112], 18/18", "+0.041 [0.014, 0.079], 16/18"]], ["30%", "35%", "35%"], "In hBCI the gain grows with calibration runs: +0.118 at k = 1 and +0.258 at k = 5."))
B.append(T("Does a person's calibration score predict their later accuracy? (Spearman)",
           ["Test", "hBCI", "Mental workload", "FineMI", "Third lab"],
           [["P5.4 cue-aligned", "0.68 [0.37, 0.86]", "0.56 [0.22, 0.78]", "0.90 [0.69, 0.97]", "not run"],
            ["P14.3 and P14.4 no cues, raw", "0.85 [0.66, 0.94]", "0.83 [0.64, 0.93]", "0.84 [0.57, 0.93]", "0.29 [−0.31, 0.73], fails"],
            ["P14.3 and P14.4 no cues, gain", "0.79 [0.51, 0.93]", "0.87 [0.69, 0.95]", "0.85 [0.55, 0.97]", "0.30 [−0.30, 0.78], fails"],
            ["P15.4 partial, removing HbO amplitude and EEG power", "0.84 [0.62, 0.94]", "0.83 [0.63, 0.94]", "0.69 [0.22, 0.89]", "not run"]], ["32%", "17%", "17%", "17%", "17%"], "Raw and gain forms are not independent in uncued windows, because a person's mean response is close to zero there."))
B.append("<p>Trial-level reliability fails in both the notebook form (P3.5) and a calibrated form (P5.7). The predicted-signal size ranks trials only weakly (0.16–0.22).</p>")
B.append(F("fig5_deployment_reliability", "Calibrated monitoring gain and person-level reliability."))

# ------------------------------------------------------------------ 5
B.append("<h2>6. Third-lab confirmation and method check</h2>")
B.append(T("P12: third-lab confirmation, n = 15, α = 0.0083.",
           ["Test", "Result", "Own rule"],
           [["P12.1 slow EEG better than band power", "+0.057 against +0.065 (difference −0.008)", "Fail; later shown uninformative"],
            ["P12.2 band-power coupling exists", "9/15 beat their null", "Pass"],
            ["P12.3 kernel repeats", "p 0.10", "Fail"],
            ["P12.4 personal delay repeats", "p 0.15", "Fail"],
            ["P12.5 frozen hBCI kernel", "−0.026", "Fail"],
            ["P12.6 calibrated hBCI kernel", "−0.037", "Fail"]], ["45%", "35%", "20%"]))
B.append("<p><b>P13 asked whether the P12.1 method could detect the advantage at all.</b> It could not: run on the cohorts where the advantage is established, it recovers it in only one of three. By the rule fixed in advance, P12.1 says nothing about the third lab, and the earlier wording “does not replicate” was withdrawn. The founding window method, on windows not aligned to cues, recovers the advantage in all three cohorts. On the third lab it finds nothing (exploratory). The infra-slow advantage is shown in hBCI, mental workload and FineMI, and is <b>not established</b> in the third lab.</p>")
B.append(T("P13: slow EEG minus band power, α = 0.0071.",
           ["Cohort", "Continuous P12.1 method", "Uncued windows"],
           [["hBCI", "+0.152 [0.086, 0.220], pass", "+0.133 [0.099, 0.170], pass"],
            ["Mental workload", "+0.061 [−0.018, 0.137], fail", "+0.084 [0.043, 0.128], pass"],
            ["FineMI", "−0.026 [−0.060, 0.008], fail", "+0.124 [0.082, 0.174], pass"],
            ["Third lab (exploratory)", "as P12.1", "−0.007 [−0.023, 0.007]; slow r −0.003, band r +0.004"]], ["28%", "36%", "36%"]))
B.append(F("fig16_method_check", "The slow-EEG advantage under two methods."))

# ------------------------------------------------------------------ 6
B.append("<h2>7. Stress tests (P14)</h2>")
B.append("<p>Twelve tests asked whether Claims 1, 2 and 5 survive their main weaknesses (α = 0.0042). Ten pass and survive. The two failures are Claim 5 in the third lab, in raw and gain form.</p>")
B.append("<p><b>Shared people (P14.1).</b> The two TU Berlin cohorts may share participants, and a shared person would leak identity into transfer. An EEG fingerprint was defined: log Welch power on the 10 common electrodes, 1–45 Hz, cohort mean removed, compared by Pearson correlation. The positive control is valid: even runs identify the same person from odd runs in 97% (hBCI) and 100% (MW) of people. Five mutual-best cross-cohort pairs exceed the threshold T = 0.68, with similarity 0.69–0.78 against 0.94 for the same person within a session. About 7 of 754 pairs would exceed the threshold by chance, so sharing is neither shown nor excluded. Transfer holds with those five pairs removed.</p>")
B.append("<p><b>Limits of the no-cue tests.</b> The windows are not aligned to cues but still contain task periods, so the correct wording is “does not need cue alignment” and not “task-free”. They use HbO only.</p>")
B.append(F("fig17_stress_tests", "Stress tests of Claims 5 and 2."))

# ------------------------------------------------------------------ 7
B.append("<h2>8. Attempts to improve (P15)</h2>")
B.append("<p>Fifteen tests (α = 0.0033). Eight pass their own rule and seven survive. Reproduction gates held: the single-source model with nobody removed reproduces P3.2 exactly, and recomputed HbO curves match the saved curves to rounding.</p>")
B.append(T("P15 results.",
           ["Test", "Result", "Own rule", "Survives α"],
           [["P15.1 MW target: Pool − hBCI only", "−0.011 [−0.018, −0.006], p 0.001 (0.193 vs 0.204)", "Fail, reliably worse", "No"],
            ["P15.1 MW target: Pool − MW only", "+0.018 [−0.034, +0.065], p 0.48 (0.193 vs 0.175)", "Fail", "No"],
            ["P15.1 hBCI target: Pool − MW only", "+0.097 [0.056, 0.139], p 0.0001 (0.274 vs 0.177)", "Pass", "Yes"],
            ["P15.1 hBCI target: Pool − hBCI only", "+0.004 [−0.004, +0.013], p 0.34 (0.274 vs 0.270)", "Fail", "No"],
            ["P15.2 HbR − HbO peak, hBCI", "+1 s [−4, +2], p 0.94", "Fail", "No"],
            ["P15.2 HbR − HbO peak, MW", "−2 s [−5, +4], p 0.76", "Fail", "No"],
            ["P15.2 HbR − HbO peak, FineMI", "−3 s [−6, 0], p 0.075", "Fail", "No"],
            ["P15.2 HbR − HbO peak, third lab", "−8 s [−13, +3], p 0.18", "Fail", "No"],
            ["P15.3 mapping gain, hBCI", "error 0.046 vs 0.106, +0.060 [0.029, 0.094], p 0.0002", "Pass", "Yes"],
            ["P15.3 mapping gain, MW", "error 0.077 vs 0.129, +0.052 [0.012, 0.095], p 0.017", "Pass", "No"],
            ["P15.3 mapping gain, FineMI", "error 0.064 vs 0.115, +0.051 [0.017, 0.085], p 0.003", "Pass", "Yes"],
            ["P15.3 mapping gain, third lab", "error 0.074 vs 0.172, +0.097 [0.057, 0.131], p 0.0002", "Pass", "Yes"],
            ["P15.4 partial ρ, hBCI", "0.84 [0.62, 0.94], p 3.5e-8", "Pass", "Yes"],
            ["P15.4 partial ρ, MW", "0.83 [0.63, 0.94], p 4.0e-7", "Pass", "Yes"],
            ["P15.4 partial ρ, FineMI", "0.69 [0.22, 0.89], p 0.0028", "Pass", "Yes"]], ["33%", "42%", "15%", "10%"]))
B.append("<ul><li><b>Pooled training (P15.1)</b> counts as an improvement only if Pool beats both single-source models. It does so for neither target cohort, so P3.2 remains the Claim 2 record.</li>"
         "<li><b>HbR timing (P15.2)</b> is inconclusive. HbR coupling is above its surrogates in all four cohorts, but the intervals are 6 to 16 s wide against an expected 1–2 s difference. Only hBCI goes in the expected direction.</li>"
         "<li><b>Mapping (P15.3)</b> fits test accuracy from calibration score on other cohorts (slopes 0.99–1.02). Coverage of an 80% interval is 0.90 hBCI, 0.58 MW, 0.83 FineMI and 0.73 third lab. MW is outside the declared 0.65–0.95 range.</li>"
         "<li><b>Post hoc limit on the third-lab mapping.</b> Against that cohort's own mean, which a user would not know, the mapping is worse (0.074 against 0.048). The gain reflects the cohort being poorly predicted overall, not a ranking of people inside it. In the other three cohorts the mapping beats the cohort's own mean too.</li></ul>")
B.append(F("fig18_improvement", "Attempts to improve.", "Left: predicted against measured later accuracy in cohorts the mapping was not fitted on. Right: HbR peak minus HbO peak."))

# ------------------------------------------------------------------ 8
B.append("<h2>9. Step-by-step record</h2>")
B.append("<p>Every step below was run in this order. Later steps were chosen because of what earlier ones showed.</p>")
B.append(T("What was done, in order.",
           ["#", "Step", "What was done", "What it showed"],
           [["1", "Deep model (to July)", "NeuroVascNet v3: EEG-only network trained with fNIRS and fMRI supervision, four outputs", "Cross-person waveform prediction near zero"],
            ["2", "Data audit (25–26 Jul)", "610 recordings, 63.2 GB, 97 subjects", "HEFMI systemic-dominated (HbO–HbR +0.90); GSR removes 83% of evoked fNIRS; 2600× unit mismatch. Switched primary fNIRS to hBCI"],
            ["3", "Merge and protocol check (4 Nov)", "Merged notebook, run with and without the blueprint band-pass", "0.019 filtered, 0.158 unfiltered: the signal is outside 0.5–45 Hz"],
            ["4", "T1–T2", "Band split, global against local target", "Signal below 0.5 Hz; mostly global, local part exists"],
            ["5", "T3–T5b", "Ocular and frontal control, topography, person-specificity (leakage fixed with leave one run out), drift control", "Not ocular, diffuse, person-specific (93%), not linear drift"],
            ["6", "T6", "Does slow EEG lead fNIRS?", "Flat, inconclusive at the 30 s window scale"],
            ["7", "R (4 Nov)", "Replication on mental workload, declared before loading", "0.136 slow against 0.006 band power: replicates"],
            ["8", "P2 (4–5 Nov)", "Infra-slow protocol; deep model rerun; continuous lag; personal calibration", "Deep model still fails all claims; calibration works (+0.176 hBCI)"],
            ["9", "P3 (5–6 Nov)", "Best route to each claim using ridge", "Claim 2 within lab passes; Claim 3a passes; Claim 1 fails; Claim 4 fails; Claim 5 partial"],
            ["10", "P4 (7 Nov)", "Added FineMI and ds002338", "Third-cohort replication (0.203); cross-lab transfer fails; fMRI arm predictable with 27 people"],
            ["11", "P5 (7 Nov)", "Confirmatory tests on fresh fNIRS cohorts, reliability at person level", "Claim 1 shared space confirmed; Claim 5 holds; Claim 4 refuted; hardware emulation makes transfer negative"],
            ["12", "P6 (7–8 Nov)", "Harmonised transfer, calibrated stacking, classical band-power coupling", "EEG leads local HbO by 6 s in 29/29 hBCI people, beyond the task"],
            ["13", "P7 (8 Nov)", "Within-person kernel reproducibility, frozen kernel transfer", "Kernel repeats beyond the task in all three cohorts (MW nominal); inverted into FineMI"],
            ["14", "P8 (8 Nov)", "Smoothed delay, CBSI target, calibrated kernel transfer", "Delay still hBCI only; calibrated cross-lab +0.020, nominal"],
            ["15", "P9 (8 Nov)", "Why do Claims 2 and 3 fail outside TU Berlin?", "Not recording length; inversion broadband; kernel lab-specific"],
            ["16", "P10 (9 Nov)", "Tested both P9 explanations", "Broadband term does not explain the inversion; weak coupling can explain delay failure"],
            ["17", "P11 (9 Nov)", "Template-alignment delay estimator with positive control", "Failed its control; personal-delay line closed"],
            ["18", "P12 (9 Nov)", "Third lab, rules declared first", "1 of 6 tests pass"],
            ["19", "P13 (10 Nov)", "Could the P12.1 method detect the advantage?", "No (1 of 3 cohorts); uncued windows recover it in 3 of 3"],
            ["20", "P14 (10 Nov)", "Stress tests: shared people, no cues, third lab", "10 of 12 pass and survive"],
            ["21", "P15 (10 Nov)", "Pooled training, HbR timing, predicted accuracy, quality control", "Claim 5 stronger; Claims 2 and 3 not improved"]],
           ["5%", "18%", "39%", "38%"]))
B.append("<h3>Errors found and corrected along the way</h3>")
B.append("<ul><li><b>T5 leakage.</b> Shuffled folds let overlapping 30 s windows sit across train and test. Rerun with leave one run out, and that run is the one the rules apply to.</li>"
         "<li><b>Wrong claim about Mendeley and Stroop.</b> They were described as never used, but June baselines exist. Corrected in the plan, summary, ledger, draft and notebooks. Two older pushed commit messages still carry the old wording.</li>"
         "<li><b>“Does not replicate” for P12.1.</b> Withdrawn by P13's rule fixed in advance.</li>"
         "<li><b>P14 Mendeley build crash.</b> It called the wrong data function. Fixed before any result existed.</li></ul>")

# ------------------------------------------------------------------ 9
B.append("<h2>10. Limits, and what to do next</h2>")
B.append("<h3>Limits that affect the claims</h3>")
B.append("<ul><li><b>FineMI fNIRS quality</b> was flagged in the audit (scalp coupling 0.023) and never re-examined.</li>"
         "<li><b>Global signal.</b> Much of the predictable signal is systemic. Only claims marked “local” were tested on the local target.</li>"
         "<li><b>Two labs.</b> The two TU Berlin cohorts may share people, so effectively the positive evidence comes from two labs. The third lab supports coupling only.</li>"
         "<li><b>Small effect sizes.</b> Cross-person r is 0.13–0.20. The usable monitoring result needs two calibration runs with fNIRS for each person.</li>"
         "<li><b>Rounds of testing.</b> Many rounds on the same data are handled by correction, and correction does not remove every risk of analytic flexibility.</li>"
         "<li><b>Seeds.</b> Most results use seed 1337. Claims 1 and 5 were rerun with five seeds and hold in 5 of 5.</li>"
         "<li><b>DC-coupled or long-time-constant EEG is required.</b> The finding cannot be applied to data recorded with a high-pass filter.</li>"
         "<li><b>Reproduction.</b> <code>RUN = ALL</code> in the full notebook has never been run end to end. The notebook's P15 Claim 2 step was not executed in the notebook either.</li></ul>")
B.append("<h3>What to do next, in order</h3>")
B.append("<ol><li><b>Check the FineMI quality flag.</b> Recompute scalp coupling with the audit method and test whether channel coupling relates to prediction score. Declare the rule first. If FineMI's signal is artefactual, the claims that rely on it must be reworded.</li>"
         "<li><b>Push the six local commits</b> on <code>data-quality-audit</code> (remote is at 7387cb3).</li>"
         "<li><b>Rewrite the title and abstract</b> around the narrowed claim, and add P15. The present title overclaims generality.</li>"
         "<li><b>Fill the [TODO] and [CITE] markers</b> in <code>merge/paper/manuscript.md</code>, including the exact filter details in section 2.2.</li>"
         "<li><b>Run <code>RUN = ALL</code> once</b> on the full data, keeping <code>BUILD_CACHES = False</code> so the 60 GB cache is not overwritten.</li>"
         "<li><b>Look for a fresh EEG–fMRI cohort</b> for Claim 1. Candidates are Gu et al. (33 people, rest and sleep) and NKI naturalistic viewing (22 people). Confirm from their methods that EEG below 0.5 Hz survives before downloading, and declare the test before loading.</li>"
         "<li><b>Consider new recordings or a larger third-lab cohort</b> for Claims 3 and 5. Further analysis of the current data is unlikely to add evidence and carries a rising risk of over-fitting the analysis to it.</li></ol>")

# ------------------------------------------------------------------ 10
B.append("<h2>11. Where everything is</h2>")
B.append(T("Files.",
           ["Path", "Contents"],
           [["<code>merge/FINAL_PLAN.md</code>", "Every declared rule, dated, and every round's result"],
            ["<code>merge/RESULTS_SUMMARY.md</code>", "Current scorecard against the five claims"],
            ["<code>merge/TEST_LEDGER.md</code>", "Multiplicity check across all attempts"],
            ["<code>merge/paper/manuscript.md</code>", "Paper draft, sections 3.8 to 3.10 cover P12 to P15"],
            ["<code>merge/NeuroVascNet_Full.ipynb</code>", "All analysis code in one notebook; displays saved results by default"],
            ["<code>merge/NeuroVascNet_Results.ipynb</code>", "Results only"],
            ["<code>merge/analysis_p2_p4/</code>", "One script per analysis; <code>README.md</code> lists run order, <code>results/</code> holds the JSON outputs"],
            ["<code>merge/paper/figures/</code>", "18 figures, built by <code>merge/paper/make_figures.py</code>"],
            ["<code>NeuroVascNet_Data_Quality_Audit.md</code>", "The July data audit"]], ["45%", "55%"]))

d.render("\n".join(B), "NeuroVascNet Technical Report")
