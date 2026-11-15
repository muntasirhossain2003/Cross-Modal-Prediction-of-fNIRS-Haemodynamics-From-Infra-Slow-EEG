"""Builds lectures/Lecture_F_Reference_General_Knowledge.pdf.

General background for interpreting results (Part F companion): negative and
null results, replication and analytic flexibility, effect-size context,
inter-individual variability, biomarker validation and clinical background on
intracerebral haemorrhage. Run:  python scripts/build_lecture_part_f_reference.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, K, M, V,
                            arrow, box, fig, qa_bank, table, title_block)

D = Doc("lecture_f_ref", "Lecture_F_Reference_General_Knowledge.pdf", "NeuroVascNet course · Lecture F reference")


def f_multiverse():
    rng = np.random.default_rng(5)
    est = np.clip(rng.normal(0.12, 0.07, 120), -0.1, 0.4)
    fig, a = plt.subplots(figsize=(W, 2.3))
    o = np.sort(est)
    a.scatter(np.arange(len(o)), o, s=10, color=[BLUE if v > 0.05 else GRAY for v in o])
    a.axhline(0, color=MUTED, lw=0.8)
    a.annotate("a researcher who reports\nonly this one pipeline", xy=(len(o) - 1, o[-1]), xytext=(45, 0.2), fontsize=8.4,
               arrowprops=dict(arrowstyle="->", color=ORANGE))
    a.set_xticks([]); a.set_xlabel("120 defensible analysis pipelines, sorted by result"); a.set_ylabel("effect estimate")
    a.set_title("Analytic flexibility: the same data, many reasonable answers", fontsize=10)
    D.save(fig, "rf1_multiverse")


def f_ladder():
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis("off")
    steps = [("Discovery", "a difference between\ngroups in one study", GRAY),
             ("Analytical validity", "measured reliably\n(test–retest, artifacts)", BLUE),
             ("Replication", "same difference in\nindependent cohorts", AQUA),
             ("Clinical validity", "predicts the condition or\noutcome in new patients", YELLOW),
             ("Clinical utility", "using it improves\ndecisions or outcomes", ORANGE)]
    for i, (n, sub, c) in enumerate(steps):
        x = 0.1 + i * 1.98; y = 0.4 + i * 0.55
        box(ax, x, y, 1.85, 0.75, n, fc=c, ec=c, color="white", weight="bold", fs=8.3)
        ax.text(x + 0.92, y - 0.08, sub, ha="center", va="top", fontsize=7.6, color=MUTED)
    ax.text(0.1, 3.55, "Our ICH HbO–HbR finding is at the first step.", fontsize=9, color=INK, weight="bold")
    D.save(fig, "rf2_ladder")


for f in [f_multiverse, f_ladder]:
    f()

QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "Negative results and interpretation"
qa(S, "What is the difference between a negative result and a null result?", "A null result fails to find an effect; it may be due to low power. A convincing negative result shows the effect is absent or too small to matter, usually with an adequately powered test and a confidence interval that excludes meaningful effect sizes.")
qa(S, "How can you show evidence of absence rather than absence of evidence?", "Use equivalence testing (for example two one-sided tests against a smallest effect of interest), report confidence intervals that exclude meaningful effects, use Bayesian estimation, and demonstrate the test could detect a real effect (positive controls).")
qa(S, "Why publish negative results?", "To prevent wasted effort, correct publication bias (the 'file drawer' of unpublished nulls inflates the literature) and define a method's limits. Several journals and formats, such as registered reports, explicitly welcome well-designed negative results.")
qa(S, "How should a correlation of 0.1–0.3 be interpreted?", "In psychology and neuroscience, correlations of about 0.1 are small, 0.2 medium and 0.3 large by empirical benchmarks (Funder and Ozer, 2019), and small effects can matter in aggregate. But for predicting individual trials, a correlation of 0.16 explains under 3% of variance and is not usable on its own.")
qa(S, "What is the difference between explanation and prediction?", "Explanatory analyses ask whether a variable is associated with an outcome in a sample (inference); predictive analyses ask how well a model performs on new data. A variable can be significantly associated yet predict poorly for individuals.")
qa(S, "What is negative transfer?", "When adding a task, dataset or modality to training makes performance on the main task worse, typically because the added source is unrelated, much noisier, much smaller, or competes for model capacity.")
qa(S, "Why do cross-dataset failures happen so often in brain data?", "Datasets differ in people, equipment, montage, task, preprocessing and target quality, and models exploit dataset-specific regularities. Unless training data cover that variability, performance usually drops sharply on a new dataset.")

S = "Replication and robustness"
qa(S, "What is the difference between reproducibility, replicability and robustness?", "Reproducibility: same data and code give the same result. Replicability: new data, same method, consistent result. Robustness: the result holds across reasonable changes in analysis. Strong findings show all three.")
qa(S, "What is a multiverse analysis?", "Running all reasonable combinations of analysis choices and reporting the distribution of results (Steegen and colleagues, 2016). A finding that holds across most of the multiverse is robust; one that appears only in a few pipelines is fragile.")
qa(S, "What did many-analysts studies show?", "When many independent teams analyse the same neuroimaging dataset to test the same hypotheses, they can reach different conclusions (Botvinik-Nezer and colleagues, 2020). Analytic choices matter, which is why independent re-analysis is valuable.")
qa(S, "Why does an independent second pipeline strengthen conclusions?", "If two teams with different code, preprocessing and model choices reach the same conclusion, it is less likely to reflect a single pipeline's bug or arbitrary choice. Where they disagree, the disagreement identifies which choices matter.")
qa(S, "Why report results over several random seeds?", "Training is stochastic; one seed can be lucky or unlucky. Averaging over seeds and showing their spread separates genuine effects from run-to-run noise. Seeds should be averaged within subjects before subject-level statistics.")

S = "Individual differences"
qa(S, "Why is between-person variability so large in brain signals?", "People differ in anatomy and cortical folding, skull and scalp thickness, vascular health, haemodynamic response timing, cognitive strategy and state. Neurovascular coupling itself varies with age, fitness, caffeine and disease.")
qa(S, "What is 'BCI illiteracy' or inefficiency?", "A sizeable minority of users (often cited as 15–30%) cannot control a motor-imagery BCI well even after training, reflecting individual differences in the relevant EEG patterns. It illustrates how person-specific brain signals are.")
qa(S, "Why do BCIs usually use a calibration session?", "Because models trained on other people transfer poorly; a few minutes of the new user's data, used to adapt or train the classifier, usually improves performance substantially. Reducing calibration is an active research goal (transfer learning, domain adaptation).")
qa(S, "How should personalisation gains be evaluated?", "Against a baseline that uses the same calibration data without the brain signal (for example the person's own mean response). Otherwise a gain might come entirely from learning the person's average level rather than from the signal.")

S = "Biomarkers and clinical translation"
qa(S, "What is a biomarker?", "A measurable characteristic that indicates a normal process, a disease process or a response to treatment, such as blood pressure or a blood test. For brain imaging, candidates include response timing, connectivity or coupling measures.")
qa(S, "What steps must a candidate biomarker pass?", "Analytical validity (reliable, accurate measurement), replication in independent cohorts, clinical validity (it distinguishes or predicts the condition in new patients) and clinical utility (using it improves decisions or outcomes). Most candidates stop at discovery.")
qa(S, "What makes a measure reliable enough to be a biomarker of individual differences?", "High test–retest reliability (ICC), so differences between people exceed measurement noise. An ICC around 0.5 is moderate: useful for group research, but marginal for decisions about individuals.")
qa(S, "What confounds threaten group comparisons in patients?", "Differences in head motion, medication, age, sex, lesion location, scalp and skull properties (which alter fNIRS optics), arousal and task performance. Reporting these as controls and matching or adjusting for them is essential.")

S = "Clinical background: intracerebral haemorrhage"
qa(S, "What is an intracerebral haemorrhage (ICH)?", "A type of stroke caused by a blood vessel rupturing inside the brain tissue, most often due to long-standing high blood pressure or cerebral amyloid angiopathy. It accounts for roughly 10–20% of strokes and carries high mortality and disability.")
qa(S, "How might ICH affect neurovascular coupling?", "Blood and its breakdown products are toxic to surrounding tissue, cause swelling and inflammation, and can damage vessels and the blood–brain barrier. Autoregulation and coupling near and sometimes away from the lesion can be impaired, changing haemodynamic responses independently of neural activity.")
qa(S, "Why is fNIRS attractive for stroke patients?", "It is portable, can be used at the bedside, tolerates some movement and allows repeated monitoring during rehabilitation, where MRI is impractical. Its limits are shallow depth and sensitivity to scalp and systemic signals.")
qa(S, "Why must optical properties be controlled in patient fNIRS studies?", "Lesions, blood products, oedema, skull changes or surgery alter how light travels (scattering, absorption, pathlength). A group difference in fNIRS signals could then reflect optics rather than physiology; DPF sensitivity analyses or short-separation and multi-distance measurements help rule this out.")

S = "Communicating results"
qa(S, "How should limitations be written?", "Specifically and with their consequences: what could bias the result, in which direction, and what would resolve it. Generic statements ('more data are needed') add little.")
qa(S, "How do you avoid overclaiming?", "Match each sentence to the evidence: 'associated with' rather than 'causes', 'in this cohort' rather than universally, report effect sizes and uncertainty, distinguish pre-planned from exploratory analyses, and state which claims were not supported.")


body = title_block("NeuroVascNet course · Lecture F · Reference", "Interpreting Results: General Knowledge",
                   "Negative results, replication, effect sizes, individual differences, biomarkers and clinical background",
                   "Companion to “The Results” · Read Lecture F first") + f"""
<div class="box">
<h3>How to use this document</h3>
<p>Lecture F reported what the project found. This reference covers the general ideas needed to interpret and defend such findings: what makes a negative result convincing, why independent pipelines matter, how big small effects are, why people differ, what it takes for a measure to become a biomarker, and the clinical background to the ICH comparison. Section 7 is a bank of {len(QA)} questions with model answers.</p>
</div>

<h2>1. Negative and null results</h2>
<p>A <b>null result</b> simply fails to find an effect, which may only mean the study was too small. A <b>convincing negative result</b> shows that an effect is absent or too small to matter: an adequately powered test, a confidence interval that excludes meaningful effects, and a positive control proving the method could have detected a real effect. Equivalence testing and Bayesian estimation formalise "evidence of absence".</p>
{K("<p>The most persuasive negative results come with controls that rule out the boring explanations: the method works elsewhere (positive control), the comparison is fair (same budget, same seeds), and a shuffled version of the ingredient performs the same as the real one.</p>")}
<p>Publishing negative results counters <b>publication bias</b> — the "file drawer" of unpublished nulls that makes published effects look larger and more consistent than they are.</p>

<h2>2. How big is a small effect?</h2>
<p>Empirical benchmarks for correlations in psychology and neuroscience put about 0.1 as small, 0.2 as medium and 0.3 as large (Funder and Ozer, 2019). Small effects can matter for understanding populations and in aggregate. But <b>prediction for an individual</b> is a different standard: a correlation of 0.16 accounts for under 3% of a single trial's variance. Always say which standard you are using, and report performance relative to the measured ceiling.</p>
{V("<p><b>“Isn't r = 0.16 too small to matter?”</b> For predicting a single trial for one person, yes — it would not be usable alone. For the scientific question 'does EEG carry information about haemodynamics in people the model has never seen?', a reliable 0.16 that survives subject demeaning, permutation tests and every fold is a clear yes, at about a fifth of what the target allows.</p>")}

<h2>3. Replication and analytic flexibility</h2>
<p>The same dataset can support different conclusions depending on reasonable analysis choices. <b>Multiverse analyses</b> run many defensible pipelines and report the spread (Steegen and colleagues, 2016); <b>many-analysts</b> studies give the same data to many teams and often find disagreement (Botvinik-Nezer and colleagues, 2020).</p>
{fig("rf1_multiverse", 1, "Analytic flexibility.", "Simulation. The same data analysed with 120 defensible pipelines give a spread of answers. Reporting only the most favourable one overstates the evidence; reporting the spread, or replicating with an independent pipeline, shows how robust a finding is.")}
<p>This is why the project's two independent pipelines matter: where different code, preprocessing and model choices reach the same conclusion, that conclusion is unlikely to be an artefact of one pipeline. Results should also be shown over several <b>random seeds</b>, and code and data versions fixed for <b>reproducibility</b>.</p>
{table(1, "Three related ideas.",
       ["Term", "Same data?", "Same method?", "Question"],
       [["Reproducibility", "yes", "yes", "does the code give the same numbers?"],
        ["Replicability", "no (new data)", "yes", "does the effect appear again?"],
        ["Robustness", "yes", "no (reasonable variations)", "does the effect survive other analysis choices?"]],
       widths=["22%", "18%", "24%", "36%"])}

<h2>4. Individual differences</h2>
<p>Brain signals vary greatly between people: anatomy and folding, skull and scalp, vascular health and haemodynamic timing, cognitive strategy, arousal, age, fitness, caffeine and disease. In BCI research a minority of users cannot control motor-imagery systems well at all, and almost all systems use a short <b>calibration session</b> per user. This background makes the project's pattern — real coupling within a person, weak transfer between people — expected rather than surprising.</p>
{M("<p>Crediting personalisation gains to the brain signal without a no-signal control. If a person's own average response, learned from the calibration data, explains most of the gain, the signal has added little.</p>")}

<h2>5. From finding to biomarker</h2>
<p>A <b>biomarker</b> is a measurable indicator of a biological state, disease or treatment response. Turning a group difference into a usable biomarker requires several steps, and most candidates stop early.</p>
{fig("rf2_ladder", 2, "The path from discovery to clinical use.", "Each step needs new evidence: reliable measurement, replication in independent cohorts, accurate classification or prediction in new patients, and finally evidence that using it improves care.")}
<p>Two properties matter from the start. <b>Reliability</b>: a measure must agree with itself across sessions (ICC) so that differences between people exceed noise; ICC around 0.5 is moderate, adequate for group research but marginal for individual decisions. <b>Confound control</b>: patients differ from controls in motion, medication, age, lesion and — for optical methods — the properties of the head itself.</p>

<h2>6. Clinical background: intracerebral haemorrhage</h2>
<p>An <b>intracerebral haemorrhage</b> is a stroke caused by bleeding into brain tissue, most often from chronic high blood pressure or amyloid deposits in vessel walls. It is a minority of strokes (roughly 10–20%) but among the most disabling and deadly. Blood and its breakdown products damage surrounding tissue, trigger swelling and inflammation, and can impair blood-vessel regulation and the blood–brain barrier — so neurovascular coupling may be altered independently of neural activity.</p>
<p>fNIRS is attractive for these patients because it is portable and repeatable at the bedside and during rehabilitation. But lesions, blood products, swelling and surgery change how light travels through the head, so a difference between patients and controls in fNIRS signals must be checked against optical explanations (pathlength sensitivity analyses, short-separation or multi-distance measurements) before it is interpreted physiologically.</p>
{A("a fogged window", "<p>If you see the street outside less clearly through one window than another, it might be because there is less happening outside — or because that window is fogged. In patients, the 'window' (scalp, skull and tissue optics) may itself be different. Before concluding that the brain is different, check the window.</p>")}

<h2>7. Question bank with model answers</h2>
{qa_bank(QA)}

<h2>8. Where to read more</h2>
<ul class="refs">
<li>Lakens D., Scheel A. M., Isager P. M. (2018). Equivalence testing for psychological research: a tutorial. <i>Advances in Methods and Practices in Psychological Science</i>.</li>
<li>Funder D. C., Ozer D. J. (2019). Evaluating effect size in psychological research: sense and nonsense. <i>Advances in Methods and Practices in Psychological Science</i>.</li>
<li>Steegen S. et al. (2016). Increasing transparency through a multiverse analysis. <i>Perspectives on Psychological Science</i>.</li>
<li>Botvinik-Nezer R. et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. <i>Nature</i>.</li>
<li>Open Science Collaboration (2015). Estimating the reproducibility of psychological science. <i>Science</i>.</li>
<li>Vidaurre C., Blankertz B. (2010). Towards a cure for BCI illiteracy. <i>Brain Topography</i>.</li>
<li>FDA–NIH Biomarker Working Group. <i>BEST (Biomarkers, EndpointS, and other Tools) Resource</i>.</li>
<li>Qureshi A. I. et al. (2009). Intracerebral haemorrhage. <i>The Lancet</i>.</li>
<li>Iadecola C. (2017). The neurovascular unit coming of age. <i>Neuron</i>.</li>
</ul>
"""

D.render(body, "Lecture F Reference: General Knowledge")
