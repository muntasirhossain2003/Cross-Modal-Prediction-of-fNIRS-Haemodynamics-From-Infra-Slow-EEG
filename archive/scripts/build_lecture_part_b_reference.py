"""Builds lectures/Lecture_B_Reference_General_Knowledge.pdf.

General background behind the research question (Part B companion): kinds of
learning problems, multimodal fusion, privileged information and distillation,
multi-task and transfer learning, uncertainty, related work and research design.
Run from the repo root:  python scripts/build_lecture_part_b_reference.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, K, M, V,
                            arrow, box, fig, qa_bank, table, title_block)

D = Doc("lecture_b_ref", "Lecture_B_Reference_General_Knowledge.pdf", "NeuroVascNet course · Lecture B reference")


# ---------------------------------------------------------------- figures
def f_fusion_levels():
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"wspace": 0.08})
    for a in ax:
        a.set_xlim(0, 4); a.set_ylim(0, 4.2); a.axis("off")
    specs = [("Early fusion", "join raw data or\nfeatures first", [(0.2, 3.0, "EEG", BLUE), (0.2, 1.6, "fNIRS", RED)], (1.6, 2.3, "combined\nfeatures"), (2.95, 2.3, "model")),
             ("Late fusion", "separate models,\njoin decisions", [(0.2, 3.0, "EEG", BLUE), (0.2, 1.6, "fNIRS", RED)], None, None),
             ("Intermediate", "separate encoders,\njoin representations", [(0.2, 3.0, "EEG", BLUE), (0.2, 1.6, "fNIRS", RED)], None, None)]
    for a, (t, sub, ins, mid, out) in zip(ax, specs):
        a.set_title(t, fontsize=10)
        a.text(2.0, 0.25, sub, ha="center", fontsize=8.3, color=MUTED)
        for x, y, lab, c in ins:
            box(a, x, y, 0.95, 0.7, lab, fc=c, ec=c, color="white", fs=8.5, weight="bold")
    a = ax[0]
    box(a, 1.55, 2.0, 1.1, 0.9, "combine", fc="#fff4e5", ec=YELLOW, fs=8.3)
    box(a, 3.0, 2.1, 0.95, 0.7, "model", fs=8.3)
    arrow(a, (1.15, 3.3), (1.55, 2.6)); arrow(a, (1.15, 1.95), (1.55, 2.3)); arrow(a, (2.65, 2.45), (3.0, 2.45))
    a = ax[1]
    box(a, 1.4, 3.0, 0.95, 0.7, "model A", fs=8.3); box(a, 1.4, 1.6, 0.95, 0.7, "model B", fs=8.3)
    box(a, 2.85, 2.1, 1.1, 0.9, "vote /\naverage", fc="#fff4e5", ec=YELLOW, fs=8.3)
    arrow(a, (1.15, 3.35), (1.4, 3.35)); arrow(a, (1.15, 1.95), (1.4, 1.95))
    arrow(a, (2.35, 3.35), (2.85, 2.75)); arrow(a, (2.35, 1.95), (2.85, 2.35))
    a = ax[2]
    box(a, 1.35, 3.0, 1.0, 0.7, "encoder\nA", fs=7.8); box(a, 1.35, 1.6, 1.0, 0.7, "encoder\nB", fs=7.8)
    box(a, 2.75, 2.1, 1.2, 0.9, "shared\nlayer", fc="#fff4e5", ec=YELLOW, fs=8.3)
    arrow(a, (1.15, 3.35), (1.35, 3.35)); arrow(a, (1.15, 1.95), (1.35, 1.95))
    arrow(a, (2.35, 3.35), (2.75, 2.75)); arrow(a, (2.35, 1.95), (2.75, 2.35))
    D.save(fig, "rb1_fusion")


def f_problem_types():
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"wspace": 0.35})
    rng = np.random.default_rng(2)
    a = ax[0]
    p0 = rng.normal([-1, 0], 0.5, (30, 2)); p1 = rng.normal([1, 0.3], 0.5, (30, 2))
    a.scatter(*p0.T, s=14, color=BLUE); a.scatter(*p1.T, s=14, color=ORANGE)
    a.plot([0.1, -0.1], [-1.6, 1.8], color=INK, lw=1.4, ls="--")
    a.set_title("Classification\n→ a category", fontsize=10); a.set_xticks([]); a.set_yticks([])
    a = ax[1]
    x = rng.uniform(0, 10, 40); y = 0.6 * x + rng.normal(0, 1, 40)
    a.scatter(x, y, s=14, color=BLUE); a.plot([0, 10], [0, 6], color=INK, lw=1.4)
    a.set_title("Regression\n→ a number", fontsize=10); a.set_xticks([]); a.set_yticks([])
    a = ax[2]
    t = np.linspace(0, 10, 200)
    a.plot(t, np.sin(t) * np.exp(-0.1 * t), color=GRAY, lw=1.6, label="input signal")
    a.plot(t, 0.8 * np.sin(t - 1.2) * np.exp(-0.1 * t) - 1.6, color=RED, lw=1.8, label="output signal")
    a.set_title("Signal-to-signal\n→ a whole time course", fontsize=10); a.set_xticks([]); a.set_yticks([])
    a.legend(fontsize=7.8, loc="upper right", bbox_to_anchor=(1.05, 1.0))
    D.save(fig, "rb2_problems")


def f_uncertainty():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(-3, 3, 70))
    x = x[(x < 0.6) | (x > 1.9)]
    noise = 0.15 + 0.35 * (x > 0)
    y = np.sin(x) + rng.normal(0, 1, len(x)) * noise
    xs = np.linspace(-3.4, 3.4, 300)
    fig, a = plt.subplots(figsize=(W, 2.6))
    a.scatter(x, y, s=14, color=GRAY, zorder=3, label="training data")
    a.plot(xs, np.sin(xs), color=BLUE, lw=2, label="model prediction")
    a.fill_between(xs, np.sin(xs) - 2 * (0.15 + 0.35 * (xs > 0)), np.sin(xs) + 2 * (0.15 + 0.35 * (xs > 0)), color=BLUE, alpha=0.12,
                   label="aleatoric: noise in the data itself")
    a.axvspan(0.6, 1.9, color=ORANGE, alpha=0.12, label="epistemic: no training data here")
    a.axvspan(3.0, 3.4, color=ORANGE, alpha=0.12)
    a.text(-3.2, 1.7, "quiet region:\nlow noise", fontsize=8.4, color=MUTED)
    a.text(2.1, -2.3, "noisy region", fontsize=8.4, color=MUTED)
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel("input"); a.set_ylabel("output")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.3)
    a.set_ylim(-2.6, 2.3)
    D.save(fig, "rb3_uncertainty")


def f_transfer():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.25})
    rng = np.random.default_rng(9)
    for a, shift, title in [(ax[0], (0, 0), "Same distribution: model transfers"), (ax[1], (2.2, 1.4), "Domain shift: model breaks")]:
        tr0 = rng.normal([-1, 0], 0.45, (25, 2)); tr1 = rng.normal([1, 0], 0.45, (25, 2))
        te0 = rng.normal([-1 + shift[0], shift[1]], 0.45, (25, 2)); te1 = rng.normal([1 + shift[0], shift[1]], 0.45, (25, 2))
        a.scatter(*tr0.T, s=14, color=BLUE); a.scatter(*tr1.T, s=14, color=ORANGE)
        a.scatter(*te0.T, s=22, facecolors="none", edgecolors=BLUE); a.scatter(*te1.T, s=22, facecolors="none", edgecolors=ORANGE)
        a.axvline(0, color=INK, lw=1.3, ls="--")
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
        a.set_xlim(-3, 5); a.set_ylim(-2, 4.2)
    ax[0].text(-2.9, 3.75, "● training people   ○ new people", fontsize=8.3, color=MUTED,
               bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
    ax[1].text(4.9, -1.8, "the boundary learned on\ntraining people now cuts\nthrough the new people", fontsize=8.3, color=MUTED, ha="right")
    D.save(fig, "rb4_shift")


for f in [f_fusion_levels, f_problem_types, f_uncertainty, f_transfer]:
    f()


# ---------------------------------------------------------------- question bank
QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "Framing the problem"
qa(S, "What is the difference between classification, regression and signal-to-signal prediction?", "Classification outputs a category (left or right hand), regression outputs a number (response amplitude), and signal-to-signal prediction outputs a whole time course (an HbO curve). They need different losses and metrics: accuracy for classification, error or correlation for regression and time courses.")
qa(S, "What is the difference between fusion and prediction (or cross-modal translation)?", "Fusion combines two modalities at use time to make a better decision, so both must be recorded. Cross-modal prediction (also called translation or synthesis) estimates one modality from another, so only the input modality is needed at use time.")
qa(S, "Why not just record fNIRS or fMRI directly?", "Because they are expensive, bulky or impossible in many settings: at home, during movement, for long periods, at the bedside or in low-resource clinics. EEG is cheap and wearable. A prediction model aims to bring some blood-based information to those settings, and the model's uncertainty tells the user how far to trust it.")
qa(S, "What is a hypothesis, and what makes it good?", "A specific, testable statement about the world. A good one is falsifiable (some outcome would show it false), precise about what is measured and compared, and decided before seeing the results. 'Cross-modal training improves held-out fNIRS prediction compared with fNIRS-only training at the same budget' is good; 'the model works' is not.")
qa(S, "Why are negative results valuable?", "They stop others wasting effort, correct over-optimistic literature and define the limits of a method. They are only valuable if the test was fair and adequately powered; a careless negative result proves nothing.")
qa(S, "What is pre-registration?", "Writing down the hypotheses, data, analysis and decision rules before seeing the results, often publicly. It prevents unconscious tuning of the analysis until it gives the hoped-for answer.")

S = "Multimodal learning"
qa(S, "What are early, late and intermediate fusion?", "Early fusion joins the raw data or features before a single model; late fusion trains separate models and combines their decisions; intermediate (hybrid) fusion uses separate encoders whose internal representations are joined. Intermediate fusion is the most common in deep learning.")
qa(S, "In neuroimaging, what are symmetric and asymmetric fusion?", "Symmetric fusion treats both modalities as equals and models them jointly (for example joint independent component analysis). Asymmetric fusion uses one modality to inform the analysis of the other, as in EEG-informed fMRI or fMRI-constrained EEG source localisation.")
qa(S, "What is a hybrid BCI?", "A brain–computer interface combining two signals, most often EEG and fNIRS, to improve accuracy or robustness. EEG reacts quickly; fNIRS adds a slower but complementary signal. Combining them typically raises classification accuracy over either alone.")
qa(S, "What is learning using privileged information?", "A framework (Vapnik and Vashist, 2009) in which extra information is available only during training. The learner uses it to learn better from the ordinary input. Our use of fNIRS and fMRI as training-only targets is in this spirit.")
qa(S, "What is knowledge distillation and cross-modal distillation?", "Distillation trains a 'student' model to imitate a 'teacher' model's outputs (Hinton and colleagues, 2015). In cross-modal distillation the teacher works on one modality and the student on another, for example teaching a depth-image network from a colour-image network (Gupta and colleagues, 2016). It transfers knowledge to a modality that lacks labels.")
qa(S, "What is contrastive learning, and how does CLIP relate?", "Contrastive learning trains encoders so that matching pairs (an image and its caption, or EEG and simultaneous fMRI) end up close in a shared space and non-matching pairs far apart. CLIP (2021) did this for images and text at very large scale. It needs many paired examples, which is the main obstacle for brain data.")

S = "Shared representations and transfer"
qa(S, "What is multi-task learning?", "Training one model on several related tasks with shared layers, so each task acts as extra supervision for the shared part (Caruana, 1997). It helps when tasks share structure and data are limited.")
qa(S, "What is negative transfer?", "When adding a task or dataset makes performance on the main task worse. Causes include unrelated tasks, conflicting gradients, imbalanced data and limited capacity: the extra task takes over the shared layers.")
qa(S, "What is a latent representation or embedding?", "An internal summary of an input as a vector of numbers learned by the network. A good one keeps information relevant to the task and discards nuisance variation. Its quality is often tested by how well a simple (linear) classifier performs on it, called a linear probe.")
qa(S, "What is transfer learning?", "Reusing a model trained on one task or dataset as the starting point for another, either by freezing it and training a small head on top, or by fine-tuning all of it. It saves data on the new task when the old one taught relevant features.")
qa(S, "What is domain shift?", "A difference between the data a model was trained on and the data it is used on: different people, devices, montages, tasks or recording conditions. Models usually degrade under shift. In EEG, the difference between people is itself a strong domain shift.")
qa(S, "What is domain adaptation, and what is CORAL?", "Methods that reduce the effect of domain shift, for example by aligning the feature statistics of the training and new domains. CORAL (correlation alignment; Sun and Saenko, 2016) matches the mean and covariance of features between domains.")
qa(S, "Why is cross-subject EEG so hard?", "Heads, brains and cortical folding differ, so the same activity produces different scalp patterns; electrode placement varies; and people use different strategies for the same task. Many EEG BCIs therefore need a short calibration session for each new user.")
qa(S, "What is zero-shot versus few-shot use on a new person?", "Zero-shot means using the model on a new person with no data from them. Few-shot means first adapting the model with a small amount of that person's data (a calibration session). Few-shot usually works better but costs time from the user.")

S = "Uncertainty and reliability"
qa(S, "What is the difference between aleatoric and epistemic uncertainty?", "Aleatoric uncertainty is noise inherent in the data; collecting more of the same data will not remove it. Epistemic uncertainty is the model's own ignorance, high where it saw little training data; more data reduces it (Kendall and Gal, 2017).")
qa(S, "How can a network estimate aleatoric uncertainty?", "By predicting a variance alongside each prediction and training with the Gaussian negative log-likelihood, which rewards admitting large variance where errors are large and penalises it where errors are small.")
qa(S, "How can a network estimate epistemic uncertainty?", "Common methods are Monte Carlo dropout (keeping dropout on at test time and measuring disagreement across repeated passes; Gal and Ghahramani, 2016), deep ensembles (training several models and measuring their disagreement; Lakshminarayanan and colleagues, 2017), and distance from the training data in the latent space.")
qa(S, "What is calibration, and how is it measured?", "Whether stated confidence matches actual accuracy. It is measured with reliability diagrams and the expected calibration error (ECE), the average gap between confidence and accuracy across confidence bins. Modern neural networks are often overconfident (Guo and colleagues, 2017).")
qa(S, "How do you fix a miscalibrated model?", "Post-hoc methods fitted on held-out data: temperature scaling, Platt scaling (a logistic fit) or isotonic regression. They change the confidence values without changing the predictions. They must be fitted and evaluated on separate data.")
qa(S, "What is selective prediction or abstention?", "Letting the model decline to answer when confidence is below a threshold. It is assessed with a risk–coverage curve: error on the answered cases as coverage falls. A useful gate lowers error as it abstains more (Geifman and El-Yaniv, 2017).")
qa(S, "Can a model be calibrated but useless for abstention?", "Yes. Calibration concerns the average match between confidence and accuracy; ranking concerns whether confidence separates good from bad cases. A model can state accurate average confidence while being unable to tell which individual cases are the bad ones.")
qa(S, "What is conformal prediction?", "A method that turns any model's outputs into prediction intervals or sets with a guaranteed coverage rate, assuming new data resemble the calibration data. The guarantee breaks under domain shift.")
qa(S, "Why does uncertainty matter more in medicine?", "A confident wrong answer can lead to a wrong decision about a patient. A system that knows when it does not know can hand difficult cases to a clinician. Regulators increasingly expect this.")

S = "Related work"
qa(S, "Has anyone predicted fMRI from EEG before?", "Yes. 'EEG fingerprint' methods learned to predict activity in a deep region such as the amygdala from EEG (Meir-Hasson and colleagues, 2014) and were later used for neurofeedback (Keynan and colleagues, 2019). More recent deep-learning work, such as NeuroBOLT (2024), reconstructs fMRI signals from EEG. Results are typically modest and strongest within subjects or for specific regions.")
qa(S, "Why is EEG-to-haemodynamic prediction hard in principle?", "EEG and blood signals are linked only indirectly through neurovascular coupling, which differs between people and regions; EEG sees mainly superficial, synchronised activity; blood signals contain systemic noise; the response is slow and overlaps across trials; and paired datasets are small.")
qa(S, "What makes a fair baseline in this kind of work?", "Baselines that could achieve the same score without the claimed ingredient: predicting each subject's mean response, a model trained on shuffled pairings, a model without the extra modality at the same training budget, and simple linear models on standard features. Beating only an untrained network is not evidence.")
qa(S, "What is neurofeedback and why is it relevant?", "Showing a person a live measure of their own brain activity so they can learn to control it. fMRI neurofeedback is precise but needs a scanner; EEG neurofeedback is portable but less specific. Predicting blood signals from EEG aims to combine the two.")

S = "Evaluation mindset"
qa(S, "What is the difference between in-distribution and out-of-distribution testing?", "In-distribution testing uses new examples from the same source as training (new trials or new people from the same dataset). Out-of-distribution testing uses a different source (another lab, device or task). The second is much harder and closer to real-world use.")
qa(S, "Why must the test set be truly unseen?", "If any information from the test data leaks into training or model selection, the score is optimistic and does not reflect real performance. For brain data, the most common leak is having the same person in training and test sets.")
qa(S, "What is an ablation study?", "Removing or replacing one component of a system at a time and measuring the effect, to show which components actually contribute.")
qa(S, "What does 'statistically significant but practically negligible' mean?", "With enough data even a tiny effect can be distinguished from zero, but it may be too small to matter. A correlation of 0.01 can be significant with thousands of trials yet explain only 0.01% of the variance. Always report effect size, not only the p-value.")


body = title_block("NeuroVascNet course · Lecture B · Reference", "Framing a Cross-Modal Prediction Problem: General Knowledge",
                   "The machine-learning and research-design ideas behind the research question, with model answers",
                   "Companion to “The Research Question” · Read Lecture B first · Use this to answer questions about the field, not only the project") + f"""
<div class="box">
<h3>How to use this document</h3>
<p>Lecture B explained what our project set out to do. This reference covers the general ideas any reviewer, examiner or collaborator will assume you know: the kinds of learning problems, ways of combining modalities, shared representations and transfer, uncertainty and reliability, related work, and how to frame claims that can be tested. Section 8 is a bank of {len(QA)} questions with model answers. For any unfamiliar question, answer in five steps: <b>define the term</b> → <b>give the intuition</b> → <b>say when it helps</b> → <b>say when it fails</b> → <b>connect it to our problem</b>.</p>
</div>

<h2>1. What kind of problem is this?</h2>
<p>Machine-learning problems are named after their output. <b>Classification</b> outputs a category; <b>regression</b> outputs a number; <b>sequence-to-sequence</b> or <b>signal-to-signal</b> prediction outputs a whole series over time. Our project is mostly the third kind: given 30 seconds of EEG, output 30 seconds of HbO, HbR or BOLD. Its simplified version, predicting one amplitude per trial, is regression.</p>
{fig("rb2_problems", 1, "Three kinds of learning problem.", "Schematic. Classification separates categories with a boundary; regression fits a number; signal-to-signal prediction maps one time course to another, which must get the shape, size and timing right.")}
<p>All three are <b>supervised learning</b>: the model learns from examples where the correct answer (the <b>target</b> or <b>label</b>) is known. The examples are split into a <b>training set</b> used to fit the model, a <b>validation set</b> used to make choices such as when to stop training, and a <b>test set</b> used once at the end to measure performance. Lecture E shows how easily this split can go wrong with brain data.</p>
{V("<p><b>“Is your project classification?”</b> No. It is cross-modal prediction: a regression problem whose output is a physiological time course (or, in the simplified version, a response amplitude). It is evaluated with correlation and error, not accuracy. Classification appears only as a secondary test of whether the learned representation is useful.</p>")}

<h2>2. Combining modalities</h2>
<h3>2.1 Fusion</h3>
<p>Most multimodal work aims to combine signals for a better decision. The standard taxonomy describes <i>where</i> they are combined:</p>
{fig("rb1_fusion", 2, "Where modalities can be combined.", "Early fusion merges inputs before one model; late fusion merges the decisions of separate models; intermediate fusion merges learned representations from separate encoders.")}
<p>In neuroimaging, fusion is also described as <b>symmetric</b> (both modalities modelled jointly, for example with joint independent component analysis) or <b>asymmetric</b> (one informs the analysis of the other). <b>EEG-informed fMRI</b> uses EEG features, convolved with the haemodynamic response function, as regressors to find where BOLD tracks them; <b>fMRI-informed EEG</b> uses fMRI maps to constrain where EEG sources are allowed to be. In BCI research, <b>hybrid EEG–fNIRS systems</b> regularly classify better than either signal alone.</p>

<h3>2.2 Cross-modal prediction, translation and synthesis</h3>
<p>A different goal is to produce one modality from another. In computer vision this is called image-to-image translation (for example turning a sketch into a photograph); in medical imaging, <b>synthesis</b> creates one scan type from another (for example a CT-like image from an MRI). Our problem is the brain-signal version: EEG in, haemodynamics out. Its practical appeal is that only the cheap modality is needed when the system is used.</p>

<h3>2.3 Learning with information you will not have later</h3>
<p>Several frameworks formalise "use the expensive signal only during training":</p>
{table(1, "Frameworks for using a modality only during training.",
       ["Framework", "Idea", "Example"],
       [["Learning using privileged information", "extra information at training time helps learn from the ordinary input (Vapnik and Vashist, 2009)", "using fNIRS during training of an EEG model"],
        ["Knowledge distillation", "a student model imitates a teacher model's outputs (Hinton and colleagues, 2015)", "a small network learning from a large one"],
        ["Cross-modal distillation", "a teacher on one modality supervises a student on another (Gupta and colleagues, 2016)", "a depth-image network taught by a colour-image network"],
        ["Contrastive alignment", "paired inputs from two modalities are pulled together in a shared space", "CLIP for images and text; EEG–fMRI alignment"],
        ["Cross-modal regression (our approach)", "the second modality is the regression target itself", "EEG → predicted HbO, HbR and BOLD"]],
       widths=["28%", "44%", "28%"])}
{A("the driving instructor's second brake pedal", "<p>A learner driver practises in a car with a second set of pedals for the instructor. The instructor's pedals are privileged information: they shape the learning but are removed for the driving test. A good learner has absorbed the lessons; a bad one only drove well because the instructor was correcting them.</p>")}

<h2>3. Shared representations</h2>
<h3>3.1 Multi-task learning and its risk</h3>
<p>In <b>multi-task learning</b> one network with shared layers learns several tasks at once (Caruana, 1997). The shared layers must serve every task, which acts as a form of regularisation and can make better use of limited data. Our shared EEG encoder with an fNIRS head and an fMRI head is a multi-task design.</p>
<p>It can also backfire. <b>Negative transfer</b> occurs when an additional task harms the main one: the tasks may not share the useful structure, their training signals may pull the shared layers in conflicting directions, one dataset may be much larger or noisier than the other, or the shared layers may simply lack capacity. The standard test is a controlled comparison: train with and without the extra task at the same budget, and, more strictly, with the extra task's targets <i>shuffled</i>. If real and shuffled targets give the same result, the extra task's content is not what matters.</p>
{K("<p>Adding a second modality is not automatically helpful. Whether it helps is an empirical question, answered only by a fair comparison against the single-modality model and a shuffled-target control.</p>")}

<h3>3.2 Latent representations and probes</h3>
<p>The internal summary a network learns is a <b>latent representation</b> or <b>embedding</b>. Its quality is usually judged indirectly with a <b>linear probe</b>: freeze the encoder and fit a simple linear classifier on the embeddings for some task. If a linear classifier succeeds, the information is laid out simply in the representation. Probes must always be compared with controls, because even a randomly initialised network can produce surprisingly useful features.</p>

<h3>3.3 Transfer learning and domain shift</h3>
<p><b>Transfer learning</b> reuses a model trained in one setting in another, either frozen with a new head or fine-tuned. Its enemy is <b>domain shift</b>: any systematic difference between training and use, such as different people, electrodes, devices, tasks or labs.</p>
{fig("rb4_shift", 3, "Domain shift.", "Schematic. Filled dots are training people, open circles new people. Left: the new people look like the training people, and the learned boundary still works. Right: the new people's data are shifted, and the same boundary fails.")}
<p>In EEG, differences between people are large enough to count as domain shift by themselves: cortical folding, skull thickness, electrode placement and mental strategy all differ. That is why many BCIs need a short calibration session for each user, and why a model that works on new trials from the same people may fail on new people. Methods to reduce shift are called <b>domain adaptation</b>; a simple one is <b>CORAL</b>, which aligns the mean and covariance of features between domains (Sun and Saenko, 2016). Using a model on a new person without any of their data is <b>zero-shot</b>; adapting with a small amount first is <b>few-shot</b>.</p>

<h2>4. Uncertainty and reliability</h2>
<h3>4.1 Two kinds of uncertainty</h3>
{fig("rb3_uncertainty", 4, "Aleatoric and epistemic uncertainty.", "Schematic. Aleatoric uncertainty (blue band) comes from noise in the data and is wider where the data are noisier. Epistemic uncertainty (orange) comes from the model's ignorance where it has seen no training data; more data would reduce it.")}
<p>Aleatoric uncertainty is estimated by having the network predict a variance as well as a mean, trained with the Gaussian negative log-likelihood. Epistemic uncertainty is estimated with Monte Carlo dropout, deep ensembles or the distance of a new input from the training data in the latent space. For a system used on new people, epistemic uncertainty is the more important one: it is what should rise when the input looks unlike anything seen before.</p>

<h3>4.2 Calibration, ranking and abstention</h3>
<p>A confidence score can be judged in two independent ways. <b>Calibration</b> asks whether stated confidence equals actual accuracy on average; it is measured with reliability diagrams and the <b>expected calibration error (ECE)</b>, and repaired post hoc with temperature scaling, Platt scaling or isotonic regression fitted on held-out data. <b>Ranking</b> (or discrimination) asks whether the score separates good from bad individual predictions; it is measured by correlating uncertainty with error, or with a <b>risk–coverage curve</b> showing error on the answered cases as the system abstains more. <b>Selective prediction</b> uses the score to abstain on low-confidence cases. <b>Conformal prediction</b> gives intervals with guaranteed coverage, but only when new data resemble the calibration data.</p>
{M("<p>Treating a low calibration error as proof that the reliability layer works. Calibration can be fixed by a simple rescaling even when the score carries no information about which cases are wrong. Always report ranking (risk–coverage or uncertainty–error correlation) alongside calibration, and test both on data from people not used to fit the calibration.</p>")}

<h2>5. Related work: predicting blood signals from EEG</h2>
<p>The idea that EEG can stand in for haemodynamic measures is not new. Knowing the main lines of work lets you place our project and answer "hasn't this been done?".</p>
{table(2, "Main lines of related work.",
       ["Line of work", "What it does", "Typical finding"],
       [["EEG-informed fMRI", "uses EEG features as regressors to locate where BOLD tracks them", "well established; maps networks behind EEG rhythms and epileptic spikes"],
        ["EEG fingerprints (EFP)", "learns to predict activity of one specific region (for example the amygdala) from EEG (Meir-Hasson and colleagues, 2014)", "works for targeted regions, enabled EEG-based neurofeedback (Keynan and colleagues, 2019)"],
        ["Deep EEG-to-fMRI synthesis", "deep networks reconstruct fMRI signals from simultaneous EEG (for example NeuroBOLT, 2024)", "some signals recoverable; generalisation across people remains difficult"],
        ["Hybrid EEG–fNIRS BCIs", "fuse the two signals to classify mental states", "fusion usually beats either alone; requires both at use time"],
        ["Our project", "one encoder supervised by both fNIRS and fMRI from different people, with a reliability layer", "see Lecture F"]],
       widths=["24%", "44%", "32%"])}
<p>The common lessons from this literature are that prediction is easier <b>within a person</b> than across people, easier for <b>targeted regions</b> than for whole-brain signals, and easier with <b>more paired data</b>. Any claim of novelty should therefore rest on something specific, such as combining two supervision modalities, the reliability layer, or a rigorous assessment of what does and does not work.</p>

<h2>6. Turning an idea into claims</h2>
<p>Strong research questions share a few properties. They name the <b>population</b> (new people, not new trials from familiar ones), the <b>comparison</b> (against what baseline), the <b>outcome measure</b> (which metric) and the <b>decision rule</b> (what result counts as support). Writing these down in advance, ideally publicly as a <b>pre-registration</b>, prevents an analysis drifting towards the answer the researcher hoped for. A claim is scientific only if some realistic outcome could refute it.</p>
{table(3, "From vague to testable.",
       ["Vague", "Testable"],
       [["The model predicts haemodynamics from EEG.", "On subjects held out from training, after removing each subject's mean, predicted and measured fNIRS correlate above a shuffled-pairing null."],
        ["Two modalities are better than one.", "At equal training budget, the fNIRS+fMRI model beats the fNIRS-only model on held-out subjects, with a confidence interval excluding zero, and beats a shuffled-fMRI control."],
        ["The reliability score is useful.", "On held-out subjects, error falls as coverage falls, and calibration error is below a stated threshold."]],
       widths=["35%", "65%"])}
<p>Two further habits mark careful work. An <b>ablation study</b> removes one component at a time to show what each contributes. And results are reported with <b>effect sizes and confidence intervals</b>, not only p-values: a correlation of 0.01 can be statistically significant with thousands of trials and still be useless.</p>

<h2>7. Where this kind of system could be used</h2>
{table(4, "Possible applications and what each would require.",
       ["Application", "Why EEG-only blood estimates help", "What it would need to work"],
       [["Home or wearable neurofeedback", "fMRI-style targets without a scanner", "reliable prediction for a specific region in each user"],
        ["Bedside monitoring after stroke or haemorrhage", "continuous estimate of neurovascular function", "robustness to altered coupling in patients; trustworthy uncertainty"],
        ["Brain–computer interfaces", "blood-related information without extra hardware", "real-time operation; benefit over EEG features alone"],
        ["Research at scale", "cheap proxy for haemodynamics in large studies", "validated accuracy against real measurements"]],
       widths=["28%", "37%", "35%"])}
{V("<p><b>“Is this ready for clinical use?”</b> No. Prediction across people is modest, generalisation to new datasets fails, and the confidence score does not yet identify bad predictions. The value at this stage is in establishing what is and is not achievable, and which ingredients (reliable targets, honest evaluation, per-person calibration) any clinical version would need.</p>")}

<h2>8. Question bank with model answers</h2>
<p>Practise answering aloud in your own words. The answers give the expected depth for a viva or supervisor meeting.</p>
{qa_bank(QA)}

<h2>9. Where to read more</h2>
<ul class="refs">
<li>Caruana R. (1997). Multitask learning. <i>Machine Learning</i>.</li>
<li>Vapnik V., Vashist A. (2009). A new learning paradigm: learning using privileged information. <i>Neural Networks</i>.</li>
<li>Hinton G., Vinyals O., Dean J. (2015). Distilling the knowledge in a neural network. <i>arXiv</i>.</li>
<li>Gupta S., Hoffman J., Malik J. (2016). Cross modal distillation for supervision transfer. <i>CVPR</i>.</li>
<li>Sun B., Saenko K. (2016). Deep CORAL: correlation alignment for deep domain adaptation. <i>ECCV Workshops</i>.</li>
<li>Gal Y., Ghahramani Z. (2016). Dropout as a Bayesian approximation. <i>ICML</i>.</li>
<li>Kendall A., Gal Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? <i>NeurIPS</i>.</li>
<li>Lakshminarayanan B., Pritzel A., Blundell C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. <i>NeurIPS</i>.</li>
<li>Guo C., Pleiss G., Sun Y., Weinberger K. Q. (2017). On calibration of modern neural networks. <i>ICML</i>.</li>
<li>Geifman Y., El-Yaniv R. (2017). Selective classification for deep neural networks. <i>NeurIPS</i>.</li>
<li>Meir-Hasson Y. et al. (2014). An EEG finger-print of fMRI deep regional activation. <i>NeuroImage</i>.</li>
<li>Keynan J. N. et al. (2019). Electrical fingerprint of the amygdala guides neurofeedback training for stress resilience. <i>Nature Human Behaviour</i>.</li>
</ul>
"""

D.render(body, "Lecture B Reference: General Knowledge")
