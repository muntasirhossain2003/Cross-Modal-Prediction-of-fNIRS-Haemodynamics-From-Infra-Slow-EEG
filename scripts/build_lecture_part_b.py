"""Builds lectures/Lecture_B_The_Research_Question.pdf (teaching notes, Part B).

Figures are schematic. Run from the repo root:  python scripts/build_lecture_part_b.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, E, K, M, P,
                            Q, arrow, badge, box, fig, hrf, table, title_block)

D = Doc("lecture_b", "Lecture_B_The_Research_Question.pdf", "NeuroVascNet course · Lecture B")


# ---------------------------------------------------------------- figures
def f_train_deploy():
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(0.05, 5.95, "TRAINING  (in the lab, once)", fontsize=10, weight="bold", color=BLUE)
    box(ax, 0.1, 4.2, 1.6, 1.0, "EEG", fc=BLUE, ec=BLUE, color="white", weight="bold")
    box(ax, 2.4, 4.2, 2.0, 1.0, "Model", fc="#fff4e5", ec=YELLOW, weight="bold")
    box(ax, 5.1, 4.2, 2.0, 1.0, "Predicted\nfNIRS / fMRI", fc="#fdeceb", ec=RED)
    box(ax, 7.8, 4.9, 2.1, 0.9, "Real fNIRS / fMRI\n(recorded with EEG)", fc=RED, ec=RED, color="white", fs=8.8)
    box(ax, 7.8, 3.75, 2.1, 0.8, "Compare:\nhow wrong? (loss)", fc="#f3f5f8", ec=GRAY, fs=8.8)
    arrow(ax, (1.7, 4.7), (2.4, 4.7)); arrow(ax, (4.4, 4.7), (5.1, 4.7))
    arrow(ax, (7.1, 4.55), (7.8, 4.2)); arrow(ax, (8.85, 4.9), (8.85, 4.55))
    arrow(ax, (8.3, 3.75), (3.4, 4.15), color=ORANGE, rad=-0.25)
    ax.text(5.6, 2.95, "adjust the model to be less wrong; repeat thousands of times", fontsize=8.5, color=ORANGE, ha="center")
    ax.plot([0, 10], [2.7, 2.7], color=LIGHT, lw=1.2)
    ax.text(0.05, 2.4, "DEPLOYMENT  (in the world, every day)", fontsize=10, weight="bold", color=AQUA)
    box(ax, 0.1, 0.6, 1.6, 1.0, "EEG\nonly", fc=BLUE, ec=BLUE, color="white", weight="bold")
    box(ax, 2.4, 0.6, 2.0, 1.0, "Trained\nmodel", fc="#fff4e5", ec=YELLOW, weight="bold")
    box(ax, 5.1, 0.15, 2.75, 1.9, "1  latent summary\n2  predicted HbO, HbR, BOLD\n3  neurovascular delay\n4  reliability (or abstain)",
        fc="#eef6f1", ec=AQUA, fs=8.4)
    box(ax, 8.2, 0.6, 1.7, 1.0, "fNIRS / fMRI\nnot needed", fc="white", ec=GRAY, color=GRAY, ls="--", fs=8.6)
    arrow(ax, (1.7, 1.1), (2.4, 1.1)); arrow(ax, (4.4, 1.1), (5.1, 1.1))
    D.save(fig, "b1_train_deploy")


def f_fusion():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.12})
    for a in ax:
        a.set_xlim(0, 5); a.set_ylim(0, 3.2); a.axis("off")
    a = ax[0]
    a.set_title("Fusion: needs both at test time", fontsize=10)
    box(a, 0.1, 1.85, 1.3, 0.75, "EEG", fc=BLUE, ec=BLUE, color="white", weight="bold")
    box(a, 0.1, 0.55, 1.3, 0.75, "fNIRS", fc=RED, ec=RED, color="white", weight="bold")
    box(a, 1.9, 1.2, 1.3, 0.75, "combine", fc="#fff4e5", ec=YELLOW)
    box(a, 3.65, 1.2, 1.3, 0.75, "decision\n(e.g. left/right)", fc="#f3f5f8", ec=GRAY, fs=8.5)
    arrow(a, (1.4, 2.2), (1.9, 1.7)); arrow(a, (1.4, 0.95), (1.9, 1.45)); arrow(a, (3.2, 1.57), (3.65, 1.57))
    a = ax[1]
    a.set_title("Prediction (our project): EEG only at test time", fontsize=10)
    box(a, 0.1, 1.2, 1.3, 0.75, "EEG", fc=BLUE, ec=BLUE, color="white", weight="bold")
    box(a, 1.9, 1.2, 1.3, 0.75, "model", fc="#fff4e5", ec=YELLOW)
    box(a, 3.65, 1.2, 1.3, 0.75, "predicted\nfNIRS / fMRI", fc="#fdeceb", ec=RED, fs=8.5)
    arrow(a, (1.4, 1.57), (1.9, 1.57)); arrow(a, (3.2, 1.57), (3.65, 1.57))
    a.text(2.5, 0.55, "fNIRS/fMRI used only as the\nanswer key during training", ha="center", fontsize=8.5, color=MUTED)
    D.save(fig, "b2_fusion")


def f_arch():
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.set_xlim(0, 10.1); ax.set_ylim(-0.1, 6.3); ax.axis("off")
    box(ax, 0.05, 2.5, 1.3, 1.1, "EEG\n32 electrodes\n30 s", fc=BLUE, ec=BLUE, color="white", fs=8.6, weight="bold")
    box(ax, 1.65, 2.5, 1.35, 1.1, "band power\n128 channels\n20 Hz", fs=8.6)
    box(ax, 3.3, 2.5, 1.25, 1.1, "adapter\n(per dataset)", fs=8.6)
    box(ax, 4.85, 2.3, 1.55, 1.5, "SHARED\nTRUNK\n(EEGNet + TCN)", fc="#fff4e5", ec=YELLOW, weight="bold", fs=8.8)
    arrow(ax, (1.35, 3.05), (1.65, 3.05)); arrow(ax, (3.0, 3.05), (3.3, 3.05)); arrow(ax, (4.55, 3.05), (4.85, 3.05))
    outs = [(5.0, "latent summary z\n(64 numbers)", "#f6f2fb", VIOLET, 1),
            (3.75, "fNIRS head → HbO, HbR\n(mean + uncertainty)", "#fdeceb", RED, 2),
            (2.5, "fMRI head → BOLD per region\n(mean + uncertainty)", "#e8f6f0", AQUA, 2),
            (1.25, "delay module → lag\n(bounded 2–8 s)", "#fff4e5", YELLOW, 3),
            (0.0, "reliability → confidence\nanswer or abstain", "#eaf2fc", BLUE, 4)]
    for y, txt, fc, ec, n in outs:
        box(ax, 7.0, y + 0.05, 2.75, 0.95, txt, fc=fc, ec=ec, fs=8.5)
        arrow(ax, (6.4, 3.05), (7.0, y + 0.52), rad=0.0)
        badge(ax, 9.75, y + 0.95, n)
    ax.text(5.62, 4.05, "learns from BOTH\nfNIRS and fMRI errors", ha="center", fontsize=8.3, color=ORANGE)
    D.save(fig, "b3_arch")


def f_gap():
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    rng = np.random.default_rng(4)
    for cx, col, lab, sub in [(1.6, RED, "Group A", "EEG + fNIRS recorded together"), (8.4, AQUA, "Group B", "EEG + fMRI recorded together")]:
        for i in range(12):
            x = cx + (i % 4 - 1.5) * 0.45 + rng.uniform(-0.05, 0.05)
            y = 2.9 - (i // 4) * 0.55
            ax.add_patch(plt.Circle((x, y + 0.18), 0.1, color=col))
            ax.add_patch(plt.Rectangle((x - 0.1, y - 0.18), 0.2, 0.28, color=col))
        ax.text(cx, 3.65, lab, ha="center", fontsize=10, weight="bold", color=col)
        ax.text(cx, 1.15, sub, ha="center", fontsize=8.6, color=MUTED)
    box(ax, 3.9, 1.9, 2.2, 1.2, "one shared\nEEG encoder", fc="#fff4e5", ec=YELLOW, weight="bold")
    arrow(ax, (2.9, 2.5), (3.9, 2.5)); arrow(ax, (7.1, 2.5), (6.1, 2.5))
    ax.text(5.0, 0.55, "Nobody is in both groups: no person has EEG, fNIRS and fMRI.\nThe two kinds of supervision can only meet inside the shared encoder.",
            ha="center", fontsize=8.8, color=INK)
    ax.text(5.0, 3.6, "different people", ha="center", fontsize=8.6, color=MUTED, style="italic")
    D.save(fig, "b4_gap")


def f_wave_amp():
    t = np.linspace(-2, 28, 600)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.3, "width_ratios": [1.5, 1]})
    a = ax[0]
    rng = np.random.default_rng(1)
    for amp, col, lab in [(1.0, RED, "trial 1"), (0.55, "#f2a09f", "trial 2")]:
        y = amp * hrf(t - 0.5) + 0.06 * rng.standard_normal(len(t))
        a.plot(t, y, color=col, lw=1.8, label=lab)
    a.axvspan(4, 10, color=YELLOW, alpha=0.13)
    a.text(7, 1.12, "response window", ha="center", fontsize=8.5, color="#8a6a00")
    a.axhline(0, color=MUTED, lw=0.8); a.set_yticks([0]); a.set_xlabel("time (s)")
    a.set_title("Waveform: the whole curve (300 numbers)", fontsize=10); a.legend(loc="upper right"); a.set_ylim(-0.4, 1.25)
    a = ax[1]
    a.bar([0, 1], [0.86, 0.47], color=[RED, "#f2a09f"], width=0.55)
    a.set_xticks([0, 1]); a.set_xticklabels(["trial 1", "trial 2"]); a.set_yticks([0])
    a.set_title("Amplitude: one number\n(average in the window)", fontsize=10); a.grid(axis="x", visible=False)
    D.save(fig, "b5_wave_amp")


def f_latent():
    rng = np.random.default_rng(7)
    fig, a = plt.subplots(figsize=(W, 2.6))
    for (cx, cy), col, lab in [((-1.6, 0.4), GRAY, "rest"), ((1.0, 1.3), BLUE, "imagine left hand"), ((1.3, -1.0), ORANGE, "imagine right hand")]:
        p = rng.normal([cx, cy], 0.45, (40, 2))
        a.scatter(p[:, 0], p[:, 1], s=22, color=col, edgecolor="white", lw=0.8, label=lab)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("latent dimension 1"); a.set_ylabel("latent dimension 2")
    a.legend(loc="upper left")
    a.text(-3.2, -2.45, "each dot = one trial, summarised by 64 numbers\n(squashed to 2 here so we can draw it)", ha="left", fontsize=8.5, color=MUTED)
    a.set_xlim(-3.3, 3.4); a.set_ylim(-2.6, 2.8)
    D.save(fig, "b6_latent")


def f_reliability():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"wspace": 0.35})
    a = ax[0]
    c = np.linspace(0.5, 1, 6)
    a.plot([0.5, 1], [0.5, 1], color=MUTED, ls="--", lw=1, label="perfectly calibrated")
    a.plot(c, c, color=AQUA, lw=2.2, marker="o", ms=5, label="well calibrated")
    a.plot(c, 0.5 + (c - 0.5) * 0.45, color=ORANGE, lw=2.2, marker="o", ms=5, label="overconfident")
    a.set_xlabel("confidence the model states"); a.set_ylabel("how often it is actually right")
    a.set_title("Calibration: are the numbers honest?", fontsize=10); a.legend(fontsize=8, loc="upper left")
    a = ax[1]
    cov = np.linspace(1, 0.2, 9)
    a.plot(cov * 100, 0.30 - 0.18 * (1 - cov), color=AQUA, lw=2.2, marker="o", ms=5, label="useful gate")
    a.plot(cov * 100, 0.30 + 0.0 * cov, color=ORANGE, lw=2.2, marker="o", ms=5, label="useless gate")
    a.invert_xaxis(); a.set_xlabel("% of cases the system answers"); a.set_ylabel("error on answered cases")
    a.set_title("Abstention: does refusing help?", fontsize=10); a.legend(fontsize=8, loc="lower left"); a.set_ylim(0.05, 0.36)
    D.save(fig, "b7_reliability")


for f in [f_train_deploy, f_fusion, f_arch, f_gap, f_wave_amp, f_latent, f_reliability]:
    f()


# ---------------------------------------------------------------- text
body = title_block("NeuroVascNet course · Lecture B", "The Research Question",
                   "What we set out to build, why it was designed this way, and how an idea becomes testable claims",
                   "Part B of 7 · Builds on Lecture A · About 75 minutes of reading") + f"""
<div class="box">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>state the project's thesis in one sentence and explain every part of it;</li>
<li>explain the difference between <i>training time</i> and <i>deployment time</i>, and why fNIRS and fMRI are only used in training;</li>
<li>tell prediction apart from fusion, and say why fusion would break the project's premise;</li>
<li>describe the four outputs the system was meant to produce and why each one is useful;</li>
<li>explain how the thesis was turned into five claims that can each pass or fail, and why the lack of a dataset with the same people in fNIRS and fMRI shapes the whole design.</li>
</ul>
</div>

<h2>1. From Lecture A to a research question</h2>
<p>Lecture A ended with one picture: a single burst of brain activity produces a fast electrical signal (EEG) and, a few seconds later, a slow blood signal (fNIRS or fMRI). Like lightning followed by thunder, the two come from the same event. This lecture asks what you would do with that fact if you were designing a research project.</p>
<p>The answer our project chose is: <b>learn to predict the blood signal from the electrical signal</b>, so that eventually only the cheap, portable EEG is needed. Everything else in the project — the datasets, the model, the evaluation — follows from that one decision.</p>

<h3>1.1 Why anyone would want this</h3>
{table(1, "The practical gap between the three machines.",
       ["", "EEG", "fNIRS", "fMRI"],
       [["Cost of equipment", "low", "moderate", "very high (a hospital scanner)"],
        ["Portable / wearable", "yes", "yes, but bulkier", "no"],
        ["Can be used at the bedside, at home, while moving", "yes", "partly", "no"],
        ["What it tells us directly", "electrical activity", "blood oxygenation (surface)", "blood oxygenation (whole brain)"]],
       widths=["40%", "17%", "20%", "23%"])}
<p>Blood-based measures are valuable: they are what doctors and neuroscientists use to map where the brain is working, and blood flow problems are central to stroke, haemorrhage and dementia. But they are expensive or awkward to record. If a model could look at an EEG and say "this is roughly what the blood is doing, and this is how confident I am", some of that value would become available wherever an EEG can be worn.</p>
{E("three places it could matter", "<ul><li><b>Neurofeedback.</b> People learn to control a brain region by watching a live signal of its activity. With fMRI this needs a scanner; with an EEG-based prediction it could be done at home. (One of our datasets, ds002336, is in fact a neurofeedback study.)</li><li><b>Bedside monitoring.</b> A patient after a brain haemorrhage could wear an EEG cap for days, with the model flagging changes in predicted blood response.</li><li><b>Brain–computer interfaces.</b> A BCI could use EEG speed together with predicted blood information without needing extra hardware.</li></ul>")}

<h2>2. The thesis, word by word</h2>
<p>A <b>thesis</b> is the central claim a project sets out to support. Ours was:</p>
{K("<p><b>One shared EEG encoder, trained with paired EEG–fNIRS and EEG–fMRI supervision, takes EEG only at deployment and outputs: (1) a hemodynamic-aware latent representation, (2) a predicted hemodynamic response (HbO/HbR-like and BOLD-like), (3) a neurovascular delay descriptor, and (4) a calibrated reliability score that allows abstention.</b></p>")}
<p>Every phrase in this sentence is a design decision. Let us take them one at a time.</p>
{table(2, "Unpacking the thesis.",
       ["Phrase", "What it means", "Why it was chosen"],
       [["<b>one shared EEG encoder</b>", "a single neural network that reads EEG and turns it into an internal summary", "so that what is learned from fNIRS and from fMRI ends up in the same place and can help each other"],
        ["<b>paired supervision</b>", "training data where EEG and a blood signal were recorded at the same time from the same person", "pairs are the only way to learn which EEG goes with which blood response"],
        ["<b>EEG–fNIRS and EEG–fMRI</b>", "two kinds of pairs, from two different groups of people", "the two blood signals have complementary strengths (Section 5)"],
        ["<b>EEG only at deployment</b>", "once trained, the system needs nothing but EEG", "this is the whole practical point: cheap input, rich output"],
        ["<b>four outputs</b>", "summary, predicted response, delay, reliability", "each serves a different user need (Section 6)"],
        ["<b>calibrated reliability … abstention</b>", "an honest confidence score, and the option to say 'I don't know'", "a medical or scientific tool must not give confident wrong answers"]],
       widths=["24%", "38%", "38%"])}

<h2>3. Training time and deployment time</h2>
<p>This is the single most important distinction in the whole project, and it is the one people most often get confused about. A machine-learning system lives two separate lives.</p>
<p><b>Training</b> happens once, in the lab. We show the model thousands of examples where we have both the EEG <i>and</i> the real blood signal recorded at the same moment. The model makes a guess, we compare the guess with the real signal, measure how wrong it was (the <b>loss</b>), and nudge the model to be less wrong. Repeat many thousands of times.</p>
<p><b>Deployment</b> (also called <b>inference</b>) is what happens afterwards, every time the system is used. Now the model receives only EEG. There is no real blood signal to compare with — producing one is the model's job.</p>
{fig("b1_train_deploy", 1, "The two lives of the system.", "Top: during training, real fNIRS and fMRI recordings act as the answer key; the model is corrected whenever its prediction differs from them. Bottom: after training, the model receives EEG only and produces its four outputs. The expensive machines are no longer needed.")}
{A("learning a language with a teacher", "<p>A student learns to translate with a teacher who marks every attempt against the correct answer. In the exam there is no teacher and no answer sheet; the student must translate alone. Training is the classroom, the real fNIRS and fMRI recordings are the teacher's answer sheet, and deployment is the exam. A student who can only translate when the answer sheet is on the desk has not learned anything useful.</p>")}
{M("<p>Saying “the system uses fNIRS and fMRI”. It uses them only to <i>learn</i>. A system that needed fNIRS or fMRI every time it ran would lose the reason for existing: it would be cheaper just to read the fNIRS or fMRI directly.</p>")}
<p>A useful technical name for this arrangement is <b>learning using privileged information</b>: extra information (here, the blood signals) is available while learning but not when the system is used. It is the same idea as a doctor training on cases where the biopsy result is known, then diagnosing new patients from symptoms alone.</p>

<h2>4. Prediction, not fusion</h2>
<p>Many studies that record EEG and fNIRS together <b>fuse</b> them: they combine both signals to make a better decision, for example whether a person is imagining moving their left or right hand. Fusion is a perfectly good idea — combining two views of the brain usually beats either alone — but it requires <b>both</b> signals every time the system runs.</p>
{fig("b2_fusion", 2, "Fusion versus prediction.", "Left: fusion combines EEG and fNIRS to make a decision, so both must be recorded every time. Right: our prediction approach maps EEG to an estimate of the blood signal; fNIRS and fMRI only appear as the answer key during training.")}
<p>Our project deliberately did <b>not</b> do fusion, because the goal was to remove the need for the blood-measuring machine. It also did not simply classify (left vs right hand); it tried to reproduce a whole <i>signal</i> — the time course of HbO, HbR or BOLD. That is a harder and more informative task.</p>
{table(3, "Three different things you can do with paired EEG and blood data.",
       ["Approach", "Input when used", "Output", "Example"],
       [["Fusion (classification)", "EEG <b>and</b> fNIRS", "a decision", "hybrid BCI: left or right hand?"],
        ["Classification from EEG", "EEG", "a decision", "standard EEG BCI"],
        ["<b>Prediction (our project)</b>", "<b>EEG</b>", "<b>a blood signal over time</b>", "<b>estimate HbO, HbR, BOLD from EEG</b>"]],
       widths=["26%", "18%", "24%", "32%"])}
{P("<p>The TU Berlin hBCI dataset we later trained on was originally collected for <i>fusion</i> (the 'h' stands for hybrid). We used it differently: as a source of EEG–fNIRS pairs to learn prediction. Reusing a dataset for a new question is normal, but it means the task design, timing and channels were chosen for someone else's purpose — a point that returns in Part C.</p>")}

<h2>5. Why supervise with both fNIRS and fMRI?</h2>
<p>We could have trained one model on EEG–fNIRS and a completely separate model on EEG–fMRI. The thesis bet that <b>one shared encoder trained on both</b> would be better than either alone. The reasoning was:</p>
<ul>
<li>fNIRS and fMRI are two views of the <i>same</i> underlying blood response (Lecture A). fNIRS sees HbO and HbR separately with good timing but only at the surface; fMRI sees the whole brain in detail but slowly and without separating HbO from HbR.</li>
<li>If the encoder must satisfy both teachers, it should be pushed to capture what is common to both — the genuine "blood-relevant" information in the EEG — rather than quirks of one machine.</li>
<li>Each dataset is small. Pooling two sources of supervision gives the shared encoder more examples to learn from.</li>
</ul>
{A("two maps of the same city", "<p>One person learns a city from a street map; another learns it from satellite photos. Each map has gaps and errors, but both describe the same city. Someone who studies both builds a better mental model than someone who studies one, because errors in one are corrected by the other. The shared encoder was meant to be that person, with fNIRS and fMRI as the two maps.</p>")}
<p>In machine-learning language this is <b>multi-task learning</b> with a shared representation. It is known to help when the tasks genuinely share structure — and known to hurt when they do not, which is called <b>negative transfer</b>. Whether it helps here is not obvious in advance, which is exactly why it became the project's first claim to test (Section 7).</p>

<h3>5.1 The subject gap: the constraint that shapes everything</h3>
<p>Ideally we would have recordings where the same person wore EEG and fNIRS inside an MRI scanner, so all three signals were measured together. No such public dataset exists. Our EEG–fNIRS data and our EEG–fMRI data come from <b>completely different people</b>.</p>
{fig("b4_gap", 3, "The subject gap.", "The fNIRS supervision and the fMRI supervision come from two groups with no person in common. The only bridge between them is the shared EEG encoder: anything the two kinds of supervision share has to be learned there.")}
{A("translating through English", "<p>Imagine learning English→French from one teacher and English→German from another, and never seeing the same sentence in French and German. You can still get better at both, because both teachers improve your understanding of English, the language they share. EEG is the shared language here. The model can never directly check that its predicted fNIRS and fMRI agree for the same moment, because no such moment exists in the data; any benefit has to come through a better understanding of EEG.</p>")}
<p>The subject gap also means we cannot rely on "the same person" to link the two kinds of supervision. And because the model will be used on new people, it must work <b>across subjects</b>: trained on some people, tested on others. Lectures E and F show that this cross-subject requirement turned out to be the hardest part of the whole project.</p>

<h2>6. The four outputs</h2>
{fig("b3_arch", 4, "Where the four outputs come from.", "EEG is turned into band power, adapted to a common format, and passed through the shared trunk. The trunk's internal summary is output 1; separate heads turn it into predicted fNIRS and fMRI signals (output 2), a delay (output 3) and a reliability score (output 4). The trunk receives corrections from both the fNIRS and fMRI errors during training. Part D explains each block in detail.")}

<h3>6.1 Output 1: a hemodynamic-aware latent summary</h3>
<p>Inside the network, each 30-second EEG window is compressed into a short list of numbers — 64 in our design — called a <b>latent representation</b> or <b>embedding</b>. "Hemodynamic-aware" means the network was trained so that this summary contains whatever in the EEG is useful for predicting blood signals.</p>
{fig("b6_latent", 5, "A latent space, illustrated.", "Schematic. If the summary is good, trials of the same kind land near each other. Here the 64-number summaries are drawn in two dimensions; rest and the two imagined movements form separate clouds.")}
{A("a fingerprint of the trial", "<p>A fingerprint is a small summary of a person that is still enough to recognise them. The latent summary is a fingerprint of a trial. Its value is that other tasks can reuse it: if it captures blood-relevant information, a simple classifier built on top of it might detect fatigue or motor intention better than one built on raw EEG.</p>")}
<p>That possibility became Claim 4: a representation trained with hemodynamic supervision should be more useful for other EEG tasks than one trained without it.</p>

<h3>6.2 Output 2: the predicted hemodynamic response</h3>
<p>The main output is a predicted blood signal over time: for fNIRS, HbO and HbR for every channel (for HEFMI, 90 channels × 2 = 180 signals, each with 300 time points per window); for fMRI, BOLD for each brain region (48 regions × 15 time points). Each head also predicts its own <b>uncertainty</b> — how much it expects to be wrong — which feeds output 4.</p>
<p>There are two levels at which a prediction can be judged, and the difference matters a great deal later:</p>
{fig("b5_wave_amp", 6, "Waveform versus amplitude.", "Left: the full waveform is the whole curve over time. Right: the amplitude is a single number summarising how big the response was, for example the average inside a response window. Predicting the waveform is much harder than predicting the amplitude.")}
{P("<p>The early versions of the project tried to predict the full <b>waveform</b> and found almost nothing across subjects. Later work showed that the <b>amplitude</b> — whether this trial's response was bigger or smaller than usual — is more recoverable. Maisha's pipeline made compact amplitude measures (HbO mean, HbR mean, HbO peak, HbR trough) its main target for this reason.</p>")}

<h3>6.3 Output 3: a neurovascular delay descriptor</h3>
<p>Lecture A showed that the blood response arrives about 5–6 seconds after neural activity, and that this delay differs between people. A system that could estimate each person's delay from their EEG would provide a potential <b>biomarker</b>: a measurable sign of how healthy their neurovascular coupling is, which might change after a stroke or haemorrhage. The network's delay module produced a value constrained to a physiological window of 2–8 seconds.</p>
{P("<p>The constraint turned out to be a trap. A value forced into 2–8 s will always look physiological, whatever the data say. Lectures D and F show that the learned delay simply sat at the edge of whatever window it was given, while a delay measured directly from the data (6.4 s, stable within a person) was the meaningful result.</p>")}

<h3>6.4 Output 4: calibrated reliability and abstention</h3>
<p>Any prediction system will sometimes be wrong. What makes it safe to use is knowing <i>when</i> it is likely to be wrong. Output 4 is a confidence score for each prediction, and a rule: if confidence is too low, <b>abstain</b> — return "insufficient confidence" instead of a number.</p>
<p>A reliability score has two separate jobs, and Lecture F shows that our system passed one and failed the other:</p>
<ul>
<li><b>Calibration</b>: are the confidence numbers honest? If the system says "80% confident" on 100 cases, about 80 should be right.</li>
<li><b>Ranking</b>: does the score separate good predictions from bad ones? If it does, refusing the least confident cases should make the remaining answers more accurate.</li>
</ul>
{fig("b7_reliability", 7, "The two jobs of a reliability score.", "Schematic. Left: a calibrated system's stated confidence matches how often it is right; an overconfident one claims more than it delivers. Right: a useful gate's error falls as it abstains on more cases; a useless gate's error stays flat however much it refuses.")}
{A("the weather forecaster and the careful doctor", "<p>A good weather forecaster who says '70% chance of rain' is right on about 70% of such days: that is calibration. A good doctor who says 'I'm not sure, let's run a test' instead of guessing is abstaining: that only helps if the doctor hesitates on the genuinely hard cases. A doctor who hesitates at random is no safer than one who never hesitates.</p>")}

<h2>7. From a thesis to claims that can fail</h2>
<p>A thesis is too broad to test directly. Good research turns it into specific <b>claims</b>, each with a test whose outcome is not known in advance and which could come out against you. A claim that cannot fail is not a scientific claim. The final version of the project (v6) turned the thesis into five claims:</p>
{table(4, "The thesis as five testable claims.",
       ["Claim", "In plain words", "How it is tested", "What would count as success"],
       [["1. Cross-modal benefit", "training on fNIRS <i>and</i> fMRI beats training on fNIRS alone", "same network, same budget, with and without fMRI; compare on the same held-out people", "the difference is reliably above zero, for most people"],
        ["2. External validation", "the system works on a dataset it has never seen", "train once, test once on new cohorts, no tuning", "a clear correlation on the new dataset"],
        ["3. Neurovascular delay", "the delay output reflects real physiology", "does the learned delay track a delay measured independently, and stay put when the allowed window changes?", "learned delay follows the data, not the window"],
        ["4. Downstream transfer", "the latent summary helps other EEG tasks", "freeze the encoder, train a simple classifier, compare with control encoders", "beats random, shuffled-target and raw-feature encoders"],
        ["5. Reliability", "the confidence score is calibrated and ranks errors", "calibration error; correlation of uncertainty with error; accuracy as abstention increases", "low calibration error and accuracy rising with abstention"]],
       widths=["17%", "24%", "33%", "26%"])}
{K("<p>Each claim was written down with its test <i>before</i> the results were known, and each was allowed to fail. Lecture F reports the outcome: Claim 1 was not supported, Claim 2 essentially failed, Claim 3 held only for the directly measured delay, Claim 4 failed, and Claim 5 passed calibration but failed ranking. Those negative results are findings, not embarrassments, precisely because the tests were fair.</p>")}
<p>Notice what is <i>not</i> on the list: "the model predicts blood signals well". On its own that statement is meaningless, because a model can look good for bad reasons — for example by recognising which person a recording came from rather than reading their brain activity. Lecture E is devoted to this problem, and it is why every claim above compares against a control.</p>

<h2>8. What a good answer to this question looks like</h2>
<p>A research project does not have to confirm its thesis to be successful. It has to answer its question convincingly. For this project, a convincing answer is one that tells the field:</p>
<ol>
<li>how much of the blood signal can be predicted from EEG in new people, measured honestly;</li>
<li>whether supervising with two kinds of blood signal helps, compared fairly against one;</li>
<li>which parts of the system work (and should be reused) and which do not (and why);</li>
<li>what limits the result — the method, the amount of data, or the quality of the recordings.</li>
</ol>
<p>Keep these four in mind through the rest of the course. Parts C to E explain how each was measured, and Part F gives the answers.</p>
{Q(["In one sentence, what is the difference between training time and deployment time?",
    "Why would a fusion approach defeat the purpose of this project?",
    "Give two reasons the project supervised with both fNIRS and fMRI rather than one.",
    "What is the subject gap, and why can't the model directly check that its fNIRS and fMRI predictions agree?",
    "Why is predicting a waveform harder than predicting an amplitude?",
    "Why does forcing the delay into 2–8 s make it impossible to judge from the output alone whether the delay is physiological?",
    "A reliability score is perfectly calibrated but abstaining does not reduce error. Which job has it failed?",
    "Why is 'the model predicts blood signals well' not a usable claim on its own?"])}

<h2>Summary</h2>
<ul>
<li>The project tries to <b>predict</b> blood signals (HbO, HbR, BOLD) from EEG, so that useful blood information becomes available wherever EEG can be worn.</li>
<li>fNIRS and fMRI are used <b>only during training</b>, as the answer key. At deployment the system sees EEG only.</li>
<li>It is <b>not fusion</b>: fusion needs both signals every time and would remove the reason for the system to exist.</li>
<li>One <b>shared encoder</b> is trained by both fNIRS and fMRI supervision, betting that two views of the blood response teach more than one. This is multi-task learning, which can also cause negative transfer.</li>
<li>The <b>subject gap</b> means the two kinds of supervision come from different people, and the model must work on people it has never seen.</li>
<li>The four outputs are a <b>latent summary</b>, a <b>predicted response</b> (waveform or amplitude), a <b>delay</b>, and a <b>calibrated reliability score with abstention</b>.</li>
<li>The thesis was turned into <b>five claims</b>, each with a fair test that could fail. Part F reports the outcomes.</li>
</ul>

<h3>Coming next: Lecture C</h3>
<p>Lecture C introduces the four datasets in detail — who was recorded, doing what, with which equipment — and the preprocessing that turns raw recordings into training pairs.</p>

<h2>Glossary</h2>
{table(5, "Terms introduced in this lecture.",
       ["Term", "Meaning"],
       [["Thesis", "the central claim a project sets out to support"],
        ["Claim", "a specific, testable statement derived from the thesis, with a test that could fail"],
        ["Supervision / target", "the correct answer the model is trained to reproduce (here, the recorded blood signal)"],
        ["Paired data", "EEG and a blood signal recorded at the same time from the same person"],
        ["Training", "the phase where the model is repeatedly corrected against the target"],
        ["Loss", "a number measuring how wrong a prediction is; training tries to reduce it"],
        ["Deployment / inference", "using the trained model on new data, without targets"],
        ["Privileged information", "information available during training but not at deployment"],
        ["Fusion", "combining two signals at deployment to make a decision"],
        ["Prediction (here)", "producing an estimate of one signal from another"],
        ["Encoder / trunk", "the part of the network that turns input into an internal summary"],
        ["Head", "a small network on top of the encoder that produces one kind of output"],
        ["Latent representation / embedding", "the internal summary of an input as a short list of numbers"],
        ["Multi-task learning", "training one shared model on several related tasks at once"],
        ["Negative transfer", "when training on an extra task makes performance worse"],
        ["Subject gap", "no person appears in both the EEG–fNIRS and EEG–fMRI data"],
        ["Cross-subject", "trained on some people, tested on different people"],
        ["Waveform / amplitude", "the full response curve / a single number summarising its size"],
        ["Biomarker", "a measurable signal that indicates a biological state or disease"],
        ["Calibration", "whether stated confidence matches actual accuracy"],
        ["Abstention", "the system declining to answer when its confidence is too low"]],
       widths=["30%", "70%"])}

<h2>Answers to "Pause and think"</h2>
<ol class="answers">
<li>During training the model sees EEG together with the real blood signal and is corrected against it; at deployment it sees EEG only and must produce the blood estimate itself.</li>
<li>Fusion needs fNIRS or fMRI every time the system runs, which brings back the cost and inconvenience the project was trying to remove.</li>
<li>They are complementary views of the same blood response (surface HbO/HbR with good timing vs whole-brain BOLD), so satisfying both should push the encoder towards genuine blood-related information; and each dataset is small, so two sources give more training examples.</li>
<li>The EEG–fNIRS and EEG–fMRI recordings come from different people, so there is no moment where both blood signals were measured together. Without such moments there is nothing to compare the two predictions against.</li>
<li>A waveform requires getting the timing, shape and size right at every point (hundreds of numbers); an amplitude only requires getting the overall size right (one number). Noise and person-to-person differences in shape hurt the waveform far more.</li>
<li>Because any value the module produces lies inside 2–8 s by construction, landing in the physiological range tells you nothing. You can only tell by checking whether the value tracks an independent measurement and stays the same when the allowed window changes.</li>
<li>Ranking: it states honest numbers but cannot tell good predictions from bad ones.</li>
<li>A model can score well for the wrong reason, for example by recognising the subject rather than reading brain activity. A claim needs a comparison against a control that rules such shortcuts out.</li>
</ol>
"""

D.render(body, "Lecture B: The Research Question")
