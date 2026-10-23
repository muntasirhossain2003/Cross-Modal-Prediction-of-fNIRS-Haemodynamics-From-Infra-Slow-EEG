"""Builds lectures/Lecture_C_The_Data.pdf (teaching notes, Part C).

Numbers come from study_reference_facts.csv, the data-quality audit and
neurovascnet_v6_outputs/data_summary.csv. Figures are schematic unless a caption
says otherwise. Run from the repo root:  python scripts/build_lecture_part_c.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, ROOT, VIOLET, W, YELLOW, A, Doc, E, K,
                            M, P, Q, arrow, box, fig, hrf, table, title_block)

D = Doc("lecture_c", "Lecture_C_The_Data.pdf", "NeuroVascNet course · Lecture C")
DS = pd.read_csv(ROOT / "neurovascnet_v6_outputs" / "data_summary.csv")


# ---------------------------------------------------------------- figures
def f_hierarchy():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")
    levels = [("Dataset", "e.g. TU Berlin hBCI", 0.2), ("Subject", "one person (29 here)", 2.15),
              ("Session", "one visit / block\n(3 per task)", 4.1), ("Run", "one continuous\nrecording file", 6.05),
              ("Trial", "one cue + response\n(one training pair)", 8.0)]
    for i, (n, sub, x) in enumerate(levels):
        box(ax, x, 2.2, 1.75, 1.0, n, fc=[BLUE, AQUA, YELLOW, ORANGE, RED][i], ec=[BLUE, AQUA, YELLOW, ORANGE, RED][i],
            color="white", weight="bold")
        ax.text(x + 0.875, 1.6, sub, ha="center", va="top", fontsize=8.6, color=MUTED)
        if i < 4:
            arrow(ax, (x + 1.77, 2.7), (x + 1.95, 2.7))
    ax.text(5.0, 3.85, "each level contains many of the next", ha="center", fontsize=8.8, color=INK, style="italic")
    D.save(fig, "c1_hierarchy")


def f_trial_window():
    fig, a = plt.subplots(figsize=(W, 2.6))
    t = np.linspace(-5, 55, 1200)
    onsets = [0, 26]
    for o in onsets:
        a.axvspan(o, o + 10, color=YELLOW, alpha=0.16)
    dt = t[1] - t[0]
    task = np.zeros_like(t)
    for o in onsets:
        task[(t >= o) & (t < o + 10)] = 1
    k = hrf(np.arange(0, 30, dt))
    resp = np.convolve(task, k)[:len(t)]
    resp /= resp.max()
    a.plot(t, resp, color=RED, lw=2, label="HbO response")
    a.axvspan(-2, 28, ymin=0.02, ymax=0.1, color=BLUE, alpha=0.5)
    a.axvspan(-2, 0, ymin=0.02, ymax=0.1, color=VIOLET, alpha=0.8)
    a.text(13, -0.2, "one 30-second window: −2 s to +28 s around the cue", ha="center", fontsize=8.6, color=BLUE)
    a.text(-3.6, -0.02, "baseline\n(2 s)", ha="center", fontsize=7.6, color=VIOLET, va="top")
    a.text(5, 1.15, "task 10 s", ha="center", fontsize=8.5, color="#8a6a00")
    a.text(31, 1.15, "next task", ha="center", fontsize=8.5, color="#8a6a00")
    a.text(19, 1.15, "rest ≈ 15–17 s", ha="center", fontsize=8.5, color=MUTED)
    a.annotate("the next trial starts\nbefore this response\nhas fully settled", xy=(26.5, 0.12), xytext=(36, 0.55), fontsize=8.4,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    a.set_xlim(-5, 55); a.set_ylim(-0.45, 1.3); a.set_yticks([])
    a.set_xlabel("time from the first cue (s)")
    D.save(fig, "c2_window")


def f_pipeline():
    fig, ax = plt.subplots(figsize=(W, 3.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.2); ax.axis("off")
    lanes = [
        ("EEG", BLUE, ["read +\npair files", "remove eye\nchannels", "scanner\nartifact fix", "cut window\n−2 to +28 s", "resample\nto 200 Hz", "z-score\nper trial"]),
        ("fNIRS", RED, ["read raw\nlight", "Beer–Lambert\nto HbO, HbR", "band-pass\n0.01–0.2 Hz", "cut window\n−2 to +28 s", "resample\nto 10 Hz", "baseline\n(no GSR)"]),
        ("fMRI", AQUA, ["read BOLD\nvolumes", "atlas\nregions", "remove slow\ndrift", "align by\ntriggers", "cut 15\nvolumes", "keep motor\nregions"]),
    ]
    for li, (name, col, steps) in enumerate(lanes):
        y = 5.6 - li * 2.35
        box(ax, 0.0, y, 0.95, 1.1, name, fc=col, ec=col, color="white", weight="bold")
        for si, s in enumerate(steps):
            x = 1.15 + si * 1.48
            box(ax, x, y, 1.35, 1.1, s, fs=7.6)
            arrow(ax, (x - 0.2 if si else 0.95, y + 0.55), (x, y + 0.55))
    D.save(fig, "c3_pipeline")


def f_montage():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"wspace": 0.15})
    for a in ax:
        a.set_xlim(-1.2, 1.2); a.set_ylim(-1.15, 1.25); a.set_aspect("equal"); a.axis("off")
    grid = {"FC3": (-0.5, 0.5), "FC1": (0, 0.5), "C3": (-0.5, 0), "C1": (0, 0), "CP3": (-0.5, -0.5), "CP1": (0, -0.5),
            "FC5": (-1.0, 0.5), "C5": (-1.0, 0), "CP5": (-1.0, -0.5)}
    a = ax[0]
    a.set_title("HEFMI: standard 10-20 positions", fontsize=10)
    for n, (x, y) in grid.items():
        if n in ("C3", "FC1", "FC5", "CP1", "CP5"):
            a.add_patch(plt.Circle((x + 0.6, y), 0.16, fc=BLUE if n == "C3" else "white", ec=BLUE, lw=1.3))
            a.text(x + 0.6, y, n, ha="center", va="center", fontsize=7.5, color="white" if n == "C3" else INK, weight="bold")
    a.text(0.1, -1.05, "C3 sits right over the left hand area", ha="center", fontsize=8.5, color=MUTED)
    a = ax[1]
    a.set_title("TU Berlin hBCI: half-way (10-5) positions", fontsize=10)
    half = {"FCC3h": (-0.25 + 0.35, 0.25), "CCP3h": (-0.25 + 0.35, -0.25), "FCC5h": (-0.75 + 0.35, 0.25), "CCP5h": (-0.75 + 0.35, -0.25)}
    for n, (x, y) in half.items():
        a.add_patch(plt.Circle((x, y), 0.19, fc="white", ec=ORANGE, lw=1.3))
        a.text(x, y, n, ha="center", va="center", fontsize=6.6, weight="bold")
    a.add_patch(plt.Circle((-0.15, 0), 0.16, fc="none", ec=GRAY, lw=1.3, ls="--"))
    a.text(-0.15, 0, "C3?", ha="center", va="center", fontsize=7.5, color=GRAY)
    for (x, y) in half.values():
        arrow(a, (x, y), (-0.15 + (x + 0.15) * 0.35, y * 0.35), color=ORANGE, lw=0.9)
    a.text(0.0, -1.05, "no C3 at all: estimate it from its neighbours\n(spherical-spline interpolation)", ha="center", fontsize=8.5, color=MUTED)
    D.save(fig, "c4_montage")


def f_baseline():
    t = np.linspace(-2, 28, 300)
    rng = np.random.default_rng(3)
    drift = 0.6 + 0.01 * t
    raw = drift + hrf(t - 0.3) * 0.5 + 0.03 * rng.standard_normal(len(t))
    base = raw[t < 0].mean()
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True, gridspec_kw={"wspace": 0.12})
    a = ax[0]
    a.plot(t, raw, color=RED, lw=1.6); a.axvspan(-2, 0, color=VIOLET, alpha=0.12)
    a.axhline(base, color=VIOLET, ls="--", lw=1)
    a.text(0.5, base - 0.1, "baseline level", fontsize=8.4, color=VIOLET)
    a.set_title("Before: sits on an arbitrary offset", fontsize=10); a.set_xlabel("time (s)"); a.set_yticks([0])
    a.axhline(0, color=MUTED, lw=0.8)
    a = ax[1]
    a.plot(t, raw - base, color=RED, lw=1.6); a.axhline(0, color=MUTED, lw=0.8)
    a.set_title("After: change from just before the cue", fontsize=10); a.set_xlabel("time (s)")
    D.save(fig, "c5_baseline")


def f_budget():
    d = DS.copy()
    names = {"TU Berlin hBCI (MI+MA)": "TU Berlin hBCI", "ds002336": "ds002336", "HEFMI": "HEFMI", "ds002739": "ds002739"}
    d["n"] = d.dataset.map(names)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.55})
    cols = [RED if m == "fnirs" else AQUA for m in d.modality]
    for a, k, title in [(ax[0], "subjects", "People"), (ax[1], "trials", "Trials (30 s windows)")]:
        a.barh(d.n, d[k], color=cols, height=0.55)
        for y, v in enumerate(d[k]):
            a.text(v * 1.02, y, f"{int(v):,}", va="center", fontsize=8.6)
        a.set_title(title, fontsize=10); a.grid(axis="y", visible=False)
        a.set_xlim(0, d[k].max() * 1.25)
    D.save(fig, "c6_budget")


for f in [f_hierarchy, f_trial_window, f_pipeline, f_montage, f_baseline, f_budget]:
    f()

r = DS.set_index("dataset")
rel = lambda k: f"{r.loc[k, 'reliability']:.2f}"

body = title_block("NeuroVascNet course · Lecture C", "The Data",
                   "Four datasets, what each one contains, and how raw recordings become training pairs",
                   "Part C of 7 · Builds on Lectures A and B · About 90 minutes of reading") + f"""
<div class="box">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>describe each of the four datasets: who was recorded, doing what, with which equipment;</li>
<li>explain the structure of a dataset (subjects, sessions, runs, trials) and why the subject is the unit that matters;</li>
<li>walk through every preprocessing step that turns a raw recording into an EEG window paired with a blood-signal window, and say why each step exists;</li>
<li>explain the three alignment problems the project had to solve: time, channels and units;</li>
<li>say which dataset trained which part of the model, why that changed, and what the data-quality audit found.</li>
</ul>
</div>

<h2>1. Why a whole lecture on data</h2>
<p>Lecture A ended with a warning: the project's biggest setback was a dataset whose blood signal was mostly heartbeat and blood pressure, not brain activity. No amount of clever modelling could have fixed that. In machine learning there is an old saying, "garbage in, garbage out", and brain recordings are particularly good at producing garbage quietly: a file loads, the numbers look reasonable, the model trains — and the result is meaningless.</p>
<p>This lecture therefore treats data as a subject in its own right. It covers what each dataset is, how recordings are organised, every step of preprocessing, and the checks that tell you whether the result is trustworthy.</p>

<h2>2. How a dataset is organised</h2>
{fig("c1_hierarchy", 1, "The levels inside a brain-recording dataset.", "A dataset contains subjects; each subject attends one or more sessions; each session contains runs (continuous recording files); each run contains many trials. In our project one trial becomes one training pair: a 30-second EEG window and the matching 30-second blood-signal window.")}
<p>Inside each run, the recording is a long stream of numbers per channel, plus a list of <b>events</b> (also called markers or triggers): time stamps saying "cue appeared", "task started", "rest started", each with a label such as "left hand". Events are what let us cut the stream into trials.</p>
{K("<p>Trials from the same person are <b>not independent</b>. They share that person's anatomy, electrode placement, skull thickness and habits. A dataset with 29 people and 3,480 trials does not contain 3,480 independent observations; for any claim about new people, the real sample size is 29. This single fact drives the whole of Lecture E.</p>")}
{A("photographs of classmates", "<p>Take 100 photos each of 30 classmates. You have 3,000 photos, but only 30 faces. A face-recognition system tested on new photos of the same 30 classmates can look excellent while being useless on a 31st person. Trials are the photos; subjects are the faces.</p>")}

<h2>3. The four datasets</h2>
<p>All four are public and were recorded by other research groups for their own purposes. That is normal and valuable, but it means every design choice — task, timing, channels, equipment — was made for someone else's question.</p>

<h3>3.1 TU Berlin hBCI (EEG + fNIRS)</h3>
<p>Recorded at the Technical University of Berlin for a hybrid brain–computer interface study (Shin and colleagues, 2017). <b>29 healthy adults</b> performed two paradigms: <b>motor imagery</b> (imagining left- or right-hand movement) and <b>mental arithmetic</b> (repeated subtraction, contrasted with rest). Each paradigm had three sessions of 20 trials, so each person contributes 60 + 60 = 120 trials. A trial is a cue followed by about <b>10 seconds of task</b> and roughly <b>15–17 seconds of rest</b>.</p>
<ul>
<li><b>EEG</b>: 30 scalp channels (plus auxiliary eye channels) at 200 Hz, placed at the finer <b>10-5 positions</b> such as FCC3h and CCP3h rather than the classic C3 and C4.</li>
<li><b>fNIRS</b>: 36 channels at 10 Hz, distributed as <b>raw light intensities at 760 and 850 nm</b>, so we did the Beer–Lambert conversion ourselves.</li>
<li><b>Quality</b>: split-half target reliability {rel('TU Berlin hBCI (MI+MA)')}, HbO/HbR correlation −0.44 (the neural signature), 78% of channels with good optode coupling.</li>
</ul>

<h3>3.2 HEFMI-ICH (EEG + fNIRS, clinical)</h3>
<p>A dataset of <b>37 people: 17 healthy controls and 20 patients with intracerebral haemorrhage (ICH)</b>, a type of stroke caused by bleeding inside the brain. All performed left/right-hand motor imagery over three sessions with 15 trials per hand per session (90 trials each).</p>
<ul>
<li><b>EEG</b>: 32 channels at 256 Hz, standard 10-20 positions including C3 and C4. Channel names are stored in the dataset's MATLAB creation script rather than in the data files, so they had to be recovered and verified.</li>
<li><b>fNIRS</b>: 90 channels at 11 Hz, distributed <b>already converted</b> to HbO and HbR and <b>already cut into trials</b>, without the original event times.</li>
<li><b>Quality</b>: split-half reliability {rel('HEFMI')}, HbO/HbR correlation <b>+0.92</b> — the systemic signature from Lecture A.</li>
</ul>
{P("<p>HEFMI was the original fNIRS training set. Its unreliable, systemic-dominated target explains much of the early 'prediction ≈ 0' result. It is still valuable: as an external test set, and for the clinical comparison between ICH patients and controls, which does not depend on prediction accuracy.</p>")}

<h3>3.3 OpenNeuro ds002336 (EEG + fMRI)</h3>
<p>From the OpenNeuro repository: <b>10 healthy adults</b> recorded with EEG <i>inside</i> an MRI scanner during a motor-imagery neurofeedback study. Each person did several tasks: a motor localiser (<code>motorloc</code>), motor imagery before and after training (<code>MIpre</code>, <code>MIpost</code>) and three neurofeedback conditions (EEG-based, fMRI-based and combined).</p>
<ul>
<li><b>EEG</b>: 64 channels at 5,000 Hz, provided both raw and in a corrected version; the raw data are dominated by the scanner's gradient artifact.</li>
<li><b>fMRI</b>: one brain volume every 2 seconds (TR = 2 s).</li>
<li><b>Quality</b>: split-half reliability {rel('ds002336')}; head motion low (mean framewise displacement 0.12 mm); but the motor localiser runs keep a large gradient residual after correction, and one recording (subject xp102, motor localiser) has completely flat EEG.</li>
</ul>

<h3>3.4 OpenNeuro ds002739 (EEG + fMRI)</h3>
<p><b>24 healthy adults (aged 20–32)</b> recorded with simultaneous EEG and fMRI while deciding which way a cloud of moving dots was drifting and then rating their confidence. This is a <b>perceptual decision task, not motor imagery</b>.</p>
<ul>
<li><b>EEG</b>: 64 channels, distributed already corrected for scanner artifacts, filtered to 0.5–40 Hz and stored at 1,000 Hz. The files carry no electrode names.</li>
<li><b>fMRI</b>: TR 2 s, excellent image quality (temporal signal-to-noise around 190, the best in the project).</li>
<li><b>Quality</b>: split-half reliability of the task-locked target only {rel('ds002739')}, because a brief perceptual decision produces a small, variable blood response.</li>
</ul>

<h3>3.5 Side by side</h3>
{table(1, "The four datasets at a glance.",
       ["", "TU Berlin hBCI", "HEFMI-ICH", "ds002336", "ds002739"],
       [["Blood signal", "fNIRS", "fNIRS", "fMRI", "fMRI"],
        ["People", "29 healthy", "17 healthy + 20 ICH", "10 healthy", "24 healthy"],
        ["Task", "motor imagery + mental arithmetic", "motor imagery", "motor imagery + neurofeedback", "perceptual decision"],
        ["EEG channels / rate", "30 / 200 Hz", "32 / 256 Hz", "64 / 5000 Hz", "64 / 1000 Hz (pre-corrected)"],
        ["Blood channels / rate", "36 / 10 Hz (raw light)", "90 / 11 Hz (converted)", "volumes / 0.5 Hz", "volumes / 0.5 Hz"],
        ["Target reliability", rel("TU Berlin hBCI (MI+MA)"), rel("HEFMI"), rel("ds002336"), rel("ds002739")],
        ["Role in final v6", "fNIRS training", "external test + clinical", "fMRI training", "external test"],
        ["Role in Maisha v2.3", "external test", "fNIRS training", "fMRI training", "external test"]],
       widths=["20%", "20%", "20%", "20%", "20%"])}
{fig("c6_budget", 2, "The data budget (final v6 run).", "Numbers of people and of 30-second trial windows used from each dataset (red = EEG + fNIRS, green = EEG + fMRI). Notice how few people the fMRI training set has: every fMRI result rests on 10 individuals.")}
{M("<p>Treating ds002739 as an external replication of a motor result. Its task is perceptual decision-making, so a model trained on motor imagery is being tested on a different kind of brain activity. It is a <b>cross-task</b> stress test, which is harder and answers a different question.</p>")}
{P("<p>A data-provenance lesson: the project folder also contains a different TU Berlin dataset — 26 people doing <i>mental-workload</i> tasks (n-back, word generation). Early documents described it as the 29-person motor dataset. The audit caught the mismatch by counting subjects and reading task names. Always check that the data on disk is the data the paper describes.</p>")}

<h2>4. Which dataset trains which part, and why it changed</h2>
<p>The original design (v3, and Maisha's pipeline) trained the fNIRS head on <b>HEFMI</b> and tested externally on TU Berlin. The final design (v6) <b>swapped them</b>: TU Berlin hBCI trains, HEFMI tests. The decision was made from target quality measured <i>before</i> any model was trained: TU Berlin's target is six times more reliable and has the neural HbO/HbR sign. The fMRI roles were kept: the reverse assignment (training on ds002739) was tried and made the fMRI arm worse.</p>
{K("<p>A role swap made after seeing results can look like cherry-picking. It is defensible here because the criterion — target reliability and the HbO/HbR sign — was measured before modelling and is stated openly, and because Maisha's pipeline keeps the original assignment, giving a direct comparison.</p>")}

<h2>5. From raw recording to training pairs</h2>
<p>Preprocessing is the sequence of steps that turns raw files into clean, aligned, comparable training examples. Figure 3 shows the whole pipeline; the subsections explain each step.</p>
{fig("c3_pipeline", 3, "The preprocessing pipeline for each signal.", "Each row is one signal type. Every step exists to remove a specific problem: wrong pairing, artifacts, mismatched timing, mismatched sampling rates or arbitrary offsets. The details below follow this figure from left to right.")}

<h3>5.1 Pairing the right files</h3>
<p>Every EEG recording must be matched to the blood recording of the same person and the same run. This sounds trivial and is not. In ds002336, one EEG recording has no fMRI partner and three fMRI recordings have no EEG partner; they are dropped, leaving 52 exact pairs. HEFMI stores its two signals in separate folders, and a person in the patient group may share a folder name such as "subject1" with a person in the control group, so the pairing must use both group and number. The pipelines write <b>pairing manifests</b> — tables listing exactly which files were paired — so the pairing can be checked.</p>

<h3>5.2 Cutting windows around events</h3>
<p>Each trial becomes a <b>30-second window from 2 s before the cue to 28 s after it</b>. Thirty seconds is chosen because the blood response takes about that long to rise, peak and settle (Lecture A). The 2 seconds before the cue are the <b>baseline</b>.</p>
{fig("c2_window", 4, "One trial window.", "Schematic for TU Berlin timing. The window (blue bar) runs from −2 s to +28 s around the cue; the first 2 s (violet) are the baseline. Because trials follow each other every 25 seconds or so, the end of one window overlaps the start of the next trial's response.")}
<p>That overlap matters. Trials are spaced closer together than the full response lasts, so the end of each window contains the rise of the next trial's response. The model must predict a blood signal that partly belongs to a neighbouring trial.</p>

<h3>5.3 EEG steps</h3>
<ul>
<li><b>Remove non-brain channels</b> such as eye (EOG) electrodes, which some datasets store alongside EEG.</li>
<li><b>Correct scanner artifacts</b> for EEG recorded inside the MRI. For ds002336 the project implemented its own average artifact subtraction (Lecture A), bringing the signal from about 8,200 µV of artifact to about 16 µV.</li>
<li><b>Filter</b>: remove slow drifts and high-frequency noise, and notch out mains interference at 50 Hz.</li>
<li><b>Resample</b> every dataset to <b>200 Hz</b>, so each window is exactly 6,000 samples per channel.</li>
<li><b>Normalise</b>: <b>z-score</b> each channel within each trial (subtract its mean, divide by its standard deviation). This removes differences in overall amplitude between people and devices, which would otherwise dominate.</li>
</ul>
{E("what z-scoring does", "<p>Suppose one person's C3 channel swings between −40 and +40 µV and another's between −10 and +10 µV, purely because of skin and electrode differences. After z-scoring, both are expressed in standard deviations around zero, so the model compares the <i>pattern</i> of activity rather than the raw size.</p>")}

<h3>5.4 fNIRS steps</h3>
<ul>
<li><b>Convert light to haemoglobin</b> with the modified Beer–Lambert law (for TU Berlin), using a 3 cm source–detector distance and differential pathlength factors of 6.4 at 760 nm and 5.8 at 850 nm.</li>
<li><b>Band-pass filter 0.01–0.2 Hz</b>, keeping the slow blood response and removing heartbeat and most breathing (Lecture A, Section 6).</li>
<li><b>Resample to 10 Hz</b>, giving 300 samples per window.</li>
<li><b>Baseline-correct</b>: subtract the average of the 2 seconds before the cue, so every trial shows the <i>change</i> caused by the task rather than an arbitrary starting level (Figure 5).</li>
<li><b>No global signal regression.</b> The audit showed GSR removed about 83% of the real response.</li>
</ul>
{fig("c5_baseline", 5, "Baseline correction.", "Schematic. Left: a raw HbO window sits on an offset that depends on the person, the channel and slow drift. Right: after subtracting the 2-second pre-cue average, the window shows only the change from just before the cue.")}

<h3>5.5 fMRI steps</h3>
<ul>
<li><b>Reduce voxels to regions</b>: each brain volume is divided into the <b>48 cortical regions of the Harvard–Oxford atlas</b>, and the voxels within each region are averaged. The final v6 run then kept only the sensorimotor regions relevant to motor imagery.</li>
<li><b>Remove slow scanner drift</b> by fitting and subtracting a cubic trend; drift alone can make a signal look falsely reliable.</li>
<li><b>Align with EEG</b> using the scanner's own triggers (in ds002336 an "R128" marker is written into the EEG each time a volume is acquired), so volume <i>k</i> is matched to the exact EEG sample at which it was recorded.</li>
<li><b>Cut 15 volumes</b> per window (30 s at one volume per 2 s), and cap trials at 100 per run, spread evenly across the run so that early or late parts of a session are not over-represented.</li>
</ul>

<h2>6. Three alignment problems</h2>
<h3>6.1 Time</h3>
<p>EEG and the blood signal must be cut at exactly corresponding moments. Each recording system has its own clock, and the two are linked only through shared triggers. A trigger offset of even one fMRI volume (2 s) would shift the blood signal relative to the EEG and quietly weaken every prediction. Maisha's pipeline checked this by finding, for each person, the delay at which EEG power and the blood signal match best: on HEFMI validation and TU Berlin, most people's best match fell outside the expected 2–8 s, so timing could not be confirmed and alignment-based claims were blocked.</p>

<h3>6.2 Channels</h3>
<p>Different datasets place electrodes at different positions. Matching by name is the obvious solution, and the early pipeline aligned everything to a common set of 32 electrode names. It breaks down for TU Berlin hBCI, which uses the finer 10-5 positions: it has no C3 or C4 at all, and only 12 names are shared with HEFMI, none of them over the motor cortex. Matching by name would hand the model exactly the wrong channels; matching by position in the file would pair electrodes far apart on the head.</p>
{fig("c4_montage", 6, "Why electrodes cannot simply be matched by name.", "Schematic of the left motor area. HEFMI has C3 directly over the hand region. TU Berlin hBCI instead has electrodes half-way between standard positions (FCC3h, CCP3h, …). A C3-like signal can be estimated from its neighbours with spherical-spline interpolation, which uses the known 3-D electrode positions.")}
<p>The final pipeline therefore gives each dataset its own small <b>input adapter</b> inside the model (Lecture D), and, where one dataset must be read in another's layout (external testing), estimates the missing channels with <b>spherical-spline interpolation</b> from the known 3-D positions. ds002739 has no electrode names at all, so its channels could only be paired by index — a weak alignment that is reported as such.</p>
<p>fNIRS has the same problem with optodes, and no published mapping between the HEFMI and TU Berlin layouts exists. For the external fNIRS test, every channel is therefore averaged into a single mean HbO and mean HbR, which is coarse but does not depend on layout.</p>

<h3>6.3 Units</h3>
<p>Concentrations can be stored in mol/L, mmol/L or µmol/L. The audit found HEFMI and TU Berlin values differing by about <b>2,600 times</b>. Within the training loss this is harmless, because the loss compares shapes (Lecture D); but any comparison of amplitudes between datasets needs explicit unit harmonisation, which Maisha's pipeline performs and checks (converting millimolar to molar and confirming the scales are within a plausible ratio).</p>

<h2>7. Two ways to define the target</h2>
<p>Once windows exist, there is still a choice of what exactly the model must predict:</p>
{table(2, "Target definitions used in the project.",
       ["Target", "What it is", "Where used"],
       [["Full waveform", "every channel's HbO and HbR (or every region's BOLD) at every time point in the window", "v3, v6 training"],
        ["Compact response", "four numbers per trial: mean HbO, mean HbR, HbO peak and HbR trough within a response window (5.2–15.2 s, chosen on training people only)", "Maisha v2.3 primary endpoint"],
        ["Pooled channels", "the average HbO and average HbR across all channels", "external fNIRS test (layouts differ)"]],
       widths=["20%", "55%", "25%"])}
<p>The compact response trades detail for reliability: averaging over channels and a time window cancels much of the noise. Choosing the response window on training people only matters; choosing it on test people would leak information about the answer.</p>

<h2>8. Checking quality before trusting the data</h2>
<p>The data-quality audit examined 610 recordings (63 GB, 97 people) before any further modelling. Its key measures, which you should be able to explain:</p>
{table(3, "Quality measures used in the audit.",
       ["Measure", "What it checks", "Good sign"],
       [["Split-half reliability", "do two halves of a person's trials give the same average response?", "high (TU Berlin {0}, HEFMI {1})".format(rel("TU Berlin hBCI (MI+MA)"), rel("HEFMI"))],
        ["HbO/HbR correlation", "is the fNIRS signal neural (opposite directions) or systemic (same direction)?", "negative"],
        ["Scalp coupling index", "is the heartbeat visible in the raw light, i.e. is the optode touching skin?", "high (TU Berlin 0.90)"],
        ["Evoked signal-to-noise", "is the average response large compared with its uncertainty?", "well above 1 standard error"],
        ["tSNR (fMRI)", "how stable is the signal over time?", "high (ds002739 ≈ 190)"],
        ["Head motion (fMRI)", "how much did the head move between volumes?", "low (ds002336 0.12 mm)"],
        ["5 mV rule (EEG)", "is the in-scanner EEG still saturated by artifacts?", "corrected data well below 5 mV"]],
       widths=["24%", "48%", "28%"])}
{M("<p>Trusting an automatic quality rule blindly. Two of the audit's own pre-set rules misfired: a line-noise threshold flagged 36 of 37 HEFMI people for interference the notch filter already removes, and a gradient-residual measure was meaningless on ds002739 because that data was already filtered. They were reported as misfires rather than quietly retuned, and the exclusion list was not used as it stood. Quality rules need sanity checks too.</p>")}

<h2>9. The split</h2>
<p>The last step before modelling is deciding which people train the model and which test it. The rule is simple and absolute: <b>a person is either in training or in testing, never both</b>. Lecture E explains why at length. One subtle example is worth knowing now: the original HEFMI split sorted people alphabetically, and because every "Normals" name sorts before every "Patients" name, the validation set contained only ICH patients. Maisha's pipeline fixed this with a split stratified by clinical group (3 of 17 controls and 4 of 20 patients held out), fixed in advance by a random seed.</p>

{Q(["Why is the number of subjects, not trials, the real sample size for claims about new people?",
    "Why are windows 30 seconds long, and why do they start 2 seconds before the cue?",
    "TU Berlin's windows overlap the next trial's response. Why does that make prediction harder?",
    "Why is each EEG channel z-scored within each trial?",
    "Why can't TU Berlin's EEG be matched to HEFMI's by electrode name, and what was done instead?",
    "What was wrong with sorting HEFMI subjects alphabetically to choose a validation set?",
    "Why should the compact-response time window be chosen on training subjects only?",
    "Why is ds002739 a cross-task test rather than a replication?"])}

<h2>Summary</h2>
<ul>
<li>Datasets are organised as subjects → sessions → runs → trials. Trials from one person are not independent; the <b>subject</b> is the unit for claims about new people.</li>
<li><b>TU Berlin hBCI</b> (29, motor imagery + arithmetic, raw fNIRS) has a reliable, neural target and trains the final model. <b>HEFMI</b> (37 incl. 20 ICH patients) has a systemic-dominated target and is used for external testing and the clinical comparison. <b>ds002336</b> (10, motor) trains the fMRI head. <b>ds002739</b> (24, perceptual) is a cross-task test.</li>
<li>Preprocessing: pair files, cut −2 to +28 s windows, clean and resample EEG to 200 Hz and z-score it, convert fNIRS with Beer–Lambert, filter 0.01–0.2 Hz, resample to 10 Hz and baseline-correct (no GSR), reduce fMRI to atlas regions, detrend and align by scanner triggers.</li>
<li>Three alignment problems: <b>time</b> (trigger clocks, not fully verifiable), <b>channels</b> (different montages; adapters and spherical-spline interpolation) and <b>units</b> (a 2,600× difference).</li>
<li>Targets can be full waveforms, compact responses or pooled channels. Quality is checked with reliability, HbO/HbR sign, coupling, SNR, tSNR and motion — and the checks themselves need checking.</li>
<li>A person is in training or testing, never both, and the split must not be confounded with clinical group.</li>
</ul>
<h3>Coming next: Lecture D</h3>
<p>Lecture D opens the model: the adapters, the shared trunk, the heads, the loss function, the delay module, the reliability layer and how joint training on two datasets actually works.</p>

<h2>Glossary</h2>
{table(4, "Terms introduced in this lecture.",
       ["Term", "Meaning"],
       [["Session / run", "one visit / one continuous recording file within it"],
        ["Event / marker / trigger", "a time stamp in the recording marking a cue or other occurrence"],
        ["Window / epoch", "a fixed stretch of signal cut around an event (here −2 to +28 s)"],
        ["Baseline correction", "subtracting the pre-cue average so the window shows change"],
        ["Z-scoring", "subtracting the mean and dividing by the standard deviation"],
        ["Resampling", "changing the number of samples per second"],
        ["Band-pass filter", "keeping only frequencies inside a range"],
        ["Pairing manifest", "a table recording which EEG file was matched to which blood file"],
        ["10-20 / 10-5 positions", "standard / finer electrode placement systems"],
        ["Spherical-spline interpolation", "estimating a missing electrode from its neighbours using head geometry"],
        ["Atlas / ROI", "a standard map dividing the brain into regions / one of those regions"],
        ["Detrending", "removing a slow drift by fitting and subtracting a smooth curve"],
        ["R128 trigger", "a marker written into the EEG each time the MRI acquires a volume"],
        ["Split-half reliability", "agreement between average responses from two halves of the trials"],
        ["Stratified split", "a split that keeps group proportions (e.g. patients and controls) balanced"],
        ["Data provenance", "knowing exactly where data came from and what it contains"]],
       widths=["30%", "70%"])}

<h2>Answers to "Pause and think"</h2>
<ol class="answers">
<li>Trials from one person share that person's anatomy, electrode placement and habits, so they are not independent. Generalising to new people means generalising across persons, and there are only as many independent examples of that as there are people.</li>
<li>The blood response takes about 30 seconds to rise, peak and settle. The 2 seconds before the cue provide a baseline so each window can be expressed as change from the pre-cue level.</li>
<li>Part of each window's blood signal is caused by the next trial, which the current window's EEG has not yet seen, so some of the target is unpredictable from the input.</li>
<li>To remove overall amplitude differences between people, electrodes and devices, so the model learns patterns of activity rather than recording-specific scale.</li>
<li>TU Berlin uses 10-5 half-way positions with no C3 or C4, sharing only 12 names with HEFMI, none over motor cortex. Each dataset got its own input adapter in the model, and missing channels were estimated by spherical-spline interpolation when one layout had to be read as another.</li>
<li>All controls sorted before all patients, so the validation set contained only ICH patients. Any validation result was confounded with clinical group.</li>
<li>Choosing it on test subjects would use information about the answers to shape the measurement, inflating the result. That is a form of leakage.</li>
<li>Its task is perceptual decision-making, not motor imagery, so it tests whether the model works on a different kind of brain activity rather than repeating the same experiment.</li>
</ol>
"""

D.render(body, "Lecture C: The Data")
