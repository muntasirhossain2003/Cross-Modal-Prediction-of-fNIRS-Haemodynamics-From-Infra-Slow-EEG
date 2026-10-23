"""Builds lectures/Lecture_G_Using_the_Results.pdf (teaching notes, Part G).

Limitations, framing, defending the work and next steps. Numbers are the same
as in Lecture F and the research report. Run:  python scripts/build_lecture_part_g.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, E, K, M,
                            P, Q, V, arrow, box, fig, qa_bank, table, title_block)

D = Doc("lecture_g", "Lecture_G_Using_the_Results.pdf", "NeuroVascNet course · Lecture G")


def f_validity():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    items = [("Internal validity", "Is the effect real in this\nstudy, not an artefact?", "leakage, timing,\ncontrols", BLUE),
             ("Statistical conclusion", "Are the statistics sound\nand powered?", "subject count,\nseeds, tests", AQUA),
             ("Construct validity", "Do the measures capture\nwhat we claim?", "is the target neural?\nis the delay real?", YELLOW),
             ("External validity", "Does it generalise to other\npeople, devices and tasks?", "cross-dataset\nfailure", ORANGE)]
    for i, (n, q, ex, c) in enumerate(items):
        x = 0.1 + i * 2.48
        box(ax, x, 2.75, 2.3, 0.85, n, fc=c, ec=c, color="white", weight="bold", fs=9)
        ax.text(x + 1.15, 2.6, q, ha="center", va="top", fontsize=8.3)
        ax.text(x + 1.15, 1.45, "here: " + ex, ha="center", va="top", fontsize=7.8, color=MUTED, style="italic")
    D.save(fig, "g1_validity")


def f_priorities():
    steps = [("Finish Maisha's\n4 seeds", 1.0, 3.0), ("Personalisation on\nTU Berlin + no-EEG control", 1.8, 4.45),
             ("Same-split reliability\ncomparison", 1.4, 3.55), ("Optical control for\nICH marker", 3.4, 3.95),
             ("Verify trigger\ntiming", 2.0, 3.95), ("Targeted motor-ROI\nfMRI prediction", 2.8, 3.15),
             ("New data with\nshort channels", 4.6, 4.6), ("Same-person EEG +\nfNIRS + fMRI data", 5.0, 5.1)]
    fig, a = plt.subplots(figsize=(W, 3.0))
    for n, eff, val in steps:
        c = BLUE if eff < 2.5 else (ORANGE if eff < 4.5 else GRAY)
        a.scatter(eff, val, s=70, color=c, edgecolor="white", lw=1.2, zorder=3)
        a.text(eff + 0.08, val, n, fontsize=7.8, va="center")
    a.axvspan(0.5, 2.5, color=BLUE, alpha=0.05)
    a.text(0.55, 5.3, "do first: high value, low effort", fontsize=8.5, color=BLUE)
    a.set_xlim(0.5, 6.4); a.set_ylim(2.7, 5.5)
    a.set_xlabel("effort (time, data, cost)  →"); a.set_ylabel("expected value  →")
    a.set_xticks([]); a.set_yticks([])
    D.save(fig, "g2_priorities")


def f_map():
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    parts = [("A", "signals"), ("B", "question"), ("C", "data"), ("D", "model"), ("E", "evaluation"), ("F", "results"), ("G", "use")]
    for i, (l, n) in enumerate(parts):
        x = 0.1 + i * 1.42
        box(ax, x, 1.5, 1.2, 0.9, f"{l}\n{n}", fc=[BLUE, AQUA, YELLOW, ORANGE, RED, VIOLET, GRAY][i],
            ec=[BLUE, AQUA, YELLOW, ORANGE, RED, VIOLET, GRAY][i], color="white", weight="bold", fs=9)
        if i < 6:
            arrow(ax, (x + 1.21, 1.95), (x + 1.42, 1.95))
    ax.text(5.0, 0.95, "every result in F traces back to a choice in C (data), D (model) or E (evaluation)", ha="center", fontsize=8.6, color=MUTED)
    D.save(fig, "g3_map")


for f in [f_validity, f_priorities, f_map]:
    f()

QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "The core claims"
qa(S, "In one sentence, what did you find?", "EEG predicts fNIRS responses in people the model has never seen (r = 0.16) when the training target is reliable, but supervising with fMRI from different people does not help, nothing transfers to other datasets, and the neurovascular delay is best measured directly, where it is a stable personal trait.")
qa(S, "What is new about this work?", "Not 'first to predict haemodynamics from EEG'. The contributions are: a controlled test of cross-modal (fNIRS + fMRI) supervision with a shuffled-target control and two independent pipelines; showing that target reliability, measured before modelling, decides whether prediction is possible; quantifying how evaluation protocol inflates scores (88%); a reliable model-free delay measure; and a candidate clinical marker.")
qa(S, "Your main positive result is r = 0.16. Why should anyone care?", "Because it survives every shortcut test — held-out people, subject and task demeaning, a no-EEG baseline at zero and a 1,000-shuffle null — in all five folds. It shows real, transferable information exists, at about a fifth of what the target allows. It is a scientific result, not yet a usable tool.")
qa(S, "Why did you change the training dataset? Isn't that cherry-picking?", "The switch was decided from target quality measured before any model was trained — split-half reliability and the HbO–HbR sign — and is documented. Maisha's pipeline keeps the original assignment and gives the comparison: on the unreliable target, cross-subject prediction equals a subject-mean baseline.")

S = "Methodological challenges"
qa(S, "How do you know the model isn't just recognising subjects?", "Subjects are held out; held-out people get the average subject embedding; scores are demeaned by subject (and by subject × task), which drives a no-EEG predictor to exactly zero; and a permutation null is used. The naive trial-level score is 88% higher, which shows how large the shortcut would have been.")
qa(S, "Isn't five held-out subjects far too few for Claim 1?", "Yes, for that test alone. That is why the 29-subject paired control was added, along with weighting and bottleneck sweeps and a shuffled-fMRI control. All point the same way, and Maisha's independent pipeline agrees.")
qa(S, "Could the fMRI arm have failed only because ds002336 is small?", "Very possibly; that is a limitation we state. The claim is scoped: at this data scale, with fMRI supervision from different people, there is no benefit. A larger or same-subject dataset could change the answer.")
qa(S, "Why didn't you use a bigger or more modern model?", "The data are small (thousands of trials, tens of people), so larger models overfit; the limiting factors identified were target reliability and person-specificity, not model capacity. The bottleneck sweep and controls showed capacity was not the issue for cross-modal benefit.")
qa(S, "Was the event timing verified?", "Not fully. Maisha's timing check could not confirm alignment for some datasets, so alignment-based claims were blocked. Verifying triggers against raw marker files is a listed next step; a fixed offset would lower, not raise, our prediction scores.")
qa(S, "Why is the external validation so weak?", "Because montage, task and target quality all change at once, and the fNIRS comparison had to be pooled across channels. We report it as a stress test that failed, and state that the available data cannot separate the causes.")

S = "Interpretation"
qa(S, "Does this mean EEG cannot predict haemodynamics?", "No. Within a person the coupling is clearly present, and across people some information transfers when the target is reliable. The limits we found are about target quality, person-specificity and data scale, not an absolute information limit.")
qa(S, "Why is the learned delay useless but the measured delay valuable?", "The learned delay is forced into a window by a sigmoid and trained on a clamped target, so it reports its bounds (it moved from 2.0 s to 0.07 s when the window changed). The measured delay comes directly from each person's data and is consistent across sessions (ICC 0.53).")
qa(S, "Is the ICH finding a biomarker?", "Not yet. It is a discovery-stage group difference that survives multiple-comparison correction and artifact checks, but haemorrhage can change head optics, and high HbO–HbR correlation also signals systemic dominance. It needs an optical control and replication.")
qa(S, "What would change your conclusions?", "A dataset recording EEG, fNIRS and fMRI in the same people; a larger fMRI training set; short-separation fNIRS data; or a demonstration that cross-modal supervision helps on a targeted region. Any of these could reopen Claim 1.")

S = "Practical"
qa(S, "If you had three more months, what would you do?", "Run Maisha's remaining seeds, repeat personalisation on TU Berlin against a no-EEG control, compare the two reliability methods on the same split, verify trigger timing, and add an optical control to the clinical analysis. All use existing data.")
qa(S, "How would this become useful clinically?", "Through per-person calibration, targeted outputs (specific regions or summary measures rather than whole waveforms), validated uncertainty that identifies bad predictions, and replication in patient cohorts with optical controls.")


body = title_block("NeuroVascNet course · Lecture G", "Using the Results",
                   "Limitations, framing the work for a paper, defending it in a viva, and what to do next",
                   "Part G of 7 · The final lecture · Builds on all previous parts") + f"""
<div class="box">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>state the project's limitations precisely, with their likely effect on the results;</li>
<li>frame the work honestly for a paper: what to claim, what not to claim, and how to present it;</li>
<li>answer the hardest questions an examiner or reviewer is likely to ask;</li>
<li>choose the most valuable next steps.</li>
</ul>
</div>

<h2>1. How the course fits together</h2>
{fig("g3_map", 1, "The course as a chain.", "Signals (A) motivate a question (B), which needs data (C), a model (D) and honest evaluation (E) to produce results (F), which this lecture turns into claims and plans (G).")}

<h2>2. Limitations, precisely</h2>
<p>A limitation is useful only when it says <b>what could be wrong</b>, <b>in which direction it would push the result</b>, and <b>what would resolve it</b>. Researchers often organise them by the four classic kinds of validity:</p>
{fig("g1_validity", 2, "Four kinds of validity.", "Each asks a different question of a result. The italic line gives this project's main concern under each.")}
{table(1, "The project's main limitations.",
       ["Limitation", "Likely effect", "What would resolve it"],
       [["Small fMRI training set (10 people, 384 trials)", "weakens Claim 1's power; fMRI results exploratory", "a larger EEG–fMRI dataset"],
        ["Primary Claim 1 test uses 5 held-out people", "wide uncertainty", "the 29-person control (done) and more seeds"],
        ["Maisha's pipeline: 1 of 5 seeds run", "unknown run-to-run variation", "run the remaining 4 seeds"],
        ["Training dataset chosen after an audit", "possible perception of selection", "documented pre-model criterion; Maisha's opposite assignment as comparison"],
        ["External tests confound montage, task and target quality", "cannot attribute the failure", "datasets that vary one factor at a time"],
        ["ds002739 EEG channels paired by index", "external fMRI test even weaker", "recover electrode positions"],
        ["Event timing not fully verified", "would lower scores if offset", "check triggers against raw markers"],
        ["Clinical difference may be optical or systemic", "candidate marker could be non-neural", "pathlength sensitivity, short-separation data"],
        ["Early-phase numbers from older protocols", "cannot be compared with final results", "do not cite as headline results"]],
       widths=["38%", "32%", "30%"])}
{M("<p>Writing 'more data are needed' as a limitation. Every study needs more data. Say which data, why, and what result it would change.</p>")}

<h2>3. Framing the work for a paper</h2>
<h3>3.1 What the paper is about</h3>
<p>The honest framing is a <b>rigorous, two-pipeline evaluation of cross-modal haemodynamic supervision for EEG</b>, not a new prediction method. A defensible abstract has five sentences: the motivation (EEG is cheap; haemodynamics informative), what was done (one encoder with fNIRS and fMRI supervision, five pre-declared claims, two independent pipelines), the positive results (cross-subject fNIRS prediction r = 0.16 on a reliable target; a reproducible delay, ICC 0.53), the negative results (no cross-modal benefit; no external transfer; protocol inflation of 88%), and the take-home (measure target reliability before modelling).</p>
{table(2, "Say this, not that.",
       ["Say", "Not"],
       [["EEG predicts trial-level fNIRS responses in held-out people (r = 0.16, 22% of the ceiling)", "EEG predicts haemodynamics"],
        ["fMRI supervision from different people did not improve fNIRS prediction at this data scale", "cross-modal learning does not work"],
        ["a model-free EEG–HbO delay is reproducible within people (ICC 0.53)", "the network learns neurovascular delay"],
        ["the reliability score is calibrated but does not rank errors", "the system knows when it is wrong"],
        ["a candidate group difference in HbO–HbR coupling after ICH, pending optical controls", "a biomarker of ICH"],
        ["coupling is person-specific and does not transfer across datasets", "EEG lacks haemodynamic information"]],
       widths=["58%", "42%"])}

<h3>3.2 Contributions to list</h3>
<ol>
<li>A controlled test of cross-modal supervision, including a shuffled-target control, replicated by an independent pipeline.</li>
<li>Evidence that target reliability, measured before modelling, determines whether EEG-to-haemodynamic prediction is possible.</li>
<li>A quantified evaluation pitfall: trial-level splitting inflates performance by 88% on these data.</li>
<li>A reproducible, model-free neurovascular delay measure, and a demonstration that a bounded learned delay is circular.</li>
<li>A candidate clinical marker in ICH, reported with its confounds.</li>
</ol>

<h3>3.3 Figures that carry the story</h3>
<p>Plan about six main figures: target quality (reliability and HbO–HbR sign); protocol inflation; cross-subject prediction against the no-EEG baseline; Claim 1 per person with the shuffled control; delay (measured, test–retest, learned); and the two-pipeline agreement table. Everything else can go in supplementary material.</p>
{P("<p>The research report (<code>NeuroVascNet_Research_Report_v2.pdf</code>) already contains these figures and a draft of the framing, and can serve as the skeleton of the manuscript.</p>")}

<h3>3.4 Where it could be published</h3>
<p>Journals in neuroimaging methods and neural engineering — for example <i>NeuroImage</i>, <i>Journal of Neural Engineering</i>, <i>IEEE Transactions on Neural Systems and Rehabilitation Engineering</i>, <i>Neurophotonics</i> or <i>Human Brain Mapping</i> — publish careful methodological evaluations, including negative ones. A venue's fit matters more than its prestige: choose one whose readers care about EEG–haemodynamic methods and rigorous evaluation.</p>

<h2>4. Defending the work</h2>
<p>Examiners and reviewers probe the same few points: novelty, shortcuts, sample size, dataset choice, timing and interpretation. Prepare short, honest answers that acknowledge limits rather than defending everything. The bank below gives model answers for this project.</p>
{K("<p>The strongest defensive move is agreeing with a valid criticism and showing what was done about it. 'Yes, five held-out people is too few on its own — which is why we added the 29-person control, the shuffled-fMRI control and an independent pipeline, all of which agree' is far stronger than arguing that five is enough.</p>")}
{A("a good witness", "<p>A credible witness says 'I don't know' when they don't, distinguishes what they saw from what they inferred, and does not change their story under pressure. An examiner trusts a candidate who does the same: clear on the evidence, honest about the gaps, consistent across questions.</p>")}
{qa_bank(QA)}

<h2>5. What to do next</h2>
{fig("g2_priorities", 3, "Next steps by value and effort.", "Schematic judgement. Steps on the left use existing data and could be done quickly; steps on the right need new data collection.")}
{table(3, "Recommended next steps, in order.",
       ["Step", "Why", "Needs"],
       [["1. Run Maisha's remaining four seeds", "gives run-to-run variation for every v2.3 number", "compute only"],
        ["2. Personalisation on TU Berlin, against a no-EEG calibration-mean control", "tests whether EEG adds anything beyond a person's own mean, with adequate power", "existing data"],
        ["3. Compare the two reliability methods on the same split", "resolves their disagreement on in-domain ranking", "existing models"],
        ["4. Verify event timing against raw markers", "removes a hidden source of lowered scores", "existing data"],
        ["5. Optical control for the ICH finding", "decides whether the marker is neurovascular", "pathlength analysis; ideally short channels"],
        ["6. Targeted motor-region fMRI prediction", "literature suggests prediction is easier for specific regions", "existing data"],
        ["7. Treat fMRI as a separate analysis unless same-subject data appear", "avoids negative transfer", "a decision"],
        ["8. New recordings with short-separation fNIRS, or same-person EEG–fNIRS–fMRI", "the only way to test Claim 1 properly", "new data collection"]],
       widths=["42%", "38%", "20%"])}

<h2>6. Key numbers to remember</h2>
{table(4, "The numbers you will be asked about.",
       ["Quantity", "Value"],
       [["Cross-subject fNIRS prediction (TU Berlin)", "r = 0.158, p = 0.001, 5/5 folds positive, ≈ 22% of ceiling"],
        ["Cross-subject fMRI prediction (ds002336)", "r = 0.064"],
        ["Protocol inflation (trial-level vs honest)", "0.297 vs 0.158 (+88%)"],
        ["Claim 1 (joint − single)", "−0.018 (5 people); −0.007, p = 0.18 (29 people); shuffled fMRI = real"],
        ["External validation", "HEFMI 0.012 (4% of ceiling); ds002739 0.000"],
        ["Measured delay", "6.4 s mean, SD 1.1 s, ICC 0.53"],
        ["Learned delay", "2.01 s (2–8 window), 0.07 s (0–20), 0.09 s (unbounded)"],
        ["Downstream probe (4 classes, chance 0.25)", "0.374 vs random 0.381 vs shuffled 0.362"],
        ["Reliability", "ECE 0.27 → 0.02; uncertainty–error r = 0.02"],
        ["Within-person (v6 / Maisha)", "0.098 / 0.094 (null 0.021)"],
        ["Clinical HbO–HbR (ICH vs controls)", "0.81 vs 0.64, Holm p = 0.003"],
        ["Target reliability (TU Berlin / HEFMI)", "0.50 / 0.08; HbO–HbR −0.44 / +0.92"],
        ["GSR effect", "removed ≈ 83% of the evoked response"]],
       widths=["45%", "55%"])}

{Q(["Rewrite this limitation precisely: 'the sample was small'.",
    "Which of the four validity types does the ICH optics concern belong to, and why?",
    "Give the five-sentence abstract outline in your own words.",
    "How would you answer 'Isn't switching datasets cherry-picking?'",
    "Which next step would you do first, and why?"])}

<h2>Course summary</h2>
<ul>
<li><b>A</b>: neurons trigger an oversupply of oxygenated blood; EEG sees the electricity, fNIRS and fMRI see the blood, seconds later, mixed with systemic noise.</li>
<li><b>B</b>: the project tried to predict the blood from EEG alone, supervising one encoder with fNIRS and fMRI from different people, and turned that into five testable claims.</li>
<li><b>C</b>: four public datasets of very different quality; preprocessing aligns time, channels and units; target reliability matters most.</li>
<li><b>D</b>: a compact convolutional network with dataset adapters, a shared trunk, per-modality heads, a correlation + uncertainty loss, and joint training.</li>
<li><b>E</b>: honest evaluation holds out people, removes the person shortcut, tests against nulls and ceilings, and decides in advance.</li>
<li><b>F</b>: real cross-subject fNIRS prediction on a reliable target; no benefit from fMRI; no external transfer; a reliable measured delay; a candidate clinical marker; agreement between two pipelines.</li>
<li><b>G</b>: claim exactly what the evidence supports, state limits precisely, and do the cheap, high-value next steps first.</li>
</ul>

<h2>Answers to "Pause and think"</h2>
<ol class="answers">
<li>For example: 'The fMRI training set has 10 people (384 trials), so the cross-modal comparison has low power and fMRI results are exploratory; a larger EEG–fMRI dataset would be needed to detect a small benefit.'</li>
<li>Construct validity: the question is whether the HbO–HbR measure captures neurovascular coupling or instead optical or systemic differences.</li>
<li>Motivation; what was done; positive results; negative results; take-home message (check target reliability first).</li>
<li>The switch was decided by a target-quality criterion measured before any modelling, it is documented, and an independent pipeline that kept the original dataset provides the comparison.</li>
<li>Running Maisha's remaining seeds or the TU Berlin personalisation test: both use existing data and directly strengthen or test current claims.</li>
</ol>
"""

D.render(body, "Lecture G: Using the Results")
