"""Builds lectures/Lecture_E_Honest_Evaluation.pdf (teaching notes, Part E).

Definitions follow scripts/build_v6_notebook.py (mean_corr, perm_null,
constant_baseline, demean_by_subject, split_half_reliability, paired_delta) and
Maisha's v2.3 reporting rules. Run:  python scripts/build_lecture_part_e.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, ROOT, VIOLET, W, YELLOW, A, Doc, E, K,
                            M, P, Q, V, arrow, box, fig, table, title_block)

D = Doc("lecture_e", "Lecture_E_Honest_Evaluation.pdf", "NeuroVascNet course · Lecture E")
V6 = ROOT / "neurovascnet_v6_outputs"
S6 = json.loads((V6 / "summary.json").read_text())
PA = pd.read_csv(V6 / "protocol_ablation.csv")
DS = pd.read_csv(V6 / "data_summary.csv").set_index("dataset")


def f_corr():
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(1, 4, figsize=(W, 1.9), gridspec_kw={"wspace": 0.12})
    for a, r in zip(ax, [0.0, 0.16, 0.5, 0.9]):
        x = rng.standard_normal(200)
        y = r * x + np.sqrt(1 - r ** 2) * rng.standard_normal(200)
        a.scatter(x, y, s=5, color=BLUE, alpha=0.6)
        a.set_title(f"r = {r:.2f}   (r² = {r*r:.0%})", fontsize=9.5)
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_ylabel("measured"); ax[1].set_xlabel("predicted")
    D.save(fig, "e1_corr")


def f_splits():
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw={"wspace": 0.15})
    rng = np.random.default_rng(0)
    for a, title in zip(ax, ["Trial-level split", "Subject-level split", "External (new dataset)"]):
        a.set_xlim(0, 6); a.set_ylim(-0.6, 5.4); a.axis("off"); a.set_title(title, fontsize=10)
    for si in range(5):
        y = 4.6 - si
        for ti in range(6):
            test = rng.random() < 0.25
            ax[0].add_patch(plt.Rectangle((ti * 0.95 + 0.2, y - 0.35), 0.8, 0.7, color=ORANGE if test else BLUE, alpha=0.85))
            ax[1].add_patch(plt.Rectangle((ti * 0.95 + 0.2, y - 0.35), 0.8, 0.7, color=ORANGE if si == 4 else BLUE, alpha=0.85))
            ax[2].add_patch(plt.Rectangle((ti * 0.95 + 0.2, y - 0.35), 0.8, 0.7,
                                          color=BLUE if si < 3 else ORANGE, alpha=0.85 if si < 3 else 0.5,
                                          hatch="" if si < 3 else "///"))
        ax[0].text(0.1, y, f"S{si+1}", ha="right", va="center", fontsize=8)
    ax[2].text(3.0, 1.15, "different lab, device,\nmontage (and task)", ha="center", va="center", fontsize=8, color=INK)
    ax[0].text(3.0, -0.45, "same person on both sides", ha="center", fontsize=8.3, color=RED)
    ax[1].text(3.0, -0.45, "new people in test", ha="center", fontsize=8.3, color=MUTED)
    ax[2].text(3.0, -0.45, "new people AND new setting", ha="center", fontsize=8.3, color=MUTED)
    D.save(fig, "e2_splits")


def f_demean():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.3})
    offs = [(-1.5, -1.4, BLUE, "person A"), (0.0, 0.1, AQUA, "person B"), (1.5, 1.4, ORANGE, "person C")]
    allp, allt, dp, dt = [], [], [], []
    for px, ty, c, lab in offs:
        p = px + 0.25 * rng.standard_normal(25)
        t = ty + 0.5 * rng.standard_normal(25)
        ax[0].scatter(p, t, s=12, color=c, label=lab)
        ax[1].scatter(p - p.mean(), t - t.mean(), s=12, color=c)
        allp += list(p); allt += list(t); dp += list(p - p.mean()); dt += list(t - t.mean())
    r0 = np.corrcoef(allp, allt)[0, 1]; r1 = np.corrcoef(dp, dt)[0, 1]
    ax[0].set_title(f"Pooled: r = {r0:.2f}\n(driven by who the person is)", fontsize=9.5)
    ax[1].set_title(f"Each person's mean removed: r = {r1:.2f}\n(what is left is trial-to-trial)", fontsize=9.5)
    ax[0].legend(fontsize=8, loc="upper left")
    for a in ax:
        a.set_xlabel("predicted"); a.set_ylabel("measured"); a.set_xticks([]); a.set_yticks([])
    D.save(fig, "e3_demean")


def f_protocol():
    labs = ["Trials split at random\n(same person in train and test)", "Subjects split\n(new people in test)",
            "Subject identity only\n(no EEG used)", "Subjects split +\nsubject mean removed"]
    fig, a = plt.subplots(figsize=(W, 2.2))
    y = np.arange(4)[::-1]
    a.barh(y, PA.r, color=[GRAY, GRAY, LIGHT, BLUE], height=0.58)
    for yi, v in zip(y, PA.r):
        a.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8.6)
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=8.6); a.set_xlim(0, 0.34)
    a.set_xlabel("r on TU Berlin hBCI (project data)"); a.grid(axis="y", visible=False)
    D.save(fig, "e4_protocol")


def f_null():
    rng = np.random.default_rng(2)
    null = rng.normal(0.0, 0.012, 1000)
    obs = 0.045
    fig, a = plt.subplots(figsize=(W, 2.3))
    a.hist(null, bins=40, color=GRAY, edgecolor="white")
    a.axvline(obs, color=ORANGE, lw=2.2)
    a.text(obs + 0.002, a.get_ylim()[1] * 0.85, "observed r\nwith the real pairing", fontsize=8.5, color=ORANGE)
    a.text(-0.045, a.get_ylim()[1] * 0.85, "1,000 shuffled pairings:\nwhat r looks like when\nEEG and target are unrelated", fontsize=8.5, color=MUTED)
    a.set_xlabel("correlation"); a.set_yticks([])
    D.save(fig, "e5_null")


def f_ceiling():
    rows = [("TU Berlin hBCI\n(internal, new people)", DS.loc["TU Berlin hBCI (MI+MA)", "ceiling"], S6["internal"]["TU Berlin hBCI (MI+MA)"]["r_demeaned"]),
            ("ds002336\n(internal, new people)", DS.loc["ds002336", "ceiling"], S6["internal"]["ds002336"]["r_demeaned"]),
            ("HEFMI\n(external)", DS.loc["HEFMI", "ceiling"], S6["external"]["HEFMI"]["external_r"]),
            ("ds002739\n(external)", DS.loc["ds002739", "ceiling"], max(S6["external"]["ds002739"]["external_r"], 0))]
    fig, a = plt.subplots(figsize=(W, 2.5))
    y = np.arange(len(rows))[::-1]
    a.barh(y, [r[1] for r in rows], color=LIGHT, height=0.6, label="ceiling = √(target reliability)")
    a.barh(y, [r[2] for r in rows], color=BLUE, height=0.32, label="model r")
    for yi, (_, c, r) in zip(y, rows):
        a.text(c + 0.01, yi, f"{r:.3f} of {c:.2f}  ({r / c:.0%})", va="center", fontsize=8.5)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=8.5); a.set_xlim(0, 1.05)
    a.legend(loc="lower right", fontsize=8.3); a.grid(axis="y", visible=False)
    a.set_xlabel("correlation (project data)")
    D.save(fig, "e6_ceiling")


for f in [f_corr, f_splits, f_demean, f_protocol, f_null, f_ceiling]:
    f()

pa = PA.r.values
c1 = S6["claim1"]
body = title_block("NeuroVascNet course · Lecture E", "Honest Evaluation",
                   "How to measure whether a model really works, and the many ways it can look good for the wrong reasons",
                   "Part E of 7 · Builds on Lectures A–D · About 100 minutes of reading · The most important lecture for defending the work") + f"""
<div class="box">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>explain what a correlation measures and what values such as r = 0.16 mean in practice;</li>
<li>explain why trial-level, subject-level and external testing answer different questions, and why the subject is the unit of analysis;</li>
<li>describe the "subject shortcut" and how the no-EEG baseline and subject demeaning expose it;</li>
<li>explain permutation and sign-flip tests, bootstrap confidence intervals, and the difference between significance and size;</li>
<li>use target reliability to compute a performance ceiling;</li>
<li>list the control comparisons that turn "the model scored X" into a claim.</li>
</ul>
</div>

<h2>1. Why evaluation is the hard part</h2>
<p>Training a model that produces a good-looking number is easy. Knowing whether the number means what you think it means is hard. Neural networks are excellent at finding shortcuts: any feature of the data that correlates with the answer will be used, whether or not it is the feature you intended.</p>
{A("Clever Hans", "<p>In the early 1900s a horse called Clever Hans appeared to solve arithmetic, tapping out answers with his hoof. Investigators eventually found that Hans was reading tiny, unconscious changes in his questioner's posture as the right number approached. The horse was genuinely clever — at reading people, not at arithmetic. Every evaluation in this lecture is a way of checking whether our model is doing arithmetic or reading the questioner.</p>")}
<p>For this project the most dangerous shortcut is <b>recognising the person</b>. Each person has their own typical blood response; a model that recognises whose EEG it is reading can predict that person's average response and score well, without learning anything about how brain activity drives blood flow.</p>

<h2>2. The basic measure: correlation</h2>
<p>Most results are reported as a <b>Pearson correlation</b>, r, between predicted and measured signals. It ranges from −1 to +1: +1 means the prediction rises and falls exactly with the measurement, 0 means no linear relationship, −1 means exact opposites. The square, r², is the fraction of the measured variation the prediction accounts for.</p>
{fig("e1_corr", 1, "What correlations look like.", "Simulated data. Each point is one predicted–measured pair. At r = 0.16 the relationship is real but weak: it accounts for under 3% of the variation, and individual predictions are poor.")}
<p>In the project, each trial's predicted time course (for every channel) is correlated with its measured time course, and these correlations are averaged across trials. Because correlation ignores scale, this measures shape and timing, matching the training loss.</p>
{K("<p>A correlation of 0.16 is not useless, and not impressive. In a field where some prior methods report zero across subjects, a reliable 0.16 on new people is a real finding; but it explains only about 2.5% of the variation in a single trial. Always say both halves of that sentence.</p>")}

<h2>3. What are we generalising to?</h2>
{fig("e2_splits", 2, "Three ways to split data.", "Schematic. Rows are people (S1–S5), boxes are trials; blue = training, orange = test. Left: trials are split at random, so every person appears in both training and test sets. Middle: whole people are held out. Right: the test data come from a different dataset recorded elsewhere, possibly with a different task.")}
{table(1, "Each split answers a different question.",
       ["Split", "Question answered", "Typical risk"],
       [["Trial-level", "can the model predict new trials from people it has already seen?", "the model can succeed by recognising the person"],
        ["Subject-level (cross-subject)", "can it predict for a new person from the same study?", "small number of people; still the same equipment and task"],
        ["External (cross-dataset)", "does it work in a new lab, device, montage or task?", "many differences at once, hard to tell which one matters"]],
       widths=["22%", "45%", "33%"])}
<p>The project's goal — an EEG-only tool for new users — demands at least subject-level testing. In <b>k-fold cross-validation by subject</b> the people are divided into k groups (k = 5 here); each group is held out once while a model is trained on the rest, so every person is tested exactly once by a model that never saw them. <b>Leave-one-subject-out (LOSO)</b> is the extreme case where each group is a single person.</p>
{M("<p>Splitting trials at random. It is the default in many machine-learning tutorials, and it is wrong for brain data whenever the claim is about new people. It is the most common reason published EEG results fail to replicate.</p>")}

<h2>4. The subject shortcut, and how to remove it</h2>
<h3>4.1 The no-EEG baseline</h3>
<p>The simplest test of a shortcut is to build a predictor that <b>cannot</b> use the brain at all, and see how well it scores. The project's <b>no-EEG baseline</b> ignores the EEG completely and predicts the average training response for every trial. Any score this baseline reaches is achievable without reading the brain.</p>

<h3>4.2 Subject demeaning</h3>
<p>The cure is to remove each person's average from both the prediction and the measurement before computing the correlation. What remains is how each trial differs from that person's usual response — the part that can only be predicted from the trial's own EEG. After demeaning, the no-EEG baseline scores exactly zero by construction.</p>
{fig("e3_demean", 3, "How the person shortcut inflates correlation.", "Simulated data. Left: three people differ in their average prediction and average measurement, so pooled trials correlate strongly even though, within each person, prediction and measurement are unrelated. Right: after removing each person's mean, the true trial-level relationship (here none) is revealed.")}
<p>When two task types are pooled (TU Berlin's motor imagery and mental arithmetic), the same logic applies one level up: a model that only recognises <i>which task</i> is being done can also score. The project therefore also demeans by <b>subject × task</b>; the result barely changed, showing the TU Berlin score was not a task-recognition shortcut.</p>
{A("grading weather forecasts", "<p>A forecaster who always predicts 'hot' in Dubai and 'cold' in Oslo will correlate well with actual temperatures across both cities, without forecasting anything. To test real skill, compare each forecast with that city's normal temperature for the day. Subject demeaning is exactly this: it asks whether the model predicts deviations from each person's normal.</p>")}

<h3>4.3 How much does protocol matter? A real example</h3>
{fig("e4_protocol", 4, "The same model scored under four protocols.", f"Project data (TU Berlin hBCI). A random trial-level split gives {pa[0]:.3f}; holding out people gives {pa[1]:.3f}; a predictor that only knows who the person is gives {pa[2]:.3f}; holding out people and removing each person's mean gives the honest {pa[3]:.3f}.")}
<p>The naive protocol reports almost twice the honest figure (an 88% inflation). Any published number from a trial-level split on data like this should be read with that in mind.</p>

<h2>5. Is it real? Null hypotheses and permutation tests</h2>
<p>A correlation from a finite sample is never exactly zero even when there is no relationship. The question is whether the observed value is larger than chance would produce. The <b>null hypothesis</b> is "EEG carries no information about the target". A <b>permutation test</b> simulates that world directly: shuffle which prediction is paired with which target, recompute the correlation, and repeat 1,000 times. The shuffled values form the <b>null distribution</b>.</p>
{fig("e5_null", 5, "A permutation test.", "Schematic. Grey: correlations obtained after randomly re-pairing predictions with targets, i.e. when there is no real link. Orange: the correlation with the real pairing. If it lies far out in the tail, the link is unlikely to be chance.")}
<p>The <b>p-value</b> is the fraction of shuffled results at least as extreme as the real one. The project computes it as (number of extreme shuffles + 1) ÷ (number of shuffles + 1), so it is never reported as exactly zero — with 1,000 shuffles the smallest possible value is about 0.001.</p>
<p>For comparing two models on the same people (for example joint versus single-modality training), the project uses a <b>sign-flip test</b>. Each person gives one difference. If the two models were equivalent, each difference would be equally likely to be positive or negative, so randomly flipping the signs generates the null distribution.</p>
{E("Claim 1 in numbers", f"<p>Five held-out people, three seeds each. Seeds are averaged within each person first, so seed noise is not counted as extra evidence. The mean difference (joint minus single) was {c1['mean_diff']:+.3f}; only {c1['n_subjects_positive']} of {c1['n_subjects']} people favoured joint training; the sign-flip p-value was {c1['p_signflip']:.2f}. No evidence that joint training helps.</p>")}

<h2>6. How sure are we? Confidence intervals</h2>
<p>A <b>95% confidence interval</b> gives a range of plausible values for the true effect. The project uses a <b>bootstrap</b>: resample the <i>people</i> with replacement 10,000 times, recompute the mean each time, and take the middle 95% of the results. Resampling people rather than trials matters for the same reason as before: trials from one person are not independent, and resampling them would produce falsely narrow intervals.</p>
{K("<p>The unit of resampling, of shuffling and of counting is the <b>person</b>, not the trial. A result built on 3,480 trials from 29 people has the statistical weight of 29 observations for any claim about new people.</p>")}

<h2>7. Significant is not the same as large</h2>
<p>With enough data, even a trivially small effect becomes statistically significant. In Maisha's pipeline the external ds002739 test gave r = 0.009 with p = 0.005: real by the test, irrelevant in practice. The project therefore reports an <b>effect size</b> label ("practically negligible", "very small"…) alongside every p-value, and uses the <b>ceiling</b> (next section) to judge size.</p>

<h2>8. The ceiling: how good could any model be?</h2>
<p>No model can predict a target better than the target agrees with itself. <b>Split-half reliability</b> measures that agreement: split each person's trials randomly into two halves, average each half, correlate the two averages, repeat 50 times and average. The square root of the reliability is an upper bound — the <b>ceiling</b> — on the correlation any model could achieve against that target.</p>
{fig("e6_ceiling", 6, "Model performance against each target's ceiling.", "Project data. Grey bars: the ceiling for each dataset. Blue bars: the model's correlation. The percentage shows how much of the achievable range the model reaches. External results reach almost none of it.")}
{A("a blurry answer key", "<p>If an exam's answer key was written by someone who copies only half the answers correctly, even a perfect student will 'disagree' with the key half the time. Before judging the student, check the answer key. Split-half reliability checks the answer key.</p>")}
{P("<p>This one calculation explained months of disappointing results. HEFMI's ceiling is only 0.29, so near-zero prediction there says little about EEG. TU Berlin's ceiling is 0.71, which is why switching datasets made a real difference.</p>")}

<h2>9. Controls: comparisons that turn a number into a claim</h2>
<p>A result becomes a claim only when it beats the comparison that would explain it away. The project's controls:</p>
{table(2, "The control comparisons used in the project.",
       ["Control", "What it rules out", "Used for"],
       [["No-EEG baseline", "scores achievable without the brain", "all prediction results"],
        ["Shuffled pairing (permutation)", "chance correlation", "significance of every r"],
        ["Single-modality model at the same budget and seed", "that any gain is due to more training rather than fMRI", "Claim 1"],
        ["Joint model with shuffled fMRI targets", "that the gain comes from extra data, not fMRI content", "Claim 1"],
        ["Subject embedding switched off", "that the gain is subject adaptation", "Claim 1"],
        ["Random-weight and shuffled-target encoders; raw band power", "that any good representation is generic", "Claim 4"],
        ["Different delay windows and an unbounded head", "that the delay simply reports its bounds", "Claim 3"],
        ["Real vs shuffled pairs at small scale (capacity test)", "that the network can learn real pairs better than random ones", "feasibility"]],
       widths=["34%", "40%", "26%"])}

<h2>10. Within-subject versus cross-subject</h2>
<p>Training and testing inside a single person (on separate blocks of their trials) gives a different upper bound. If within-subject prediction is good but cross-subject is poor, the signal exists but is person-specific; if both are poor, the target itself is the problem. Maisha's pipeline separates three quantities for amplitude predictions:</p>
<ul>
<li><b>Between-subject ranking</b>: does the model rank people correctly by their average response? (High values mean it learned <i>who</i>.)</li>
<li><b>Within-subject</b>: within each person, does it track which trials are larger?</li>
<li><b>Cross-subject centred</b>: within-person tracking on people it has never seen — the only one that supports a claim about new users.</li>
</ul>
{E("learning who, not what", "<p>On the ds002336 fMRI training data, Maisha's model ranked people by average response with r = 0.98 but tracked trials within a person at only 0.25. It had mostly learned to recognise people.</p>")}

<h2>11. Several tests at once</h2>
<p>Testing many measures at p &lt; 0.05 will produce some false positives by chance. The clinical comparison (Lecture F) tests four primary measures and applies the <b>Holm correction</b>: sort the p-values, and compare the smallest with 0.05/4, the next with 0.05/3, and so on. Only results that survive are reported as significant. Artifact checks (motion, amplitudes) are reported separately as controls.</p>

<h2>12. Evaluating the other outputs</h2>
<ul>
<li><b>Reliability score</b>: calibration error (ECE) with the calibration fitted on one half of the data and measured on the other; the correlation between uncertainty and error (ranking); and accuracy as coverage falls (risk–coverage curve). Calibrating and judging on the same data is not allowed.</li>
<li><b>Delay</b>: compared with a <b>model-free</b> delay (the lag at which EEG power and HbO correlate best), checked for consistency across sessions with the intraclass correlation (ICC) against surrogate data, and stress-tested by changing the allowed window.</li>
<li><b>Representation</b>: linear-probe accuracy against random, shuffled and raw-feature controls.</li>
</ul>

<h2>13. Deciding in advance</h2>
<p>Every claim in the final pipeline had its test and its decision rule written before the results were seen: for example, Claim 1 "holds" only if the confidence interval excludes zero, the sign-flip p-value is below 0.05, and the difference favours joint training. Maisha's pipeline goes further with an automatic <b>claim guard</b> that blocks wording the evidence does not support (for example, it will not print "information limit" when fewer than 15 subjects were tested, and it blocked alignment-based claims when timing checks failed). Deciding in advance prevents the analysis from drifting towards the answer you hoped for.</p>
{V("<p><b>“How do you know your result isn't just subject identity?”</b> Subjects are held out, each held-out person is given the average subject embedding, each person's mean is removed before scoring so a no-EEG baseline scores exactly zero, and the result is also demeaned by subject and task. The 0.158 survives all of these and beats a 1,000-shuffle permutation null.</p>")}

{Q(["What does r = 0.16 mean in terms of explained variation?",
    "Why does a trial-level split overstate performance for new people?",
    "Explain the no-EEG baseline and why subject demeaning sets it to zero.",
    "Why was the score also demeaned by subject × task for TU Berlin?",
    "What does a permutation test simulate, and how is its p-value computed here?",
    "Why are people, not trials, resampled in the bootstrap?",
    "A result is r = 0.009, p = 0.005. How would you describe it?",
    "HEFMI's split-half reliability is 0.08. What is the best correlation any model could reach?",
    "Why compare joint training with a shuffled-fMRI control and not only with fNIRS-only training?",
    "What does high between-subject ranking with low within-subject tracking tell you?",
    "Why must calibration be fitted and judged on different data?"])}

<h2>Summary</h2>
<ul>
<li>Models find shortcuts; for brain data the main one is <b>recognising the person</b>.</li>
<li><b>Correlation</b> measures shape agreement; r² gives explained variation. 0.16 is real but small.</li>
<li>Trial-level, subject-level and external splits answer different questions; claims about new people need <b>subject-level</b> testing.</li>
<li>The <b>no-EEG baseline</b> reveals the person shortcut; <b>subject demeaning</b> removes it. On TU Berlin the naive protocol inflates the score by 88%.</li>
<li><b>Permutation</b> and <b>sign-flip</b> tests ask "could this be chance?"; <b>bootstrap</b> intervals over people ask "how sure are we?"; effect sizes ask "does it matter?".</li>
<li>The <b>ceiling</b> (√ split-half reliability) bounds every result and explains why the target dataset mattered so much.</li>
<li>Claims require <b>controls</b> decided in advance; corrections such as Holm handle multiple tests.</li>
</ul>
<h3>Coming next: Lecture F</h3>
<p>Lecture F tells the whole story of the project's results, claim by claim, for both pipelines — what held, what failed and why.</p>

<h2>Glossary</h2>
{table(3, "Terms introduced in this lecture.",
       ["Term", "Meaning"],
       [["Pearson correlation (r)", "strength of linear agreement, from −1 to +1"],
        ["r²", "fraction of variation accounted for"],
        ["Cross-validation (k-fold, LOSO)", "repeatedly holding out part of the data for testing"],
        ["No-EEG baseline", "a predictor that ignores EEG; its score is achievable without the brain"],
        ["Subject demeaning", "removing each person's average before scoring"],
        ["Null hypothesis", "the assumption that there is no real effect"],
        ["Permutation test", "rebuilding the null by shuffling pairings"],
        ["Sign-flip test", "a permutation test for paired differences"],
        ["p-value", "how often chance alone would give a result at least this extreme"],
        ["Bootstrap", "estimating uncertainty by resampling the data with replacement"],
        ["Confidence interval", "a range of plausible values for the true effect"],
        ["Effect size", "how large an effect is, independent of significance"],
        ["Split-half reliability", "agreement of a target with itself across halves of trials"],
        ["Ceiling", "the square root of reliability: the best achievable correlation"],
        ["Holm correction", "a step-down adjustment for testing several hypotheses"],
        ["ICC", "intraclass correlation: consistency of a measure across repeated sessions"],
        ["Claim guard", "an automatic check that blocks unsupported conclusions"]],
       widths=["30%", "70%"])}

<h2>Answers to "Pause and think"</h2>
<ol class="answers">
<li>r² = 0.0256: the prediction accounts for about 2.5% of the variation in a single trial.</li>
<li>The model can learn each person's typical response and recognise the person in the test set, so it scores well without learning anything that transfers to new people.</li>
<li>It predicts the average training response for every trial, ignoring EEG. Demeaning removes each person's average from both prediction and target; a prediction that is the same for every trial of a person becomes zero, so it correlates with nothing.</li>
<li>Motor imagery and mental arithmetic have different average responses, so a model that only recognised the task could score. Demeaning by subject × task removes that route; the score barely changed.</li>
<li>A world where EEG and target are unrelated. p = (number of shuffled correlations at least as large as the real one + 1) ÷ (number of shuffles + 1).</li>
<li>Trials from one person are not independent; resampling them would treat 120 correlated trials as 120 independent observations and give falsely narrow intervals.</li>
<li>Statistically detectable but practically negligible: it explains less than 0.01% of the variation.</li>
<li>√0.08 ≈ 0.29.</li>
<li>A gain over fNIRS-only training could come simply from having more data or more training steps. If real fMRI and shuffled fMRI give the same result, the fMRI content is not what helps.</li>
<li>The model has learned who each person is (their average level) rather than what drives trial-to-trial changes.</li>
<li>Otherwise the calibration can memorise the data it is judged on and report an optimistic error.</li>
</ol>
"""

D.render(body, "Lecture E: Honest Evaluation")
