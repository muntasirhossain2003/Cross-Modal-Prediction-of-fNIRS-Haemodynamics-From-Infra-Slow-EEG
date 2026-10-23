"""Builds lectures/Lecture_F_The_Results.pdf (teaching notes, Part F).

All result figures read neurovascnet_v6_outputs/ and Maisha's output/ JSON.
Run from the repo root:  python scripts/build_lecture_part_f.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, ROOT, VIOLET, W, YELLOW, A, Doc, E, K,
                            M, P, Q, V, box, arrow, fig, table, title_block)

D = Doc("lecture_f", "Lecture_F_The_Results.pdf", "NeuroVascNet course · Lecture F")
V6 = ROOT / "neurovascnet_v6_outputs"
MA = ROOT / "output"
S6 = json.loads((V6 / "summary.json").read_text())
SM = json.loads((MA / "summary_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json").read_text())
PH = json.loads((MA / "personalized_physio_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json").read_text())
csv = lambda n, **k: pd.read_csv(V6 / n, **k)


def f_timeline():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    phases = [("Phase 1\nv3 – v3.3", "Jun–Jul", "waveform ≈ 0;\n3 fixes fail;\n'EEG limit?'", GRAY),
              ("Phase 2\nData audit", "late Jul", "target is the\nproblem; GSR\nharmful", ORANGE),
              ("Phase 3\nv6 final", "Aug", "TU Berlin trains:\nr = 0.158;\n5 claims tested", BLUE),
              ("Phase 4\nMaisha v2.3", "Aug", "independent\nrebuild; same\nconclusions", AQUA)]
    for i, (n, when, what, c) in enumerate(phases):
        x = 0.15 + i * 2.5
        box(ax, x, 2.15, 2.15, 1.15, n, fc=c, ec=c, color="white", weight="bold", fs=9)
        ax.text(x + 1.07, 1.95, when, ha="center", va="top", fontsize=8.3, color=MUTED)
        ax.text(x + 1.07, 1.55, what, ha="center", va="top", fontsize=8.5)
        if i < 3:
            arrow(ax, (x + 2.17, 2.72), (x + 2.48, 2.72))
    D.save(fig, "f1_timeline")


def f_quality():
    d = csv("data_summary.csv")
    names = {"TU Berlin hBCI (MI+MA)": "TU Berlin hBCI", "ds002336": "ds002336", "HEFMI": "HEFMI", "ds002739": "ds002739"}
    d["n"] = d.dataset.map(names)
    o = d.sort_values("reliability")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.45})
    a = ax[0]
    a.barh(o.n, o.reliability, color=[ORANGE if v < 0.3 else BLUE for v in o.reliability], height=0.55)
    for y, v in enumerate(o.reliability):
        a.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8.6)
    a.set_xlim(0, 0.85); a.set_title("Split-half reliability of the target", fontsize=10); a.grid(axis="y", visible=False)
    a = ax[1]
    f = d.dropna(subset=["hbo_hbr"]).sort_values("hbo_hbr")
    a.barh(f.n, f.hbo_hbr, color=[BLUE if v < 0 else ORANGE for v in f.hbo_hbr], height=0.45)
    a.axvline(0, color=MUTED, lw=1)
    for y, v in enumerate(f.hbo_hbr):
        a.text(v + (0.05 if v > 0 else -0.05), y, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8.6)
    a.set_xlim(-1, 1.15); a.set_title("HbO–HbR correlation (fNIRS)", fontsize=10); a.grid(axis="y", visible=False)
    D.save(fig, "f2_quality")


def f_cv():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    cols = [("r_raw", "raw"), ("r_no_eeg_baseline", "no EEG"), ("r_demeaned", "EEG\ncontribution")]
    for a, (f, title) in zip(ax, [("cv_TU_Berlin_hBCI_(MI+MA).csv", "TU Berlin hBCI (fNIRS)"), ("cv_ds002336.csv", "ds002336 (fMRI)")]):
        c = csv(f)
        for i, (k, _) in enumerate(cols):
            col = BLUE if k == "r_demeaned" else (LIGHT if "no_eeg" in k else GRAY)
            a.scatter(i + np.linspace(-0.12, 0.12, len(c)), c[k], s=30, color=col, edgecolor="white", lw=1, zorder=3)
            m = c[k].mean(); a.plot([i - 0.25, i + 0.25], [m, m], color=INK, lw=2)
            a.text(i + 0.28, m, f"{m:.3f}", va="center", fontsize=8.3)
        a.set_xticks(range(3)); a.set_xticklabels([t for _, t in cols], fontsize=8.5); a.set_xlim(-0.5, 2.75)
        a.axhline(0, color=MUTED, lw=0.8); a.set_title(title, fontsize=10); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("r per fold (5 folds)")
    D.save(fig, "f3_cv")


def f_claim1():
    arms = csv("controls_paired_arms.csv")
    f = arms[arms.cohort == "fnirs"].groupby(["subject", "arm"]).r.mean().unstack()
    lw = csv("loss_weight_search.csv"); bs = csv("bottleneck_sweep.csv")
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"wspace": 0.45, "width_ratios": [1.15, 1, 1]})
    a = ax[0]
    lim = (-0.05, 0.45)
    a.plot(lim, lim, color=MUTED, ls="--", lw=1)
    a.scatter(f.single, f.joint, s=22, color=BLUE, edgecolor="white", lw=0.9, zorder=3)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("fNIRS only"); a.set_ylabel("fNIRS + fMRI")
    a.set_title(f"Per person ({(f.joint > f.single).sum()}/{len(f)} above line)", fontsize=9.5)
    a = ax[1]
    a.plot(lw.w_fmri, lw["diff"], color=BLUE, lw=2, marker="o", ms=5); a.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log"); a.set_xticks(lw.w_fmri); a.set_xticklabels(lw.w_fmri, fontsize=8)
    a.set_xlabel("fMRI loss weight"); a.set_ylabel("joint − single r"); a.set_title("Weighting", fontsize=9.5)
    a = ax[2]
    a.plot(bs.bottleneck, bs.joint_minus_single, color=BLUE, lw=2, marker="o", ms=5); a.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log", base=2); a.set_xticks(bs.bottleneck); a.set_xticklabels(bs.bottleneck, fontsize=8)
    a.set_xlabel("bottleneck width"); a.set_title("Shared capacity", fontsize=9.5)
    D.save(fig, "f4_claim1")


def f_claim2():
    ev = csv("external_validation.csv", index_col=0)
    fs = csv("claim2_fewshot_calibration.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.4, "width_ratios": [1.3, 1]})
    a = ax[0]
    labs = ["fNIRS\n(TU Berlin → HEFMI)", "fMRI\n(ds002336 → ds002739)"]
    x = np.arange(2); w = 0.36
    a.bar(x - w / 2, ev.internal_r, w, color=GRAY, label="internal (new people)")
    a.bar(x + w / 2, ev.external_r.clip(lower=0.0005), w, color=ORANGE, label="external (new dataset)")
    for xi, (i, e) in enumerate(zip(ev.internal_r, ev.external_r)):
        a.text(xi - w / 2, i + 0.004, f"{i:.3f}", ha="center", fontsize=8.2)
        a.text(xi + w / 2, max(e, 0) + 0.004, f"{abs(e) if abs(e) < 5e-4 else e:.3f}", ha="center", fontsize=8.2)
    a.set_xticks(x); a.set_xticklabels(labs, fontsize=8.3); a.legend(fontsize=8); a.grid(axis="x", visible=False)
    a.set_title("Internal vs external", fontsize=9.5)
    a = ax[1]
    a.plot(fs.calibration_subjects, fs.r_external, color=ORANGE, lw=2, marker="o", ms=5); a.axhline(0, color=MUTED, lw=1)
    a.set_xlabel("HEFMI people used to recalibrate"); a.set_ylabel("external r"); a.set_title("Recalibration does not help", fontsize=9.5)
    D.save(fig, "f5_claim2")


def f_claim3():
    rec = csv("clinical_per_recording.csv"); per = csv("clinical_per_person.csv")
    s = rec.assign(sess=rec.recording.str.extract(r"(\d)").astype(int)).pivot_table(index="person", columns="sess", values="coupling_lag_s").dropna(subset=[1, 2])
    icc = csv("delay_model_free_icc.csv").iloc[0]; dl = csv("claim3_delay.csv"); dfree = csv("claim3_delay_free.csv")
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"wspace": 0.45})
    a = ax[0]
    a.axvspan(2, 8, color=BLUE, alpha=0.07)
    a.hist(per.coupling_lag_s, bins=np.arange(3, 10.01, 0.5), color=BLUE, edgecolor="white")
    a.axvline(icc.mean_s, color=INK, lw=1.4); a.set_xlabel("measured delay (s)"); a.set_ylabel("people")
    a.set_title(f"Measured: mean {icc.mean_s:.1f} s", fontsize=9.5)
    a = ax[1]
    a.plot([3.5, 9.5], [3.5, 9.5], color=MUTED, ls="--", lw=1); a.scatter(s[1], s[2], s=18, color=BLUE)
    a.set_xlim(3.5, 9.5); a.set_ylim(3.5, 9.5); a.set_xlabel("session 1 (s)"); a.set_ylabel("session 2 (s)")
    a.set_title(f"Repeatable: ICC {icc.icc:.2f}", fontsize=9.5)
    a = ax[2]
    labs = ["bounded\n2–8 s", "bounded\n0–20 s", "unbounded"]
    vals = [dl.mean_s[0], dl.mean_s[1], dfree.mean_s[0]]
    a.bar(range(3), vals, color=ORANGE, width=0.55)
    a.axhline(icc.mean_s, color=BLUE, ls="--", lw=1.3); a.text(2.4, icc.mean_s + 0.2, "measured", fontsize=8, color=BLUE, ha="right")
    for i, v in enumerate(vals):
        a.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8.3)
    a.set_xticks(range(3)); a.set_xticklabels(labs, fontsize=8); a.set_ylim(0, 7.5)
    a.set_title("Learned: follows the window", fontsize=9.5); a.grid(axis="x", visible=False)
    D.save(fig, "f6_claim3")


def f_claim4():
    c4 = csv("claim4_downstream.csv"); dp = csv("claim4_depth_probe.csv")
    names = {"cross-modal supervised": "fNIRS+fMRI encoder", "SHUFFLED-target control": "shuffled targets",
             "random init (untrained)": "random weights", "raw band-power": "raw band power",
             "FINE-TUNED from supervised": "fine-tuned (from encoder)", "FINE-TUNED from scratch": "fine-tuned (scratch)"}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.6, "width_ratios": [1.4, 1]})
    a = ax[0]
    o = c4.iloc[[3, 2, 1, 0, 5, 4]].reset_index(drop=True)
    y = np.arange(len(o))[::-1]
    a.barh(y, o.accuracy, xerr=o.sd, color=[GRAY, GRAY, GRAY, BLUE, LIGHT, LIGHT], height=0.6,
           error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    a.axvline(0.25, color=RED, ls=":", lw=1.2); a.text(0.255, 5.45, "chance 0.25", fontsize=7.8, color=RED)
    a.set_yticks(y); a.set_yticklabels([names[e] for e in o.encoder], fontsize=8.2); a.set_xlim(0, 0.55)
    a.set_xlabel("4-class accuracy"); a.grid(axis="y", visible=False); a.set_title("Linear probe", fontsize=9.5)
    a = ax[1]
    m = dp[dp.readout == "mean"]
    a.plot(range(len(m)), m.accuracy, color=BLUE, lw=2, marker="o", ms=5)
    a.set_xticks(range(len(m))); a.set_xticklabels(m.stage, fontsize=8); a.set_ylim(0.2, 0.4)
    a.axhline(0.25, color=RED, ls=":", lw=1.2); a.set_title("Depth probe: flat", fontsize=9.5); a.set_ylabel("accuracy")
    D.save(fig, "f7_claim4")


def f_claim5():
    cov = csv("coverage_curve.csv"); c5 = S6["claim5"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.4})
    a = ax[0]
    a.plot(cov.coverage * 100, cov.r, color=BLUE, lw=2, marker="o", ms=5); a.invert_xaxis(); a.set_ylim(0.14, 0.19)
    a.set_xlabel("% of trials answered"); a.set_ylabel("r on answered"); a.set_title("Abstaining does not help", fontsize=9.5)
    a = ax[1]
    a.bar([0, 1], [c5["ece_before"], c5["ece_after"]], color=[GRAY, BLUE], width=0.5)
    for x, v in enumerate([c5["ece_before"], c5["ece_after"]]):
        a.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.3)
    a.set_xticks([0, 1]); a.set_xticklabels(["before", "after Platt"]); a.set_ylabel("ECE")
    a.set_title("Calibration works", fontsize=9.5); a.grid(axis="x", visible=False)
    D.save(fig, "f8_claim5")


def f_clinical():
    pp = csv("clinical_per_person.csv")
    fig, a = plt.subplots(figsize=(W, 2.0))
    rng = np.random.default_rng(1)
    for i, (g, lab, c) in enumerate([("Normals", "controls (17)", BLUE), ("Patients", "ICH patients (20)", ORANGE)]):
        v = pp[pp.group == g].hbo_hbr.values
        a.scatter(v, i + rng.uniform(-0.12, 0.12, len(v)), s=24, color=c, edgecolor="white", lw=0.9, zorder=3)
        a.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        a.text(v.mean() + 0.01, i + 0.33, f"mean {v.mean():.2f}", fontsize=8.3)
    a.set_yticks([0, 1]); a.set_yticklabels(["controls (17)", "ICH patients (20)"]); a.set_ylim(-0.55, 1.7)
    a.set_xlabel("HbO–HbR correlation per person"); a.grid(axis="y", visible=False)
    D.save(fig, "f9_clinical")


def f_maisha():
    lo, ws = SM["cross_subject_loso"], SM["within_subject_probe"]
    pc = PH["personalized_calibration_curve"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35})
    a = ax[0]
    items = [("HEFMI\nnew people", lo["hefmi_fnirs"]["mean_r"], lo["hefmi_fnirs"]["subject_mean_baseline_r"]),
             ("HEFMI\nsame person", ws["hefmi_fnirs"]["pearson"], 0.0213),
             ("ds002336\nnew people", lo["ds002336_fmri"]["mean_r"], lo["ds002336_fmri"]["subject_mean_baseline_r"])]
    x = np.arange(3); w = 0.36
    a.bar(x - w / 2, [i[1] for i in items], w, color=BLUE, label="model")
    a.bar(x + w / 2, [i[2] for i in items], w, color=LIGHT, label="reference")
    for xi, (_, m, b) in zip(x, items):
        a.text(xi - w / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=7.8)
        a.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=7.8, color=MUTED)
    a.set_xticks(x); a.set_xticklabels([i[0] for i in items], fontsize=8.3); a.legend(fontsize=8, loc="upper left")
    a.set_ylim(0, 0.22); a.grid(axis="x", visible=False); a.set_title("LOSO and within-person", fontsize=9.5)
    a = ax[1]
    fr = [float(k) * 100 for k in pc]
    for key, lab, col, ls in [("global_unadapted", "general", GRAY, "-"), ("calibration_mean_baseline", "own mean, no EEG", AQUA, "--"),
                              ("personalized_residual_ridge", "personalised + EEG", BLUE, "-")]:
        a.plot(fr, [pc[k][key]["mean_subject_r"] for k in pc], color=col, ls=ls, lw=2, marker="o", ms=5, label=lab)
    a.set_xticks(fr); a.set_xticklabels([f"{int(v)}%" for v in fr]); a.set_xlim(5, 45); a.set_ylim(0, 0.26)
    a.set_xlabel("calibration data"); a.legend(fontsize=7.8, loc="center"); a.set_title("Personalisation", fontsize=9.5)
    D.save(fig, "f10_maisha")


for fn in [f_timeline, f_quality, f_cv, f_claim1, f_claim2, f_claim3, f_claim4, f_claim5, f_clinical, f_maisha]:
    fn()

c1 = S6["claim1"]; cs = csv("controls_summary.csv"); icc = csv("delay_model_free_icc.csv").iloc[0]
c5 = S6["claim5"]; cl = csv("clinical_comparison.csv").set_index("measure")
body = title_block("NeuroVascNet course · Lecture F", "The Results",
                   "What happened, claim by claim, in both pipelines — and why each result came out the way it did",
                   "Part F of 7 · Builds on Lectures A–E · About 100 minutes of reading") + f"""
<div class="box">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>tell the project's story in four phases, and explain how the interpretation changed;</li>
<li>state each main result with its key number, its control and its verdict;</li>
<li>give the most likely <i>reasons</i> each claim succeeded or failed;</li>
<li>explain what Maisha's independent pipeline added and where the two agree;</li>
<li>summarise what the evidence, taken together, supports.</li>
</ul>
</div>

<h2>1. The story in four phases</h2>
{fig("f1_timeline", 1, "The project's four phases.", "Each phase changed what the team believed. The most important turn was Phase 2, when the explanation for poor results moved from 'EEG cannot do this' to 'the training target cannot support it'.")}

<h2>2. Phase 1: early versions and a misleading conclusion</h2>
<p>The first full pipeline (v3) trained the deep model on HEFMI fNIRS and ds002336 fMRI, with global signal regression and per-window z-scoring of targets. Across held-out people, prediction of the full waveform was essentially <b>zero</b> for both modalities. Three fixes were tried, and none changed it:</p>
{table(1, "Fixes tried in Phase 1.",
       ["Fix", "Idea", "Result"],
       [["Remove or strengthen cross-modal alignment", "make the latent spaces of the two datasets match", "no change"],
        ["Rebalance training between datasets (capped sampling)", "fMRI was being shown far more often than fNIRS", "less overfitting (fMRI train–validation gap 0.40 → 0.31) but validation still ≈ 0"],
        ["Deep CORAL domain adaptation", "align feature statistics of held-out people to training people", "fNIRS 0.064 → 0.065; fMRI −0.017 → −0.033"]],
       widths=["30%", "38%", "32%"])}
<p>A parallel line of simpler models (ridge regression, small MLPs) predicted <b>amplitude</b> rather than waveform and found more: within-person trial-level fNIRS r ≈ 0.33, cross-person fNIRS amplitude with CORAL r ≈ 0.25. However, these used older protocols and were never reproduced under the stricter evaluation of Lecture E, so they are not headline results.</p>
<p>At this point the team's working explanation was that EEG simply lacks the information — an "EEG information limit". That explanation turned out to be premature.</p>
{M("<p>Concluding that a signal is unpredictable without first checking that the target is reliable. 'We could not predict it' and 'it cannot be predicted' are very different claims; the second needs evidence about the target, not just about the model.</p>")}

<h2>3. Phase 2: the data audit changes the explanation</h2>
<p>A full audit of every recording (Lecture C) measured the targets themselves, before any model:</p>
{fig("f2_quality", 2, "Target quality, measured before modelling.", "Project data. Left: how well each target agrees with itself across halves of trials (orange = below 0.3). Right: the fNIRS HbO–HbR correlation; HEFMI is strongly positive, the systemic signature.")}
<ul>
<li>HEFMI's fNIRS target had HbO–HbR correlation <b>+0.92</b> and reliability <b>0.08</b>: mostly systemic signal, barely repeatable. Its best possible correlation is about 0.29.</li>
<li>The pipeline itself made things worse: reliability fell from 0.26 raw to 0.13 after processing, and GSR removed about 83% of the real response.</li>
<li>No preprocessing variant rescued HEFMI (best of 12: cross-subject r = 0.06).</li>
<li>TU Berlin hBCI, with raw light data, showed good optode coupling (0.90) and twice HEFMI's reliability; in the final pipeline its reliability is 0.50 with the neural HbO–HbR sign.</li>
</ul>
{K("<p>The early 'prediction ≈ 0' was mainly a statement about the <b>target</b>, not about EEG. The fix was a different data source, not a different model.</p>")}

<h2>4. Phase 3: the final pipeline (v6)</h2>
<h3>4.1 Prediction for new people</h3>
{fig("f3_cv", 3, "Cross-subject prediction, fold by fold.", "Project data. 'raw' is the uncorrected score; 'no EEG' is what subject identity alone achieves; 'EEG contribution' is the subject-demeaned score. For TU Berlin fNIRS the EEG contribution is about the size of the raw score; for ds002336 fMRI most of the raw score is identity.")}
<p><b>TU Berlin fNIRS: r = 0.158</b> on people never seen in training (p = 0.001), positive in all five folds (0.13–0.24), about 22% of the dataset's ceiling, and unchanged when also demeaned by task. This is the project's main positive result. <b>ds002336 fMRI: r = 0.064</b>: statistically present but small; the raw 0.23 is almost entirely the person shortcut.</p>

<h3>4.2 Claim 1 — does fMRI supervision help? <span class="no">Not supported</span></h3>
{fig("f4_claim1", 4, "Claim 1 and its follow-up tests.", "Project data. Left: each held-out person's correlation with fNIRS-only training (x) and joint training (y); points below the line mean fMRI hurt. Middle: joint minus single as the fMRI loss weight changes. Right: the same as the shared bottleneck narrows. No setting gives a reliable benefit.")}
<p>Primary test: Δr = {c1['mean_diff']:+.3f}, 95% CI {c1['ci_low']:.3f} to {c1['ci_high']:.4f}, 1 of 5 people positive. Larger control (29 people): Δr = {cs.mean_diff[0]:+.3f} (p = {cs.p_signflip[0]:.2f}). Joint training with <b>shuffled</b> fMRI targets scores the same as with real ones (Δ = {cs.mean_diff[1]:+.3f}, p = {cs.p_signflip[1]:.2f}), and shuffled-joint is significantly <i>worse</i> than fNIRS alone (p = {cs.p_signflip[2]:.3f}).</p>
<p><b>Why, most likely?</b> The fMRI arm offers little to learn from: only 10 people and 384 trials, a target the model itself can barely predict (0.064), recorded from different people doing a different study (the subject gap). The shuffled control shows the network does not use the fMRI <i>content</i>; the extra data mostly compete for the shared trunk's capacity — a case of negative transfer. Weighting and bottleneck sweeps rule out the two obvious mechanical explanations.</p>

<h3>4.3 Claim 2 — does it work on another dataset? <span class="no">Essentially fails</span></h3>
{fig("f5_claim2", 5, "Claim 2: external validation.", "Project data. Left: internal (new people, same dataset) versus external (new dataset). Right: recalibrating with 2–8 HEFMI people does not recover performance.")}
<p>HEFMI: r = 0.012 (significant but 4% of its ceiling; 92% drop). ds002739: r = 0.000. <b>Why?</b> Three things change at once — montage, task (ds002739 is not motor; TU Berlin pooled two tasks), and target quality (HEFMI is systemic-dominated) — and the fNIRS comparison had to be pooled to mean HbO/HbR. The current data cannot separate these causes.</p>

<h3>4.4 Claim 3 — neurovascular delay <span class="mid">Holds for the measured delay only</span></h3>
{fig("f6_claim3", 6, "Claim 3: measured versus learned delay.", "Project data. Left: the EEG-to-HbO delay measured directly for each person. Middle: the same measurement in two sessions. Right: the delay the network learned under three settings, against the measured value (dashed).")}
<p>The <b>model-free</b> delay is {icc.mean_s:.1f} s on average, differs between people (SD {icc.between_subject_sd:.1f} s) and is consistent across sessions (ICC {icc.icc:.2f}, p = 0.0005 against surrogates): a candidate individual biomarker. The <b>learned</b> delay sits at the lower edge of whatever window it is given and collapses near zero when unbounded, with almost no variation between people. <b>Why?</b> The bounded sigmoid and the clamped training target make the output circular; and with weak waveform prediction there is little timing information for the head to learn from.</p>

<h3>4.5 Claim 4 — a better EEG representation? <span class="no">Fails</span></h3>
{fig("f7_claim4", 7, "Claim 4: downstream task decoding.", "Project data, 4-class task (chance 0.25). Left: a frozen encoder trained with haemodynamic targets (blue) performs like random-weight and shuffled-target encoders; full fine-tuning helps equally from either start. Right: probing each depth of the trunk gives flat accuracy.")}
<p>Frozen hemodynamic encoder 0.374 vs shuffled-target 0.362 vs random weights 0.381; raw band power 0.329; fine-tuning 0.474 from either start. <b>Why?</b> The supervision signal is weak (r ≈ 0.16), so it barely shapes the trunk; random convolutional features already capture band power, which is most of what the task needs.</p>

<h3>4.6 Claim 5 — reliability <span class="mid">Calibrates, does not rank</span></h3>
{fig("f8_claim5", 8, "Claim 5: reliability.", "Project data. Left: correlation on the trials the system chooses to answer barely changes as it abstains more. Right: Platt scaling (fitted and judged on separate halves) fixes calibration.")}
<p>ECE {c5['ece_before']:.2f} → {c5['ece_after']:.2f} after Platt scaling, but uncertainty and error correlate at only {c5['corr_unc_err']:.2f}, and subject-level scores point the wrong way. <b>Why?</b> When predictions are weak everywhere, there are few systematically "good" trials for uncertainty to find; calibration can still be fixed by rescaling, which is why the two properties diverge.</p>

<h3>4.7 Supporting result: personal models</h3>
<p>Models trained and tested inside one person average r = {S6['within_subject']['within_r']:.3f} — positive for 27 of 29 people but <i>below</i> the cross-subject 0.158. With about 100 training trials per person, pooling across people is worth more than personalising. On TU Berlin, the bottleneck is data per person, not the target.</p>

<h3>4.8 Clinical comparison</h3>
{fig("f9_clinical", 9, "ICH patients versus controls.", "Project data (HEFMI). One dot per person; black lines are group means.")}
<p>Of four primary coupling measures, only <b>HbO–HbR correlation</b> differs: 0.81 in patients vs 0.64 in controls (Holm-corrected p = {cl.loc['hbo_hbr','p_holm']:.3f}). Coupling strength and delay do not differ, and motion, amplitudes and trial counts are similar. <b>Caution:</b> haemorrhage changes head optics, and a high HbO–HbR correlation is also the systemic signature, so this is a candidate marker that needs an optical control.</p>

<h2>5. Phase 4: Maisha's independent pipeline</h2>
<p>Maisha rebuilt the pipeline independently, keeping HEFMI as the fNIRS training set and adding stricter safeguards (Lectures C–E). One of five planned seeds has been run.</p>
{fig("f10_maisha", 10, "Maisha's main results.", "Project data (seed 1337). Left: on HEFMI, prediction for new people equals a subject-mean baseline, while within-person coupling clearly beats its shuffled null; on ds002336 the model is below its baseline. Right: personalisation raises r, but the person's own mean without EEG gets most of the way.")}
<ul>
<li><b>New people, HEFMI (LOSO, 30 people):</b> r = 0.026 vs subject-mean baseline 0.029 (p = 0.70) — a well-powered null on this target.</li>
<li><b>Same person, HEFMI:</b> r = 0.094 vs null 0.021 (p = 0.005) — real, person-specific coupling.</li>
<li><b>Cross-modal vs fNIRS-only:</b> fNIRS-only better (0.071 vs 0.024; decoding 0.502 vs 0.456).</li>
<li><b>External:</b> TU Berlin −0.002, ds002739 0.009 (significant, negligible).</li>
<li><b>Reliability:</b> ranks on internal test people (0.46) but reverses externally (−0.20); the locked threshold answers only 0.9% of external windows.</li>
<li><b>Learned delay:</b> 5.00 s for everyone (zero spread); disabled in favour of a fixed value.</li>
<li><b>Personalisation:</b> 10% of a new person's data lifts r from 0.03 to 0.20, but the person's own mean alone reaches 0.17; EEG adds about 0.03 (7 people, not significant).</li>
<li><b>Protocol findings:</b> the alphabetical validation split contained only patients; timing checks failed, so alignment-based claims were blocked.</li>
</ul>

<h2>6. Putting it together</h2>
{table(2, "What the two pipelines agree on.",
       ["Question", "v6 (TU Berlin trains)", "Maisha (HEFMI trains)", "Conclusion"],
       [["Does fMRI supervision help?", "no; shuffled = real", "no; fNIRS-only better", "<span class='no'>replicated negative</span>"],
        ["Prediction for new people", "yes, r = 0.158", "no, = baseline", "depends on target quality"],
        ["Same-person coupling", "0.098 (27/29 positive)", "0.094 vs null 0.021", "<span class='yes'>replicated positive</span>"],
        ["Other datasets", "≈ 0", "≈ 0", "<span class='no'>replicated negative</span>"],
        ["Learned delay", "follows bounds", "constant 5 s", "<span class='no'>replicated negative</span>"],
        ["Reliability ranking", "fails", "internal only", "<span class='mid'>both fail externally</span>"]],
       widths=["24%", "24%", "24%", "28%"])}
{K("<p>The evidence supports four statements. (1) EEG carries some information about fNIRS responses in new people when the target is reliable. (2) fMRI supervision from different people does not help, at this data scale. (3) EEG–haemodynamic coupling is real but strongly person-specific, and does not transfer across datasets. (4) The neurovascular delay is best measured directly, where it is a stable trait.</p>")}
{V("<p><b>“So did the project fail?”</b> The central hypothesis was not supported, but the question was answered convincingly: by two independent pipelines, with controls, against measured ceilings. It also produced positive results (cross-subject fNIRS prediction, a reliable delay measure, a candidate clinical marker) and methodological ones (target-reliability checks, an 88% protocol inflation). A clear negative answer to a well-posed question is a result.</p>")}

{Q(["Why did the Phase 1 conclusion ('EEG information limit') turn out to be premature?",
    "Which single control most directly shows the network does not use the fMRI content?",
    "Give two reasons fMRI supervision might fail to help here.",
    "Why can't the external failure be pinned on one cause?",
    "How can the measured delay be a reliable trait while the learned delay is useless?",
    "Why is a frozen encoder trained on blood targets no better than a random one?",
    "Personal models scored below the cross-subject model on TU Berlin. What does that tell you?",
    "Why is the ICH HbO–HbR difference only a 'candidate' marker?",
    "On HEFMI, Maisha found within-person coupling but no cross-person prediction. What does this pattern mean?"])}

<h2>Summary</h2>
<ul>
<li>Phase 1 found nothing and blamed EEG; Phase 2 found the target was the problem; Phase 3 found r = 0.158 on a reliable target; Phase 4 independently confirmed the main negatives.</li>
<li>Claim 1 not supported (shuffled fMRI = real fMRI); Claim 2 fails (montage, task and target confounded); Claim 3 holds for the model-free delay only; Claim 4 fails (random encoder equally good); Claim 5 calibrates but does not rank.</li>
<li>Same-person coupling is real in both pipelines; personalisation gains come mostly from the person's own mean.</li>
<li>One clinical difference survives correction but needs an optical control.</li>
</ul>
<h3>Coming next: Lecture G</h3>
<p>Lecture G covers what to do with these results: limitations, how to frame and defend them in a paper and a viva, and the most valuable next steps.</p>

<h2>Answers to "Pause and think"</h2>
<ol class="answers">
<li>It was drawn without checking the target. The audit later showed HEFMI's target was systemic-dominated and unreliable (ceiling ≈ 0.29), so poor prediction said little about EEG.</li>
<li>Joint training with shuffled fMRI targets scoring the same as with real fMRI targets.</li>
<li>Any two of: very small fMRI data (10 people), a weakly predictable fMRI target (0.064), different people from the fNIRS data (subject gap), a different study and task, and competition for shared capacity (negative transfer).</li>
<li>Montage, task and target quality all differ between training and external data at the same time, and no dataset varies only one of them.</li>
<li>The measured delay is computed directly from each person's data; the learned delay is forced into a window by a sigmoid and trained towards a clamped target, so it reports its bounds rather than the data.</li>
<li>The blood-signal supervision is weak (r ≈ 0.16), so it hardly changes the features; random convolutions with pooling already capture band power, which drives task decoding.</li>
<li>With ~100 trials per person there is not enough data to fit a personal model; pooling many people helps more. The limit there is data per person, not the target.</li>
<li>Haemorrhage alters optical properties, and high HbO–HbR correlation is also the signature of systemic signal; without an optical control the difference may not be neurovascular.</li>
<li>The coupling exists inside each person but differs between people, so a model from others does not transfer; it is a transfer limit, not an absence of signal.</li>
</ol>
"""

D.render(body, "Lecture F: The Results")
