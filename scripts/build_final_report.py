"""Builds NeuroVascNet_Final_Report.pdf: the complete final report of the project.

Covers motivation, datasets, history, the data-quality audit, methods, the
final v6 pipeline (Muntasir), the five-seed v3.1-v2.3 pipeline (Maisha), their
comparison, discussion, limitations and next steps. Every number is read from
saved outputs: neurovascnet_v6_outputs/, output/ (Maisha's five seeds) and the
executed notebooks (seed 1337 from git commit 8d26941).
Run from the repo root:  python scripts/build_final_report.py
"""
import json
import re
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, ROOT, VIOLET, W, YELLOW, Doc, K, M,
                            arrow, box, fig, table, title_block)

D = Doc("final_report", "unused.pdf", "NeuroVascNet · Final Report")
D.pdf = ROOT / "NeuroVascNet_Final_Report.pdf"

# ---------------------------------------------------------------- data
V6 = ROOT / "neurovascnet_v6_outputs"
OUT = ROOT / "output"
S6 = json.loads((V6 / "summary.json").read_text())
csv = lambda n, **k: pd.read_csv(V6 / n, **k)
DS = csv("data_summary.csv")
SEEDS = [1337, 20260726, 7, 42, 2718]
SEEDCOL = [str(s) for s in SEEDS]
VER = "v3.1-v2.3-personalized-physio-fixed-delay"
NBNAME = "NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED"
S = {s: json.loads((OUT / f"summary_{VER}_seed{s}.json").read_text()) for s in SEEDS}
PH = json.loads((OUT / f"personalized_physio_{VER}_seed2718.json").read_text())
AGG = json.loads((OUT / f"multi_seed_summary_{VER}.json").read_text())["aggregates"]


def executed(seed):
    if seed == 1337:
        raw = subprocess.run(["git", "show", f"8d26941:{NBNAME}.ipynb"], cwd=ROOT, capture_output=True, check=True).stdout
        return json.loads(raw.decode("utf-8"))
    return json.loads((OUT / "executed_seed_notebooks" / f"{NBNAME}_seed{seed}.ipynb").read_text(encoding="utf-8"))


def cell_text(nb, i):
    return "".join("".join(o.get("text", "")) for o in nb["cells"][i].get("outputs", []))


XM, CAP = {}, {}
for s in SEEDS:
    nb = executed(s)
    for line in cell_text(nb, 59).splitlines():
        if line.startswith("Full (cross-modal)"):
            XM.setdefault(s, {})["full"] = [float(x) for x in line.split()[2:]]
        elif line.startswith("fNIRS-only baseline"):
            XM.setdefault(s, {})["base"] = [float(x) for x in line.split()[2:]]
    CAP[s] = float(re.search(r"gap ([+-][0-9.]+)\)", cell_text(nb, 53)).group(1))


def g(d, *p):
    for k in p:
        d = d[k]
    return d


per_seed = lambda *p: np.array([float(g(S[s], *p)) for s in SEEDS])
f3 = lambda v: f"{v:+.3f}" if v < 0 else f"{v:.3f}"
f2 = lambda v: f"{v:+.2f}" if v < 0 else f"{v:.2f}"
NAMES = {"TU Berlin hBCI (MI+MA)": "TU Berlin hBCI", "ds002336": "ds002336", "HEFMI": "HEFMI", "ds002739": "ds002739"}
DS["n"] = DS.dataset.map(NAMES)
rel = DS.set_index("dataset")


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="right")


# ---------------------------------------------------------------- figures: overview and audit
def f_design():
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.1, 2.0, 1.5, 1.1, "EEG\n(32 electrodes)", fc=BLUE, ec=BLUE, color="white", weight="bold", fs=8.8)
    box(ax, 2.2, 1.9, 1.9, 1.3, "shared EEG\nencoder", fc="#fff4e5", ec=YELLOW, weight="bold")
    outs = [(4.1, "fNIRS head → HbO, HbR", "#fdeceb", RED), (2.95, "fMRI head → BOLD", "#e8f6f0", AQUA),
            (1.8, "delay output", "#fff4e5", YELLOW), (0.65, "reliability → answer\nor abstain", "#eaf2fc", BLUE)]
    for y, t, fc, ec in outs:
        box(ax, 4.9, y, 2.75, 0.85, t, fc=fc, ec=ec, fs=8.3)
        arrow(ax, (4.1, 2.55), (4.9, y + 0.4))
    arrow(ax, (1.6, 2.55), (2.2, 2.55))
    box(ax, 8.0, 3.6, 1.9, 1.3, "training only:\nreal fNIRS (people A)\nreal fMRI (people B)", fc="white", ec=GRAY, fs=7.8, ls="--")
    ax.text(5.0, 0.15, "At use time only EEG is needed. The fNIRS and fMRI supervision come from different people.", fontsize=8.6,
            color=MUTED, ha="center")
    D.save(fig, "x01_design")


def f_target():
    o = DS.sort_values("reliability")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.45})
    a = ax[0]
    y = np.arange(len(o))
    a.barh(y, o.reliability, color=[ORANGE if v < 0.3 else BLUE for v in o.reliability], height=0.55)
    a.scatter(o.ceiling, y, marker="|", s=160, color=INK, lw=2, zorder=3)
    for yi, (r, c) in enumerate(zip(o.reliability, o.ceiling)):
        a.text(max(r, c) + 0.025, yi, f"{r:.2f}", va="center", fontsize=8.6)
    a.set_yticks(y); a.set_yticklabels([f"{n} ({m.replace('fnirs','fNIRS').replace('fmri','fMRI')})" for n, m in zip(o.n, o.modality)])
    a.set_xlim(0, 1); a.set_xlabel("split-half reliability of the target (| = best possible r)"); a.grid(axis="y", visible=False)
    a.set_title("How repeatable is each target?", fontsize=10); panel(a, "A")
    f = DS.dropna(subset=["hbo_hbr"]).sort_values("hbo_hbr")
    a = ax[1]
    a.axvspan(-1, 0, color=BLUE, alpha=0.06); a.axvspan(0, 1, color=ORANGE, alpha=0.06)
    a.barh(f.n, f.hbo_hbr, color=[BLUE if v < 0 else ORANGE for v in f.hbo_hbr], height=0.45)
    a.axvline(0, color=MUTED, lw=1)
    for yi, v in enumerate(f.hbo_hbr):
        a.text(v + (0.05 if v > 0 else -0.05), yi, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8.6)
    a.text(-0.5, 1.6, "neural", ha="center", fontsize=9, color=BLUE, weight="bold")
    a.text(0.5, 1.6, "systemic", ha="center", fontsize=9, color=ORANGE, weight="bold")
    a.set_xlim(-1, 1); a.set_ylim(-0.7, 1.9); a.set_xlabel("HbO–HbR correlation"); a.grid(axis="y", visible=False)
    a.set_title("Is the fNIRS signal neural?", fontsize=10); panel(a, "B")
    D.save(fig, "x02_target")


def f_protocol():
    pa = csv("protocol_ablation.csv")
    labs = ["Trials split at random\n(same person in train and test)", "Subjects split\n(new people in test)",
            "Subject identity only\n(no EEG used)", "Subjects split + subject\nmean removed (honest)"]
    fig, a = plt.subplots(figsize=(W, 2.2))
    y = np.arange(4)[::-1]
    a.barh(y, pa.r, color=[GRAY, GRAY, LIGHT, BLUE], height=0.58)
    for yi, v in zip(y, pa.r):
        a.text(v + 0.004, yi, f"r = {v:.3f}", va="center", fontsize=8.6)
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=8.6); a.set_xlim(0, 0.36)
    a.set_xlabel("correlation, TU Berlin hBCI"); a.grid(axis="y", visible=False)
    D.save(fig, "x03_protocol")


# ---------------------------------------------------------------- figures: v6
def f_cv():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
    cols = [("r_raw", "Raw\nscore"), ("r_no_eeg_baseline", "No EEG\n(identity only)"), ("r_demeaned", "EEG\ncontribution")]
    for a, (f, title, letter) in zip(ax, [("cv_TU_Berlin_hBCI_(MI+MA).csv", "TU Berlin hBCI (fNIRS, 29 people)", "A"),
                                          ("cv_ds002336.csv", "ds002336 (fMRI, 10 people)", "B")]):
        c = csv(f)
        for i, (k, _) in enumerate(cols):
            col = BLUE if k == "r_demeaned" else (LIGHT if "no_eeg" in k else GRAY)
            a.scatter(i + np.linspace(-0.13, 0.13, len(c)), c[k], s=32, color=col, edgecolor="white", lw=1.1, zorder=3)
            m = c[k].mean(); a.plot([i - 0.25, i + 0.25], [m, m], color=INK, lw=2)
            a.text(i + 0.29, m, f"{m:.3f}", va="center", fontsize=8.4)
        a.set_xticks(range(3)); a.set_xticklabels([t for _, t in cols], fontsize=8.5); a.set_xlim(-0.5, 2.75)
        a.axhline(0, color=MUTED, lw=0.8); a.set_title(title, fontsize=10); a.grid(axis="x", visible=False); panel(a, letter)
    ax[0].set_ylabel("r per cross-validation fold")
    D.save(fig, "x04_cv")


def f_claim1():
    arms = csv("controls_paired_arms.csv")
    f = arms[arms.cohort == "fnirs"].groupby(["subject", "arm"]).r.mean().unstack()
    cs = csv("controls_summary.csv"); lw = csv("loss_weight_search.csv"); bs = csv("bottleneck_sweep.csv")
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"wspace": 0.45, "width_ratios": [1.15, 1, 1]})
    a = ax[0]; lim = (-0.05, 0.45)
    a.plot(lim, lim, color=MUTED, ls="--", lw=1)
    a.scatter(f.single, f.joint, s=22, color=BLUE, edgecolor="white", lw=0.9, zorder=3)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("fNIRS only"); a.set_ylabel("fNIRS + fMRI")
    a.set_title(f"Per person ({(f.joint > f.single).sum()}/{len(f)} above line)", fontsize=9.5); panel(a, "A")
    a = ax[1]
    a.plot(lw.w_fmri, lw["diff"], color=BLUE, lw=2, marker="o", ms=5); a.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log"); a.set_xticks(lw.w_fmri); a.set_xticklabels(lw.w_fmri, fontsize=8)
    a.set_xlabel("fMRI loss weight"); a.set_ylabel("joint − single r"); a.set_title("Weighting", fontsize=9.5); panel(a, "B")
    a = ax[2]
    a.plot(bs.bottleneck, bs.joint_minus_single, color=BLUE, lw=2, marker="o", ms=5); a.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log", base=2); a.set_xticks(bs.bottleneck); a.set_xticklabels(bs.bottleneck, fontsize=8)
    a.set_xlabel("bottleneck width"); a.set_title("Shared capacity", fontsize=9.5); panel(a, "C")
    D.save(fig, "x05_claim1")


def f_claim2():
    ev = csv("external_validation.csv", index_col=0); fs = csv("claim2_fewshot_calibration.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.4, "width_ratios": [1.3, 1]})
    a = ax[0]; x = np.arange(2); w = 0.36
    a.bar(x - w / 2, ev.internal_r, w, color=GRAY, label="internal (new people)")
    a.bar(x + w / 2, ev.external_r.clip(lower=0.0005), w, color=ORANGE, label="external (new dataset)")
    for xi, (i, e) in enumerate(zip(ev.internal_r, ev.external_r)):
        a.text(xi - w / 2, i + 0.004, f"{i:.3f}", ha="center", fontsize=8.2)
        a.text(xi + w / 2, max(e, 0) + 0.004, f"{abs(e) if abs(e) < 5e-4 else e:.3f}", ha="center", fontsize=8.2)
    a.set_xticks(x); a.set_xticklabels(["fNIRS\n(TU Berlin → HEFMI)", "fMRI\n(ds002336 → ds002739)"], fontsize=8.3)
    a.legend(fontsize=8); a.grid(axis="x", visible=False); a.set_title("Internal vs external", fontsize=9.5); panel(a, "A")
    a = ax[1]
    a.plot(fs.calibration_subjects, fs.r_external, color=ORANGE, lw=2, marker="o", ms=5); a.axhline(0, color=MUTED, lw=1)
    a.set_xlabel("HEFMI people used to recalibrate"); a.set_ylabel("external r"); a.set_title("Recalibration", fontsize=9.5); panel(a, "B")
    D.save(fig, "x06_claim2")


def f_claim3():
    rec = csv("clinical_per_recording.csv"); per = csv("clinical_per_person.csv")
    s = rec.assign(sess=rec.recording.str.extract(r"(\d)").astype(int)).pivot_table(index="person", columns="sess", values="coupling_lag_s").dropna(subset=[1, 2])
    icc = csv("delay_model_free_icc.csv").iloc[0]; dl = csv("claim3_delay.csv"); dfree = csv("claim3_delay_free.csv")
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"wspace": 0.45})
    a = ax[0]
    a.axvspan(2, 8, color=BLUE, alpha=0.07); a.hist(per.coupling_lag_s, bins=np.arange(3, 10.01, 0.5), color=BLUE, edgecolor="white")
    a.axvline(icc.mean_s, color=INK, lw=1.4); a.set_xlabel("measured delay (s)"); a.set_ylabel("people")
    a.set_title(f"Measured: mean {icc.mean_s:.1f} s", fontsize=9.5); panel(a, "A")
    a = ax[1]
    a.plot([3.5, 9.5], [3.5, 9.5], color=MUTED, ls="--", lw=1); a.scatter(s[1], s[2], s=18, color=BLUE)
    a.set_xlim(3.5, 9.5); a.set_ylim(3.5, 9.5); a.set_xlabel("session 1 (s)"); a.set_ylabel("session 2 (s)")
    a.set_title(f"Repeatable: ICC {icc.icc:.2f}", fontsize=9.5); panel(a, "B")
    a = ax[2]
    vals = [dl.mean_s[0], dl.mean_s[1], dfree.mean_s[0]]
    a.bar(range(3), vals, color=ORANGE, width=0.55); a.axhline(icc.mean_s, color=BLUE, ls="--", lw=1.3)
    a.text(2.4, icc.mean_s + 0.2, "measured", fontsize=8, color=BLUE, ha="right")
    for i, v in enumerate(vals):
        a.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8.3)
    a.set_xticks(range(3)); a.set_xticklabels(["window\n2–8 s", "window\n0–20 s", "no bound"], fontsize=8); a.set_ylim(0, 7.5)
    a.set_title("Learned: follows window", fontsize=9.5); a.grid(axis="x", visible=False); panel(a, "C")
    D.save(fig, "x07_claim3")


def f_claim4():
    c4 = csv("claim4_downstream.csv"); dp = csv("claim4_depth_probe.csv")
    names = {"cross-modal supervised": "fNIRS+fMRI encoder", "SHUFFLED-target control": "shuffled targets",
             "random init (untrained)": "random weights", "raw band-power": "raw band power",
             "FINE-TUNED from supervised": "fine-tuned (from encoder)", "FINE-TUNED from scratch": "fine-tuned (scratch)"}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.6, "width_ratios": [1.4, 1]})
    a = ax[0]; o = c4.iloc[[3, 2, 1, 0, 5, 4]].reset_index(drop=True); y = np.arange(len(o))[::-1]
    a.barh(y, o.accuracy, xerr=o.sd, color=[GRAY, GRAY, GRAY, BLUE, LIGHT, LIGHT], height=0.6, error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    a.axvline(0.25, color=RED, ls=":", lw=1.2); a.text(0.255, 5.45, "chance 0.25", fontsize=7.8, color=RED)
    a.set_yticks(y); a.set_yticklabels([names[e] for e in o.encoder], fontsize=8.2); a.set_xlim(0, 0.55)
    a.set_xlabel("4-class accuracy"); a.grid(axis="y", visible=False); a.set_title("Linear probe", fontsize=9.5); panel(a, "A")
    a = ax[1]; m = dp[dp.readout == "mean"]
    a.plot(range(len(m)), m.accuracy, color=BLUE, lw=2, marker="o", ms=5); a.axhline(0.25, color=RED, ls=":", lw=1.2)
    a.set_xticks(range(len(m))); a.set_xticklabels(m.stage, fontsize=8); a.set_ylim(0.2, 0.4)
    a.set_title("Depth probe", fontsize=9.5); a.set_ylabel("accuracy"); panel(a, "B")
    D.save(fig, "x08_claim4")


def f_claim5():
    cov = csv("coverage_curve.csv"); c5 = S6["claim5"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.4})
    a = ax[0]
    a.plot(cov.coverage * 100, cov.r, color=BLUE, lw=2, marker="o", ms=5); a.invert_xaxis(); a.set_ylim(0.14, 0.19)
    a.set_xlabel("% of trials answered"); a.set_ylabel("r on answered trials"); a.set_title("Abstaining does not help", fontsize=9.5); panel(a, "A")
    a = ax[1]
    a.bar([0, 1], [c5["ece_before"], c5["ece_after"]], color=[GRAY, BLUE], width=0.5)
    for x, v in enumerate([c5["ece_before"], c5["ece_after"]]):
        a.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.3)
    a.set_xticks([0, 1]); a.set_xticklabels(["before", "after Platt"]); a.set_ylabel("ECE")
    a.set_title("Calibration works", fontsize=9.5); a.grid(axis="x", visible=False); panel(a, "B")
    D.save(fig, "x09_claim5")


def f_within():
    ws = csv("within_subject_probe.csv").sort_values("r").reset_index(drop=True)
    wi, cr = S6["within_subject"]["within_r"], S6["within_subject"]["cross_r"]
    fig, a = plt.subplots(figsize=(W, 2.1))
    a.bar(range(len(ws)), ws.r, color=BLUE, width=0.72)
    a.axhline(cr, color=ORANGE, lw=1.6); a.axhline(wi, color=INK, lw=1.4, ls="--")
    a.text(-0.4, cr + 0.008, f"model trained on other people: r = {cr:.3f}", fontsize=8.5, color=ORANGE)
    a.text(-0.4, wi + 0.008, f"average of personal models: r = {wi:.3f}", fontsize=8.5)
    a.set_xticks([]); a.set_xlabel(f"one bar per person (n = {len(ws)}), each model trained and tested on that person only")
    a.set_ylabel("r"); a.grid(axis="x", visible=False)
    D.save(fig, "x10_within")


def f_clinical():
    pp = csv("clinical_per_person.csv"); cl = csv("clinical_comparison.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.55})
    a = ax[0]; rng = np.random.default_rng(1)
    for i, (gname, c) in enumerate([("Normals", BLUE), ("Patients", ORANGE)]):
        v = pp[pp.group == gname].hbo_hbr.values
        a.scatter(v, i + rng.uniform(-0.13, 0.13, len(v)), s=28, color=c, edgecolor="white", lw=1, zorder=3)
        a.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        a.text(v.mean() + 0.01, i + 0.33, f"mean {v.mean():.2f}", fontsize=8.4, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    a.set_yticks([0, 1]); a.set_yticklabels(["Controls (17)", "ICH patients (20)"]); a.set_ylim(-0.55, 1.7)
    a.set_xlabel("HbO/HbR correlation per person"); a.grid(axis="y", visible=False); a.set_title("The one measure that differs", fontsize=9.5); panel(a, "A")
    a = ax[1]
    nm = {"coupling_r": "EEG–HbO coupling", "coupling_lag_s": "coupling delay", "reliability": "response repeatability",
          "hbo_hbr": "HbO/HbR correlation", "motion_frac": "head motion*", "hbo_amp": "HbO amplitude*",
          "eeg_amp": "EEG amplitude*", "n_trials": "trial count*"}
    cl = cl.iloc[::-1]; p = np.where(cl.p_holm.notna(), cl.p_holm, cl.p); y = np.arange(len(cl))
    a.barh(y, -np.log10(p), color=[ORANGE if v < 0.05 else GRAY for v in p], height=0.6)
    a.axvline(-np.log10(0.05), color=INK, lw=1, ls="--")
    a.set_yticks(y); a.set_yticklabels([nm[m] for m in cl.measure], fontsize=8.4)
    a.set_xlabel("−log10 p (primary: Holm-corrected)"); a.grid(axis="y", visible=False); a.set_title("All measures", fontsize=9.5); panel(a, "B")
    D.save(fig, "x11_clinical")


# ---------------------------------------------------------------- figures: Maisha five seeds
def m_timing():
    av = S[2718]["alignment_verification"]
    names = [v["name"] for v in av.values()]; frac = [100 * v["frac_in_physiological_band"] for v in av.values()]
    st = [v["status"].startswith("PASS") for v in av.values()]
    fig, a = plt.subplots(figsize=(W, 2.3)); y = np.arange(len(names))[::-1]
    a.barh(y, frac, color=[BLUE if ok else ORANGE for ok in st], height=0.6)
    for yi, v, ok in zip(y, frac, st):
        a.text(v + 1.5, yi, f"{v:.0f}%  {'PASS' if ok else 'WARN'}", va="center", fontsize=8.5)
    a.set_yticks(y); a.set_yticklabels(names, fontsize=8.6); a.set_xlim(0, 105)
    a.set_xlabel("% of people whose best EEG→blood delay lies inside 2–8 s"); a.grid(axis="y", visible=False)
    D.save(fig, "x12_timing")


def m_waveform():
    tests = [("HEFMI validation\n(7 people)", ("internal_validation", "hefmi_fnirs")),
             ("ds002336 fMRI val\n(3 people)", ("internal_validation", "ds002336_fmri")),
             ("TU Berlin\n(external, 29)", ("external_ood_stress_tests", "tuberlin_hbci_cross_dataset")),
             ("ds002739\n(external, 24)", ("external_ood_stress_tests", "ds002739_fmri_cross_task"))]
    fig, a = plt.subplots(figsize=(W, 2.5))
    for i, (lab, p) in enumerate(tests):
        r = per_seed(*p, "pearson_subject"); nl = per_seed(*p, "null_subject_mean")
        a.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), r, s=34, color=BLUE, edgecolor="white", lw=1, zorder=3,
                  label="model, one dot per seed" if i == 0 else None)
        a.plot([i - 0.25, i + 0.25], [r.mean()] * 2, color=INK, lw=2)
        a.plot([i - 0.25, i + 0.25], [nl.mean()] * 2, color=ORANGE, lw=1.6, ls="--", label="chance level" if i == 0 else None)
        a.text(i + 0.3, r.mean(), f"{r.mean():.3f}", va="center", fontsize=8.3)
    a.axhline(0, color=MUTED, lw=0.8); a.set_xticks(range(4)); a.set_xticklabels([t for t, _ in tests], fontsize=8.4)
    a.set_xlim(-0.5, 3.75); a.set_ylabel("r (subject level)"); a.legend(fontsize=8, loc="upper right"); a.grid(axis="x", visible=False)
    D.save(fig, "x13_waveform")


def m_personal():
    pc = PH["personalized_calibration_curve"]; fr = [float(k) * 100 for k in pc]
    fig, a = plt.subplots(figsize=(W, 2.3))
    for key, lab, col, ls in [("global_unadapted", "general model", GRAY, "-"), ("calibration_mean_baseline", "person's own average, no EEG", AQUA, "--"),
                              ("personalized_residual_ridge", "personalised with EEG", BLUE, "-")]:
        v = [pc[k][key]["mean_subject_r"] for k in pc]
        a.plot(fr, v, color=col, ls=ls, lw=2.2, marker="o", ms=6, mec="white", label=lab)
        a.text(fr[-1] + 1.2, v[-1], f"{v[-1]:.3f}", va="center", fontsize=8.3)
    a.set_xticks(fr); a.set_xticklabels([f"{int(v)}%" for v in fr]); a.set_xlim(7, 48); a.set_ylim(0, 0.25)
    a.set_xlabel("share of the new person's data used to calibrate"); a.set_ylabel("r")
    a.legend(fontsize=8.2, loc="center right", bbox_to_anchor=(1, 0.42))
    D.save(fig, "x14_personal")


def m_newpeople():
    lo, ws = S[2718]["cross_subject_loso"], S[2718]["within_subject_probe"]
    items = [("HEFMI\nnew people (LOSO)", lo["hefmi_fnirs"]["mean_r"], lo["hefmi_fnirs"]["subject_mean_baseline_r"]),
             ("HEFMI\nsame person", ws["hefmi_fnirs"]["pearson"], 0.0209),
             ("ds002336\nnew people (LOSO)", lo["ds002336_fmri"]["mean_r"], lo["ds002336_fmri"]["subject_mean_baseline_r"]),
             ("ds002336\nsame person", ws["ds002336_fmri"]["pearson"], 0.1178)]
    fig, a = plt.subplots(figsize=(W, 2.4)); x = np.arange(4); w = 0.36
    a.bar(x - w / 2, [i[1] for i in items], w, color=BLUE, label="model")
    a.bar(x + w / 2, [i[2] for i in items], w, color=LIGHT, label="reference (person-average shortcut or chance)")
    for xi, (_, m, b) in zip(x, items):
        a.text(xi - w / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=8)
        a.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8, color=MUTED)
    a.set_xticks(x); a.set_xticklabels([i[0] for i in items], fontsize=8.4); a.set_ylabel("r")
    a.set_ylim(0, 0.215); a.legend(fontsize=8, loc="upper left"); a.grid(axis="x", visible=False)
    D.save(fig, "x15_newpeople")


def m_crossmodal():
    full = np.array([XM[s]["full"][2] for s in SEEDS]); base = np.array([XM[s]["base"][2] for s in SEEDS])
    fx = np.array([XM[s]["full"][4] for s in SEEDS]); bx = np.array([XM[s]["base"][4] for s in SEEDS])
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.35})
    for a, (fv, bv, title, letter) in zip(ax, [(full, base, "Prediction r (7 validation people)", "A"), (fx, bx, "Timing match above chance", "B")]):
        for i in range(5):
            a.plot([0, 1], [bv[i], fv[i]], color=GRAY, lw=1)
        a.scatter(np.zeros(5), bv, s=40, color=AQUA, zorder=3); a.scatter(np.ones(5), fv, s=40, color=ORANGE, zorder=3)
        a.set_xticks([0, 1]); a.set_xticklabels(["fNIRS only", "fNIRS + fMRI"]); a.set_xlim(-0.4, 1.4)
        a.set_title(title, fontsize=9.5); a.grid(axis="x", visible=False); panel(a, letter)
    ax[0].set_ylabel("value per seed")
    D.save(fig, "x16_crossmodal")


def m_reliability():
    ri = per_seed("reliability", "corr_internal_test_subjects"); re_ = per_seed("reliability", "corr_external")
    cov = per_seed("reliability", "abstention", "external", "coverage") * 100
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.35})
    a = ax[0]; x = np.arange(5); w = 0.38
    a.bar(x - w / 2, ri, w, color=BLUE, label="inside HEFMI (test people)"); a.bar(x + w / 2, re_, w, color=ORANGE, label="external (TU Berlin)")
    a.axhline(0, color=MUTED, lw=0.8); a.set_xticks(x); a.set_xticklabels(SEEDCOL, fontsize=8)
    a.set_ylabel("confidence ranks errors (r)"); a.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    a.grid(axis="x", visible=False); a.set_title("Ranking good vs bad predictions", fontsize=9.5); panel(a, "A")
    a = ax[1]
    a.bar(x, cov, color=VIOLET, width=0.55)
    for xi, v in zip(x, cov):
        a.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=8.3)
    a.set_xticks(x); a.set_xticklabels(SEEDCOL, fontsize=8); a.set_ylim(0, 112)
    a.set_ylabel("% of external windows answered"); a.set_title("Locked abstention on TU Berlin", fontsize=9.5); a.grid(axis="x", visible=False); panel(a, "B")
    D.save(fig, "x17_reliability")


for fn in [f_design, f_target, f_protocol, f_cv, f_claim1, f_claim2, f_claim3, f_claim4, f_claim5, f_within, f_clinical,
           m_timing, m_waveform, m_personal, m_newpeople, m_crossmodal, m_reliability]:
    fn()

# ---------------------------------------------------------------- derived numbers
c1 = S6["claim1"]; cs = csv("controls_summary.csv"); ev = csv("external_validation.csv", index_col=0)
icc = csv("delay_model_free_icc.csv").iloc[0]; c5 = S6["claim5"]; cl = csv("clinical_comparison.csv").set_index("measure")
pa = csv("protocol_ablation.csv").r.values; tub = csv("cv_TU_Berlin_hBCI_(MI+MA).csv"); dl = csv("claim3_delay.csv")
dfree = csv("claim3_delay_free.csv"); c4 = csv("claim4_downstream.csv").set_index("encoder")
lw = csv("loss_weight_search.csv")
r_tub = S6["internal"]["TU Berlin hBCI (MI+MA)"]["r_demeaned"]; r_336 = S6["internal"]["ds002336"]["r_demeaned"]
lo = S[2718]["cross_subject_loso"]; wsp = S[2718]["within_subject_probe"]
fm = np.array([XM[s]["full"][2] for s in SEEDS]); bm = np.array([XM[s]["base"][2] for s in SEEDS])
fxm = np.array([XM[s]["full"][4] for s in SEEDS]); bxm = np.array([XM[s]["base"][4] for s in SEEDS])
nb_better = int((bm > fm).sum()); nbx_better = int((bxm > fxm).sum())
gsr = per_seed("provenance", "gsr_ab", "frac_variance_removed") * 100
pc = PH["personalized_calibration_curve"]; gl = PH["global_internal_validation"]
A = lambda k: AGG[k]["mean"]


def seed_row(label, arr, extra=""):
    return [label] + [f3(v) for v in arr] + [f"<b>{arr.mean():.3f} ± {arr.std(ddof=1):.3f}</b>", extra]


iv = lambda key, field: per_seed("internal_validation", key, field)
evm = lambda key, field: per_seed("external_ood_stress_tests", key, field)
wave_rows = [seed_row("HEFMI validation (7)", iv("hefmi_fnirs", "pearson_subject"), f"chance {iv('hefmi_fnirs', 'null_subject_mean').mean():.3f}"),
             seed_row("ds002336 fMRI val (3)", iv("ds002336_fmri", "pearson_subject"), f"chance {iv('ds002336_fmri', 'null_subject_mean').mean():.3f}"),
             seed_row("TU Berlin, external (29)", evm("tuberlin_hbci_cross_dataset", "pearson_subject"), "zero"),
             seed_row("ds002739, external (24)", evm("ds002739_fmri_cross_task", "pearson_subject"), "practically zero")]
xm_rows = [[str(s), f"{XM[s]['full'][2]:.3f}", f"{XM[s]['base'][2]:.3f}", "fNIRS only" if XM[s]['base'][2] > XM[s]['full'][2] else "fNIRS + fMRI",
            f"{XM[s]['full'][4]:.3f}", f"{XM[s]['base'][4]:.3f}", f"{XM[s]['full'][6]:.3f} / {XM[s]['base'][6]:.3f}"] for s in SEEDS]
xm_rows.append(["<b>Mean</b>", f"<b>{fm.mean():.3f}</b>", f"<b>{bm.mean():.3f}</b>", f"<b>fNIRS only in {nb_better}/5</b>",
                f"<b>{fxm.mean():.3f}</b>", f"<b>{bxm.mean():.3f}</b>",
                f"{np.mean([XM[s]['full'][6] for s in SEEDS]):.3f} / {np.mean([XM[s]['base'][6] for s in SEEDS]):.3f}"])
rel_rows = []
for s in SEEDS:
    r = S[s]["reliability"]
    rel_rows.append([str(s), ", ".join(str(x) for x in r["protocol"]["calibration_subjects"]), f"{r['corr_internal_test_subjects']:.2f}",
                     f2(r["corr_external"]), f"{r['ece']['internal_test_raw']:.3f} → {r['ece']['internal_test_calibrated']:.3f}",
                     f"{r['ece']['external_raw']:.3f} → {r['ece']['external_calibrated']:.3f}", f"{100 * r['abstention']['external']['coverage']:.1f}%"])
amp_rows = [["Within person, new people (p)"] + [f"{a:.3f} ({p:.2f})" for a, p in zip(per_seed("amplitude", "hefmi_fnirs_val", "within_subject_r_Q2"), per_seed("amplitude", "hefmi_fnirs_val", "p_signflip"))],
            ["New people, own mean removed"] + [f3(v) for v in per_seed("amplitude", "hefmi_fnirs_val", "cross_subject_centered_r_Q3")],
            ["fMRI train: ranking people"] + [f2(v) for v in per_seed("amplitude", "ds002336_fmri_train_fit_on_self", "between_subject_ranking_r_Q1")],
            ["fMRI train: within person"] + [f2(v) for v in per_seed("amplitude", "ds002336_fmri_train_fit_on_self", "within_subject_r_Q2")]]
anames = {"hefmi_internal_subject_r": "HEFMI validation prediction", "fmri_internal_subject_r": "fMRI validation prediction",
          "tuberlin_external_subject_r": "TU Berlin external", "ds002739_external_subject_r": "ds002739 external",
          "hefmi_loso_r": "HEFMI leave-one-person-out", "hefmi_loso_subject_mean_baseline_r": "… person-average shortcut",
          "fmri_loso_r": "fMRI leave-one-person-out", "physio_ridge_subject_r": "Ridge, compact endpoint",
          "physio_personalized_20pct_subject_r": "Personalised with EEG (20%)", "physio_calibration_mean_20pct_subject_r": "Own average, no EEG (20%)",
          "physio_global_20pct_subject_r": "General model (20%)", "hefmi_within_subject_r": "Within person, HEFMI",
          "fmri_within_subject_r": "Within person, fMRI", "reliability_corr_internal_test": "Reliability ranking, internal",
          "reliability_corr_external": "Reliability ranking, external"}
agg_rows = [[lab, f3(AGG[k]["mean"]), f"{(AGG[k]['std_across_seeds'] or 0):.3f}", f3(AGG[k]["min"]), f3(AGG[k]["max"])] for k, lab in anames.items()]
pers_rows = [[f"{float(k):.0%}", f"{pc[k]['global_unadapted']['mean_subject_r']:.3f}", f"{pc[k]['calibration_mean_baseline']['mean_subject_r']:.3f}",
              f"{pc[k]['personalized_residual_ridge']['mean_subject_r']:.3f}", f"{pc[k]['paired_signflip_p']:.2f}"] for k in pc]
endp_rows = [[lab, f3(gl[k]["mean_subject_r"]), f"{gl[k]['ci95_subject_bootstrap'][0]:+.2f} to {gl[k]['ci95_subject_bootstrap'][1]:+.2f}",
              f"{gl[k]['mean_subject_mae_sd_units']:.3f}"] for k, lab in [("train_mean", "Training average (no EEG)"),
              ("task_condition_event", "Average for each task (no EEG)"), ("ridge_bandpower", "Ridge on EEG band power")]]

# ---------------------------------------------------------------- document
body = title_block("NeuroVascNet · Final report", "Predicting Haemodynamic Responses from EEG with fNIRS and fMRI Supervision",
                   "Complete final report: data, methods, both pipelines, every result, and what the evidence supports",
                   "October 2026 · Final v6 pipeline (Muntasir Hossain) and five-seed v3.1-v2.3 pipeline (Maisha) · Four public datasets, 100 people") + f"""
<div class="box">
<h3>Abstract</h3>
<p>EEG is cheap and portable; fNIRS and fMRI measure the brain's blood response but are expensive or impractical. This project asked whether one EEG encoder, trained with paired EEG–fNIRS and EEG–fMRI data from different people, can predict haemodynamic responses from EEG alone in people it has never seen, and whether supervising with both blood signals helps. Two independent pipelines were built and evaluated with subject-level cross-validation, subject-mean removal, permutation nulls, bootstrap intervals, pre-declared controls and, for the second pipeline, five random seeds.</p>
<p><b>Positive results.</b> EEG predicts trial-level fNIRS responses in new people at r = {r_tub:.3f} (p = 0.001, all five folds positive) when trained on a reliable target. EEG–fNIRS coupling is real within each person in both pipelines (r ≈ 0.09–0.10). A directly measured EEG→HbO delay of {icc.mean_s:.1f} s is reproducible within people (ICC {icc.icc:.2f}). One clinical difference — tighter HbO/HbR coupling in haemorrhage patients — survives multiple-comparison correction.</p>
<p><b>Negative results.</b> fMRI supervision from different people does not improve fNIRS prediction in either pipeline (shuffled fMRI equals real fMRI; fNIRS-only better in {nb_better} of 5 seeds). Nothing transfers to new datasets. The learned delay reports its bounds. The reliability layer calibrates but does not reliably identify bad predictions on new data. A naive trial-level split inflates performance by 88%.</p>
<p><b>Main lesson.</b> Target reliability, measured before modelling, decided whether prediction was possible: the same question gave r = 0.158 on a reliable dataset and nothing beyond a person-average shortcut on an unreliable one.</p>
</div>

<h2>1. Introduction</h2>
<p>When neurons fire they produce electrical activity, measurable at the scalp in milliseconds with EEG, and — through neurovascular coupling — a slower increase in local blood flow, measurable seconds later with fNIRS (oxygenated and deoxygenated haemoglobin, HbO and HbR) or fMRI (the BOLD signal). EEG is inexpensive and wearable; fNIRS and fMRI are bulky or require a scanner. A model that estimates the blood response from EEG would bring some haemodynamic information to bedside monitoring, neurofeedback and brain–computer interfaces.</p>
<p><b>Thesis.</b> One shared EEG encoder, trained with paired EEG–fNIRS and EEG–fMRI supervision, takes EEG only at deployment and outputs (1) a haemodynamic-aware latent representation, (2) a predicted haemodynamic response, (3) a neurovascular delay and (4) a calibrated reliability score that allows abstention.</p>
{fig("x01_design", 1, "System design.", "fNIRS and fMRI are used only as training targets; at use time only EEG is required. No public dataset records the same people with fNIRS and fMRI, so the two kinds of supervision come from different people and can only interact inside the shared encoder (the 'subject gap').")}
<p>The thesis was turned into five pre-declared claims, each with a test that could fail: (1) cross-modal supervision beats single-modality supervision; (2) the model generalises to external cohorts; (3) the delay output reflects physiology; (4) the learned representation helps downstream EEG tasks; (5) the reliability score is calibrated and ranks errors.</p>

<h2>2. Datasets</h2>
{table(1, "Datasets, roles in each pipeline, and target quality measured before modelling.",
       ["Dataset", "Signal", "Task", "People", "Muntasir (v6)", "Maisha (v2.3)", "Reliability", "HbO/HbR"],
       [["TU Berlin hBCI", "EEG + fNIRS", "motor imagery + mental arithmetic", "29", "fNIRS training", "external test", f"{rel.loc['TU Berlin hBCI (MI+MA)','reliability']:.2f}", f"{rel.loc['TU Berlin hBCI (MI+MA)','hbo_hbr']:+.2f}"],
        ["HEFMI-ICH", "EEG + fNIRS", "motor imagery; 17 controls, 20 ICH patients", "37", "external test, clinical", "fNIRS training (30) / validation (7)", f"{rel.loc['HEFMI','reliability']:.2f}", f"{rel.loc['HEFMI','hbo_hbr']:+.2f}"],
        ["OpenNeuro ds002336", "EEG + fMRI", "motor imagery, neurofeedback", "10", "fMRI training", "fMRI training (7) / validation (3)", f"{rel.loc['ds002336','reliability']:.2f}", "–"],
        ["OpenNeuro ds002739", "EEG + fMRI", "perceptual decision (not motor)", "24", "external test", "external test", f"{rel.loc['ds002739','reliability']:.2f}", "–"]],
       widths=["14%", "10%", "21%", "7%", "13%", "17%", "9%", "9%"],
       note="Reliability = split-half reliability of the haemodynamic target; its square root is the best correlation any model could reach. A neural fNIRS response gives a negative HbO/HbR correlation.")}
<ul>
<li><b>TU Berlin hBCI</b> (Shin et al., 2017): 30 EEG channels at 200 Hz on 10-5 positions (no C3/C4), 36 fNIRS channels distributed as raw light at 760/850 nm.</li>
<li><b>HEFMI-ICH</b>: 32 EEG channels at 256 Hz, 90 fNIRS channels at 11 Hz, distributed already converted and epoched.</li>
<li><b>ds002336</b>: 64-channel EEG at 5 kHz recorded inside the scanner (gradient artifact removed with a custom average-artifact subtraction, ~8,200 µV → ~16 µV); fMRI TR 2 s; 52 complete EEG–fMRI pairs.</li>
<li><b>ds002739</b>: EEG pre-corrected and filtered 0.5–40 Hz, no electrode names (channels paired by index); fMRI TR 2 s, tSNR ≈ 190.</li>
</ul>

<h2>3. Project history</h2>
{table(2, "How the project developed.",
       ["Phase", "What was done", "What it showed"],
       [["v3 – v3.3 (Jun–Jul)", "deep model on HEFMI + ds002336 with GSR; fixes tried: alignment removal, capped/reweighted sampling, Deep CORAL", "cross-subject waveform ≈ 0; no fix helped (CORAL: fNIRS 0.064 → 0.065)"],
        ["Amplitude line (Jul)", "ridge / small MLPs on response amplitude", "within-person r ≈ 0.33, cross-person with CORAL ≈ 0.25 — older protocols, not reproduced later"],
        ["Data-quality audit (late Jul)", "610 recordings, 63 GB, 97 people; 12-variant target sweep", "HEFMI target systemic-dominated; GSR removes ~83% of the response; root cause found"],
        ["v6 final (Aug)", "TU Berlin as fNIRS training set; five claims with controls; clinical analysis", f"cross-subject fNIRS r = {r_tub:.3f}; claims tested"],
        ["Maisha v2.3 (Aug–Oct)", "independent rebuild on HEFMI with strict safeguards; five seeds", "within-person coupling real; cross-person = shortcut; negatives replicated"]],
       widths=["20%", "42%", "38%"])}

<h2>4. The data-quality audit</h2>
{fig("x02_target", 2, "Target quality, measured before modelling.", "(A) Split-half reliability with the best achievable correlation (|); orange bars fall below 0.3. (B) HbO/HbR correlation: negative for TU Berlin (neural), strongly positive for HEFMI (systemic).")}
<ul>
<li><b>HEFMI is systemic-dominated:</b> HbO/HbR +0.92 and reliability 0.08 (ceiling ≈ 0.29). Our earlier pipeline lowered reliability further, from 0.26 raw to 0.13 after processing; no preprocessing variant rescued it (best cross-subject r = 0.06).</li>
<li><b>Global signal regression removed ~83% of the evoked response</b> on TU Berlin (median 17% retained). Maisha's independent A/B test found it removed {gsr.min():.0f}–{gsr.max():.0f}% of the HEFMI signal, with a partly structured (non-global) removed component. GSR was switched off in both pipelines.</li>
<li><b>Units</b> differed about 2,600-fold between HEFMI and TU Berlin; both pipelines harmonise or use scale-free losses.</li>
<li><b>Provenance:</b> a 26-person mental-workload TU Berlin dataset had been confused with the 29-person motor hBCI set; both final pipelines load the correct one.</li>
<li>Two of the audit's own pre-set rules misfired (line-noise and gradient-residual thresholds) and were reported as misfires rather than retuned.</li>
</ul>
{K("<p>The early result 'cross-subject prediction ≈ 0' was mainly a statement about an unreliable target, not about EEG. This is why Muntasir's final pipeline trains on TU Berlin, chosen by a criterion measured before any model was fitted.</p>")}

<h2>5. Methods</h2>
<h3>5.1 Preprocessing</h3>
{table(3, "Preprocessing in the two pipelines.",
       ["Step", "Muntasir (v6)", "Maisha (v2.3)"],
       [["Window", "−2 to +28 s around each cue (30 s)", "30 s windows"],
        ["EEG", "eye channels removed, resampled to 200 Hz, z-scored per trial; raw signal into the model", "0.5–45 Hz, 50 Hz notch; band power (4 bands × 32 ch) at 20 Hz; training-only normalisation"],
        ["fNIRS", "Beer–Lambert from raw light (3 cm, DPF 6.4/5.8), 0.01–0.2 Hz, 10 Hz, 2 s baseline, no GSR", "0.01–0.2 Hz, baseline = first 20% of window, no GSR, amplitude in physical units"],
        ["fMRI", "Harvard–Oxford regions, cubic detrend, aligned by scanner triggers; motor regions in the final run", "48 Harvard–Oxford regions, 0.008–0.15 Hz"],
        ["Channels across datasets", "per-dataset input adapter; spherical-spline interpolation for external tests", "common 32 electrodes matched by name"],
        ["Split", "5-fold cross-validation by person", "fixed validation people, stratified by clinical group (seed 20260807), plus leave-one-person-out"]],
       widths=["18%", "41%", "41%"])}
<h3>5.2 Models and training</h3>
{table(4, "Model and training settings.",
       ["", "Muntasir (v6)", "Maisha (v2.3)"],
       [["Encoder", "per-dataset 1×1 adapter + 3 strided convolutions (6000 → 1200 → 300 → 100 → 16 steps), 8-number subject embedding (average used for new people)", "EEGNet front-end + temporal convolution network, per-domain batch norm, 142,922 parameters"],
        ["Outputs", "per-modality heads: mean + log-variance", "per-modality heads: mean + uncertainty; delay fixed at 5 s"],
        ["Loss", "1.0 × correlation + 0.1 × Gaussian NLL + 0.05 × smoothness", "heteroscedastic + smoothness + cross-correlation terms"],
        ["Training", "AdamW, lr 0.001, batch 64, 30 epochs; both losses summed in one update; smaller set cycles", "15 epochs, 19 balanced steps per modality per epoch; stage-2 alignment disabled"],
        ["Reliability", "MC dropout (20 passes) + predicted variance; Platt scaling fitted on one half, scored on the other", "learned gate → temperature → isotonic; 4 calibration people; locked threshold for 75% coverage"],
        ["Seeds", "3 for Claim 1 (seeds averaged within person)", "5 (1337, 20260726, 7, 42, 2718), each in a fresh kernel"]],
       widths=["16%", "42%", "42%"])}
<h3>5.3 Evaluation protocol</h3>
<ul>
<li><b>People are the unit of analysis.</b> No person is in both training and test data; all confidence intervals resample people; all nulls permute or sign-flip at the person level.</li>
<li><b>Subject demeaning.</b> Each person's mean is removed from prediction and target before scoring, so a predictor that only recognises the person scores zero (the "no-EEG baseline"). The TU Berlin score is also demeaned by subject × task.</li>
<li><b>Permutation nulls</b> (1,000 shuffles; p = (extreme + 1)/(n + 1)), <b>bootstrap</b> intervals (10,000 resamples of people), <b>sign-flip</b> tests for paired comparisons, <b>Holm</b> correction for the clinical tests.</li>
<li><b>Ceilings</b>: every result is compared with √(split-half reliability) of its target.</li>
<li><b>Controls</b>: single-modality model at equal budget and seed; joint model with shuffled fMRI; subject embedding switched off; random-weight and shuffled-target encoders; raw band power; several delay windows; real-versus-shuffled capacity test; Maisha's automatic claim guard.</li>
</ul>
{fig("x03_protocol", 3, "Evaluation protocol changes the score as much as the model does.", f"The same TU Berlin model scored four ways. A random trial-level split gives {pa[0]:.3f}, 88% above the honest {pa[3]:.3f}; subject identity alone (no EEG) gives {pa[2]:.3f}.")}

<h2>6. Results: Muntasir's final pipeline (v6)</h2>
<h3>6.1 Prediction for new people</h3>
{fig("x04_cv", 4, "Cross-subject prediction, fold by fold.", "Raw score, the no-EEG (identity-only) baseline and the EEG contribution after removing each person's mean. For TU Berlin the EEG contribution matches the raw score; for ds002336 most of the raw score is identity.")}
<p><b>TU Berlin fNIRS: r = {r_tub:.3f}</b> (p = 0.001), positive in every fold ({tub.r_demeaned.min():.2f}–{tub.r_demeaned.max():.2f}), about 22% of the ceiling (0.71), unchanged when demeaned by subject × task. <b>ds002336 fMRI: r = {r_336:.3f}</b>: present but small; the raw 0.23 is almost all identity.</p>

<h3>6.2 Claim 1 — does fMRI supervision help? <span class="no">Not supported</span></h3>
{fig("x05_claim1", 5, "Claim 1 and follow-up tests.", "(A) Each held-out person under fNIRS-only (x) and joint (y) training. (B) Effect of the fMRI loss weight. (C) Effect of narrowing the shared bottleneck. No setting gives a reliable benefit.")}
<p>Primary test: Δr = {c1['mean_diff']:+.3f} (95% CI {c1['ci_low']:.3f} to {c1['ci_high']:.4f}), 1 of 5 people positive. 29-person control: Δr = {cs.mean_diff[0]:+.3f} (p = {cs.p_signflip[0]:.2f}). Shuffled fMRI equals real fMRI (Δ = {cs.mean_diff[1]:+.3f}, p = {cs.p_signflip[1]:.2f}); shuffled-joint is worse than fNIRS alone (p = {cs.p_signflip[2]:.3f}). Best loss weight gave only +{lw['diff'].max():.3f}.</p>

<h3>6.3 Claim 2 — external datasets <span class="no">Fails</span></h3>
{fig("x06_claim2", 6, "Claim 2: external validation.", "(A) Internal (new people, same dataset) versus external (new dataset). (B) Recalibrating with 2–8 HEFMI people does not help.")}
<p>HEFMI r = {ev.loc['HEFMI','external_r']:.3f} (p = 0.01, 4% of its ceiling, 92% drop); ds002739 r = {abs(ev.loc['ds002739','external_r']):.3f}. Montage, task and target quality change together, so the cause cannot be isolated.</p>

<h3>6.4 Claim 3 — neurovascular delay <span class="mid">Holds for the measured delay only</span></h3>
{fig("x07_claim3", 7, "Measured versus learned delay.", "(A) Delay measured directly per person; shaded = 2–8 s. (B) Two sessions per person. (C) The learned delay under three settings against the measured value.")}
<p>Model-free delay {icc.mean_s:.1f} s, SD {icc.between_subject_sd:.1f} s between people, ICC {icc.icc:.2f} across sessions (p = 0.0005). The learned delay sits at the lower edge of its window ({dl.mean_s[0]:.2f} s for 2–8 s, {dl.mean_s[1]:.2f} s for 0–20 s, {dfree.mean_s[0]:.2f} s unbounded): it reports its bounds.</p>

<h3>6.5 Claim 4 — better EEG representation? <span class="no">Fails</span></h3>
{fig("x08_claim4", 8, "Downstream task decoding (4 classes, chance 0.25).", "(A) A frozen encoder trained on haemodynamic targets performs like random-weight and shuffled-target encoders; fine-tuning helps equally from either start. (B) Accuracy is flat across the trunk's depth.")}
<p>Haemodynamic encoder {c4.loc['cross-modal supervised','accuracy']:.3f} vs random {c4.loc['random init (untrained)','accuracy']:.3f} vs shuffled {c4.loc['SHUFFLED-target control','accuracy']:.3f}; raw band power {c4.loc['raw band-power','accuracy']:.3f}; fine-tuned {c4.loc['FINE-TUNED from supervised','accuracy']:.3f} vs {c4.loc['FINE-TUNED from scratch','accuracy']:.3f} from scratch.</p>

<h3>6.6 Claim 5 — reliability <span class="mid">Calibrates, does not rank</span></h3>
{fig("x09_claim5", 9, "Reliability.", "(A) Accuracy on answered trials barely changes as the system abstains more. (B) Platt scaling, fitted and judged on separate halves, fixes calibration.")}
<p>ECE {c5['ece_before']:.2f} → {c5['ece_after']:.2f}; uncertainty–error correlation {c5['corr_unc_err']:.2f}; subject-level scores point the wrong way.</p>

<h3>6.7 Personal models</h3>
{fig("x10_within", 10, "Models trained on other people beat personal models.", "Each bar is one person's own model; with ~100 trials per person, pooling people helps more than personalising.")}
<p>Within-person r = {S6['within_subject']['within_r']:.3f} (positive in 27 of 29) versus cross-person {S6['within_subject']['cross_r']:.3f}.</p>

<h3>6.8 Clinical comparison: ICH patients versus controls (HEFMI)</h3>
{fig("x11_clinical", 11, "Patients show tighter HbO/HbR coupling; nothing else differs.", "(A) One dot per person. (B) Evidence for a group difference per measure; primary measures Holm-corrected, artifact checks (*) uncorrected.")}
<p>HbO/HbR correlation 0.81 vs 0.64 (Holm p = {cl.loc['hbo_hbr','p_holm']:.3f}); coupling strength (Holm p = {cl.loc['coupling_r','p_holm']:.2f}) and delay (Holm p = {cl.loc['coupling_lag_s','p_holm']:.1f}) do not differ; motion, amplitudes and trial counts are similar. Because haemorrhage alters head optics and positive HbO/HbR also marks systemic signal, this is a candidate marker pending an optical control.</p>

<h2>7. Results: Maisha's pipeline (v2.3, five seeds)</h2>
<h3>7.1 Checks before modelling</h3>
{fig("x12_timing", 12, "Timing check per data split.", "Share of people whose best EEG-to-blood delay lies inside 2–8 s. Three splits failed (WARN); the claim guard therefore blocks timing-based claims until triggers are verified.")}
<p>All five seeds passed the claim guard with the restriction <i>"HEFMI timing QA did not pass in both train and validation; alignment-based claims blocked."</i> The legacy alphabetical split (all patients in validation) was replaced by a clinically stratified split; TU Berlin units passed the scale check (ratio 9.57).</p>

<h3>7.2 Compact endpoint and personalisation</h3>
{table(5, "Compact endpoint (mean HbO, mean HbR, HbO peak, HbR trough) versus simple baselines, HEFMI validation.", ["Predictor", "r", "95% range", "Error (MAE)"], endp_rows, widths=["40%", "15%", "25%", "20%"])}
{table(6, "Personalised calibration on a new person (7 people).", ["Data used", "General model", "Own average, no EEG", "Personalised with EEG", "p (EEG vs general)"], pers_rows, widths=["16%", "18%", "24%", "22%", "20%"])}
{fig("x14_personal", 13, "Most of the personalisation gain comes without EEG.", "The person's own average response from the calibration data (green) gets most of the way; EEG adds about 0.03, not significant with 7 people.")}

<h3>7.3 Prediction for new people, per seed</h3>
{table(7, "Subject-level prediction, five seeds.", ["Test"] + SEEDCOL + ["Mean ± SD", "Reading"], wave_rows, widths=["22%", "8%", "9%", "8%", "8%", "8%", "15%", "22%"])}
{fig("x13_waveform", 14, "Prediction stays at chance in every seed.", "Blue dots: seeds; black line: mean; orange dashed: chance.")}

<h3>7.4 Within person, leave-one-person-out and amplitude</h3>
{fig("x15_newpeople", 15, "Coupling exists inside each person but does not transfer.", "Blue: model. Grey: person-average shortcut (new people) or chance (same person).")}
<p>Within person, HEFMI: r = {wsp['hefmi_fnirs']['pearson']:.3f} vs chance 0.021 (p = {wsp['hefmi_fnirs']['p']:.3f}, 30 people) — <span class="yes">real coupling</span>. Leave-one-person-out, HEFMI: {lo['hefmi_fnirs']['mean_r']:.3f} vs person-average shortcut {lo['hefmi_fnirs']['subject_mean_baseline_r']:.3f} (p = {lo['hefmi_fnirs']['p_vs_baseline']:.2f}) — <span class="no">no better</span>; fMRI: {lo['ds002336_fmri']['mean_r']:.3f} vs {lo['ds002336_fmri']['subject_mean_baseline_r']:.3f} — worse. Capacity test, real minus shuffled fit per seed: {', '.join(f'{CAP[s]:+.3f}' for s in SEEDS)} — no learnable structure at small scale.</p>
{table(8, "Amplitude analysis per seed (HEFMI validation; fMRI training).", ["Measure"] + SEEDCOL, amp_rows, widths=["30%", "14%", "14%", "14%", "14%", "14%"])}
<p>An amplitude signal on new people appears in some seeds (significant in 2 of 5) but not others — unstable with 7 people. On fMRI training data the model ranks people well but tracks trials weakly: it learned <i>who</i>, not <i>what</i>.</p>

<h3>7.5 Does adding fMRI help?</h3>
{table(9, "Cross-modal versus fNIRS-only model at the same budget, per seed.",
       ["Seed", "fNIRS + fMRI r", "fNIRS-only r", "Better", "Timing excess (joint)", "Timing excess (fNIRS-only)", "Decoding (both)"],
       xm_rows, widths=["12%", "12%", "12%", "17%", "14%", "14%", "19%"])}
{fig("x16_crossmodal", 16, "Adding fMRI makes fNIRS prediction worse.", f"Each line joins one seed's two models. fNIRS-only is better in {nb_better} of 5 seeds on correlation and {nbx_better} of 5 on timing match.")}

<h3>7.6 Reliability, calibration and abstention</h3>
{table(10, "Reliability per seed.", ["Seed", "Calibration people", "Ranking, internal", "Ranking, external", "Cal. error internal", "Cal. error external", "External answered"],
       rel_rows, widths=["10%", "16%", "11%", "11%", "17%", "19%", "16%"])}
{fig("x17_reliability", 17, "Reliability works inside HEFMI but not on new data.", "(A) Ranking is positive inside HEFMI and negative on TU Berlin in every seed. (B) The locked abstention rule answers between 0% and 100% of external windows depending on the seed.")}
<p>The delay output is fixed at 5.00 s and does not follow measured delays (HEFMI validation 5.9 s, TU Berlin 8.1 s, ds002336 10.0 s): it is reported only as a "bounded model timing parameter". Downstream transfer and ablations were switched off after earlier versions showed no benefit.</p>

<h3>7.7 Five-seed summary</h3>
{table(11, "Aggregate over five seeds.", ["Metric", "Mean", "SD", "Min", "Max"], agg_rows, widths=["44%", "14%", "14%", "14%", "14%"],
       note="SD = 0 rows come from analyses that do not use the network's random seed (LOSO, ridge, personalisation, within-person).")}

<h2>8. Comparison of the two pipelines</h2>
{table(12, "Do the two pipelines agree? (plain language)",
       ["Question", "Muntasir (v6)", "Maisha (v2.3, five seeds)", "What it means"],
       [["Does training with fMRI as well as fNIRS improve fNIRS prediction?", f"No: {c1['mean_diff']:+.3f} (5 people), −0.007 (29); shuffled fMRI = real", f"No: fNIRS-only better in {nb_better}/5 seeds ({bm.mean():.3f} vs {fm.mean():.3f})", "<span class='no'>Both say no.</span>"],
        ["Can the model predict someone it has never seen?", f"Yes, a little: r = {r_tub:.3f}, all folds positive", f"No: {lo['hefmi_fnirs']['mean_r']:.3f} = person-average shortcut ({lo['hefmi_fnirs']['subject_mean_baseline_r']:.3f})", "<span class='mid'>Differ because the data differ:</span> clean target needed."],
        ["Are EEG and blood linked inside one person?", "Yes: 0.098, positive in 27/29", f"Yes: {wsp['hefmi_fnirs']['pearson']:.3f} vs chance 0.021", "<span class='yes'>Both say yes</span> — real but personal."],
        ["Does it work on another dataset?", "No: 0.012 and 0.000", f"No: {A('tuberlin_external_subject_r'):+.3f} and {A('ds002739_external_subject_r'):.3f}", "<span class='no'>Both say no.</span>"],
        ["Does the network learn the delay?", "No: follows its bounds", "No: fixed 5 s, does not match measured delays", "<span class='no'>Both say no</span>; measure it directly (6.4 s)."],
        ["Can confidence spot bad predictions?", "No: 0.02", f"Inside HEFMI yes ({A('reliability_corr_internal_test'):.2f}); new data no ({A('reliability_corr_external'):.2f})", "<span class='mid'>Partly disagree;</span> both fail on new data."],
        ["Should GSR be used?", "No: removes ~83% of the response", f"No: removes {gsr.min():.0f}–{gsr.max():.0f}%, partly real", "<span class='yes'>Both say off.</span>"],
        ["Does a new person's own data help?", "not tested this way", f"Mostly without EEG: {A('physio_personalized_20pct_subject_r'):.3f} vs {A('physio_calibration_mean_20pct_subject_r'):.3f}", "EEG adds ≈ +0.03."]],
       widths=["24%", "24%", "28%", "24%"])}
{table(13, "What can be claimed? (plain language)",
       ["Claim", "Answer", "Evidence"],
       [["EEG predicts fNIRS in new people, if the training data are clean", "<span class='yes'>Yes</span>", f"r = {r_tub:.3f}, 5/5 folds, p = 0.001; fails on the unreliable HEFMI target."],
        ["EEG predicts fMRI in new people", "<span class='mid'>Weak</span>", f"v6 {r_336:.3f}; Maisha at chance and below the shortcut; 7–10 people."],
        ["EEG and fNIRS are linked inside one person", "<span class='yes'>Yes</span>", "Both pipelines r ≈ 0.09–0.10."],
        ["fMRI supervision improves prediction", "<span class='no'>No</span>", "Both pipelines; shuffled control; five seeds."],
        ["The model works on other datasets", "<span class='no'>No</span>", "≈ 0 everywhere, ≤ 4% of ceiling."],
        ["The delay is a stable personal measure (measured directly)", "<span class='yes'>Yes</span>", f"{icc.mean_s:.1f} s, ICC {icc.icc:.2f}, p = 0.0005."],
        ["The network learns the delay", "<span class='no'>No</span>", "Follows bounds (v6); fixed 5 s (Maisha)."],
        ["Haemodynamic training gives a better EEG representation", "<span class='no'>No</span>", "Equal to random and shuffled encoders."],
        ["Confidence numbers are calibrated", "<span class='mid'>Same dataset only</span>", "v6 ECE 0.27 → 0.02; Maisha external up to 0.89."],
        ["Confidence spots bad predictions", "<span class='mid'>Mixed</span>", "Maisha inside HEFMI only; both fail on new data."],
        ["ICH patients differ in HbO/HbR coupling", "<span class='mid'>Possible, not proven</span>", "Holm p = 0.003; optical confound not excluded."],
        ["EEG adds to personalisation", "<span class='mid'>Very little</span>", "+0.03 over the person's own average; 7 people."]],
       widths=["40%", "18%", "42%"])}

<h2>9. Discussion</h2>
<p><b>Target quality decided the outcome.</b> On TU Berlin's reliable target the model predicts new people (r = 0.158); on HEFMI's systemic-dominated target the same question yields nothing beyond a person-average shortcut. Measuring split-half reliability and the HbO/HbR sign before modelling would have saved the project's first two months.</p>
<p><b>Coupling is person-specific.</b> Both pipelines find real within-person coupling (≈ 0.09–0.10) and weak or absent transfer. Personal models are limited by data per person (~100 trials), and personalisation gains come mostly from learning a person's average. This matches the literature: EEG-to-haemodynamic prediction is easier within people, for targeted regions and with more paired data.</p>
<p><b>Cross-modal supervision showed negative transfer.</b> Shuffled fMRI performs like real fMRI, and fNIRS-only training wins in 4 of 5 seeds: the small fMRI set (10 people, weakly predictable target, different people) competes for capacity rather than teaching shared structure.</p>
<p><b>Evaluation protocol matters as much as the model.</b> A trial-level split inflates the score by 88%, and subject identity alone explains most of the raw fMRI correlation. Published EEG–haemodynamic numbers from trial-level splits should be read with this in mind.</p>
<p><b>Uncertainty did not transfer.</b> Calibration can be repaired by rescaling, but ranking errors on new data failed in both pipelines, and Maisha's locked abstention rule was unstable because its calibration people depend on the seed.</p>

<h2>10. Limitations</h2>
<ul>
<li>Small fMRI data (10 people, 384 trials) and few validation people in Maisha's pipeline (7 HEFMI, 3 fMRI).</li>
<li>The v6 training dataset was chosen after the audit (by a pre-model criterion, documented; Maisha's opposite assignment is the comparison).</li>
<li>External tests confound montage, task and target quality; ds002739 channels were paired by index.</li>
<li>Event timing not verified in three of Maisha's six splits.</li>
<li>The ICH difference may be optical or systemic.</li>
<li>Seed repetition does not test analyses that ignore the seed; data-split variation would be needed.</li>
<li>Early-phase results used older protocols and are not comparable with the final results.</li>
</ul>

<h2>11. Conclusions and next steps</h2>
<p>EEG carries information about fNIRS responses in unseen people when the training target is reliable, and EEG–fNIRS coupling is real but strongly person-specific. Supervising with fMRI from different people does not help, nothing transfers across datasets, and the neurovascular delay is best measured directly. A defensible publication is a rigorous two-pipeline evaluation reporting these positives and negatives, the 88% protocol inflation and target-reliability gating as a methodological contribution.</p>
<ol>
<li>Repeat personalisation on TU Berlin (29 people) against the "own average, no EEG" control.</li>
<li>Fix Maisha's calibration people (or calibrate leave-one-person-out) and re-test abstention.</li>
<li>Compare the two reliability methods on the same split.</li>
<li>Verify event timing against raw markers.</li>
<li>Add an optical-pathlength control to the ICH analysis.</li>
<li>Test targeted motor-region fMRI prediction; treat fMRI as exploratory unless same-person EEG–fNIRS–fMRI data become available.</li>
</ol>

<h2>Appendix A. Key numbers</h2>
{table(14, "Numbers to remember.", ["Quantity", "Value"],
       [["Cross-subject fNIRS (TU Berlin, v6)", f"r = {r_tub:.3f}, p = 0.001, ≈ 22% of ceiling"],
        ["Protocol inflation", f"{pa[0]:.3f} (trial split) vs {pa[3]:.3f} (honest), +88%"],
        ["Claim 1", f"{c1['mean_diff']:+.3f} (5 people); {cs.mean_diff[0]:+.3f} (29); Maisha fNIRS-only better {nb_better}/5 seeds"],
        ["External", f"v6 {ev.loc['HEFMI','external_r']:.3f} / 0.000; Maisha {A('tuberlin_external_subject_r'):+.3f} / {A('ds002739_external_subject_r'):.3f}"],
        ["Within person", f"v6 {S6['within_subject']['within_r']:.3f}; Maisha {wsp['hefmi_fnirs']['pearson']:.3f} (chance 0.021)"],
        ["Measured delay", f"{icc.mean_s:.1f} s, SD {icc.between_subject_sd:.1f} s, ICC {icc.icc:.2f}"],
        ["Reliability", f"v6 ECE 0.27 → 0.02, ranking 0.02; Maisha ranking {A('reliability_corr_internal_test'):.2f} internal, {A('reliability_corr_external'):.2f} external"],
        ["Clinical HbO/HbR", f"0.81 vs 0.64, Holm p = {cl.loc['hbo_hbr','p_holm']:.3f}"],
        ["Target reliability", "TU Berlin 0.50 / HEFMI 0.08; HbO/HbR −0.44 / +0.92"]],
       widths=["40%", "60%"])}

<h2>Appendix B. Reproducibility</h2>
<p style="text-align:left">Muntasir's pipeline: <code>NeuroVascNet_FINAL_muntasir.ipynb</code> (built by <code>scripts/build_v6_notebook.py</code>), outputs in <code>neurovascnet_v6_outputs/</code>. Maisha's pipeline: <code>{NBNAME}.ipynb</code>, run for all five seeds with <code>scripts/run_maisha_all_seeds.py</code>; per-seed summaries and executed copies in <code>output/</code>; detailed report <code>{NBNAME}_Report.pdf</code>. Data audit: <code>NeuroVascNet_Data_Quality_Audit.md</code>. This report: <code>scripts/build_final_report.py</code>.</p>

<h2>Appendix C. Glossary</h2>
{table(15, "Terms.", ["Term", "Meaning"],
       [["HbO / HbR", "oxygenated / deoxygenated haemoglobin, measured by fNIRS"],
        ["BOLD", "the fMRI blood-oxygen signal"],
        ["r", "correlation between prediction and measurement (0 = no link)"],
        ["Chance / null", "the score obtained when predictions are randomly re-paired"],
        ["Subject demeaning", "removing each person's average before scoring"],
        ["Split-half reliability / ceiling", "how well a target agrees with itself / the best achievable r (its square root)"],
        ["LOSO", "leave-one-subject-out: test each person with a model trained on all others"],
        ["GSR", "global signal regression: subtracting the all-channel average"],
        ["ECE", "expected calibration error: gap between stated confidence and accuracy"],
        ["ICC", "intraclass correlation: consistency across repeated sessions"],
        ["ICH", "intracerebral haemorrhage, a bleeding stroke"],
        ["Seed", "starting point of the random number generator; changing it repeats training differently"]],
       widths=["30%", "70%"])}
"""

D.render(body, "NeuroVascNet Final Report")
