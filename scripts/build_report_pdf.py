"""Builds NeuroVascNet_Research_Report.pdf from saved results (no retraining).

Figures are redrawn as SVG from neurovascnet_v6_outputs/ and output/ (Maisha v2.3),
the report is typeset as print HTML, and headless Chrome renders the PDF.
Run from the repo root:  python scripts/build_report_pdf.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT / "neurovascnet_v6_outputs"
MA = ROOT / "output"
BUILD = ROOT / "report_build"
FIGDIR = BUILD / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "NeuroVascNet_Research_Report.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

S6 = json.loads((V6 / "summary.json").read_text())
SM = json.loads((MA / "summary_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json").read_text())
PH = json.loads((MA / "personalized_physio_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json").read_text())
csv = lambda n, **k: pd.read_csv(V6 / n, **k)

# ------------------------------------------------------------------ style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT, INK, MUTED, GRID = "#a8a6a0", "#d9d7d1", "#1a1a1a", "#55534f", "#e8e7e3"
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 10, "svg.fonttype": "none",
    "axes.edgecolor": "#bdbbb5", "axes.labelcolor": MUTED, "axes.labelsize": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
W = 6.9  # inches, full text width


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def label(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9)
    kw.setdefault("color", INK)
    ax.text(x, y, s, **kw)


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="right")


# ------------------------------------------------------------------ figures
def fig_target_quality():
    d = csv("data_summary.csv")
    names = {"TU Berlin hBCI (MI+MA)": "TU Berlin hBCI", "ds002336": "ds002336", "HEFMI": "HEFMI", "ds002739": "ds002739"}
    d["n"] = d.dataset.map(names)
    o = d.sort_values("reliability")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.45})
    a = ax[0]
    y = np.arange(len(o))
    a.barh(y, o.reliability, color=[ORANGE if v < 0.3 else BLUE for v in o.reliability], height=0.55)
    a.scatter(o.ceiling, y, marker="|", s=160, color=INK, lw=2, zorder=3)
    for yi, (r, c, m) in enumerate(zip(o.reliability, o.ceiling, o.modality)):
        label(a, max(r, c) + 0.025, yi, f"{r:.2f}", va="center")
    a.set_yticks(y)
    a.set_yticklabels([f"{n}  ({m})" for n, m in zip(o.n, o.modality.str.replace('fnirs', 'fNIRS').str.replace('fmri', 'fMRI'))])
    a.set_xlim(0, 1)
    a.set_xlabel("split-half reliability of the training target")
    a.set_title("How repeatable is each target?")
    a.grid(axis="y", visible=False)
    label(a, 0.70, -0.95, "| = best achievable r (√reliability)", fontsize=8.5, color=MUTED)
    a.set_ylim(-1.2, len(o) - 0.4)
    panel(a, "A")

    f = d.dropna(subset=["hbo_hbr"]).sort_values("hbo_hbr")
    a = ax[1]
    a.axvspan(-1, 0, color=BLUE, alpha=0.06)
    a.axvspan(0, 1, color=ORANGE, alpha=0.06)
    a.barh(f.n, f.hbo_hbr, color=[BLUE if v < 0 else ORANGE for v in f.hbo_hbr], height=0.45)
    a.axvline(0, color=MUTED, lw=1)
    for yi, v in enumerate(f.hbo_hbr):
        label(a, v + (0.05 if v > 0 else -0.05), yi, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right")
    a.set_xlim(-1, 1)
    a.set_ylim(-0.7, 1.9)
    label(a, -0.5, 1.6, "neural", fontsize=9, color=BLUE, va="center", ha="center", fontweight="bold")
    label(a, 0.5, 1.6, "systemic", fontsize=9, color=ORANGE, va="center", ha="center", fontweight="bold")
    a.set_xlabel("correlation between HbO and HbR")
    a.set_title("Is the fNIRS signal neural?")
    a.grid(axis="y", visible=False)
    panel(a, "B")
    save(fig, "f1_target")


def fig_protocol():
    pa = csv("protocol_ablation.csv")
    labs = ["Trials split at random\n(same person in train and test)", "Subjects split\n(new people in test)",
            "Subject identity only\n(no EEG used)", "Subjects split + subject\nmean removed (honest)"]
    fig, a = plt.subplots(figsize=(W, 2.3))
    y = np.arange(4)[::-1]
    a.barh(y, pa.r, color=[GRAY, GRAY, LIGHT, BLUE], height=0.58)
    for yi, v in zip(y, pa.r):
        label(a, v + 0.004, yi, f"r = {v:.3f}", va="center")
    a.set_yticks(y)
    a.set_yticklabels(labs)
    a.set_xlim(0, 0.36)
    a.set_xlabel("correlation between predicted and measured fNIRS response (TU Berlin hBCI)")
    a.grid(axis="y", visible=False)
    a.annotate("", xy=(0.158, 0.0), xytext=(0.297, 3.0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3, connectionstyle="arc3,rad=-0.25"))
    label(a, 0.300, 1.55, "+88% inflation\nfrom the naive split", color=ORANGE, fontsize=9, ha="left", va="center")
    save(fig, "f2_protocol")


def fig_cv():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    cols = [("r_raw", "Raw\nscore"), ("r_no_eeg_baseline", "No EEG\n(identity only)"), ("r_demeaned", "EEG\ncontribution")]
    for a, (f, title, letter) in zip(ax, [("cv_TU_Berlin_hBCI_(MI+MA).csv", "TU Berlin hBCI (fNIRS, 29 subjects)", "A"),
                                          ("cv_ds002336.csv", "ds002336 (fMRI, 10 subjects)", "B")]):
        c = csv(f)
        for i, (k, _) in enumerate(cols):
            col = BLUE if k == "r_demeaned" else (LIGHT if "no_eeg" in k else GRAY)
            jit = np.linspace(-0.13, 0.13, len(c))
            a.scatter(i + jit, c[k], s=34, color=col, edgecolor="white", lw=1.2, zorder=3)
            m = c[k].mean()
            a.plot([i - 0.25, i + 0.25], [m, m], color=INK, lw=2, zorder=4)
            label(a, i + 0.29, m, f"{m:.3f}", va="center", fontsize=8.5)
        a.set_xticks(range(3))
        a.set_xticklabels([t for _, t in cols], fontsize=8.5)
        a.set_xlim(-0.5, 2.75)
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_title(title, fontsize=10.5)
        a.grid(axis="x", visible=False)
        panel(a, letter)
    ax[0].set_ylabel("r per cross-validation fold")
    save(fig, "f3_cv")


def fig_crossmodal():
    arms = csv("controls_paired_arms.csv")
    f = arms[arms.cohort == "fnirs"].groupby(["subject", "arm"]).r.mean().unstack()
    cs = csv("controls_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.25})
    a = ax[0]
    lim = (-0.05, 0.46)
    a.plot(lim, lim, color=MUTED, lw=1, ls="--")
    above = (f.joint > f.single)
    a.scatter(f.single, f.joint, s=30, color=BLUE, edgecolor="white", lw=1.1, zorder=3)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("r, fNIRS-only training")
    a.set_ylabel("r, fNIRS + fMRI training")
    a.set_title("Each dot is one held-out subject")
    label(a, -0.03, 0.44, f"above the line: fMRI helped ({above.sum()} of {len(f)})", fontsize=8.5, color=MUTED, va="top")
    label(a, 0.44, -0.03, "below: fMRI hurt", fontsize=8.5, color=MUTED, ha="right")
    panel(a, "A")

    a = ax[1]
    rows = [("fNIRS + fMRI  vs  fNIRS only", cs.iloc[0]), ("fNIRS + real fMRI  vs  fNIRS + shuffled fMRI", cs.iloc[1]),
            ("fNIRS + shuffled fMRI  vs  fNIRS only", cs.iloc[2])]
    for i, (lab, r) in enumerate(rows):
        yy = 2 - i
        sig = r.ci_low > 0 or r.ci_high < 0
        a.plot([r.ci_low, r.ci_high], [yy, yy], color=INK, lw=1.6)
        a.scatter(r.mean_diff, yy, s=48, color=ORANGE if sig else BLUE, zorder=3, edgecolor="white", lw=1.2)
        label(a, -0.0235, yy + 0.3, lab, fontsize=8.5, color=INK)
        label(a, 0.0255, yy, f"p = {r.p_signflip:.3f}", va="center", ha="right", fontsize=8.5, color=MUTED)
    a.axvline(0, color=MUTED, lw=1)
    a.set_yticks([])
    a.set_xlim(-0.024, 0.026)
    a.set_ylim(-0.5, 2.6)
    a.set_xlabel("difference in r (mean and 95% CI, 29 subjects)")
    a.set_title("Paired comparisons")
    a.grid(axis="y", visible=False)
    panel(a, "B")
    save(fig, "f4_crossmodal")


def fig_external():
    ev = csv("external_validation.csv", index_col=0)
    fig, a = plt.subplots(figsize=(W, 2.5))
    groups = [("fNIRS: train TU Berlin  →  test HEFMI", "HEFMI"), ("fMRI: train ds002336  →  test ds002739", "ds002739")]
    for i, (lab, k) in enumerate(groups):
        r = ev.loc[k]
        y0 = 1 - i
        a.barh(y0 + 0.17, r.internal_r, height=0.3, color=GRAY)
        a.barh(y0 - 0.17, max(r.external_r, 0.0008), height=0.3, color=ORANGE)
        label(a, r.internal_r + 0.004, y0 + 0.17, f"internal (new subjects, same dataset)  {r.internal_r:.3f}", va="center", fontsize=8.5)
        label(a, max(r.external_r, 0) + 0.004, y0 - 0.17, f"external (new dataset)  {abs(r.external_r) if abs(r.external_r) < 5e-4 else r.external_r:.3f}", va="center", fontsize=8.5)
    a.set_yticks([1, 0]); a.set_yticklabels([g for g, _ in groups])
    a.set_xlim(0, 0.32)
    a.set_xlabel("subject-demeaned r")
    a.grid(axis="y", visible=False)
    save(fig, "f5_external")


def fig_delay():
    rec = csv("clinical_per_recording.csv")
    per = csv("clinical_per_person.csv")
    s = rec.assign(sess=rec.recording.str.extract(r"(\d)").astype(int)).pivot_table(index="person", columns="sess", values="coupling_lag_s")
    s = s.dropna(subset=[1, 2])
    icc = csv("delay_model_free_icc.csv").iloc[0]
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"wspace": 0.45, "width_ratios": [1.1, 1, 1.05]})
    a = ax[0]
    a.axvspan(2, 8, color=BLUE, alpha=0.07)
    a.hist(per.coupling_lag_s, bins=np.arange(3, 10.01, 0.5), color=BLUE, edgecolor="white", lw=1)
    a.axvline(icc.mean_s, color=INK, lw=1.5)
    a.set_ylim(0, a.get_ylim()[1] * 1.18)
    label(a, icc.mean_s + 0.15, a.get_ylim()[1] * 0.93, f"mean {icc.mean_s:.1f} s", fontsize=8.5,
          bbox=dict(facecolor="white", edgecolor="none", pad=1))
    label(a, 2.2, a.get_ylim()[1] * 0.92, "2–8 s band", fontsize=8.5, color=MUTED)
    a.set_xlim(2, 10)
    a.set_xlabel("EEG→HbO delay per person (s)")
    a.set_ylabel("people")
    a.set_title("Measured delay")
    panel(a, "A")
    a = ax[1]
    a.plot([3.5, 9.5], [3.5, 9.5], color=MUTED, lw=1, ls="--")
    a.scatter(s[1], s[2], s=26, color=BLUE, edgecolor="white", lw=1, zorder=3)
    a.set_xlim(3.5, 9.5); a.set_ylim(3.5, 9.5)
    a.set_xlabel("session 1 delay (s)"); a.set_ylabel("session 2 delay (s)")
    a.set_title(f"Repeatable (ICC {icc.icc:.2f})")
    panel(a, "B")
    a = ax[2]
    dl = csv("claim3_delay.csv")
    wins = [(2, 8, dl.mean_s[0]), (0, 20, dl.mean_s[1])]
    for i, (lo, hi, m) in enumerate(wins):
        a.barh(i, hi - lo, left=lo, height=0.42, color=LIGHT)
        a.scatter(m, i, s=55, color=ORANGE, zorder=3, edgecolor="white", lw=1.2)
        label(a, m + 0.6, i + 0.33, f"learned {m:.2f} s", fontsize=8.5)
    a.axvline(icc.mean_s, color=BLUE, lw=1.4, ls="--")
    label(a, icc.mean_s + 0.4, -0.62, "measured 6.4 s", fontsize=8.5, color=BLUE)
    a.set_yticks([0, 1]); a.set_yticklabels(["window\n2–8 s", "window\n0–20 s"])
    a.set_xlim(-0.5, 20.5); a.set_ylim(-0.8, 1.6)
    a.set_xlabel("delay (s)")
    a.set_title("Learned delay")
    a.grid(axis="y", visible=False)
    panel(a, "C")
    save(fig, "f6_delay")


def fig_downstream():
    c = csv("claim4_downstream.csv")
    names = {"cross-modal supervised": "Trained with fNIRS + fMRI targets", "SHUFFLED-target control": "Trained with shuffled targets",
             "random init (untrained)": "Untrained (random weights)", "raw band-power": "Raw EEG band-power (no network)",
             "FINE-TUNED from supervised": "Fine-tuned, from fNIRS+fMRI encoder", "FINE-TUNED from scratch": "Fine-tuned, from scratch"}
    c["n"] = c.encoder.map(names)
    order = [3, 2, 1, 0, 5, 4]
    c = c.iloc[order].reset_index(drop=True)
    fig, a = plt.subplots(figsize=(W, 2.5))
    y = np.arange(len(c))[::-1]
    cols = [GRAY, GRAY, GRAY, BLUE, LIGHT, LIGHT]
    a.barh(y, c.accuracy, xerr=c.sd, color=cols, height=0.6, error_kw=dict(ecolor=MUTED, lw=1, capsize=2.5))
    for yi, v, sd in zip(y, c.accuracy, c.sd):
        label(a, v + sd + 0.008, yi, f"{v:.3f}", va="center", fontsize=8.5)
    a.set_yticks(y); a.set_yticklabels(c.n)
    a.axhline(1.5, color=GRID, lw=1)
    label(a, 0.505, 0.5, "full network\nretrained", fontsize=8.5, color=MUTED, va="center")
    label(a, 0.505, 3.5, "encoder frozen,\nlinear read-out", fontsize=8.5, color=MUTED, va="center")
    a.set_xlim(0, 0.58)
    a.set_xlabel("task classification accuracy (mean ± SD across folds)")
    a.grid(axis="y", visible=False)
    save(fig, "f7_downstream")


def fig_reliability():
    cov = csv("coverage_curve.csv")
    c5 = S6["claim5"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.4})
    a = ax[0]
    a.plot(cov.coverage * 100, cov.r, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.3, label="actual")
    a.plot([100, 20], [cov.r[0], cov.r[0] + 0.03], color=MUTED, lw=1, ls="--", label="what a working gate would do")
    a.invert_xaxis()
    a.set_xlabel("% of trials the model answers (rest abstained)")
    a.set_ylabel("r on answered trials")
    a.set_ylim(0.14, 0.2)
    a.legend(loc="upper left")
    a.set_title("Abstaining does not improve accuracy")
    panel(a, "A")
    a = ax[1]
    a.bar([0, 1], [c5["ece_before"], c5["ece_after"]], color=[GRAY, BLUE], width=0.55)
    for x, v in enumerate([c5["ece_before"], c5["ece_after"]]):
        label(a, x, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5)
    a.set_xticks([0, 1]); a.set_xticklabels(["before", "after Platt\nscaling"])
    a.set_ylabel("calibration error (ECE)")
    a.set_ylim(0, 0.32)
    a.set_title("Confidence is well scaled")
    a.grid(axis="x", visible=False)
    panel(a, "B")
    save(fig, "f8_reliability")


def fig_within():
    ws = csv("within_subject_probe.csv").sort_values("r").reset_index(drop=True)
    wi, cr = S6["within_subject"]["within_r"], S6["within_subject"]["cross_r"]
    fig, a = plt.subplots(figsize=(W, 2.2))
    a.bar(range(len(ws)), ws.r, color=BLUE, width=0.72)
    a.axhline(cr, color=ORANGE, lw=1.6)
    a.axhline(wi, color=INK, lw=1.4, ls="--")
    label(a, -0.4, cr + 0.008, f"model trained on other people: r = {cr:.3f}", fontsize=8.5, color=ORANGE)
    label(a, -0.4, wi + 0.008, f"average of personal models: r = {wi:.3f}", fontsize=8.5)
    a.set_xticks([])
    a.set_xlabel(f"subjects, sorted (n = {len(ws)}; each model trained and tested on one person)")
    a.set_ylabel("r")
    a.grid(axis="x", visible=False)
    save(fig, "f9_within")


def fig_clinical():
    pp = csv("clinical_per_person.csv")
    cl = csv("clinical_comparison.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.55})
    a = ax[0]
    rng = np.random.default_rng(1)
    for i, (g, lab, c) in enumerate([("Normals", "Controls (17)", BLUE), ("Patients", "ICH patients (20)", ORANGE)]):
        v = pp[pp.group == g].hbo_hbr.values
        a.scatter(v, i + rng.uniform(-0.13, 0.13, len(v)), s=30, color=c, edgecolor="white", lw=1.1, zorder=3)
        a.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        label(a, v.mean() + 0.01, i + 0.33, f"mean {v.mean():.2f}", ha="left", fontsize=8.5,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    a.set_yticks([0, 1]); a.set_yticklabels(["Controls (17)", "ICH patients (20)"])
    a.set_ylim(-0.55, 1.65)
    a.set_xlabel("HbO/HbR correlation per person")
    a.set_title("The one measure that differs")
    a.grid(axis="y", visible=False)
    panel(a, "A")
    a = ax[1]
    names = {"coupling_r": "EEG–HbO coupling", "coupling_lag_s": "coupling delay", "reliability": "model reliability",
             "hbo_hbr": "HbO/HbR correlation", "motion_frac": "head motion*", "hbo_amp": "HbO amplitude*",
             "eeg_amp": "EEG amplitude*", "n_trials": "trial count*"}
    cl = cl.iloc[::-1]
    p = np.where(cl.p_holm.notna(), cl.p_holm, cl.p)
    y = np.arange(len(cl))
    a.barh(y, -np.log10(p), color=[ORANGE if v < 0.05 else GRAY for v in p], height=0.6)
    a.axvline(-np.log10(0.05), color=INK, lw=1, ls="--")
    label(a, -np.log10(0.05) + 0.05, len(cl) - 0.6, "p = 0.05", fontsize=8.5)
    a.set_yticks(y); a.set_yticklabels([names[m] for m in cl.measure], fontsize=8.5)
    a.set_xlabel("−log10 p  (primary: Holm-corrected)")
    a.set_title("All tested measures")
    a.grid(axis="y", visible=False)
    panel(a, "B")
    save(fig, "f10_clinical")


def fig_maisha():
    lo, ws = SM["cross_subject_loso"], SM["within_subject_probe"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    a = ax[0]
    items = [("HEFMI\nnew people", lo["hefmi_fnirs"]["mean_r"], lo["hefmi_fnirs"]["subject_mean_baseline_r"], "subject-mean\nbaseline"),
             ("HEFMI\nsame person", ws["hefmi_fnirs"]["pearson"], 0.0213, "shuffled\nnull"),
             ("ds002336\nnew people", lo["ds002336_fmri"]["mean_r"], lo["ds002336_fmri"]["subject_mean_baseline_r"], ""),
             ("ds002336\nsame person", ws["ds002336_fmri"]["pearson"], 0.1176, "")]
    x = np.arange(4)
    w = 0.36
    a.bar(x - w / 2, [i[1] for i in items], w, color=BLUE, label="model")
    a.bar(x + w / 2, [i[2] for i in items], w, color=LIGHT, label="baseline or null")
    for xi, (_, m, b, _) in zip(x, items):
        label(a, xi - w / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=8)
        label(a, xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8, color=MUTED)
    a.set_xticks(x); a.set_xticklabels([i[0] for i in items], fontsize=8.5)
    a.set_ylabel("r")
    a.set_ylim(0, 0.21)
    a.legend(loc="upper left")
    a.set_title("Model vs reference")
    a.grid(axis="x", visible=False)
    panel(a, "A")
    a = ax[1]
    pc = PH["personalized_calibration_curve"]
    fr = [float(k) * 100 for k in pc]
    for key, lab, col, ls in [("global_unadapted", "general model", GRAY, "-"),
                              ("calibration_mean_baseline", "person's own mean (no EEG)", AQUA, "--"),
                              ("personalized_residual_ridge", "personalised with EEG", BLUE, "-")]:
        v = [pc[k][key]["mean_subject_r"] for k in pc]
        a.plot(fr, v, color=col, lw=2, ls=ls, marker="o", ms=5.5, mec="white", mew=1.2, label=lab)
    a.set_xticks(fr); a.set_xticklabels([f"{int(f)}%" for f in fr])
    a.set_xlabel("new person's data used to calibrate")
    a.set_xlim(5, 45)
    a.set_ylabel("r")
    a.set_ylim(0, 0.26)
    a.legend(loc="center", bbox_to_anchor=(0.5, 0.42), fontsize=8.5)
    a.set_title("Personalisation (7 subjects)")
    panel(a, "B")
    save(fig, "f11_maisha")


for f in [fig_target_quality, fig_protocol, fig_cv, fig_crossmodal, fig_external, fig_delay,
          fig_downstream, fig_reliability, fig_within, fig_clinical, fig_maisha]:
    f()

# ------------------------------------------------------------------ numbers used in text
c1 = S6["claim1"]
cs = csv("controls_summary.csv")
ev = csv("external_validation.csv", index_col=0)
icc = csv("delay_model_free_icc.csv").iloc[0]
c5 = S6["claim5"]
cl = csv("clinical_comparison.csv").set_index("measure")
iv, ex = SM["internal_validation"], SM["external_ood_stress_tests"]
lo, wsp, rel = SM["cross_subject_loso"], SM["within_subject_probe"], SM["reliability"]
pc = PH["personalized_calibration_curve"]["0.1"]
tub = csv("cv_TU_Berlin_hBCI_(MI+MA).csv")


def fig(name, num, title, body):
    return f"""<figure><img src="figures/{name}.svg"><figcaption><b>Figure {num}. {title}</b> {body}</figcaption></figure>"""


def tbl(num, title, head, rows, note="", widths=None, cls=""):
    cg = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    n = f'<div class="tnote">{note}</div>' if note else ""
    return f"""<div class="table {cls}"><div class="tcap"><b>Table {num}.</b> {title}</div><table><colgroup>{cg}</colgroup><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>{n}</div>"""


f3 = lambda v: f"{v:.3f}"
ds = csv("data_summary.csv")

# ------------------------------------------------------------------ document
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>NeuroVascNet Research Report</title>
<style>
@page {{ size: A4; margin: 22mm 20mm 20mm 20mm;
  @bottom-center {{ content: counter(page); font-family: Calibri, sans-serif; font-size: 9pt; color: #777; }}
  @top-right {{ content: "NeuroVascNet: Research Report"; font-family: Calibri, sans-serif; font-size: 8.5pt; color: #999; }} }}
@page :first {{ @top-right {{ content: none; }} @bottom-center {{ content: none; }} }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: Cambria, Georgia, serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; margin: 0; }}
h1, h2, h3, .tcap, figcaption, table, .kicker, .meta, .summary h3 {{ font-family: Calibri, "Segoe UI", sans-serif; }}
h1 {{ font-size: 23pt; line-height: 1.15; margin: 0 0 6pt; font-weight: 700; color: #111; }}
.kicker {{ font-size: 10pt; letter-spacing: .08em; text-transform: uppercase; color: {BLUE}; font-weight: 700; margin-top: 8mm; }}
.subtitle {{ font-size: 13pt; color: #444; margin: 0 0 10pt; font-style: italic; }}
.meta {{ font-size: 9.5pt; color: #555; border-top: 1.5px solid #222; border-bottom: .6px solid #bbb; padding: 5pt 0; margin-bottom: 12pt; }}
h2 {{ font-size: 15pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: .8px solid #ccc; break-after: avoid; color: #111; }}
h3 {{ font-size: 11.5pt; margin: 12pt 0 3pt; break-after: avoid; color: #222; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }}
ul, ol {{ margin: 0 0 7pt 0; padding-left: 16pt; }} li {{ margin-bottom: 2.5pt; }}
.summary {{ background: #f4f6f9; border-left: 3px solid {BLUE}; padding: 8pt 12pt 4pt; margin: 4pt 0 10pt; break-inside: avoid; }}
.summary h3 {{ margin-top: 0; }}
figure {{ margin: 8pt 0 12pt; break-inside: avoid; }}
figure img {{ width: 100%; display: block; }}
figcaption {{ font-size: 9pt; line-height: 1.35; color: #333; margin-top: 4pt; text-align: left; }}
.table {{ margin: 6pt 0 12pt; break-inside: avoid; }}
.tcap {{ font-size: 9pt; margin-bottom: 3pt; color: #333; }}
.tnote {{ font-family: Calibri, sans-serif; font-size: 8.3pt; color: #666; margin-top: 2pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; line-height: 1.3; table-layout: fixed; }}
thead th {{ border-top: 1.3px solid #222; border-bottom: .7px solid #222; text-align: left; padding: 3pt 5pt; font-weight: 700; vertical-align: bottom; }}
td {{ padding: 2.6pt 5pt; vertical-align: top; border-bottom: .4px solid #e2e2e2; }}
tbody tr:last-child td {{ border-bottom: 1.3px solid #222; }}
.num td:not(:first-child), .num th:not(:first-child) {{ text-align: right; }}
.yes {{ color: #12754f; font-weight: 700; }} .no {{ color: #b2431b; font-weight: 700; }} .mid {{ color: #8a6a00; font-weight: 700; }}
.keep {{ break-inside: avoid; }}
code {{ font-family: Consolas, monospace; font-size: 8.8pt; }}
</style></head><body>

<div class="kicker">NeuroVascNet project · Research report</div>
<h1>Predicting hemodynamic responses from EEG with fNIRS and fMRI supervision</h1>
<div class="subtitle">What two independent pipelines show, and what the data can and cannot support</div>
<div class="meta">October 2026 &nbsp;·&nbsp; Final v6 pipeline (Muntasir Hossain) and v3.1-v2.3 pipeline (Maisha) &nbsp;·&nbsp; Four public datasets, 100 subjects</div>

<div class="summary">
<h3>Summary</h3>
<p>We asked whether a single EEG encoder, trained using paired EEG–fNIRS and EEG–fMRI recordings, can predict hemodynamic responses from EEG alone in people it has never seen. The answer is a qualified yes for fNIRS and no for the cross-modal part of the idea.</p>
<ul>
<li><b>EEG predicts fNIRS responses in new people</b> with r = {S6['internal']['TU Berlin hBCI (MI+MA)']['r_demeaned']:.3f} (p = 0.001, all five folds positive), once the model is trained on a reliable target.</li>
<li><b>Adding fMRI supervision does not help.</b> Two independent pipelines agree, and training with shuffled fMRI targets works just as well as with real ones.</li>
<li><b>Nothing transfers to a different dataset.</b> External correlations are 0.012 (HEFMI) and 0.000 (ds002739).</li>
<li><b>The neurovascular delay is a stable personal trait</b> when measured directly (mean {icc.mean_s:.1f} s, test–retest ICC {icc.icc:.2f}), but the network's learned delay only reports the range it was given.</li>
<li><b>The main obstacle was data quality, not the model.</b> The original fNIRS training set (HEFMI) has a split-half reliability of 0.08 and is dominated by systemic physiology.</li>
</ul>
</div>

<h2>1. Aim and design</h2>
<p>The aim was an EEG-only system that, after training, outputs four things: a hemodynamic-aware representation of the EEG, a predicted fNIRS (HbO/HbR) and fMRI (BOLD) response, a neurovascular delay, and a reliability score that allows the system to decline to answer. fNIRS and fMRI are used only to supervise training. At deployment the system sees EEG and nothing else, so this is a prediction problem rather than multimodal fusion.</p>
<p>The architecture is a shared EEG trunk with a small adapter per dataset (because electrode montages differ), followed by one output head for fNIRS and one for fMRI. Each head predicts a mean and a variance; the variance feeds the reliability score. A key constraint shapes everything that follows: no public dataset records the same people with both fNIRS and fMRI. The two supervision streams therefore come from different subjects, and the shared trunk is the only place where they can interact.</p>

<h2>2. Datasets</h2>
<p>Four public EEG datasets with simultaneous hemodynamic recordings were used (Table 1). The two pipelines differ in which fNIRS dataset they train on. The final v6 pipeline trains on TU Berlin hBCI and holds out HEFMI. Maisha's pipeline trains on HEFMI and holds out TU Berlin. This turns out to be the most important difference between them.</p>
{tbl(1, "Datasets, their roles in each pipeline, and target quality measured before any model was trained.",
     ["Dataset", "Signal", "Task", "Subjects", "v6 role", "Maisha role", "Reliability", "HbO/HbR"],
     [["TU Berlin hBCI", "EEG + fNIRS", "motor imagery, mental arithmetic", "29", "train", "external test", f"{ds.reliability[0]:.2f}", f"{ds.hbo_hbr[0]:+.2f}"],
      ["HEFMI-ICH", "EEG + fNIRS", "motor imagery; 17 controls, 20 ICH patients", "37", "external test, clinical", "train (30) / val (7)", f"{ds.reliability[2]:.2f}", f"{ds.hbo_hbr[2]:+.2f}"],
      ["OpenNeuro ds002336", "EEG + fMRI", "motor imagery with neurofeedback", "10", "train", "train (7) / val (3)", f"{ds.reliability[1]:.2f}", "–"],
      ["OpenNeuro ds002739", "EEG + fMRI", "perceptual decision (not motor)", "24", "external test", "external test", f"{ds.reliability[3]:.2f}", "–"]],
     note="Reliability = split-half reliability of the hemodynamic target. HbO/HbR = correlation between oxygenated and deoxygenated haemoglobin; a genuine neural response gives a negative value.",
     widths=["14%", "10%", "21%", "8%", "13%", "13%", "10%", "11%"])}

<h2>3. The data-quality audit</h2>
<p>A model cannot predict a signal more accurately than that signal agrees with itself. We therefore measured, for every dataset, the split-half reliability of the target: how well the average response from one half of a subject's trials matches the other half. Its square root is the highest correlation any predictor could reach. We also checked the sign of the HbO/HbR correlation. In a genuine brain response HbO rises while HbR falls, so the two are negatively correlated; when both move together the signal is dominated by systemic physiology such as blood pressure and breathing.</p>
{fig("f1_target", 1, "HEFMI's fNIRS target is unreliable and mostly systemic.",
     "<b>(A)</b> Bars show split-half reliability; the vertical tick marks the best correlation any model could achieve on that target. Orange bars fall below 0.3. <b>(B)</b> HbO/HbR correlation for the two fNIRS datasets. TU Berlin is negative, as expected for a neural response; HEFMI is strongly positive, the signature of systemic signal.")}
<p>The audit, which covered 610 recordings (63 GB, 97 subjects), found that HEFMI's fNIRS target has a reliability of only 0.08 and an HbO/HbR correlation of +0.92. Our own preprocessing made the situation worse: global-signal regression (GSR) removed about 83% of the evoked HbO response, and per-window z-scoring removed most of what was left, lowering HEFMI's reliability from 0.26 to 0.13. The early project result that “cross-subject waveform prediction is near zero” was therefore mainly a statement about the target, not about EEG. None of the 12 preprocessing variants we tried recovered a usable HEFMI target (best cross-subject r = 0.06).</p>
<p>Both later pipelines switched GSR off. Maisha's pipeline confirmed the decision independently: an A/B test showed that GSR removed 41% of the target variance and that the removed component was only weakly uniform across channels (r = 0.47), meaning it contained spatially structured signal rather than pure global noise. The v6 pipeline also moved fNIRS training to TU Berlin hBCI, whose target is six times more reliable. This choice was made from the reliability figures above, before any model was fitted.</p>

<h2>4. Results of the final v6 pipeline</h2>
<h3>4.1 How the models are scored</h3>
<p>Trials from the same person share an average response. A model that simply recognises who the subject is can therefore score well without using any information about brain activity. To remove this effect we test on subjects who were never in the training set, and we subtract each subject's mean response before computing the correlation. Figure 2 shows how much the choice of protocol matters. Splitting trials at random, so that every person appears in both training and test data, gives r = 0.297, which is 88% higher than the honest figure of 0.158. All results below use the honest protocol.</p>
{fig("f2_protocol", 2, "Evaluation protocol changes the score as much as the model does.",
     "Each bar scores the same TU Berlin model under a different evaluation rule. The light bar is what subject identity alone achieves without any EEG input. The blue bar is the protocol used throughout this report.")}

<h3>4.2 Prediction for new people</h3>
<p>On TU Berlin hBCI, EEG predicts trial-by-trial fNIRS responses in held-out subjects with a mean correlation of {tub.r_demeaned.mean():.3f}; every one of the five folds is positive ({tub.r_demeaned.min():.2f} to {tub.r_demeaned.max():.2f}). This is about a fifth of the best value the target allows (0.71) and is the project's first robust positive result for unseen subjects. For fMRI (ds002336) the picture is different. The raw score of about 0.23 looks promising, but subject identity alone reaches the same value, and only r = 0.064 remains once the subject mean is removed (Figure 3).</p>
{fig("f3_cv", 3, "Cross-subject prediction is real for fNIRS, and mostly subject identity for fMRI.",
     "Each dot is one cross-validation fold; black lines are means. <i>Raw score</i>: correlation without correction. <i>Subject identity only</i>: a predictor that knows who the subject is but sees no EEG. <i>EEG contribution</i>: the score after removing each subject's mean, which is the part attributable to EEG. (A) For fNIRS the EEG contribution is close to the raw score. (B) For fMRI most of the raw score is explained by subject identity.")}

<h3>4.3 Does fMRI supervision help fNIRS prediction?</h3>
<p>This is the central claim of the project. We trained the same network, with the same budget, once on fNIRS alone and once on fNIRS plus fMRI, and compared them on the same held-out subjects. The cross-modal model was not better. In the primary five-subject test the difference was Δr = {c1['mean_diff']:.3f} (95% CI {c1['ci_low']:.3f} to {c1['ci_high']:.4f}). In the larger 29-subject comparison it was {cs.mean_diff[0]:.3f} (CI {cs.ci_low[0]:.3f} to {cs.ci_high[0]:.3f}, p = {cs.p_signflip[0]:.2f}). The most telling control replaces the fMRI targets with shuffled ones: performance is unchanged (Figure 4B), so the network gains nothing from the real fMRI signal that it would not gain from noise. Changing the fMRI loss weight or the size of the shared trunk did not alter this.</p>
{fig("f4_crossmodal", 4, "Adding fMRI supervision does not improve fNIRS prediction.",
     "(A) Each dot is one held-out subject (mean of two seeds). Points above the dashed line are subjects for whom fMRI supervision helped. (B) Mean paired difference in r with 95% confidence interval; an interval that crosses zero means no reliable difference. Orange marks the one interval that excludes zero: training with shuffled fMRI is slightly worse than fNIRS alone, suggesting the extra dataset mainly uses up network capacity.")}

<h3>4.4 Generalisation to other datasets</h3>
<p>The trained models were applied once, without tuning, to the two external datasets. Performance fell to {ev.loc['HEFMI','external_r']:.3f} on HEFMI and {abs(ev.loc['ds002739','external_r']):.3f} on ds002739 (Figure 5). The HEFMI value is statistically above zero (p = 0.01) but reaches only 4% of that dataset's own ceiling, and recalibrating on 2–8 HEFMI subjects did not help. Three factors change at once between training and external data — electrode and optode montage, task (ds002739 is a perceptual decision task) and target quality — so the current data cannot tell which of them is responsible.</p>
{fig("f5_external", 5, "Performance collapses on a different dataset.",
     "Grey: performance on new subjects from the training dataset. Orange: performance on a different dataset, without retraining.")}

<h3>4.5 Neurovascular delay</h3>
<p>We measured the EEG-to-HbO delay directly, without any model, from the cross-correlation between EEG power and the fNIRS response in each HEFMI recording. The mean delay is {icc.mean_s:.1f} s, inside the expected 2–8 s physiological range. It differs between people (SD {icc.between_subject_sd:.1f} s) and is consistent within a person across sessions (ICC = {icc.icc:.2f}, {int(icc.n_people)} people, p = 0.0005 against surrogate data). This makes it a candidate individual biomarker.</p>
<p>The delay learned inside the network behaves very differently. Because it is produced by a bounded function, it must fall inside whatever window it is given. When the window was 2–8 s, it settled on the lower edge (2.0 s); when the window was widened to 0–20 s it moved to 0.07 s, and it showed almost no variation between subjects (Figure 6C). The learned value therefore reflects the chosen bounds, not physiology.</p>
{fig("f6_delay", 6, "The measured delay is a stable trait; the learned delay is not.",
     "(A) Distribution of the directly measured delay across people; the shaded band is the 2–8 s physiological range. (B) The same measurement in two sessions for each person; points close to the dashed line indicate good test–retest agreement. (C) Grey bars show the window the network was allowed to use and orange dots the delay it learned. The learned value follows the window, while the measured delay (blue) stays at 6.4 s.")}

<h3>4.6 Is the learned representation useful for other tasks?</h3>
<p>Even an imperfect predictor can produce a useful EEG representation, so we froze the trained trunk and fitted a simple linear classifier on the task labels. The trunk trained with hemodynamic targets (0.374) performed no better than one trained on shuffled targets (0.362) or one with random, untrained weights (0.381). It did beat raw band-power features (0.329). When the whole network was retrained for the task, starting from the hemodynamic trunk gave no advantage over starting from scratch (0.474 vs 0.473).</p>
{fig("f7_downstream", 7, "Hemodynamic training does not give a better EEG representation.",
     "Top four rows: the encoder is frozen and only a linear read-out is trained; blue is the encoder trained on fNIRS + fMRI targets, grey are control encoders. Bottom two rows: the whole network is retrained for the task. Error bars show standard deviation across folds.")}

<h3>4.7 Reliability score and abstention</h3>
<p>A reliability score has two separate jobs. It should be <i>calibrated</i>, meaning a stated 70% confidence is right about 70% of the time, and it should <i>rank</i> predictions, so that refusing to answer the least confident trials leaves the remaining answers more accurate. After Platt scaling, the score is well calibrated (expected calibration error falls from {c5['ece_before']:.2f} to {c5['ece_after']:.2f}). It does not rank: its correlation with prediction error is {c5['corr_unc_err']:.2f}, and abstaining on up to 80% of trials leaves accuracy unchanged (Figure 8). A gate that cannot separate good predictions from bad ones is of no practical use for abstention.</p>
{fig("f8_reliability", 8, "The reliability score is calibrated but cannot identify bad predictions.",
     "(A) Accuracy on the trials the model chooses to answer, as it abstains on more of them (moving right). A useful gate would follow the dashed line upwards; the actual curve is flat. (B) Calibration error before and after Platt scaling, fitted and scored on separate halves of the data.")}

<h3>4.8 Personal models</h3>
<p>To see whether the signal exists but differs between people, we trained a separate model for each subject using only that person's data. These personal models are positive in 27 of 29 subjects but average r = {S6['within_subject']['within_r']:.3f}, below the 0.158 achieved by the model trained on other people (Figure 9). With roughly 100 training trials per person, pooling data across subjects is worth more than tailoring to the individual. On TU Berlin, the limiting factor is the amount of data per person, not the quality of the target.</p>
{fig("f9_within", 9, "Models trained on other people beat personal models.",
     "Each bar is one subject's personal model, trained and tested on that person's data only. The orange line is the cross-subject model; the dashed line is the average of the personal models.")}

<h3>4.9 Clinical comparison: ICH patients and controls</h3>
<p>HEFMI includes 20 patients with intracerebral haemorrhage (ICH) and 17 healthy controls recorded with the same protocol. Coupling measures computed directly from the data do not depend on prediction accuracy, so this comparison is meaningful even though prediction on HEFMI is weak. Of the four primary measures, only one differs after correction for multiple comparisons: patients show a higher HbO/HbR correlation (0.81 vs 0.64, Holm-corrected p = {cl.loc['hbo_hbr','p_holm']:.3f}). EEG–hemodynamic coupling strength and delay do not differ. Head motion, signal amplitudes and trial counts were also compared to rule out acquisition differences, and none differ.</p>
<p>This result should be interpreted with care. A haemorrhage changes the optical properties of the head, and a high HbO/HbR correlation is also the marker of systemic signal seen in Section 3. Before it can be reported as altered neurovascular coupling it needs a control for optical path length.</p>
{fig("f10_clinical", 10, "ICH patients show tighter HbO/HbR coupling; nothing else differs.",
     "(A) One dot per person; black lines are group means. (B) Strength of evidence for a group difference on each measure (longer bar = smaller p). Primary measures are Holm-corrected; measures marked * are artefact checks reported uncorrected. Only bars past the dashed line are significant.")}

<h2>5. Maisha's v3.1-v2.3 pipeline</h2>
<p>Maisha's pipeline is an independent rebuild that keeps HEFMI as the fNIRS training set. It adds several safeguards: a pre-declared, clinically stratified data split; a GSR A/B test; a check that EEG and hemodynamic events are correctly aligned in time; leave-one-subject-out testing; compact response measures (HbO mean, HbR mean, HbO peak, HbR trough) compared against simple baselines; personalised calibration; a locked abstention threshold; and an automatic check that blocks claims the evidence does not support. Only the first of five planned random seeds (1337) has been run, so all numbers in this section come from a single training run.</p>

<h3>5.1 Corrections to the protocol</h3>
<p>Three corrections from this work apply to the project as a whole. First, the earlier HEFMI validation split was ordered alphabetically, and because every “Normals” ID sorts before every “Patients” ID, the validation set contained only ICH patients. Any earlier HEFMI validation figure from that split is confounded. The corrected split ranks subjects within each clinical group (3 of 17 controls and 4 of 20 patients go to validation). Second, the GSR A/B test supports switching GSR off (Section 3). Third, the timing check could not confirm event alignment: on the HEFMI validation set and on TU Berlin, only 14% and 24% of subjects show a peak EEG-to-hemodynamic delay inside 2–8 s. Until trigger timing is verified against the raw marker files, claims that depend on alignment are blocked.</p>

<h3>5.2 Results</h3>
{tbl(2, "Main results of Maisha's pipeline (seed 1337). r is computed per subject and then averaged.",
     ["Test", "Subjects", "Model r", "Reference", "Reference r", "p", "Reading"],
     [["New people, HEFMI fNIRS (LOSO)", "30", f3(lo['hefmi_fnirs']['mean_r']), "subject-mean baseline", f3(lo['hefmi_fnirs']['subject_mean_baseline_r']), f"{lo['hefmi_fnirs']['p_vs_baseline']:.2f}", "no better than baseline"],
      ["New people, ds002336 fMRI (LOSO)", "7", f3(lo['ds002336_fmri']['mean_r']), "subject-mean baseline", f3(lo['ds002336_fmri']['subject_mean_baseline_r']), f"{lo['ds002336_fmri']['p_vs_baseline']:.3f}", "worse than baseline"],
      ["Same person, HEFMI fNIRS", "30", f3(wsp['hefmi_fnirs']['pearson']), "shuffled null", "0.021", f"{wsp['hefmi_fnirs']['p']:.3f}", "<span class='yes'>real coupling</span>"],
      ["Same person, ds002336 fMRI", "7", f3(wsp['ds002336_fmri']['pearson']), "shuffled null", "0.118", f"{wsp['ds002336_fmri']['p']:.2f}", "not distinguishable"],
      ["External: TU Berlin fNIRS", "29", f3(ex['tuberlin_hbci_cross_dataset']['pearson_subject']), "pairing null", f3(ex['tuberlin_hbci_cross_dataset']['null_subject_mean']), f"{ex['tuberlin_hbci_cross_dataset']['p_subject_pairing']:.2f}", "none"],
      ["External: ds002739 fMRI", "24", f3(ex['ds002739_fmri_cross_task']['pearson_subject']), "pairing null", f3(ex['ds002739_fmri_cross_task']['null_subject_mean']), f"{ex['ds002739_fmri_cross_task']['p_subject_pairing']:.3f}", "significant but negligible"],
      ["Cross-modal vs fNIRS-only (HEFMI val)", "7", "0.024", "fNIRS-only model", "0.071", "–", "fMRI supervision hurts"]],
     note="LOSO = leave-one-subject-out. Subject-mean baseline = predicting each held-out trial from that subject's average response, with no EEG.",
     widths=["27%", "8%", "9%", "17%", "10%", "8%", "21%"], cls="num")}
<p>On HEFMI, prediction for new people is no better than the subject-mean baseline (0.026 vs 0.029, p = 0.70). With 30 subjects this test has adequate power, so for this target the null result is genuine. On the other hand, EEG–fNIRS coupling within the same person is clearly real (r = {wsp['hefmi_fnirs']['pearson']:.3f}, 95% CI 0.065–0.127, against a null of 0.021). The coupling exists, but it is specific to each person and does not carry over to others. A capacity test points the same way: with up to 128 training windows the network fits real EEG–fNIRS pairs no better than randomly re-paired ones. On the fMRI training data, between-subject ranking reaches r = 0.98 while within-subject prediction is 0.25, showing that the fMRI head mostly learns who the subject is.</p>
{fig("f11_maisha", 11, "Coupling is person-specific, and personalisation gains come mostly from the person's own average.",
     "(A) Blue: model; light grey: the appropriate reference (subject-mean baseline for new people, shuffled null for the same person). Only the HEFMI same-person result clearly exceeds its reference. (B) Accuracy on a new person after using part of their data for calibration. The personalised model (blue) improves strongly on the general model (grey), but simply using that person's own average response, with no EEG (green, dashed), achieves most of the same gain.")}

<h3>5.3 Compact measures and personalisation</h3>
<p>For the compact response measures, no predictor beats simple baselines: EEG ridge regression (r = 0.045) does no better than predicting the mean for each task condition (r = 0.058). Personalisation looks promising at first. Calibrating on 10% of a new person's data raises r from {pc['global_unadapted']['mean_subject_r']:.2f} to {pc['personalized_residual_ridge']['mean_subject_r']:.2f}. However, the same calibration data used without any EEG, simply taking that person's average response, already reaches {pc['calibration_mean_baseline']['mean_subject_r']:.2f} (Figure 11B). EEG therefore adds only about 0.03; most of the gain comes from knowing the person's baseline. With seven subjects none of these differences is statistically significant.</p>

<h3>5.4 Reliability and delay</h3>
<p>Maisha's reliability score does rank predictions on held-out internal subjects (correlation with accuracy {rel['corr_internal_test_subjects']:.2f}), but the ranking reverses on TU Berlin ({rel['corr_external']:.2f}). With the threshold fixed in advance, the system declines to answer 99% of external windows, which is the correct cautious behaviour but means it cannot operate on new datasets. The learned delay produced exactly 5.00 s for every subject, so, as in the v6 pipeline, it is reported only as a fixed model parameter and not as a neurovascular delay.</p>

<h2>6. Comparing the two pipelines</h2>
<p>The two pipelines differ in training data, code and model details. Where they agree, the conclusion is stronger than either result alone (Table 3).</p>
{tbl(3, "Agreement between the final v6 pipeline and Maisha's pipeline.",
     ["Question", "v6 (trains on TU Berlin)", "Maisha (trains on HEFMI)", "Conclusion"],
     [["Does fMRI supervision help?", "No: Δr −0.018 (5 subjects), −0.007 (29); shuffled fMRI equals real", "No: fNIRS-only model is better (0.071 vs 0.024)", "<span class='no'>Replicated negative</span>"],
      ["Prediction for new people", "Yes: r = 0.158 on a reliable target", "No: equal to subject-mean baseline", "Depends on target quality"],
      ["Same-person coupling", "r = 0.098; positive in 27/29", "r = 0.094 vs null 0.021", "<span class='yes'>Replicated positive</span>"],
      ["Transfer to other datasets", "0.012 and 0.000", "−0.002 and 0.009", "<span class='no'>Replicated negative</span>"],
      ["Learned delay", "Follows its bounds", "Constant 5.00 s", "<span class='no'>Replicated negative</span>"],
      ["Reliability ranking", "Fails (r = 0.02)", "Works internally (0.46), reverses externally", "<span class='mid'>Disagree internally; both fail externally</span>"],
      ["GSR", "Off (removes 83% of response)", "Off (A/B test)", "<span class='yes'>Agree</span>"]],
     widths=["21%", "29%", "27%", "23%"])}

<h3>Overall status of each claim</h3>
{tbl(4, "Status of the project's claims.",
     ["Claim", "Status", "Evidence"],
     [["EEG predicts fNIRS in new people (reliable target)", "<span class='yes'>Supported</span>", "r = 0.158, 5/5 folds positive, p = 0.001"],
      ["EEG predicts fMRI in new people", "<span class='mid'>Weak</span>", "r = 0.064; Maisha's LOSO below baseline"],
      ["Same-person EEG–fNIRS coupling", "<span class='yes'>Supported</span>", "Both pipelines, r ≈ 0.09–0.10"],
      ["fMRI supervision improves prediction", "<span class='no'>Not supported</span>", "Both pipelines; shuffled fMRI equals real"],
      ["Generalisation to other datasets", "<span class='no'>Not supported</span>", "≤ 4% of ceiling"],
      ["Measured neurovascular delay is a stable trait", "<span class='yes'>Supported</span>", "6.4 s, ICC 0.53, p = 0.0005"],
      ["Network learns the neurovascular delay", "<span class='no'>Not supported</span>", "Output follows the allowed window"],
      ["Hemodynamic training gives a better EEG representation", "<span class='no'>Not supported</span>", "Equal to random and shuffled encoders"],
      ["Reliability score is calibrated", "<span class='yes'>Supported</span>", "ECE 0.27 → 0.02"],
      ["Reliability score identifies bad predictions", "<span class='mid'>Mixed</span>", "v6 fails; Maisha internal only"],
      ["ICH patients differ in HbO/HbR coupling", "<span class='mid'>Candidate</span>", "Holm p = 0.003; optical confound not excluded"],
      ["EEG adds to personalised calibration", "<span class='mid'>Weak</span>", "≈ +0.03 over the person's own mean; 7 subjects"]],
     widths=["45%", "17%", "38%"])}

<h2>7. Limitations</h2>
<ul>
<li><b>Small samples.</b> ds002336 has 10 subjects (3 in Maisha's validation set), and the primary cross-modal test uses 5 held-out subjects. All fMRI results are exploratory.</li>
<li><b>One seed.</b> Only one of Maisha's five planned seeds has run, so run-to-run variation is unknown for those results.</li>
<li><b>Dataset choice.</b> The v6 decision to train on TU Berlin was made from target quality before modelling and is documented, but reviewers may still see it as a choice made with knowledge of the data. Maisha's HEFMI-trained pipeline provides the comparison.</li>
<li><b>Confounded external tests.</b> Montage, task and target quality all change between training and external data. For ds002739, EEG channels were matched by index rather than by electrode name.</li>
<li><b>Unverified timing.</b> Event alignment could not be confirmed for HEFMI validation, TU Berlin and ds002336 validation. A fixed timing offset would lower every prediction score.</li>
<li><b>Clinical confound.</b> The ICH difference may reflect optics or systemic physiology rather than neurovascular coupling.</li>
<li><b>Earlier results.</b> Figures from the early project phase (for example amplitude r = 0.33, or 0.25 with CORAL) used older protocols and have not been reproduced under the current one. They should not be cited.</li>
</ul>

<h2>8. Conclusions and next steps</h2>
<p>EEG does carry information about hemodynamic responses in people the model has never seen, with r = 0.16 for fNIRS, but only when the training target is reliable. Across the project, the quality of the target data mattered more than any change to the model. The central idea, that fMRI supervision from other subjects would improve EEG-only prediction, is not supported: two independent pipelines reach the same conclusion, and the fMRI signal behaves like noise. EEG–hemodynamic coupling is real but largely person-specific, and nothing transfers across datasets. The neurovascular delay is best measured directly, where it is a stable individual trait, and the learned delay module should be removed or redesigned. Finally, evaluation protocol alone can almost double the apparent performance, which matters when comparing with published work.</p>
<p>A defensible framing for publication is therefore a careful two-pipeline evaluation: EEG predicts fNIRS responses in unseen subjects when the target is reliable; the measured neurovascular delay is a reproducible individual trait; fMRI co-supervision from different subjects does not help; external transfer fails; and naive trial-level evaluation inflates performance by 88%. Measuring target reliability before modelling is a practical contribution that does not depend on the original hypothesis.</p>
<h3>Recommended next steps</h3>
<ol>
<li>Run Maisha's four remaining seeds, so that every v2.3 result has a measure of run-to-run variation.</li>
<li>Repeat the personalisation analysis on TU Berlin (29 subjects, reliable target), always comparing against the person's own average response without EEG.</li>
<li>Compare the two reliability methods on the same data split to resolve their disagreement.</li>
<li>Control the ICH finding for optical path length before interpreting it as a neurovascular effect.</li>
<li>Verify event timing for HEFMI and TU Berlin against the raw marker files.</li>
<li>Treat fMRI as a separate exploratory analysis rather than a co-supervision signal, unless a dataset with the same people in EEG, fNIRS and fMRI becomes available.</li>
</ol>

<h2>Appendix: sources</h2>
<p style="text-align:left">All numbers are read from saved outputs; no model was retrained for this report. Final v6 pipeline: <code>NeuroVascNet_FINAL_muntasir.ipynb</code>, results in <code>neurovascnet_v6_outputs/</code>. Maisha's pipeline: <code>NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED.ipynb</code>, results in <code>output/summary_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json</code> and the accompanying JSON files. Data-quality audit: <code>NeuroVascNet_Data_Quality_Audit.md</code>. Throughout, the sample-size unit is the subject, confidence intervals are subject-level bootstrap intervals, and null tests permute or sign-flip at subject level. This report is generated by <code>scripts/build_report_pdf.py</code>.</p>
</body></html>"""

(BUILD / "report.html").write_text(html, encoding="utf-8")
tmp = BUILD / "report.pdf"
tmp.unlink(missing_ok=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={tmp}", (BUILD / "report.html").as_uri()], check=True, capture_output=True)
if not tmp.exists():
    raise SystemExit("Chrome did not produce a PDF")
try:
    PDF.write_bytes(tmp.read_bytes())
    print("wrote", PDF)
except PermissionError:
    print(f"{PDF.name} is open in another program; new version left at {tmp}")
