"""Builds NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED_Report.pdf.

Final report for Maisha's v3.1-v2.3 pipeline over five seeds, with the
comparison against Muntasir's v6 pipeline. Every per-seed number is read from
output/summary_*_seed<seed>.json, output/personalized_physio_*_seed<seed>.json
and the executed notebooks (seed 1337 from git commit 8d26941, whose notebook
holds that run's outputs). Run from the repo root:
    python scripts/build_maisha_report_pdf.py
"""
import json
import re
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, ROOT, VIOLET, W, YELLOW, Doc, K, M,
                            fig, table, title_block)

D = Doc("maisha_report", "unused.pdf", "NeuroVascNet v3.1-v2.3 (Maisha) · Five-seed report")
D.pdf = ROOT / "NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED_Report.pdf"

SEEDS = [1337, 20260726, 7, 42, 2718]
VER = "v3.1-v2.3-personalized-physio-fixed-delay"
OUT = ROOT / "output"
NBNAME = "NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED"
S = {s: json.loads((OUT / f"summary_{VER}_seed{s}.json").read_text()) for s in SEEDS}
PH = json.loads((OUT / f"personalized_physio_{VER}_seed2718.json").read_text())
AGG = json.loads((OUT / f"multi_seed_summary_{VER}.json").read_text())["aggregates"]
V6 = json.loads((ROOT / "neurovascnet_v6_outputs" / "summary.json").read_text())


def executed(seed):
    if seed == 1337:   # this run's outputs live in the committed notebook
        raw = subprocess.run(["git", "show", f"8d26941:{NBNAME}.ipynb"], cwd=ROOT, capture_output=True,
                             check=True).stdout.decode("utf-8")
        return json.loads(raw)
    return json.loads((OUT / "executed_seed_notebooks" / f"{NBNAME}_seed{seed}.ipynb").read_text(encoding="utf-8"))


def cell_text(nb, i):
    return "".join("".join(o.get("text", "")) for o in nb["cells"][i].get("outputs", []))


XM, CAP = {}, {}
for s in SEEDS:
    nb = executed(s)
    for line in cell_text(nb, 59).splitlines():
        parts = line.split()
        if line.startswith("Full (cross-modal)"):
            XM.setdefault(s, {})["full"] = [float(x) for x in parts[2:]]
        elif line.startswith("fNIRS-only baseline"):
            XM.setdefault(s, {})["base"] = [float(x) for x in parts[2:]]
    CAP[s] = float(re.search(r"gap ([+-][0-9.]+)\)", cell_text(nb, 53)).group(1))
# columns of the relative table: MAE RMSE r_subj peakXC XCexcess XCp decode


def g(d, *p):
    for k in p:
        d = d[k]
    return d


def per_seed(*p):
    return np.array([float(g(S[s], *p)) for s in SEEDS])


f3 = lambda v: f"{v:+.3f}" if v < 0 else f"{v:.3f}"
f2 = lambda v: f"{v:+.2f}" if v < 0 else f"{v:.2f}"
SEEDCOL = ["1337", "20260726", "7", "42", "2718"]

# ---------------------------------------------------------------- figures
def f_timing():
    av = S[2718]["alignment_verification"]
    names = [v["name"] for v in av.values()]
    frac = [100 * v["frac_in_physiological_band"] for v in av.values()]
    st = [v["status"].startswith("PASS") for v in av.values()]
    fig, a = plt.subplots(figsize=(W, 2.4))
    y = np.arange(len(names))[::-1]
    a.barh(y, frac, color=[BLUE if ok else ORANGE for ok in st], height=0.6)
    for yi, v, ok in zip(y, frac, st):
        a.text(v + 1.5, yi, f"{v:.0f}%  {'PASS' if ok else 'WARN'}", va="center", fontsize=8.5)
    a.set_yticks(y); a.set_yticklabels(names, fontsize=8.6); a.set_xlim(0, 105)
    a.set_xlabel("% of people whose best EEG→blood delay falls inside 2–8 s"); a.grid(axis="y", visible=False)
    D.save(fig, "m1_timing")


def f_waveform():
    tests = [("HEFMI validation\n(7 people)", ("internal_validation", "hefmi_fnirs")),
             ("ds002336 fMRI val\n(3 people)", ("internal_validation", "ds002336_fmri")),
             ("TU Berlin\n(external, 29)", ("external_ood_stress_tests", "tuberlin_hbci_cross_dataset")),
             ("ds002739\n(external, 24)", ("external_ood_stress_tests", "ds002739_fmri_cross_task"))]
    fig, a = plt.subplots(figsize=(W, 2.6))
    for i, (lab, p) in enumerate(tests):
        r = per_seed(*p, "pearson_subject"); nl = per_seed(*p, "null_subject_mean")
        a.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), r, s=34, color=BLUE, edgecolor="white", lw=1, zorder=3,
                  label="model, one dot per seed" if i == 0 else None)
        a.plot([i - 0.25, i + 0.25], [r.mean()] * 2, color=INK, lw=2)
        a.plot([i - 0.25, i + 0.25], [nl.mean()] * 2, color=ORANGE, lw=1.6, ls="--", label="chance level" if i == 0 else None)
        a.text(i + 0.3, r.mean(), f"{r.mean():.3f}", va="center", fontsize=8.3)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(range(4)); a.set_xticklabels([t for t, _ in tests], fontsize=8.4); a.set_xlim(-0.5, 3.75)
    a.set_ylabel("r (subject level)"); a.legend(fontsize=8, loc="upper right"); a.grid(axis="x", visible=False)
    D.save(fig, "m2_waveform")


def f_personal():
    pc = PH["personalized_calibration_curve"]
    fr = [float(k) * 100 for k in pc]
    fig, a = plt.subplots(figsize=(W, 2.4))
    for key, lab, col, ls in [("global_unadapted", "general model", GRAY, "-"),
                              ("calibration_mean_baseline", "person's own average, no EEG", AQUA, "--"),
                              ("personalized_residual_ridge", "personalised with EEG", BLUE, "-")]:
        v = [pc[k][key]["mean_subject_r"] for k in pc]
        a.plot(fr, v, color=col, ls=ls, lw=2.2, marker="o", ms=6, mec="white", label=lab)
        a.text(fr[-1] + 1.2, v[-1], f"{v[-1]:.3f}", va="center", fontsize=8.3)
    a.annotate("", xy=(20, pc["0.2"]["personalized_residual_ridge"]["mean_subject_r"]),
               xytext=(20, pc["0.2"]["calibration_mean_baseline"]["mean_subject_r"]),
               arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
    a.text(21, 0.115, "EEG adds ≈ +0.03", fontsize=8.4, color=ORANGE)
    a.set_xticks(fr); a.set_xticklabels([f"{int(v)}%" for v in fr]); a.set_xlim(7, 48); a.set_ylim(0, 0.25)
    a.set_xlabel("share of the new person's data used to calibrate"); a.set_ylabel("r")
    a.legend(fontsize=8.2, loc="center right", bbox_to_anchor=(1, 0.42))
    D.save(fig, "m3_personal")


def f_newpeople():
    lo, ws = S[2718]["cross_subject_loso"], S[2718]["within_subject_probe"]
    items = [("HEFMI\nnew people (LOSO)", lo["hefmi_fnirs"]["mean_r"], lo["hefmi_fnirs"]["subject_mean_baseline_r"], "person-average shortcut"),
             ("HEFMI\nsame person", ws["hefmi_fnirs"]["pearson"], 0.0209, "chance"),
             ("ds002336\nnew people (LOSO)", lo["ds002336_fmri"]["mean_r"], lo["ds002336_fmri"]["subject_mean_baseline_r"], ""),
             ("ds002336\nsame person", ws["ds002336_fmri"]["pearson"], 0.1178, "")]
    fig, a = plt.subplots(figsize=(W, 2.5))
    x = np.arange(4); w = 0.36
    a.bar(x - w / 2, [i[1] for i in items], w, color=BLUE, label="model")
    a.bar(x + w / 2, [i[2] for i in items], w, color=LIGHT, label="reference (shortcut or chance)")
    for xi, (_, m, b, _) in zip(x, items):
        a.text(xi - w / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=8)
        a.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8, color=MUTED)
    a.set_xticks(x); a.set_xticklabels([i[0] for i in items], fontsize=8.4); a.set_ylabel("r")
    a.set_ylim(0, 0.215); a.legend(fontsize=8, loc="upper left"); a.grid(axis="x", visible=False)
    D.save(fig, "m4_newpeople")


def f_crossmodal():
    full = np.array([XM[s]["full"][2] for s in SEEDS]); base = np.array([XM[s]["base"][2] for s in SEEDS])
    fx = np.array([XM[s]["full"][4] for s in SEEDS]); bx = np.array([XM[s]["base"][4] for s in SEEDS])
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.35})
    for a, (fv, bv, title) in zip(ax, [(full, base, "Prediction r (7 validation people)"), (fx, bx, "Timing match above chance")]):
        for i in range(5):
            a.plot([0, 1], [bv[i], fv[i]], color=GRAY, lw=1)
        a.scatter(np.zeros(5), bv, s=40, color=AQUA, zorder=3, label="fNIRS only")
        a.scatter(np.ones(5), fv, s=40, color=ORANGE, zorder=3, label="fNIRS + fMRI")
        a.set_xticks([0, 1]); a.set_xticklabels(["fNIRS only", "fNIRS + fMRI"]); a.set_xlim(-0.4, 1.4)
        a.set_title(title, fontsize=9.5); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("value per seed")
    D.save(fig, "m5_crossmodal")


def f_reliability():
    ri = per_seed("reliability", "corr_internal_test_subjects"); re_ = per_seed("reliability", "corr_external")
    cov = per_seed("reliability", "abstention", "external", "coverage") * 100
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.35})
    a = ax[0]
    x = np.arange(5); w = 0.38
    a.bar(x - w / 2, ri, w, color=BLUE, label="inside HEFMI (test people)")
    a.bar(x + w / 2, re_, w, color=ORANGE, label="external (TU Berlin)")
    a.axhline(0, color=MUTED, lw=0.8); a.set_xticks(x); a.set_xticklabels(SEEDCOL, fontsize=8)
    a.set_ylabel("confidence ranks errors (r)"); a.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); a.grid(axis="x", visible=False)
    a.set_title("Ranking good vs bad predictions", fontsize=9.5)
    a = ax[1]
    a.bar(x, cov, color=VIOLET, width=0.55)
    for xi, v in zip(x, cov):
        a.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=8.3)
    a.set_xticks(x); a.set_xticklabels(SEEDCOL, fontsize=8); a.set_ylim(0, 112)
    a.set_ylabel("% of external windows answered"); a.set_title("Locked abstention rule on TU Berlin", fontsize=9.5)
    a.grid(axis="x", visible=False)
    D.save(fig, "m6_reliability")


def f_amplitude():
    q2 = per_seed("amplitude", "hefmi_fnirs_val", "within_subject_r_Q2"); q3 = per_seed("amplitude", "hefmi_fnirs_val", "cross_subject_centered_r_Q3")
    p = per_seed("amplitude", "hefmi_fnirs_val", "p_signflip")
    fig, a = plt.subplots(figsize=(W, 2.3))
    x = np.arange(5); w = 0.36
    a.bar(x - w / 2, q2, w, color=[BLUE if pv < 0.05 else LIGHT for pv in p], edgecolor=BLUE, label="within person (dark = p < 0.05)")
    a.bar(x + w / 2, q3, w, color=AQUA, label="new people, own mean removed")
    a.axhline(0, color=MUTED, lw=0.8); a.set_xticks(x); a.set_xticklabels([f"seed {s}" for s in SEEDCOL], fontsize=8)
    a.set_ylabel("r"); a.legend(fontsize=8, loc="upper left"); a.grid(axis="x", visible=False)
    D.save(fig, "m7_amplitude")


def f_delay():
    dv = json.loads((OUT / "delay_validation.json").read_text())
    rows = [("HEFMI val", 5.93), ("TU Berlin", 8.14), ("ds002336 val", 10.0), ("HEFMI train", 5.24)]
    fig, a = plt.subplots(figsize=(W, 2.1))
    y = np.arange(4)[::-1]
    a.barh(y, [r[1] for r in rows], color=AQUA, height=0.5, label="measured from the data (average)")
    a.axvline(5.0, color=ORANGE, lw=2, ls="--", label="model output: 5.00 s for everyone")
    for yi, (_, v) in zip(y, rows):
        a.text(v + 0.15, yi, f"{v:.1f} s", va="center", fontsize=8.4)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows]); a.set_xlim(0, 12.5); a.set_xlabel("delay (s)")
    a.legend(fontsize=8, loc="lower right"); a.grid(axis="y", visible=False)
    D.save(fig, "m8_delay")


for fn in [f_timing, f_waveform, f_personal, f_newpeople, f_crossmodal, f_reliability, f_amplitude, f_delay]:
    fn()

# ---------------------------------------------------------------- tables built from data
iv = lambda key, field: per_seed("internal_validation", key, field)
ev = lambda key, field: per_seed("external_ood_stress_tests", key, field)


def seed_row(label, arr, extra=""):
    return [label] + [f3(v) for v in arr] + [f"<b>{arr.mean():.3f} ± {arr.std(ddof=1):.3f}</b>", extra]


wave_rows = [
    seed_row("HEFMI validation (7)", iv("hefmi_fnirs", "pearson_subject"), f"chance {iv('hefmi_fnirs', 'null_subject_mean').mean():.3f}"),
    seed_row("ds002336 fMRI val (3)", iv("ds002336_fmri", "pearson_subject"), f"chance {iv('ds002336_fmri', 'null_subject_mean').mean():.3f}"),
    seed_row("TU Berlin, external (29)", ev("tuberlin_hbci_cross_dataset", "pearson_subject"), "zero"),
    seed_row("ds002739, external (24)", ev("ds002739_fmri_cross_task", "pearson_subject"), "practically zero"),
]
pval_rows = [
    ["HEFMI validation: p (pairing)"] + [f"{v:.3f}" for v in iv("hefmi_fnirs", "p_subject_pairing")],
    ["HEFMI validation: p (sign-flip)"] + [f"{v:.3f}" for v in iv("hefmi_fnirs", "p_subject_signflip")],
    ["ds002739: p (pairing)"] + [f"{v:.3f}" for v in ev("ds002739_fmri_cross_task", "p_subject_pairing")],
]
xm_rows = []
for s in SEEDS:
    fu, ba = XM[s]["full"], XM[s]["base"]
    xm_rows.append([str(s), f"{fu[2]:.3f}", f"{ba[2]:.3f}", "fNIRS only" if ba[2] > fu[2] else "fNIRS + fMRI",
                    f"{fu[4]:.3f}", f"{ba[4]:.3f}", f"{fu[6]:.3f} / {ba[6]:.3f}"])
fm = np.array([XM[s]["full"][2] for s in SEEDS]); bm = np.array([XM[s]["base"][2] for s in SEEDS])
fxm = np.array([XM[s]["full"][4] for s in SEEDS]); bxm = np.array([XM[s]["base"][4] for s in SEEDS])
n_base_better = int((bm > fm).sum())
xm_rows.append(["<b>Mean</b>", f"<b>{fm.mean():.3f}</b>", f"<b>{bm.mean():.3f}</b>", f"<b>fNIRS only in {n_base_better}/5</b>",
                f"<b>{fxm.mean():.3f}</b>", f"<b>{bxm.mean():.3f}</b>",
                f"{np.mean([XM[s]['full'][6] for s in SEEDS]):.3f} / {np.mean([XM[s]['base'][6] for s in SEEDS]):.3f}"])

rel_rows = []
for s in SEEDS:
    r = S[s]["reliability"]
    rel_rows.append([str(s), ", ".join(str(x) for x in r["protocol"]["calibration_subjects"]),
                     f"{r['corr_internal_test_subjects']:.2f}", f2(r["corr_external"]),
                     f"{r['ece']['internal_test_raw']:.3f} → {r['ece']['internal_test_calibrated']:.3f}",
                     f"{r['ece']['external_raw']:.3f} → {r['ece']['external_calibrated']:.3f}",
                     f"{r['abstention']['locked_threshold']:.2f}",
                     f"{100 * r['abstention']['external']['coverage']:.1f}%"])

amp_rows = [
    ["Within person, new people (p)"] + [f"{a:.3f} ({p:.2f})" for a, p in zip(per_seed("amplitude", "hefmi_fnirs_val", "within_subject_r_Q2"), per_seed("amplitude", "hefmi_fnirs_val", "p_signflip"))],
    ["New people, own mean removed"] + [f3(v) for v in per_seed("amplitude", "hefmi_fnirs_val", "cross_subject_centered_r_Q3")],
    ["Ranking people (HEFMI val)"] + [f2(v) for v in per_seed("amplitude", "hefmi_fnirs_val", "between_subject_ranking_r_Q1")],
    ["fMRI train: ranking people"] + [f2(v) for v in per_seed("amplitude", "ds002336_fmri_train_fit_on_self", "between_subject_ranking_r_Q1")],
    ["fMRI train: within person"] + [f2(v) for v in per_seed("amplitude", "ds002336_fmri_train_fit_on_self", "within_subject_r_Q2")],
]
gsr = per_seed("provenance", "gsr_ab", "frac_variance_removed") * 100
gsr_u = per_seed("provenance", "gsr_ab", "removed_spatial_uniformity_r")

names = {"hefmi_internal_subject_r": "HEFMI validation prediction", "fmri_internal_subject_r": "fMRI validation prediction",
         "tuberlin_external_subject_r": "TU Berlin external prediction", "ds002739_external_subject_r": "ds002739 external prediction",
         "hefmi_loso_r": "HEFMI leave-one-person-out", "hefmi_loso_subject_mean_baseline_r": "… its person-average shortcut",
         "fmri_loso_r": "fMRI leave-one-person-out", "physio_ridge_subject_r": "Ridge on compact endpoint",
         "physio_personalized_20pct_subject_r": "Personalised with EEG (20%)",
         "physio_calibration_mean_20pct_subject_r": "Person's own average, no EEG (20%)",
         "physio_global_20pct_subject_r": "General model (20%)", "hefmi_within_subject_r": "Within-person, HEFMI",
         "fmri_within_subject_r": "Within-person, fMRI", "reliability_corr_internal_test": "Reliability ranking, internal",
         "reliability_corr_external": "Reliability ranking, external"}
agg_rows = []
for k, lab in names.items():
    a = AGG[k]
    sd = a["std_across_seeds"] or 0.0
    agg_rows.append([lab, f3(a["mean"]), f"{sd:.3f}", f3(a["min"]), f3(a["max"])])

pc = PH["personalized_calibration_curve"]
pers_rows = [[f"{float(k):.0%}", f"{pc[k]['global_unadapted']['mean_subject_r']:.3f}",
              f"{pc[k]['calibration_mean_baseline']['mean_subject_r']:.3f}",
              f"{pc[k]['personalized_residual_ridge']['mean_subject_r']:.3f}",
              f"{pc[k]['paired_signflip_p']:.2f}"] for k in pc]
gl = PH["global_internal_validation"]
endp_rows = [[lab, f3(gl[k]["mean_subject_r"]), f"{gl[k]['ci95_subject_bootstrap'][0]:+.2f} to {gl[k]['ci95_subject_bootstrap'][1]:+.2f}",
              f"{gl[k]['mean_subject_mae_sd_units']:.3f}"]
             for k, lab in [("train_mean", "Training average (no EEG)"), ("task_condition_event", "Average for each task (no EEG)"),
                            ("ridge_bandpower", "Ridge on EEG band power")]]
lo = S[2718]["cross_subject_loso"]; wsp = S[2718]["within_subject_probe"]
cap_txt = ", ".join(f"{CAP[s]:+.3f}" for s in SEEDS)
v6i = V6["internal"]; v6c1 = V6["claim1"]

body = title_block("NeuroVascNet · Final report · Maisha's pipeline", "NeuroVascNet v3.1-v2.3: Five-Seed Results",
                   "Everything the notebook found — positive and negative — with the comparison against Muntasir's final pipeline",
                   f"Notebook: {NBNAME}.ipynb · Model version {VER} · Seeds 1337, 20260726, 7, 42, 2718 (all complete, one fresh kernel each) · October 2026") + f"""
<div class="box">
<h3>Summary</h3>
<p>This report collects every result of Maisha's v3.1-v2.3 NeuroVascNet notebook, repeated with five random seeds, and compares it with Muntasir's final (v6) pipeline. The notebook trains one EEG encoder with fNIRS (HEFMI-ICH) and fMRI (ds002336) supervision, predicts haemodynamic responses from EEG, and tests the predictions on new people and on two external datasets.</p>
<ul>
<li><b>Positive:</b> EEG and fNIRS are clearly linked <i>inside each person</i> (r = {wsp['hefmi_fnirs']['pearson']:.3f} vs chance 0.021, p = 0.005, 30 people). The reliability score ranks predictions within HEFMI in every seed (mean {AGG['reliability_corr_internal_test']['mean']:.2f}). The safeguards — clinically balanced split, leakage-free normalisation, unit and timing checks, a claim guard and five seeds — all worked.</li>
<li><b>Negative:</b> for new people the model is no better than predicting each person's own average ({lo['hefmi_fnirs']['mean_r']:.3f} vs {lo['hefmi_fnirs']['subject_mean_baseline_r']:.3f}); adding fMRI makes fNIRS prediction worse in {n_base_better} of 5 seeds; nothing transfers to other datasets (TU Berlin {AGG['tuberlin_external_subject_r']['mean']:+.3f}, ds002739 {AGG['ds002739_external_subject_r']['mean']:.3f}); EEG adds only about +0.03 to personalisation; the delay output does not follow measured delays; the abstention rule is unstable on new data.</li>
<li><b>Comparison:</b> the two pipelines agree on every negative result. They disagree on prediction for new people, because Muntasir trained on clean TU Berlin data (r = {v6i['TU Berlin hBCI (MI+MA)']['r_demeaned']:.3f}) while this notebook trains on HEFMI, whose fNIRS signal is mostly body physiology.</li>
</ul>
</div>

<h2>1. How to read the numbers</h2>
<ul>
<li><b>r (correlation)</b> runs from −1 to +1; <b>0 means no link</b>. 0.1 is weak, 0.3 moderate, 0.5 strong. Almost every useful number in this project lies between 0 and 0.2.</li>
<li><b>Chance (null)</b> is the score obtained when predictions are randomly re-paired with the wrong trials. A result only counts if it is clearly above chance.</li>
<li><b>p-value</b>: how likely the result is under chance alone; below 0.05 is called significant. A tiny effect can be significant and still useless, so every effect is also labelled by size.</li>
<li><b>People, not trials, are the sample size.</b> 540 trials from 7 people are 7 independent observations.</li>
<li><b>Five seeds</b>: the whole training is repeated five times from different random starting points. A result that changes a lot between seeds is not stable.</li>
</ul>

<h2>2. What the notebook does</h2>
<h3>2.1 Data and roles</h3>
{table(1, "Datasets as loaded by the notebook.",
       ["Role", "Dataset", "Signal", "People", "30 s windows"],
       [["fNIRS training", "HEFMI-ICH", "EEG + fNIRS (90 channels, HbO + HbR)", "30", "2,400"],
        ["fNIRS validation", "HEFMI-ICH", "same", "7", "540"],
        ["External fNIRS test", "TU Berlin hBCI", "EEG + fNIRS (36 real channels)", "29", "3,480"],
        ["fMRI training", "ds002336", "EEG + fMRI (48 brain regions)", "7", "305"],
        ["fMRI validation", "ds002336", "same", "3", "124"],
        ["External fMRI test", "ds002739 (perceptual task, not motor)", "EEG + fMRI", "24", "2,448"]],
       widths=["20%", "28%", "30%", "10%", "12%"])}
<ul>
<li>HEFMI validation people were chosen <b>within each clinical group</b> (3 of 17 controls, 4 of 20 haemorrhage patients) with a fixed seed (20260807). The old alphabetical split had put only patients into validation and is marked invalid.</li>
<li>No person appears in both training and validation (checked by the notebook).</li>
<li>All datasets use the same 32 EEG electrodes, matched by name; the motor electrodes (C3, C4, Cz and neighbours) are present.</li>
<li>ds002336: 52 EEG–fMRI pairs kept; 1 EEG-only and 3 fMRI-only recordings dropped because their partner file is missing. ds002739: 48 pairs.</li>
</ul>
<h3>2.2 Preprocessing</h3>
<ul>
<li><b>Target:</b> response amplitude in physical units; no per-trial z-scoring of the target; baseline = first 20% of each window.</li>
<li><b>Global signal regression: off.</b> The notebook's A/B test on training data found GSR removed {gsr.min():.0f}–{gsr.max():.0f}% of the fNIRS signal across seeds, and the removed part was only moderately uniform across channels (r ≈ {gsr_u.mean():.2f}): it contained real, spatially structured signal.</li>
<li><b>Filters:</b> fNIRS 0.01–0.2 Hz, fMRI 0.008–0.15 Hz, EEG 0.5–45 Hz with a 50 Hz notch.</li>
<li><b>EEG input:</b> band power, 4 bands × 32 electrodes = 128 channels at 20 samples/s (600 per window), normalised with training-people statistics only.</li>
<li><b>Units:</b> TU Berlin converted from millimolar to molar; scale check passed (external/train ratio 9.57).</li>
</ul>
<h3>2.3 Model and training</h3>
<ul>
<li>Shared EEG encoder (EEGNet front-end + temporal convolution network), <b>142,922 parameters</b>, separate normalisation statistics for fNIRS and fMRI, equal training steps per modality (19 per epoch), 15 epochs.</li>
<li>Heads: fNIRS 180 signals × 300 time points; fMRI 48 regions × 15 time points; each predicts a value and its uncertainty.</li>
<li>Switched off in this version: stage-2 cross-modal alignment, the learned delay (fixed at 5 s), the downstream-transfer test and ablations — all because earlier versions showed no benefit.</li>
<li>Reliability: learned gate → temperature scaling → isotonic calibration, fitted on 4 validation people and tested on the other 3.</li>
</ul>

<h2>3. Checks before modelling</h2>
{fig("m1_timing", 1, "Timing check: does the blood response peak at a sensible delay after EEG?", "For each data split, the share of people whose best-matching EEG-to-blood delay lies inside the normal 2–8 s range. Orange splits failed (WARN): most people peak outside the range, which can happen when event clocks are slightly offset. Until trigger timing is verified against raw marker files, the notebook blocks any claim that depends on timing.")}
<p>Every seed passed the automatic <b>claim guard</b> with one restriction: <i>"HEFMI timing QA did not pass in both train and validation; alignment-based claims blocked."</i></p>

<h2>4. Results</h2>
<h3>4.1 Main endpoint: compact response versus simple baselines</h3>
<p>The main target is four numbers per trial — mean HbO, mean HbR, HbO peak, HbR trough — inside a 5.2–15.2 s window chosen on training people only (HEFMI validation, 7 people; identical in every seed because no neural network is involved).</p>
{table(2, "Compact endpoint versus simple baselines.", ["Predictor", "r", "95% range", "Error (MAE)"], endp_rows, widths=["40%", "15%", "25%", "20%"])}
<p><b>Meaning:</b> the EEG model does <b>not</b> beat the simple "average for this task" guess; its error is 0.07 units worse (p ≈ 0.06). <span class="no">Negative.</span></p>

<h3>4.2 Personalisation with part of a new person's data</h3>
{table(3, "Personalised calibration (7 validation people).",
       ["Data used to calibrate", "General model", "Person's own average, no EEG", "Personalised with EEG", "p (EEG model vs general)"],
       pers_rows, widths=["20%", "18%", "24%", "20%", "18%"])}
{fig("m3_personal", 2, "Most of the personalisation gain comes without EEG.", "Grey: general model. Green dashed: simply using the new person's own average response from the calibration data, with no EEG. Blue: the personalised EEG model. EEG adds only about 0.03 on top of the person's own average, and with 7 people none of the differences is significant.")}
<p><span class="mid">Weak.</span></p>

<h3>4.3 Predicting the full blood-signal time course, five seeds</h3>
{table(4, "Subject-level prediction on new people, per seed.", ["Test"] + SEEDCOL + ["Mean ± SD", "Reading"], wave_rows,
       widths=["22%", "8%", "9%", "8%", "8%", "8%", "15%", "22%"])}
{fig("m2_waveform", 3, "Prediction for new people stays at chance in every seed.", "Each blue dot is one seed; the black line is the mean, the orange dashed line the chance level. Internal HEFMI is barely above chance, fMRI is at chance, and both external datasets are essentially zero.")}
{table(5, "Significance of the internal and external results, per seed.", ["Test"] + SEEDCOL, pval_rows, widths=["35%", "13%", "13%", "13%", "13%", "13%"])}
<p><b>Meaning:</b> the HEFMI result is statistically above chance in most seeds but labelled "practically negligible" (mean 0.029 vs chance 0.021); ds002739 is significant in 4 of 5 seeds at r ≈ 0.007, which is also negligible. <span class="no">Negative, stable across seeds.</span></p>

<h3>4.4 Size of the response instead of its full shape</h3>
<p>Three questions are separated: does the model rank <i>people</i> by their average response; within a person, does it track which trials are larger; and does that hold for people it has never seen?</p>
{table(6, "Amplitude analysis, per seed.", ["Measure"] + SEEDCOL, amp_rows, widths=["30%", "14%", "14%", "14%", "14%", "14%"])}
{fig("m7_amplitude", 4, "The amplitude signal is not stable across seeds.", "Within-person tracking (dark bars are significant) and tracking on new people with each person's own mean removed. Two of five seeds show a significant effect; the others do not.")}
<p><b>Meaning:</b> a small amplitude signal appears in some seeds but not others; with 7 validation people it is not stable enough to claim. On the fMRI training data the model ranks people well but tracks trials only weakly: it mostly learned <b>who</b> the person is, not <b>what</b> their brain is doing. <span class="mid">Unclear.</span></p>

<h3>4.5 Can the network learn real EEG–fNIRS pairs at all?</h3>
<p>The network is trained on a few windows with the real pairing and again with the pairing shuffled. Real minus shuffled fit at 128 windows, per seed: <b>{cap_txt}</b>. Real pairs are learned no better than random ones: at this scale the network finds no usable EEG→fNIRS structure. <span class="no">Negative.</span></p>

<h3>4.6 Inside one person, and the strict new-person test</h3>
{table(7, "Within-person coupling and leave-one-person-out (LOSO).",
       ["Test", "People", "Model r", "Reference", "Reference r", "p", "Verdict"],
       [["Within person, HEFMI fNIRS", "30", f"{wsp['hefmi_fnirs']['pearson']:.3f}", "chance", "0.021", f"{wsp['hefmi_fnirs']['p']:.3f}", "<span class='yes'>real coupling</span>"],
        ["Within person, ds002336 fMRI", "7", f"{wsp['ds002336_fmri']['pearson']:.3f}", "chance", "0.118", f"{wsp['ds002336_fmri']['p']:.2f}", "not distinguishable"],
        ["LOSO, HEFMI fNIRS", "30", f"{lo['hefmi_fnirs']['mean_r']:.3f}", "person-average shortcut", f"{lo['hefmi_fnirs']['subject_mean_baseline_r']:.3f}", f"{lo['hefmi_fnirs']['p_vs_baseline']:.2f}", "<span class='no'>no better than shortcut</span>"],
        ["LOSO, ds002336 fMRI", "7", f"{lo['ds002336_fmri']['mean_r']:.3f}", "person-average shortcut", f"{lo['ds002336_fmri']['subject_mean_baseline_r']:.3f}", f"{lo['ds002336_fmri']['p_vs_baseline']:.3f}", "<span class='no'>worse than shortcut</span>"]],
       widths=["26%", "8%", "10%", "20%", "11%", "8%", "17%"])}
{fig("m4_newpeople", 5, "Coupling exists inside each person but does not transfer.", "Blue: model. Grey: the relevant reference — the person-average shortcut for new people, chance for the same person. Only within-person HEFMI clearly beats its reference.")}
<p><b>Meaning:</b> EEG and fNIRS are linked inside each person; the link is personal and does not carry over to other people. With 30 people, the HEFMI new-person null is well powered. <span class="yes">Positive</span> (within person) and <span class="no">negative</span> (across people).</p>

<h3>4.7 Does adding fMRI help fNIRS prediction?</h3>
{table(8, "Cross-modal (fNIRS + fMRI) model versus fNIRS-only model at the same training budget.",
       ["Seed", "fNIRS + fMRI r", "fNIRS-only r", "Better", "Timing excess (fNIRS + fMRI)", "Timing excess (fNIRS-only)", "Task decoding (both)"],
       xm_rows, widths=["12%", "12%", "12%", "17%", "14%", "14%", "19%"])}
{fig("m5_crossmodal", 6, "Adding fMRI makes fNIRS prediction worse.", "Each line joins the two models of one seed. Left: prediction correlation (fNIRS-only better in 4 of 5 seeds). Right: timing match above chance (fNIRS-only better in all 5).")}
<p><b>Meaning:</b> fMRI supervision does not help; task decoding is at chance (≈ 0.50) for both models. <span class="no">Negative, replicated across seeds.</span></p>

<h3>4.8 Reliability score, calibration and abstention</h3>
{table(9, "Reliability per seed.",
       ["Seed", "Calibration people", "Ranking, internal", "Ranking, external", "Calibration error internal (raw → cal.)", "Calibration error external (raw → cal.)", "Locked threshold", "External answered"],
       rel_rows, widths=["10%", "14%", "10%", "10%", "16%", "16%", "10%", "14%"])}
{fig("m6_reliability", 7, "The reliability score works inside HEFMI but not on new data.", "Left: correlation between confidence and accuracy, positive inside HEFMI in every seed and negative on TU Berlin in every seed. Right: the share of external windows the locked abstention rule agreed to answer swings from 0% to 100% depending on the seed.")}
<ul>
<li><span class="yes">Positive:</span> inside HEFMI the score puts better predictions ahead of worse ones (mean {AGG['reliability_corr_internal_test']['mean']:.2f}, positive in every seed).</li>
<li><span class="no">Negative:</span> on the external dataset the ranking reverses in every seed (mean {AGG['reliability_corr_external']['mean']:.2f}): the score is most confident when most wrong.</li>
<li><span class="mid">Unstable:</span> the 4 calibration people change with the seed, and so does everything downstream. In seed 2718 the rule answered every external window, 97% of them outside the error tolerance, with calibration error 0.89.</li>
</ul>

<h3>4.9 Delay</h3>
{fig("m8_delay", 8, "The delay output does not follow measured delays.", "Green: each group's average delay measured directly from the data. Orange: the model's delay, fixed at 5.00 s for everyone because the learned delay is switched off in this version. It does no better than a constant guess.")}
<p>Report the delay output only as a "bounded model timing parameter", never as a neurovascular delay. <span class="no">Negative.</span></p>

<h3>4.10 Not tested in this version</h3>
<p>The downstream-transfer test (is the learned EEG representation useful for other tasks?) and the ablations (removing parts of the model) were switched off because earlier versions showed no benefit over fair comparisons.</p>

<h2>5. Five-seed summary</h2>
{table(10, "Aggregate over the five seeds (from the notebook's last cell).", ["Metric", "Mean", "SD across seeds", "Min", "Max"], agg_rows,
       widths=["44%", "14%", "14%", "14%", "14%"])}
<p>Rows with SD = 0 come from analyses that do not use the network's random seed (leave-one-person-out, ridge, personalisation, within-person), so they give identical results in every run. The five seeds measure run-to-run variation only for the neural-network rows; the stability of the others would need different data splits.</p>

<h2>6. All findings</h2>
<h3>Positive</h3>
<ul>
<li>Real EEG–fNIRS coupling inside each person (r = 0.094, p = 0.005, 30 people).</li>
<li>The reliability score ranks predictions within the training dataset (0.42 average, positive in all seeds).</li>
<li>Personalisation improves prediction (0.035 → 0.191), mostly by learning the person's average.</li>
<li>Protocol safeguards work: balanced clinical split, subject-level statistics, leakage-free normalisation, unit and timing checks, claim guard, per-prediction audit files, five seeds.</li>
<li>GSR correctly identified as harmful and switched off.</li>
<li>Loaders, file pairing and channel alignment pass their checks.</li>
</ul>
<h3>Negative</h3>
<ul>
<li>Prediction for new people on HEFMI is no better than knowing each person's average.</li>
<li>fMRI supervision does not help; fNIRS-only training is better in 4 of 5 seeds.</li>
<li>No transfer to TU Berlin or ds002739.</li>
<li>EEG ridge does not beat the task-average guess on the compact endpoint.</li>
<li>Real EEG–fNIRS pairs are learned no better than shuffled pairs.</li>
<li>The delay output does not follow measured delays.</li>
<li>The reliability score reverses on new datasets; the abstention rule is unpredictable there.</li>
</ul>
<h3>Unclear or unstable</h3>
<ul>
<li>Amplitude tracking for new people (significant in 2 of 5 seeds only).</li>
<li>All fMRI results (3–7 people).</li>
<li>EEG–fNIRS timing in three splits (WARN).</li>
</ul>

<h2>7. Comparison with Muntasir's final pipeline</h2>
{table(11, "How the two pipelines differ.",
       ["", "Muntasir (final v6)", "Maisha (v3.1-v2.3)"],
       [["fNIRS training data", "TU Berlin hBCI (29 people)", "HEFMI-ICH (30 train + 7 validation)"],
        ["fNIRS test data", "HEFMI (external)", "TU Berlin (external)"],
        ["fMRI training data", "ds002336", "ds002336"],
        ["Quality of the fNIRS training target", "good: reliability 0.50, neural signature", "poor: reliability 0.08, mostly body signal"],
        ["EEG input", "raw EEG, 200 Hz", "band power, 20 Hz"],
        ["Main target", "full time course", "four compact numbers per trial"],
        ["Evaluation", "5-fold by person, person average removed", "fixed validation people + leave-one-person-out"],
        ["Seeds", "3 for the cross-modal test", "5 for every analysis"]],
       widths=["30%", "35%", "35%"])}
<p>Because the two pipelines differ in data, code and model, a conclusion they share is much stronger than either alone.</p>

<h3>7.1 Do the two pipelines agree?</h3>
{table(12, "Agreement between the pipelines (rebuilt in plain language; Maisha's side uses the five-seed results).",
       ["Question", "Muntasir's answer", "Maisha's answer (five seeds)", "What it means"],
       [["<b>Does training with fMRI as well as fNIRS make fNIRS prediction better?</b>", f"<b>No.</b> Slightly worse ({v6c1['mean_diff']:+.3f} on 5 people; −0.007 on 29). Shuffled fMRI works just as well as real fMRI.", f"<b>No.</b> fNIRS-only better in {n_base_better} of 5 seeds ({bm.mean():.3f} vs {fm.mean():.3f}).", "<span class='no'>Both say no.</span> Adding fMRI does not help."],
        ["<b>Can the model predict the blood signal of someone it has never seen?</b>", f"<b>Yes, a little:</b> r = {v6i['TU Berlin hBCI (MI+MA)']['r_demeaned']:.3f}, all 5 test groups positive.", f"<b>No:</b> r = {lo['hefmi_fnirs']['mean_r']:.3f}, same as guessing each person's average ({lo['hefmi_fnirs']['subject_mean_baseline_r']:.3f}).", "<span class='mid'>Different, because the data differ.</span> Clean, repeatable training data are needed."],
        ["<b>Are EEG and blood signals linked inside one person?</b>", "<b>Yes:</b> r = 0.098, positive in 27 of 29 people.", f"<b>Yes:</b> r = {wsp['hefmi_fnirs']['pearson']:.3f} vs chance 0.021 (p = 0.005).", "<span class='yes'>Both say yes.</span> The link is real but personal."],
        ["<b>Does it work on a completely different dataset?</b>", "<b>No:</b> 0.012 (HEFMI) and 0.000 (ds002739).", f"<b>No:</b> {AGG['tuberlin_external_subject_r']['mean']:+.3f} (TU Berlin) and {AGG['ds002739_external_subject_r']['mean']:.3f} (ds002739).", "<span class='no'>Both say no.</span> Nothing transfers between datasets."],
        ["<b>Does the network learn the brain–blood delay?</b>", "<b>No:</b> its answer follows the limits it is given (2.0 s, then 0.07 s).", "<b>No:</b> fixed at 5.00 s; does not match measured delays.", "<span class='no'>Both say no.</span> Measure it directly instead (6.4 s, stable within a person)."],
        ["<b>Can the confidence score tell good predictions from bad?</b>", "<b>No:</b> confidence–error link only 0.02.", f"<b>Inside HEFMI, yes</b> ({AGG['reliability_corr_internal_test']['mean']:.2f}); <b>on new data, no</b> ({AGG['reliability_corr_external']['mean']:.2f}; abstention unstable).", "<span class='mid'>Partly disagree.</span> Both fail on new datasets."],
        ["<b>Should global signal regression be used?</b>", "<b>No:</b> removes about 83% of the real response.", f"<b>No:</b> removes {gsr.min():.0f}–{gsr.max():.0f}% of the signal, part of it real.", "<span class='yes'>Both say turn it off.</span>"],
        ["<b>Does using some of a new person's data help?</b>", "Not tested this way.", "<b>Yes, mostly without EEG:</b> 0.191 with EEG vs 0.160 from the person's own average.", "EEG adds only about +0.03."]],
       widths=["24%", "25%", "27%", "24%"])}

<h3>7.2 What can be claimed?</h3>
{table(13, "Status of each claim (rebuilt in plain language).",
       ["Claim in plain words", "Answer", "Why"],
       [["EEG can predict fNIRS in new people, if the training data are clean", "<span class='yes'>Yes</span>", "Muntasir: r = 0.158 on clean data, all 5 test groups positive, p = 0.001. Maisha's noisy data shows the opposite, which is why clean data matter."],
        ["EEG can predict fMRI in new people", "<span class='mid'>Weak</span>", f"Muntasir: r = 0.064. Maisha: at chance ({AGG['fmri_internal_subject_r']['mean']:.3f} vs 0.088) and worse than the person-average shortcut. Only 7–10 people."],
        ["EEG and fNIRS are linked inside one person", "<span class='yes'>Yes</span>", "Both pipelines find r ≈ 0.09–0.10, clearly above chance."],
        ["Training with fMRI makes predictions better", "<span class='no'>No</span>", "Both pipelines: no gain. Muntasir's shuffled test and Maisha's five seeds agree."],
        ["The model works on other datasets", "<span class='no'>No</span>", "Both pipelines ≈ 0 on every outside dataset."],
        ["The brain–blood delay is a stable personal measure", "<span class='yes'>Yes, measured directly</span>", "Muntasir: 6.4 s on average, consistent between sessions (ICC 0.53)."],
        ["The network itself learns the delay", "<span class='no'>No</span>", "Muntasir: follows its limits. Maisha: fixed 5 s, does not match measured delays."],
        ["Training with blood signals gives a better EEG representation", "<span class='no'>No</span>", "Muntasir: no better than a random network. Maisha: switched off after earlier tests showed no benefit."],
        ["The confidence numbers are honest (calibrated)", "<span class='mid'>Same dataset only</span>", "Muntasir: calibration error 0.27 → 0.02. Maisha: internal 0.07–0.14, external large and unstable (up to 0.89)."],
        ["The confidence score can spot bad predictions", "<span class='mid'>Mixed</span>", "Maisha: yes inside HEFMI (0.42, every seed); both pipelines: no on new datasets."],
        ["Haemorrhage patients differ from healthy people", "<span class='mid'>Possible, not proven</span>", "Muntasir: HbO–HbR link 0.81 vs 0.64 (corrected p = 0.003), but light-travel differences in injured heads could explain it."],
        ["EEG adds useful information when personalising", "<span class='mid'>Very little</span>", "Maisha: +0.03 over the person's own average, 7 people, not significant."]],
       widths=["34%", "16%", "50%"])}

<h2>8. Limitations</h2>
<ul>
<li><b>Noisy training target.</b> HEFMI fNIRS is mostly body signal (reliability 0.08); this caps every prediction result in this notebook.</li>
<li><b>Few validation people:</b> 7 for HEFMI and 3 for fMRI; several effects swing between seeds because of this.</li>
<li><b>Unverified timing</b> in three of six splits.</li>
<li><b>Seed-dependent calibration people</b> make the abstention rule unstable.</li>
<li><b>External tests change several things at once</b> (dataset, device, electrode layout, task), so the cause of failure cannot be isolated.</li>
<li><b>Deterministic analyses</b> (SD = 0 across seeds) are not tested for stability by the seed repetition.</li>
</ul>

<h2>9. Recommended next steps</h2>
<ol>
<li>Repeat the personalisation test on TU Berlin (29 people, clean target), always against the "person's own average, no EEG" shortcut.</li>
<li>Fix the calibration people (not seed-dependent) or use leave-one-person-out calibration, then re-test abstention.</li>
<li>Verify event timing against raw marker files for the three WARN splits.</li>
<li>Report fMRI analyses as exploratory unless more people are added.</li>
</ol>

<h2>Appendix: sources</h2>
<p style="text-align:left">Notebook <code>{NBNAME}.ipynb</code> (saved outputs: seed 2718); per-seed summaries <code>output/summary_{VER}_seed&lt;seed&gt;.json</code>; executed copies <code>output/executed_seed_notebooks/</code> (seed 1337 from git commit 8d26941); five-seed aggregate <code>output/multi_seed_summary_{VER}.json</code>; Muntasir's results <code>neurovascnet_v6_outputs/</code> and <code>NeuroVascNet_Research_Report_v2.pdf</code>. All statistics use people as the unit; confidence intervals are subject-level bootstraps; p-values are permutation-based and never reported as zero. This report is generated by <code>scripts/build_maisha_report_pdf.py</code>.</p>
"""

D.render(body, "NeuroVascNet v3.1-v2.3: Five-Seed Results")
