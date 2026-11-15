"""Builds NeuroVascNet_Research_Report.ipynb (narrative + live result loading)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    c = nbf.v4.new_code_cell(s.strip("\n"))
    c.metadata["jupyter"] = {"source_hidden": True}
    c.metadata["tags"] = ["hide-input"]
    cells.append(c)


# ---------------------------------------------------------------- title
md(r"""
# NeuroVascNet — Research Report

### Learning hemodynamic responses from EEG with cross-modal (fNIRS + fMRI) supervision: what the evidence supports

**Project:** NeuroVascNet · **Report date:** 2 October 2026
**Pipelines covered:** Final v6 pipeline (`NeuroVascNet_FINAL_muntasir.ipynb`, `neurovascnet_v6_outputs/`) and Maisha's v3.1-v2.3 pipeline (`NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED.ipynb`, `output/`)

---

| | |
|---|---|
| **Question** | Can one EEG encoder, trained with paired EEG–fNIRS *and* EEG–fMRI data, predict hemodynamic responses from EEG alone on new people? |
| **Short answer** | **Partly.** EEG predicts trial-to-trial fNIRS response variation on unseen subjects at **r = 0.158** when the training target is clean. Adding fMRI supervision does **not** help, and nothing transfers to a different cohort. |
| **Strongest positive results** | Cross-subject fNIRS prediction on TU Berlin hBCI (r = 0.158, p = 0.001); a reliable model-free neurovascular delay (6.4 s, ICC = 0.53); real within-subject coupling (Maisha: r = 0.094, p = 0.005); one clinical marker that survives correction (ICH vs controls HbO/HbR coupling, p_Holm = 0.003). |
| **Main negative results** | Cross-modal supervision does not beat single modality, in both pipelines. External validation is ≈ 0 for both external cohorts. The learned delay follows its bounds, not the physiology. The reliability gate calibrates but does not rank errors. |
| **Root cause** | Most of the earlier failure came from the **target data**, not the model: HEFMI fNIRS has split-half reliability 0.08 and is dominated by systemic signal. |

> All numbers in this report are loaded from the saved result files when the notebook runs. Nothing is retrained. Code cells are hidden by default; expand them to see the source of every number.
""")

code(r"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
V6 = ROOT / "neurovascnet_v6_outputs"
MAISHA = ROOT / "output"
FIG = V6 / "figures"

V6_SUMMARY = json.loads((V6 / "summary.json").read_text())
M_SUMMARY = json.loads((MAISHA / "summary_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json").read_text())
M_PHYSIO = json.loads((MAISHA / "personalized_physio_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json").read_text())
csv = lambda name, **kw: pd.read_csv(V6 / name, **kw)

# reference categorical palette (validated, fixed order) + recessive neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED, GRID = "#9a9893", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

def table(df, caption=None, fmt=3):
    if caption:
        display(Markdown(f"**{caption}**"))
    display(df.style.format(precision=fmt, na_rep="—").hide(axis="index"))

print("Loaded:", V6_SUMMARY["mode"], "| Maisha seed", M_SUMMARY["provenance"]["seed"],
      "|", M_SUMMARY["provenance"]["model_version"])
""")

# ---------------------------------------------------------------- contents
md(r"""
## Contents

1. Research question and system design
2. Datasets and their roles
3. How the project got here: timeline of attempts
4. Data-quality audit: the target data limits the result
5. Final v6 pipeline results (claims 1–5, clinical application, supporting analyses)
6. Maisha's v3.1-v2.3 pipeline findings
7. Where the two pipelines agree and disagree
8. Scorecard
9. Limitations and threats to validity
10. Conclusions, recommended paper framing, and next steps
11. Appendix: original figures and file index
""")

# ---------------------------------------------------------------- 1
md(r"""
## 1. Research question and system design

**Thesis.** One shared EEG encoder, supervised during training by paired **EEG–fNIRS** and **EEG–fMRI** recordings, takes **EEG only** at deployment and outputs:

1. a hemodynamic-aware latent representation,
2. a predicted hemodynamic response (HbO/HbR-like and BOLD-like),
3. a neurovascular delay descriptor,
4. a calibrated reliability score that lets the system abstain.

fNIRS and fMRI are used **only as training supervision**. This is prediction, not fusion: fusion would need fNIRS or fMRI at deployment and would break the EEG-only premise.

```
EEG (n_ch × 6000 @ 200 Hz)
      │
      ▼  per-dataset adapter (1×1 conv; montages differ)
      ▼  SHARED TRUNK  ◄── receives gradients from both modalities
      ├──► fNIRS head → HbO/HbR (mean + log-variance)
      ├──► fMRI head  → ROI BOLD (mean + log-variance)
      ├──► delay module (bounded timing parameter)
      └──► reliability gate (MC-dropout + predicted variance → calibration → abstain)
```

**Structural constraint.** No public dataset records the same people in both fNIRS and fMRI. The two supervision streams therefore come from **disjoint subjects**. The shared trunk is the only place they can interact.
""")

# ---------------------------------------------------------------- 2
md(r"""
## 2. Datasets and their roles

Four public datasets, 100 subjects in total. The two pipelines assign the fNIRS roles differently, and this difference matters for interpreting every result that follows.

| Dataset | Modality | Task | Subjects | **Final v6 role** | **Maisha v2.3 role** |
|---|---|---|---|---|---|
| TU Berlin hBCI | EEG–fNIRS | motor imagery + mental arithmetic | 29 | **fNIRS training** | external fNIRS test |
| HEFMI-ICH | EEG–fNIRS | motor imagery (17 controls + 20 ICH patients) | 37 | external fNIRS test + clinical analysis | **fNIRS training** (30 train / 7 val) |
| OpenNeuro ds002336 | EEG–fMRI | motor imagery + neurofeedback | 10 | fMRI training | fMRI training (7 train / 3 val) |
| OpenNeuro ds002739 | EEG–fMRI | perceptual decision (**not motor**) | 24 | external fMRI test (cross-task) | external fMRI test (cross-task) |

**Why v6 swapped the fNIRS roles.** The decision was made on target quality measured **before any model was fitted** (Section 4). HEFMI's fNIRS target is unreliable and systemic-dominated; TU Berlin hBCI's target is not. The swap is declared in the v6 notebook. It was not chosen after seeing model results. The reverse fMRI assignment was also tested and was worse (fMRI arm r = 0.0001), so the original fMRI roles were kept.

**Note on the TU Berlin directory.** An earlier concern was that the external TU Berlin set loaded was the 26-subject *mental-workload* dataset rather than the 29-subject motor hBCI set. Both current pipelines load `TU Berlin hBCI EEG+FNRI` and report 29 subjects, so the results below use the correct dataset.
""")

code(r"""
ds = csv("data_summary.csv").rename(columns={
    "role": "v6 role", "dataset": "Dataset", "modality": "Modality", "subjects": "Subjects",
    "trials": "Trials", "eeg_ch": "EEG ch", "targets": "Targets",
    "reliability": "Split-half reliability", "ceiling": "Prediction ceiling", "hbo_hbr": "HbO/HbR corr"})
table(ds, "Table 2.1 — Datasets as loaded by the v6 pipeline (from data_summary.csv)")
""")

# ---------------------------------------------------------------- 3
md(r"""
## 3. How the project got here: timeline of attempts

| Phase | What was tried | What it showed | Status now |
|---|---|---|---|
| **v3 – v3.3** (Jun–Jul) | Full deep model (EEGNet + TCN) trained on HEFMI + ds002336 with GSR and per-window z-scoring. Fixes tested: alignment removal, capped/reweighted sampling, Deep CORAL | Cross-subject **waveform** prediction ≈ 0 for both modalities. None of the three fixes moved it. Capped sampling cut fMRI overfitting by 24% | Superseded: the target was the problem (Section 4) |
| **Amplitude line** (Jul) | Ridge / small MLP on response amplitude: trial-level, CORAL, shared latent | Within-subject trial-level fNIRS r = 0.33; cross-subject fNIRS + CORAL r = 0.25; fMRI LOSO r = 0.15 | Exploratory. Not reproduced under the stricter v6 / v2.3 protocols, so not used as headline numbers |
| **Data-quality audit** (25–26 Jul) | 610 recordings / 63 GB / 97 subjects audited; 12-variant target-preprocessing sweep | HEFMI target reliability 0.257 raw → **0.127 after our own pipeline**. GSR removes 83% of the evoked response. HbO/HbR = +0.90 (systemic) | **Root cause.** Led to GSR off and to the TU Berlin swap |
| **Maisha v3.1-v2.3** (Aug) | Independent rebuild on HEFMI: clinically stratified split, GSR A/B test, alignment QA, LOSO, compact endpoint, personalised calibration, locked abstention | See Section 6. Within-subject coupling is real; cross-subject is at the subject-mean baseline | 1 of 5 seeds run |
| **v6 hBCI → FINAL** (Aug) | TU Berlin hBCI as fNIRS training set, subject-wise 5-fold CV with subject demeaning, five pre-declared claims, controls, clinical ICH analysis | See Section 5. First robust positive cross-subject result (r = 0.158) | **Current reference pipeline** |
""")

# ---------------------------------------------------------------- 4
md(r"""
## 4. Data-quality audit: the target data limits the result

A model cannot predict a target better than that target agrees with itself. **Split-half reliability** measures how well two independent halves of a subject's trials agree. It sets a hard ceiling on any achievable correlation. A real evoked fNIRS response should also show **negative** HbO/HbR correlation: oxygenated haemoglobin rises while deoxygenated haemoglobin falls. A strongly positive correlation means both chromophores move together, which points to systemic physiology (blood pressure, breathing) rather than neural activity.
""")

code(r"""
d = csv("data_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
order = d.sort_values("reliability")
cols = [ORANGE if r < 0.3 else BLUE for r in order.reliability]
ax = axes[0]
ax.barh(order.dataset, order.reliability, color=cols, height=0.55)
for y, (v, c) in enumerate(zip(order.reliability, order.ceiling)):
    ax.text(v + 0.01, y, f"{v:.2f}  (ceiling {c:.2f})", va="center", color=INK, fontsize=8.5)
ax.set_xlim(0, 1); ax.set_xlabel("split-half reliability of the target")
ax.set_title("Target reliability by dataset"); ax.grid(axis="y", visible=False)

f = d.dropna(subset=["hbo_hbr"])
ax = axes[1]
ax.barh(f.dataset, f.hbo_hbr, color=[ORANGE if v > 0 else BLUE for v in f.hbo_hbr], height=0.45)
ax.axvline(0, color=MUTED, lw=1)
for y, v in enumerate(f.hbo_hbr):
    ax.text(v + (0.03 if v > 0 else -0.03), y, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", color=INK, fontsize=8.5)
ax.set_xlim(-1, 1); ax.set_xlabel("HbO/HbR correlation  (neural response < 0 < systemic)")
ax.set_title("fNIRS signal origin"); ax.grid(axis="y", visible=False)
fig.suptitle("Figure 4.1 — HEFMI's fNIRS target is unreliable and systemic-dominated (orange = fails)",
             x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(); plt.show()
""")

md(r"""
**Findings of the audit**

1. **HEFMI fNIRS is systemic-dominated.** HbO/HbR correlation is **+0.92** and split-half reliability is **0.083**, so the best achievable correlation is about 0.29. TU Berlin hBCI has HbO/HbR **−0.44** and reliability **0.50**, so its ceiling is about 0.71.
2. **Our own preprocessing made it worse.** Global-signal regression (GSR) removed about **83%** of the evoked HbO amplitude. Per-window z-scoring then removed what remained. On HEFMI, target reliability fell from 0.257 (raw) to **0.127** (after the pipeline). "Cross-subject waveform ≈ 0" in the v3 phase was therefore mostly a statement about the target, not about EEG.
3. **Preprocessing alone cannot rescue HEFMI.** The best of 12 preprocessing variants still gave cross-subject amplitude r = 0.062. The fix had to be the data source.
4. **fMRI targets are usable.** ds002336 reliability is 0.70. ds002739 has excellent image quality (tSNR 190) but low task-locked reliability (0.19) because its task is not motor.
5. **Two of the audit's own thresholds misfired** and are not used as exclusion rules. The line-noise rule flagged 36/37 HEFMI subjects for an artefact the notch filter already removes. The gradient-residual metric is invalid on ds002739 because that data is already band-limited.

**Consequence.** Both later pipelines turned GSR **off**. Maisha's notebook confirmed this independently with an A/B test: GSR removed 41% of target variance, and the removed component had a spatial uniformity of only r = 0.47, so it was not purely global noise. v6 also moved fNIRS training to TU Berlin hBCI.
""")

# ---------------------------------------------------------------- 5
md(r"""
## 5. Final v6 pipeline results

**Configuration:** fNIRS training = TU Berlin hBCI (29 subjects, 3,480 trials, MI + MA); fMRI training = ds002336 (10 subjects, 384 trials); external tests = HEFMI (37) and ds002739 (24); GSR off; subject-wise cross-validation.

**The metric that matters is the subject-demeaned correlation.** Trials from one person share a mean, so a model that only learns *who* the subject is can score well while knowing nothing about the brain. Removing each subject's mean drives this "no-EEG baseline" to zero. Whatever correlation survives is the EEG's contribution.

### 5.1 Internal cross-subject prediction
""")

code(r"""
cvs = {"TU Berlin hBCI (fNIRS)": "cv_TU_Berlin_hBCI_(MI+MA).csv",
       "ds002336 (fMRI)": "cv_ds002336.csv",
       "ds002739 (fMRI, as train — reference)": "cv_ds002739.csv"}
rows = []
for name, f in cvs.items():
    c = csv(f)
    rows.append({"Dataset": name, "Folds": len(c), "Test subjects": int(c.test_subjects.sum()),
                 "r raw": c.r_raw.mean(), "No-EEG baseline": c.r_no_eeg_baseline.mean(),
                 "r subject-demeaned (honest)": c.r_demeaned.mean(),
                 "min fold": c.r_demeaned.min(), "max fold": c.r_demeaned.max()})
cvt = pd.DataFrame(rows)
table(cvt, "Table 5.1 — Subject-wise 5-fold CV, mean over folds")

fig, ax = plt.subplots(figsize=(8.5, 3))
for i, (name, f) in enumerate(cvs.items()):
    c = csv(f)
    ax.scatter(c.r_demeaned, [i] * len(c), s=46, color=BLUE, edgecolor="#fcfcfb", lw=2, zorder=3)
    m = c.r_demeaned.mean()
    ax.plot([m, m], [i - 0.25, i + 0.25], color=INK, lw=2)
    ax.text(m, i + 0.32, f"mean {m:.3f}", ha="center", fontsize=8.5, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(cvs))); ax.set_yticklabels(list(cvs)); ax.set_ylim(-0.6, len(cvs) - 0.1)
ax.set_xlabel("subject-demeaned r per fold (dots)"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 5.1 — Real for TU Berlin fNIRS, small for ds002336, absent for ds002739")
fig.tight_layout(); plt.show()
""")

md(r"""
- **TU Berlin hBCI fNIRS: r = 0.158 (p = 0.001)** on subjects never seen in training. All five folds are positive (0.134 – 0.240). This is the project's first robust cross-subject positive result. It sits at about 22% of the dataset's reliability ceiling (0.71).
- **ds002336 fMRI: r = 0.064 pooled (fold mean 0.062; p = 0.001)**. The raw r of about 0.23 is almost entirely subject identity; the no-EEG baseline alone scores 0.23.
- **ds002739** shows nothing as a training set (≈ 0.00). This is consistent with its non-motor task and low target reliability.

### 5.2 How much does evaluation protocol inflate the score?
""")

code(r"""
pa = csv("protocol_ablation.csv")
labels = ["Trial-level split\n(subject on both sides)", "+ subject-wise CV",
          "No-EEG baseline\n(subject identity only)", "+ subject demeaning\n(HONEST)"]
fig, ax = plt.subplots(figsize=(8.5, 2.9))
colors = [GRAY, GRAY, GRAY, BLUE]
ax.barh(labels[::-1], pa.r.values[::-1], color=colors[::-1], height=0.55)
for y, v in enumerate(pa.r.values[::-1]):
    ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 0.34); ax.set_xlabel("r on TU Berlin hBCI"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 5.2 — A naive trial-level split nearly doubles the score (0.297 vs 0.158)")
fig.tight_layout(); plt.show()
""")

md(r"""
Splitting trials rather than subjects gives **0.297**, which is 88% higher than the honest **0.158**. Any published number from a trial-level split on this kind of data should be read with that inflation in mind. All later results use the honest protocol.

### 5.3 Claim 1 — Does cross-modal supervision beat single modality?

This is the core thesis. The design keeps the trunk, training budget and evaluation fixed. The only difference is whether the encoder also receives fMRI supervision.
""")

code(r"""
c1 = V6_SUMMARY["claim1"]
cs = csv("controls_summary.csv")
cs.insert(0, "Analysis", ["29-subject control", "29-subject control", "29-subject control", "fMRI cohort (10)"])
cs = cs.rename(columns={"comparison": "Comparison", "n_subjects": "n subj", "mean_diff": "Mean Δr",
                        "ci_low": "CI low", "ci_high": "CI high", "p_signflip": "p (sign-flip)",
                        "n_positive": "Subjects favouring 1st arm", "cohort": "Cohort"})
head = pd.DataFrame([{"Analysis": "Primary (5 held-out subj × 3 seeds)", "Comparison": "cross-modal − single",
                      "n subj": c1["n_subjects"], "Mean Δr": c1["mean_diff"], "CI low": c1["ci_low"],
                      "CI high": c1["ci_high"], "p (sign-flip)": c1["p_signflip"],
                      "Subjects favouring 1st arm": c1["n_subjects_positive"], "Cohort": "fnirs"}])
table(pd.concat([head, cs], ignore_index=True), "Table 5.3 — Cross-modal vs single-modality supervision")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
lw = csv("loss_weight_search.csv")
ax = axes[0]
ax.plot(lw.w_fmri, lw["diff"], color=BLUE, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=2)
ax.axhline(0, color=MUTED, lw=1); ax.set_xscale("log"); ax.set_xticks(lw.w_fmri); ax.set_xticklabels(lw.w_fmri)
ax.set_xlabel("fMRI loss weight"); ax.set_ylabel("joint − single r")
ax.set_title("Loss-weight search")
bs = csv("bottleneck_sweep.csv")
ax = axes[1]
ax.plot(bs.bottleneck, bs.joint_minus_single, color=BLUE, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=2)
ax.axhline(0, color=MUTED, lw=1); ax.set_xscale("log", base=2); ax.set_xticks(bs.bottleneck); ax.set_xticklabels(bs.bottleneck)
ax.set_xlabel("trunk bottleneck width"); ax.set_title("Bottleneck sweep")
fig.suptitle("Figure 5.3 — No setting makes fMRI supervision reliably help fNIRS prediction (above 0 = helps)",
             x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(); plt.show()
""")

md(r"""
**Verdict: NOT CONFIRMED.** The primary test gives Δr = −0.018 (95% CI −0.035 to −0.0003), with only 1 of 5 subjects favouring cross-modal training. The larger 29-subject control gives Δr = −0.007 (CI −0.017 to +0.003, p = 0.18). Joint training with **shuffled** fMRI targets scores the same as real joint training (Δ = +0.006, p = 0.18), so the fMRI signal adds nothing that noise would not. Shuffled-joint is significantly *worse* than single (p = 0.002), which suggests a second dataset mainly costs trunk capacity. A loss-weight search found only a negligible +0.008 at weight 1.0. Every bottleneck width favours single modality.

### 5.4 Claim 2 — External validation on held-out cohorts
""")

code(r"""
ev = csv("external_validation.csv", index_col=0).reset_index().rename(columns={"index": "Cohort"})
ev = ev[["Cohort", "modality", "internal_r", "external_r", "p", "drop_pct", "external_ceiling",
         "pct_of_external_ceiling", "subjects", "montage_alignment"]]
ev.columns = ["Cohort", "Modality", "Internal r", "External r", "p", "Drop %", "External ceiling",
              "% of ceiling", "Subjects", "Montage alignment"]
table(ev, "Table 5.4 — Train once, evaluate once on unseen cohorts")
fs = csv("claim2_fewshot_calibration.csv")
fs.columns = ["Calibration subjects used", "External r", "Subjects scored"]
table(fs, "Table 5.5 — Few-shot calibration on HEFMI does not recover performance", fmt=4)
""")

md(r"""
**Verdict: PARTIAL (in practice, fails).** HEFMI r = 0.012 is statistically above zero (p = 0.01) but 92% below internal performance and only 4% of HEFMI's own ceiling. ds002739 is exactly zero. Calibrating on 2–8 HEFMI subjects does not help. Three things are confounded here and cannot be separated with the current data: different montages (HEFMI fNIRS had to be pooled to mean HbO/HbR), different tasks (ds002739 is not motor), and different target quality (HEFMI is systemic-dominated).

### 5.5 Claim 3 — Neurovascular delay
""")

code(r"""
c3 = V6_SUMMARY["claim3"]
icc = csv("delay_model_free_icc.csv")
dl = csv("claim3_delay.csv"); dl.insert(0, "Variant", "learned, bounded")
df_ = csv("claim3_delay_free.csv"); df_.insert(0, "Variant", "learned, free")
mf = pd.DataFrame([{"Variant": "MODEL-FREE (cross-correlation)", "window": "—",
                    "mean_s": icc.mean_s[0], "between_subject_sd": icc.between_subject_sd[0], "frac_of_range": np.nan}])
t = pd.concat([mf, dl, df_], ignore_index=True)
t.columns = ["Delay estimate", "Window", "Mean delay (s)", "Between-subject SD (s)", "Fraction of window moved"]
table(t, "Table 5.6 — Model-free delay vs learned delay module")
display(Markdown(f"Model-free delay reliability across sessions: **ICC = {icc.icc[0]:.2f}** "
                 f"({icc.n_people[0]} people, {icc.mean_sessions[0]:.1f} sessions each; surrogate null ICC "
                 f"{icc.null_icc_mean[0]:.3f}, p = {icc.surrogate_p[0]:.4f})."))
""")

md(r"""
**Verdict: HOLDS for the model-free delay; FAILS for the learned module.** These are two different results and should be reported separately.

- **Model-free delay is a real, stable trait.** The EEG→HbO peak lag measured directly from cross-correlation is **6.4 s**, inside the physiological 2–8 s band, varies between people (SD 1.06 s), and is reproducible across sessions (**ICC 0.53**, p = 0.0005 against surrogates).
- **The learned delay module reports its own bounds.** With a [2, 8] s window it sits at 2.01 s (the lower bound). With [0, 20] s it moves to 0.07 s. Unbounded, it goes to 0.09 s. Between-subject spread is ≈ 0. Landing inside 2–8 s therefore proves nothing about physiology. Maisha's pipeline found the same (Section 6).

### 5.6 Claim 4 — Downstream EEG-only transfer
""")

code(r"""
c4 = csv("claim4_downstream.csv")
fig, ax = plt.subplots(figsize=(8.5, 3.1))
o = c4.iloc[::-1]
col = [BLUE if "supervised" == e.split()[-1] or e.startswith("cross") else GRAY for e in o.encoder]
ax.barh(o.encoder, o.accuracy, xerr=o.sd, color=col, height=0.55, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
for y, (v, s) in enumerate(zip(o.accuracy, o["sd"])):
    ax.text(v + s + 0.006, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 0.56); ax.set_xlabel("linear-probe task accuracy (mean ± SD)"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 5.6 — Cross-modal encoder (blue) does not beat a random or shuffled-target encoder")
fig.tight_layout(); plt.show()
""")

md(r"""
**Verdict: FAILS.** The frozen cross-modal encoder (0.374) is no better than an encoder trained on **shuffled** targets (0.362) or a **randomly initialised** one (0.381). It does beat raw band-power (0.329). Fine-tuning reaches 0.474 whether it starts from the supervised encoder or from scratch, so hemodynamic pre-training gives no head start. A layer-by-layer probe (`claim4_depth_probe.csv`) shows accuracy flat at 0.28–0.31 from the first convolution onward.

### 5.7 Claim 5 — Reliability, calibration and selective prediction
""")

code(r"""
c5 = V6_SUMMARY["claim5"]
cov = csv("coverage_curve.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
ax = axes[0]
ax.plot(cov.coverage * 100, cov.r, color=BLUE, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=2)
ax.invert_xaxis(); ax.set_ylim(0.10, 0.20)
ax.set_xlabel("coverage kept (%)  → more abstention"); ax.set_ylabel("r on kept trials")
ax.set_title("Selective prediction: flat (gate does not rank)")
ax = axes[1]
ax.bar(["ECE before", "ECE after Platt"], [c5["ece_before"], c5["ece_after"]], color=[GRAY, BLUE], width=0.5)
for x, v in enumerate([c5["ece_before"], c5["ece_after"]]):
    ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
ax.set_ylabel("expected calibration error"); ax.set_title("Calibration: fixed by Platt scaling")
ax.grid(axis="x", visible=False)
fig.suptitle("Figure 5.7 — The reliability layer calibrates but cannot tell good predictions from bad ones",
             x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(); plt.show()
rk = pd.DataFrame([
    ("trial: MC-dropout uncertainty vs error", c5["corr_unc_err"]),
    ("trial: combined score vs error", c5["corr_trial_combined"]),
    ("trial: domain score vs error", c5["corr_trial_domain"]),
    ("subject: combined score vs error", c5["corr_subject_combined"]),
    ("subject: domain score vs error", c5["corr_subject_domain"])], columns=["Ranking test", "Correlation"])
table(rk, f"Table 5.7 — Ranking tests (n = {c5['n_subjects_ranked']} subjects; a working gate needs a clearly positive value)")
""")

md(r"""
**Verdict: FAILS on ranking, passes on calibration.** Platt scaling cuts ECE from 0.271 to 0.021, so the confidence numbers are on the right scale. But uncertainty does not track error (trial-level r = 0.02), and abstaining on the least confident 80% of trials leaves r unchanged (0.160 → 0.154). The subject-level scores even point the wrong way (−0.28, −0.33). A calibrated gate that cannot rank is not useful for abstention.

### 5.8 Supporting analysis — within-subject probe
""")

code(r"""
ws = csv("within_subject_probe.csv")
within, cross = V6_SUMMARY["within_subject"]["within_r"], V6_SUMMARY["within_subject"]["cross_r"]
fig, ax = plt.subplots(figsize=(8.5, 2.8))
s = ws.sort_values("r").reset_index(drop=True)
ax.bar(range(len(s)), s.r, color=[BLUE if v > 0 else ORANGE for v in s.r], width=0.7)
ax.axhline(within, color=INK, lw=1.5, ls="--"); ax.text(0, within + 0.008, f"within-subject mean {within:.3f}", fontsize=8.5)
ax.axhline(cross, color=AQUA, lw=1.5); ax.text(0, cross + 0.008, f"cross-subject {cross:.3f}", fontsize=8.5, color=INK)
ax.set_xticks([]); ax.set_xlabel(f"subjects sorted by r (n = {len(ws)}, 36 test trials each)"); ax.set_ylabel("r")
ax.set_title("Figure 5.8 — Within-subject models (trained on one person) score below the cross-subject model")
ax.grid(axis="x", visible=False); fig.tight_layout(); plt.show()
""")

md(r"""
Within-subject mean r = **0.098**; 27 of 29 subjects are positive, but it is below the cross-subject **0.158**. With about 100 training trials per person, a personal model has too little data. Pooling 24 other subjects is worth more than matching the individual. **The target is not the bottleneck on TU Berlin; per-person data volume is.**

### 5.9 Clinical application — ICH patients vs healthy controls (HEFMI)

These coupling descriptors are measured directly from the data, so this analysis does not depend on prediction accuracy.
""")

code(r"""
cl = csv("clinical_comparison.csv")
cl = cl.rename(columns={"measure": "Measure", "kind": "Kind", "controls": "Controls (17)", "patients": "ICH patients (20)",
                        "p_holm": "p (Holm)", "survives_correction": "Survives correction"}).drop(columns=["sig"])
table(cl, "Table 5.9 — ICH vs controls; primary measures Holm-corrected, artefact controls reported alongside", fmt=4)
pp = csv("clinical_per_person.csv")
fig, ax = plt.subplots(figsize=(6.5, 3))
for i, (g, c) in enumerate([("Normals", BLUE), ("Patients", ORANGE)]):
    v = pp[pp.group == g].hbo_hbr.values
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
    ax.scatter(v, i + jitter, s=40, color=c, edgecolor="#fcfcfb", lw=2, zorder=3)
    ax.plot([np.mean(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Controls", "ICH patients"]); ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("per-person HbO/HbR correlation"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 5.9 — Patients show tighter HbO/HbR coupling")
fig.tight_layout(); plt.show()
""")

md(r"""
**One measure survives correction:** ICH patients show higher HbO/HbR correlation (0.81 vs 0.64, p = 0.0009, **p_Holm = 0.003**). Head motion, EEG amplitude, fNIRS amplitude and trial count do not differ between groups, so the most obvious acquisition confounds are ruled out. EEG–hemodynamic coupling strength (p_Holm = 0.29) and coupling lag (p_Holm = 1.0) do **not** differ.

**Caution:** a haemorrhage changes skull and tissue optics, and higher HbO/HbR correlation is also the signature of systemic dominance (Section 4). This result is a candidate marker, not yet evidence of altered neurovascular coupling. It needs an optical-pathlength control before it can be interpreted physiologically.
""")

# ---------------------------------------------------------------- 6
md(r"""
## 6. Maisha's v3.1-v2.3 pipeline findings

Maisha's notebook is an **independent rebuild** that keeps HEFMI as the fNIRS training set. It adds a set of rigour safeguards: a pre-declared clinically stratified split, a GSR A/B test, timing QA, LOSO, compact physiological endpoints with strong baselines, personalised calibration, a locked abstention rule, delay validation against a pre-registered reference, and an automatic claim guard that blocks wording the evidence does not support.

**Run status:** seed 1337 completed; the other 4 seeds of the declared 5-seed protocol (20260726, 7, 42, 2718) have **not** been run. Every number below is from a single seed.

### 6.1 Protocol corrections that matter beyond this notebook

1. **Split confound found and fixed.** The legacy alphabetical holdout put **only ICH patients** into validation, because every `Normals_*` ID sorts before every `Patients_*` ID. Any earlier HEFMI validation number from that split is clinically confounded. The fix is a SHA-256 ranking within clinical group (frozen seed 20260807): 3 of 17 controls and 4 of 20 patients go to validation.
2. **GSR A/B confirms the audit.** GSR removed 41% of target variance, and the removed part had spatial uniformity of only r = 0.47, so it contained spatially structured signal. Decision: GSR off.
3. **Timing QA blocks alignment claims.** Peak EEG→hemodynamic lag falls inside 2–8 s for only 14% of HEFMI-validation and 24% of TU Berlin subjects (median 9.7 s). Event clocks could not be verified, so the claim guard restricts every alignment-based claim.
4. **TU Berlin units harmonised** (mM → M, ×0.001; external/train scale ratio 9.6, inside QA bounds).
""")

code(r"""
av = M_SUMMARY["alignment_verification"]
t = pd.DataFrame([{"Split": v["name"], "Subjects": int(v["n_subjects"]), "Median peak lag (s)": v["median_peak_lag_s"],
                   "% in 2–8 s": 100 * v["frac_in_physiological_band"], "% at grid edge": 100 * v["frac_at_grid_edge"],
                   "Median peak |r|": v["median_peak_abs_r"], "Status": v["status"].split(" -")[0]} for v in av.values()])
table(t, "Table 6.1 — Maisha: EEG→hemodynamic timing QA", fmt=2)
""")

md(r"""
### 6.2 Prediction results
""")

code(r"""
iv, ex = M_SUMMARY["internal_validation"], M_SUMMARY["external_ood_stress_tests"]
rows = []
for lab, d in [("HEFMI fNIRS — internal val", iv["hefmi_fnirs"]), ("ds002336 fMRI — internal val", iv["ds002336_fmri"]),
               ("TU Berlin fNIRS — external", ex["tuberlin_hbci_cross_dataset"]), ("ds002739 fMRI — external (cross-task)", ex["ds002739_fmri_cross_task"])]:
    rows.append({"Test set": lab, "Subjects": d["n_subjects"], "r (subject-level)": d["pearson_subject"],
                 "CI low": d["pearson_subject_ci95_subject_bootstrap"][0], "CI high": d["pearson_subject_ci95_subject_bootstrap"][1],
                 "Null mean": d["null_subject_mean"], "p (pairing)": d["p_subject_pairing"], "Effect size": d["practical_size"]})
table(pd.DataFrame(rows), "Table 6.2 — Maisha: waveform prediction vs a subject-level pairing null", fmt=4)

lo, wsp = M_SUMMARY["cross_subject_loso"], M_SUMMARY["within_subject_probe"]
rows = [
    {"Analysis": "LOSO cross-subject — HEFMI fNIRS", "Subjects": lo["hefmi_fnirs"]["n_subjects"], "Model r": lo["hefmi_fnirs"]["mean_r"],
     "Reference": lo["hefmi_fnirs"]["subject_mean_baseline_r"], "Reference is": "subject-mean baseline", "p": lo["hefmi_fnirs"]["p_vs_baseline"]},
    {"Analysis": "LOSO cross-subject — ds002336 fMRI", "Subjects": lo["ds002336_fmri"]["n_subjects"], "Model r": lo["ds002336_fmri"]["mean_r"],
     "Reference": lo["ds002336_fmri"]["subject_mean_baseline_r"], "Reference is": "subject-mean baseline", "p": lo["ds002336_fmri"]["p_vs_baseline"]},
    {"Analysis": "Within-subject — HEFMI fNIRS", "Subjects": wsp["hefmi_fnirs"]["n_subjects"], "Model r": wsp["hefmi_fnirs"]["pearson"],
     "Reference": 0.0213, "Reference is": "shuffled null", "p": wsp["hefmi_fnirs"]["p"]},
    {"Analysis": "Within-subject — ds002336 fMRI", "Subjects": wsp["ds002336_fmri"]["n_subjects"], "Model r": wsp["ds002336_fmri"]["pearson"],
     "Reference": 0.1176, "Reference is": "shuffled null", "p": wsp["ds002336_fmri"]["p"]},
]
table(pd.DataFrame(rows), "Table 6.3 — Maisha: cross-subject (LOSO) vs within-subject", fmt=4)
""")

code(r"""
fig, ax = plt.subplots(figsize=(8.5, 3.0))
items = [("HEFMI LOSO\n(cross-subject)", lo["hefmi_fnirs"]["mean_r"], lo["hefmi_fnirs"]["subject_mean_baseline_r"]),
         ("HEFMI\nwithin-subject", wsp["hefmi_fnirs"]["pearson"], 0.0213),
         ("ds002336 LOSO\n(cross-subject)", lo["ds002336_fmri"]["mean_r"], lo["ds002336_fmri"]["subject_mean_baseline_r"]),
         ("ds002336\nwithin-subject", wsp["ds002336_fmri"]["pearson"], 0.1176)]
x = np.arange(len(items)); w = 0.36
ax.bar(x - w / 2 - 0.01, [i[1] for i in items], w, color=BLUE, label="model")
ax.bar(x + w / 2 + 0.01, [i[2] for i in items], w, color=GRAY, label="baseline / null")
for xi, (_, m, b) in zip(x, items):
    ax.text(xi - w / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=8.5)
    ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items]); ax.set_ylabel("r"); ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
ax.set_title("Figure 6.1 — Only HEFMI within-subject coupling clearly beats its reference (Maisha)")
fig.tight_layout(); plt.show()
""")

md(r"""
**What this shows**

- **Cross-subject prediction on HEFMI equals the subject-mean baseline.** LOSO over all 30 training subjects gives r = 0.026 against a baseline of 0.029 (model − baseline = −0.003, p = 0.70). This test has adequate power, so on *this target* the cross-subject result is a genuine null. On ds002336 the model (0.055) is significantly **worse** than the subject-mean baseline (0.189, p = 0.013), with only 7 subjects.
- **Within-subject coupling is real.** HEFMI within-subject r = **0.094** (CI 0.065–0.127) vs shuffled null 0.021, p = 0.005, 30 subjects. The EEG–fNIRS coupling exists but is person-specific. As Maisha's notebook words it, this is "a transfer/protocol limit, not an EEG information limit."
- **Capacity diagnostic.** With 8–128 windows, the model fits real EEG–fNIRS pairs no better than shuffled pairs (gap +0.004). Nothing learnable is visible at that scale.
- **External tests ≈ 0.** TU Berlin −0.002 (inside null); ds002739 +0.009, statistically above null (p = 0.005) but practically negligible.
- **Amplitude decomposition.** On ds002336 training data, between-subject ranking r = 0.98 while within-subject r = 0.25. Most of what the fMRI head learns is *who* the subject is, not *what* their brain is doing.

### 6.3 Compact endpoint and personalised calibration (v2.3 primary endpoint)

v2.3 moved the primary endpoint from the 180-channel waveform to four compact quantities (HbO mean, HbR mean, HbO peak, HbR trough) inside a response window chosen on training subjects only.
""")

code(r"""
g = M_PHYSIO["global_internal_validation"]
def arm(d, lab):
    return {"Predictor": lab, "r (mean over subjects)": d["mean_subject_r"], "CI low": d["ci95_subject_bootstrap"][0],
            "CI high": d["ci95_subject_bootstrap"][1], "MAE (SD units)": d["mean_subject_mae_sd_units"], "Subjects": d["n_subjects"]}
rows = [arm(g["train_mean"], "Training mean"), arm(g["task_condition_event"], "Task-condition / event mean"),
        arm(g["ridge_bandpower"], "Ridge on EEG band-power")]
table(pd.DataFrame(rows), "Table 6.4 — Maisha: compact endpoint vs simple baselines (HEFMI internal validation)", fmt=4)
display(Markdown(f"Ridge minus best simple baseline MAE: {M_PHYSIO['best_simple_baseline_minus_ridge_mae_sd_units']:+.3f} SD units "
                 f"(p = {M_PHYSIO['ridge_vs_best_simple_baseline_mae_p']:.3f}); best simple baseline = `{M_PHYSIO['best_simple_baseline_by_mae']}`."))

rows = []
for frac, d in M_PHYSIO["personalized_calibration_curve"].items():
    rows.append({"Calibration data from new subject": f"{float(frac):.0%}",
                 "Global model r": d["global_unadapted"]["mean_subject_r"],
                 "Calibration-block mean only r": d["calibration_mean_baseline"]["mean_subject_r"],
                 "Personalised residual ridge r": d["personalized_residual_ridge"]["mean_subject_r"],
                 "Personalised CI low": d["personalized_residual_ridge"]["ci95_subject_bootstrap"][0],
                 "Personalised CI high": d["personalized_residual_ridge"]["ci95_subject_bootstrap"][1],
                 "Δr vs global": d["personalized_minus_global_mean_subject_r"], "p (paired sign-flip)": d["paired_signflip_p"]})
table(pd.DataFrame(rows), "Table 6.5 — Maisha: personalised residual calibration (7 validation subjects; first block calibrates, later windows test)", fmt=4)
""")

md(r"""
- **No predictor beats simple baselines.** EEG ridge (r = 0.045) is no better than predicting the task-condition mean (r = 0.058). The ridge MAE is 0.07 SD *worse* (p = 0.066).
- **Personalisation lifts r, but mostly without EEG.** Calibrating on 10% of a new person's data lifts r from about 0.03 to about **0.20** (Δ ≈ +0.17, p = 0.16–0.22, 7 subjects). However, simply using the mean of that person's calibration block, with **no EEG**, already reaches r ≈ 0.16–0.18. EEG adds only about **+0.03** on top. The gain is mostly "know this person's baseline", not EEG-driven prediction.

### 6.4 Cross-modal, reliability and delay checks
""")

code(r"""
rel = M_SUMMARY["reliability"]
rows = [
    ("Cross-modal model r_subj (HEFMI val)", 0.024, "fNIRS-only baseline: 0.071 (same budget)"),
    ("Cross-modal task decoding (GroupKFold)", 0.456, "fNIRS-only baseline: 0.502 (chance ≈ 0.50)"),
    ("Reliability vs −error, calibration subjects (fit on self)", rel["corr_calibration_subjects_FIT_ON_SELF"], "optimistic by construction"),
    ("Reliability vs −error, internal TEST subjects", rel["corr_internal_test_subjects"], "useful in-domain ranking"),
    ("Reliability vs −error, external (TU Berlin)", rel["corr_external"], "ranking reverses OOD"),
    ("ECE internal test, raw → calibrated", rel["ece"]["internal_test_calibrated"], f"raw {rel['ece']['internal_test_raw']:.3f}"),
    ("ECE external, raw → calibrated", rel["ece"]["external_calibrated"], f"raw {rel['ece']['external_raw']:.3f}"),
    ("Locked abstention: external coverage", 0.0092, "gate refuses to answer 99% of OOD windows"),
    ("Learned delay: spread across subjects (s)", 0.0, "constant 5.00 s for every subject → not validated"),
]
table(pd.DataFrame(rows, columns=["Check", "Value", "Read"]), "Table 6.6 — Maisha: cross-modal, reliability and delay checks")
""")

md(r"""
- **Cross-modal training hurts here too.** At an identical training budget, the fNIRS-only baseline beats the cross-modal model on waveform r (0.071 vs 0.024) and on task decoding (0.502 vs 0.456).
- **Reliability ranks in-domain, not out of domain.** On untouched internal test subjects, reliability correlates with accuracy at 0.46. On TU Berlin it reverses (−0.20). The locked abstention threshold answers only 0.9% of external windows, which is the correct conservative behaviour, but means the system cannot be used out of domain.
- **The learned delay collapses to a constant 5.00 s** for every subject (zero spread). It is labelled a "bounded model timing parameter", not a neurovascular delay.
- **Stage-2 cross-modal latent alignment and downstream transfer were disabled** in v2.3, as neither showed benefit.
""")

# ---------------------------------------------------------------- 7
md(r"""
## 7. Where the two pipelines agree and disagree

The two pipelines differ in training data, architecture and code, which makes their agreements stronger evidence than either result alone.

| Question | Final v6 (TU Berlin trains) | Maisha v2.3 (HEFMI trains) | Joint conclusion |
|---|---|---|---|
| Does cross-modal (fNIRS + fMRI) supervision help? | No: Δr = −0.018 (5 subj), −0.007 (29 subj); shuffled fMRI equals real | No: fNIRS-only beats cross-modal (0.071 vs 0.024; decoding 0.502 vs 0.456) | **Replicated negative.** The core thesis is not supported at this data scale |
| Cross-subject prediction on new people | **Yes**: r = 0.158 (p = 0.001) on a clean target | No: LOSO 0.026 = subject-mean baseline | **Depends on target quality.** It works where the target is reliable (0.50) and fails where it is not (0.08) |
| Within-subject coupling | r = 0.098, positive in 27/29 subjects | r = 0.094 vs null 0.021, p = 0.005 | **Replicated positive.** Person-specific EEG–fNIRS coupling is real |
| External / cross-dataset transfer | HEFMI 0.012, ds002739 0.000 | TU Berlin −0.002, ds002739 0.009 | **Replicated negative.** Nothing transfers across cohorts, montages or tasks |
| Learned delay module | Follows its bounds (2.0 s → 0.07 s) | Constant 5.00 s, zero spread | **Replicated negative.** The module reports its prior |
| Model-free delay | 6.4 s, ICC 0.53, p = 0.0005 | Timing QA: physiological band only partly met | v6's model-free estimate is the defensible delay result |
| Reliability gate | Calibrates (ECE 0.27 → 0.02); does not rank (r = 0.02) | Ranks in-domain (0.46); reverses OOD (−0.20) | **Disagree on in-domain ranking.** Both fail out of domain. Needs a matched comparison |
| GSR | Off (audit: removes 83% of the evoked response) | Off (A/B: removes 41%, structured) | **Agree.** GSR is harmful for these targets |
| Personalisation | — | Δr ≈ +0.17 with 10% calibration (p = 0.16, n = 7), but a no-EEG calibration-mean baseline gets most of it (0.17 vs 0.20) | EEG's share of the gain is small (≈ +0.03). Worth testing on TU Berlin with the no-EEG control |
""")

# ---------------------------------------------------------------- 8
md(r"""
## 8. Scorecard
""")

code(r"""
sc = pd.DataFrame([
    ("Cross-subject fNIRS prediction (clean target)", "SUPPORTED", "v6: r = 0.158, 5/5 folds positive, p = 0.001"),
    ("Cross-subject fMRI prediction", "WEAK", "v6: r = 0.064 (p = 0.001); Maisha LOSO below baseline"),
    ("Within-subject EEG–fNIRS coupling", "SUPPORTED", "Both pipelines: r ≈ 0.09–0.10"),
    ("Claim 1 — cross-modal beats single modality", "NOT SUPPORTED", "Negative in both pipelines; shuffled fMRI = real fMRI"),
    ("Claim 2 — external generalisation", "NOT SUPPORTED", "≤ 4% of ceiling; both external cohorts ≈ 0"),
    ("Claim 3a — model-free neurovascular delay", "SUPPORTED", "6.4 s, ICC 0.53, surrogate p = 0.0005"),
    ("Claim 3b — learned delay module", "NOT SUPPORTED", "Follows bounds / constant output"),
    ("Claim 4 — hemodynamic-aware representation helps downstream", "NOT SUPPORTED", "= random and shuffled encoders"),
    ("Claim 5 — reliability gate ranks errors", "MIXED", "v6 fails; Maisha 0.46 in-domain; both fail OOD"),
    ("Claim 5 — calibration", "SUPPORTED", "v6 ECE 0.27 → 0.02 (Platt, split fit/score)"),
    ("Clinical: ICH vs control HbO/HbR coupling", "CANDIDATE", "p_Holm = 0.003; optics confound not yet excluded"),
    ("Personalised calibration (EEG contribution)", "WEAK", "Δr ≈ +0.17, but no-EEG calibration mean gives most of it; EEG adds ≈ +0.03 (n = 7)"),
], columns=["Claim", "Status", "Evidence"])
pal = {"SUPPORTED": "#d7ecdf", "NOT SUPPORTED": "#f8dcd0", "WEAK": "#efeeea", "MIXED": "#efeeea",
       "CANDIDATE": "#dbe8f8", "PROMISING": "#dbe8f8"}
display(sc.style.hide(axis="index").map(lambda v: f"background-color:{pal.get(v, '')}; font-weight:600" if v in pal else "", subset=["Status"]))
""")

# ---------------------------------------------------------------- 9
md(r"""
## 9. Limitations and threats to validity

1. **Small samples.** ds002336 has 10 subjects (3 in Maisha's validation split). v6's primary claim-1 test uses 5 held-out subjects. fMRI results are exploratory.
2. **Single seed in Maisha's pipeline.** 1 of 5 declared seeds has run. Seed-to-seed variance is unknown for every v2.3 number.
3. **Dataset role swap.** v6's fNIRS swap was decided on pre-model target-quality metrics and is declared. A reviewer may still view it as a researcher degree of freedom. Maisha's HEFMI-trained run is the useful counterfactual.
4. **External tests are confounded.** Montage (HEFMI pooled to mean HbO/HbR), task (ds002739 is perceptual decision) and target quality all differ simultaneously. For ds002739 the EEG channels were paired **by index**, not by electrode name, which is a weak alignment.
5. **Unverified event timing.** Maisha's timing QA could not confirm trigger clocks (HEFMI val, TU Berlin and ds002336 val all WARN). A fixed offset would depress every prediction score.
6. **Clinical marker confound.** Higher HbO/HbR correlation in ICH patients may reflect altered optical pathlength or systemic physiology, not neurovascular coupling.
7. **The two pipelines are not matched.** They differ in data roles, endpoints (waveform vs compact) and gate design, so their disagreement on in-domain reliability ranking cannot be resolved from existing outputs.
8. **Earlier phase numbers** (amplitude r = 0.33, CORAL 0.25) came from older protocols and have not been reproduced under the current subject-level protocol. They should not be cited as headline results.
""")

# ---------------------------------------------------------------- 10
md(r"""
## 10. Conclusions, recommended paper framing, and next steps

### Conclusions

1. **EEG carries information about hemodynamic responses in unseen people** (r = 0.158 cross-subject on TU Berlin fNIRS), but only when the training target is reliable. **Target quality, not model design, was the dominant factor** across this project.
2. **The central thesis, that adding fMRI supervision improves EEG-only hemodynamic prediction, is not supported.** Two independent pipelines with different data and code agree. The fMRI stream behaves like noise (real = shuffled targets).
3. **Coupling is person-specific.** Within-subject coupling is replicated. Personalised calibration raises r by about 0.17, but most of that comes from learning the person's baseline response, not from EEG (EEG adds ≈ +0.03).
4. **Nothing transfers across cohorts** with current data.
5. **The neurovascular delay is a stable, measurable trait** (6.4 s, ICC 0.53) when measured model-free. The learned delay module should be dropped or redesigned.
6. **Evaluation protocol matters as much as the model.** A trial-level split nearly doubles the apparent score (0.297 vs 0.158). Subject identity alone explains most of the raw fMRI correlation.

### Recommended paper framing

> *A rigorous, two-pipeline evaluation of cross-modal hemodynamic supervision for EEG. EEG predicts trial-level fNIRS responses in unseen subjects (r = 0.16) when the target is reliable, and a model-free neurovascular delay is a reproducible individual trait (ICC = 0.53). Adding fMRI supervision from disjoint subjects does not help, external transfer fails, and a naive trial-level evaluation inflates performance by 88%. Target reliability, quantified before modelling, predicts where EEG→hemodynamic prediction can succeed.*

This framing uses every robust positive result, reports the negatives as findings, and offers a methodological contribution (target-reliability gating and the protocol-inflation measurement) that does not depend on the thesis holding.

### Next steps (highest value first)

1. **Finish Maisha's 4 remaining seeds** so v2.3 numbers carry seed variance. Low cost, high credibility.
2. **Run the personalised-calibration analysis on TU Berlin** (29 subjects, clean target), always reporting it against the no-EEG calibration-mean baseline, to measure what EEG itself adds with adequate power.
3. **Resolve the reliability disagreement** by running v6's MC-dropout gate and Maisha's learned gate on the same data split.
4. **Control the clinical finding** for optical pathlength (e.g. short-separation channels or DPF sensitivity analysis) before reporting it as neurovascular.
5. **Verify trigger timing** on HEFMI and TU Berlin against raw marker files.
6. **Reposition fMRI** as a separate, exploratory analysis rather than a co-supervision stream, unless a same-subject EEG–fNIRS–fMRI dataset becomes available.
""")

# ---------------------------------------------------------------- 11
md(r"""
## 11. Appendix

### A. Original v6 figures
The figures produced by the v6 pipeline, reproduced unchanged.
""")

code(r"""
captions = {
    "fig1_target_quality.png": "A1 — Target quality by dataset",
    "fig2_cross_subject_cv.png": "A2 — Cross-subject CV",
    "fig3_claim1_cross_modal.png": "A3 — Claim 1: cross-modal vs single modality",
    "fig4_claim2_external.png": "A4 — Claim 2: external validation",
    "fig5_claim3_delay.png": "A5 — Claim 3: delay",
    "fig6_claim4_downstream.png": "A6 — Claim 4: downstream transfer",
    "fig7_claim5_reliability.png": "A7 — Claim 5: reliability",
    "fig8_clinical.png": "A8 — Clinical comparison",
}
for f, cap in captions.items():
    p = FIG / f
    if p.exists():
        display(Markdown(f"**Figure {cap}**")); display(Image(filename=str(p), width=820))
""")

md(r"""
### B. Source files

| What | File |
|---|---|
| Final v6 notebook | `NeuroVascNet_FINAL_muntasir.ipynb` (generated by `build_v6_notebook.py`) |
| v6 results | `neurovascnet_v6_outputs/summary.json` and `*.csv`; figures in `neurovascnet_v6_outputs/figures/` |
| Maisha notebook | `NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED.ipynb` |
| Maisha results | `output/summary_v3.1-v2.3-personalized-physio-fixed-delay_seed1337.json`, `output/personalized_physio_…_seed1337.json`, `output/alignment_verification.json`, `output/delay_validation.json` |
| Data-quality audit | `NeuroVascNet_Data_Quality_Audit.md`, code in `qc/`, results in `qc_outputs/` |
| Earlier history | `NeuroVascNet_Findings_Summary.md` (v3 era) |

### C. Reporting rules used throughout
- The sample-size unit is **subjects**; window counts are never used as independent samples.
- Confidence intervals are subject-bootstrap; nulls are subject-level (pairing permutation or sign-flip).
- External sets are reported as out-of-distribution stress tests, not replications.
- Any score computed on data the model was fitted to is labelled as such and not used as evidence.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, r"G:\research_bio_medical\NeuroVascNet_Research_Report.ipynb")
print("written", len(cells), "cells")
