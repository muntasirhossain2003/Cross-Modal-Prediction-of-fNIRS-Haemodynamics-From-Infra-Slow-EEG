"""Paper figures for NeuroVascNet. Reads the result JSONs in ../analysis_p2_p4 (and shard files); numbers that exist
only in RESULTS_SUMMARY.md tables are typed in below with their source section noted."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
A = HERE.parent / "analysis_p2_p4"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
SAVED = HERE.parent / "analysis_p2_p4" / "results"


def RJ(name):
    p = A / name
    return json.loads((p if p.exists() else SAVED / name).read_text())

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COH = {"hbci": ("hBCI (TU Berlin)", "#2a78d6"), "mental_workload": ("Mental workload (TU Berlin)", "#eb6834"),
       "finemi": ("FineMI (other lab)", "#1baf7a")}
NEUTRAL = "#a3a29c"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "legend.frameon": False, "lines.linewidth": 2})


def boot_ci(v, rng=np.random.default_rng(1337), n=5000):
    v = np.asarray(v, float)
    return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)], [2.5, 97.5])


def shards(pattern):
    d = {}
    roots = (A, HERE.parent / "analysis_p2_p4", SAVED / "shards")
    files = next((g for g in (glob.glob(str(r / pattern)) for r in roots) if g), [])
    for f in sorted(files):
        d.update(json.loads(Path(f).read_text()))
    return d


def save(fig, name):
    fig.tight_layout(rect=(0, 0.06, 1, 1) if name.startswith("fig4") else (0, 0, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# Fig 1 — where the information lives (RESULTS_SUMMARY, "The finding the paper is built on"; ridge, cross-person)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
R1 = {"mental_workload": RJ("replication_results.json")["R1"], "finemi": RJ("replication_finemi_results.json")["R1"]}
slow = {"hbci": 0.197, **{c: v["slow_lt0.5"]["r_subject"] for c, v in R1.items()}}      # hBCI ridge: FINAL_PLAN P4.A
bp = {"hbci": 0.02, **{c: v["bandpower_1-45"]["r_subject"] for c, v in R1.items()}}     # hBCI: FINAL_PLAN T1
x = np.arange(3)
ax.bar(x - 0.19, [slow[c] for c in COH], 0.36, color="#2a78d6", label="infra-slow EEG (< 0.5 Hz)")
ax.bar(x + 0.19, [bp[c] for c in COH], 0.36, color=NEUTRAL, label="band power 1–45 Hz")
for i, c in enumerate(COH):
    ax.text(i - 0.19, slow[c] + 0.004, f"{slow[c]:.2f}", ha="center", va="bottom", color=INK, fontsize=8)
    ax.text(i + 0.19, bp[c] + 0.004, f"{bp[c]:.2f}", ha="center", va="bottom", color=INK2, fontsize=8)
ax.set_xticks(x, ["hBCI", "Mental\nworkload", "FineMI"])
ax.set_ylabel("cross-person prediction of HbO (r)")
ax.set_ylim(0, 0.30)
ax.legend(loc="upper center", fontsize=8, ncol=2)
ax.grid(axis="x", visible=False)
save(fig, "fig1_slow_band")

# Fig 2 — band-power -> local HbO lag curves (P6.4)
nvc = shards("nvc_shard_*_of_11.json")
lags = np.arange(-10, 21)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for c, (lab, col) in COH.items():
    C = np.array([v["curve_local"] for v in nvc.values() if v["cohort"] == c])
    m, se = C.mean(0), C.std(0, ddof=1) / np.sqrt(len(C))
    ax.fill_between(lags, m - se, m + se, color=col, alpha=0.18, linewidth=0)
    ax.plot(lags, m, color=col, label=f"{lab}, n = {len(C)}")
    k = int(np.argmax(m))
    ax.plot(lags[k], m[k], "o", ms=5, color=col, mec=SURF, mew=1.5)
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.set_xlabel("lag (s); positive = EEG band power leads HbO")
ax.set_ylabel("held-out-run r (local HbO)")
ax.legend(fontsize=7.5, loc="upper right")
save(fig, "fig2_coupling_lag")

# Fig 3 — kernel reproducibility within person (P7.1): d = k - mean(k_null)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
pvals = {("", "hbci"): "p < 0.0001", ("", "mental_workload"): "p 0.09", ("", "finemi"): "p 0.033",
         ("_resid", "hbci"): "p < 0.0001", ("_resid", "mental_workload"): "p 0.006", ("_resid", "finemi"): "p 0.001"}
rng = np.random.default_rng(0)
for ax, (tag, title) in zip(axes, (("", "raw recordings"), ("_resid", "task response removed"))):
    K = shards(f"nvc_kernel{tag}_shard_*_of_*.json")
    for i, (c, (lab, col)) in enumerate(COH.items()):
        d = np.array([v["d"] for v in K.values() if v["cohort"] == c])
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=14, color=col, alpha=0.75, edgecolors=SURF, linewidths=0.5)
        ax.plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color=INK, lw=2)
        ax.text(i, 1.27, pvals[(tag, c)], ha="center", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(3), ["hBCI", "Mental\nworkload", "FineMI"])
    ax.set_title(title, fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-0.9, 1.38)
axes[0].set_ylabel("kernel reproducibility d\n(half-A vs half-B r, minus shifted null)")
save(fig, "fig3_kernel_repro")

# Fig 4 — where the coupling kernel generalises (P7.2, P9.2, P8.3)
tr = RJ("nvc_transfer_results.json")
bd = RJ("nvc_bands_results.json")
t2 = RJ("nvc_transfer2_results.json")
lopo_h = tr["within_hbci_lopo"]
lh_ci = lopo_h["ci"] if "ci" in lopo_h else boot_ci([q["r"] for q in lopo_h["people"].values()]) if "people" in lopo_h else [np.nan] * 2
rows = [("hBCI -> hBCI, held-out person", lopo_h["mean_r"], lh_ci, "hbci"),
        ("FineMI -> FineMI, held-out person", bd["lopo | finemi"]["mean_r"], bd["lopo | finemi"]["ci"], "finemi"),
        ("hBCI -> mental workload (same lab)", tr["hbci_to_mental_workload"]["mean_r"], tr["hbci_to_mental_workload"]["ci"], "mental_workload"),
        ("hBCI -> FineMI, frozen (other lab)", tr["hbci_to_finemi"]["mean_r"], tr["hbci_to_finemi"]["ci"], "finemi"),
        ("hBCI -> FineMI, calibrated gain*", t2["calibrated | finemi"]["diff"], t2["calibrated | finemi"]["ci"], "finemi")]
fig, ax = plt.subplots(figsize=(5.6, 3.1))
for i, (lab, m, ci, c) in enumerate(rows[::-1]):
    col = COH[c][1]
    ax.plot(ci, [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(m, i, "o", ms=7, color=col, mec=SURF, mew=1.5)
    if m < 0:
        ax.text(ci[0] - 0.006, i, f"{m:+.3f}", va="center", ha="right", fontsize=7.5, color=INK)
    else:
        ax.text(ci[1] + 0.006, i, f"{m:+.3f}", va="center", fontsize=7.5, color=INK)
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8)
ax.set_xlabel("r with global HbO (95% CI)")
ax.grid(axis="y", visible=False)
ax.set_xlim(-0.13, 0.23)
fig.text(0.01, 0.01, "*added to a person's own two-run model; nominal (does not survive multiplicity correction)",
         fontsize=6.5, color=INK2)
save(fig, "fig4_kernel_transfer")

# Fig 5 — deployment value and person-level reliability (RESULTS_SUMMARY deployment table; P5.4)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
G, GL = RJ("calibration_results.json"), RJ("calibration_results_local.json")
gain = {c: tuple(x for d in (G, GL) for x in (d[c]["gains"]["k2_personal_minus_baseline"]["mean"], *d[c]["gains"]["k2_personal_minus_baseline"]["ci"])) for c in COH}
ax = axes[0]
for i, (c, (lab, col)) in enumerate(COH.items()):
    g = gain[c]
    for j, (m, lo, hi, mk) in enumerate(((g[0], g[1], g[2], "o"), (g[3], g[4], g[5], "s"))):
        xx = i + (j - 0.5) * 0.3
        ax.plot([xx, xx], [lo, hi], color=col, lw=2)
        ax.plot(xx, m, mk, ms=6, color=col, mec=SURF, mew=1.2)
ax.plot([], [], "o", color=INK2, label="raw HbO")
ax.plot([], [], "s", color=INK2, label="local HbO (systemic removed)")
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(3), ["hBCI", "Mental\nworkload", "FineMI"])
ax.set_ylabel("gain over person's own mean response (r)")
ax.set_title("EEG-only monitoring after 2 calibration runs", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.grid(axis="x", visible=False)
ax = axes[1]
REL = RJ("calib_reliability_results.json")
rel = {c: (REL[c]["spearman_score_vs_minus_error"], *REL[c]["ci95"]) for c in COH}
for i, (c, (lab, col)) in enumerate(COH.items()):
    m, lo, hi = rel[c]
    ax.plot([i, i], [lo, hi], color=col, lw=2)
    ax.plot(i, m, "o", ms=7, color=col, mec=SURF, mew=1.2)
    ax.text(i + 0.12, m, f"{m:.2f}", va="center", fontsize=7.5, color=INK)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(3), ["hBCI", "Mental\nworkload", "FineMI"])
ax.set_ylabel("Spearman ρ (95% CI)")
ax.set_title("calibration-run score predicts later accuracy", fontsize=9)
ax.set_ylim(-0.1, 1.05)
ax.grid(axis="x", visible=False)
save(fig, "fig5_deployment_reliability")
# Fig 6 — which slow band carries the signal
B = RJ("bands_results.json")
R6 = {c: RJ(f)["R1"] for c, f in (("mental_workload", "replication_results.json"), ("finemi", "replication_finemi_results.json"))}
vals = {"hbci": [B[k]["r_subject"] for k in ("lt0.5", "lt0.1", "0.1-0.5")], **{c: [R6[c][k]["r_subject"] for k in ("slow_lt0.5", "slow_lt0.1", "slow_0.1-0.5")] for c in R6}}
fig, ax = plt.subplots(figsize=(5.4, 3.1))
x = np.arange(3)
for i, (c, (lab, col)) in enumerate(COH.items()):
    xs = x + (i - 1) * 0.27
    ax.bar(xs, vals[c], 0.25, color=col, label=lab + (", deep model" if c == "hbci" else ", ridge"))
    for a, v in zip(xs, vals[c]):
        ax.text(a, v + 0.004, f"{v:.2f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x, ["< 0.5 Hz", "< 0.1 Hz", "0.1–0.5 Hz"])
ax.set_ylabel("cross-person prediction of HbO (r)")
ax.set_ylim(0, 0.31)
ax.legend(fontsize=7, loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "fig6_slow_bands")

# Fig 7 — person specificity
M7 = np.array(RJ("person_specific_results_runs.json")["slow"]["matrix"], float)
fig, ax = plt.subplots(figsize=(4.3, 3.6))
im = ax.imshow(M7, cmap="Blues", vmin=0, vmax=np.nanpercentile(M7, 99))
ax.grid(False)
ax.set_xlabel("person whose data is predicted")
ax.set_ylabel("person whose model is used")
fig.colorbar(im, ax=ax, label="r", shrink=0.85)
save(fig, "fig7_person_matrix")

# Fig 8 — gain with more calibration runs
G8 = RJ("calibration_results.json")
fig, ax = plt.subplots(figsize=(5.0, 3.1))
for c, (lab, col) in COH.items():
    K, m, lo, hi = [], [], [], []
    for k in range(1, 6):
        v = G8[c]["gains"].get(f"k{k}_personal_minus_baseline")
        if v and v.get("n", 0) >= 10:
            K.append(k); m.append(v["mean"]); lo.append(v["ci"][0]); hi.append(v["ci"][1])
    ax.fill_between(K, lo, hi, color=col, alpha=0.18, linewidth=0)
    ax.plot(K, m, "-o", color=col, ms=5, mec=SURF, mew=1.2, label=lab)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(1, 6))
ax.set_xlabel("calibration runs")
ax.set_ylabel("gain over person's own mean (r)")
ax.legend(fontsize=7.5, loc="upper left")
save(fig, "fig8_calibration_runs")

# Fig 9 — every form of transfer into the other lab
T9, T2, B9 = RJ("nvc_transfer_results.json"), RJ("nvc_transfer2_results.json"), RJ("nvc_bands_results.json")
H9, C9, BB = RJ("harmonised_transfer_results.json")["hbci_to_finemi | primary: window + montage"], RJ("calibrated_transfer_results.json")["hbci_to_finemi"], RJ("nvc_broadband_results.json")
rows = [("Band-power kernel, calibrated gain*", T2["calibrated | finemi"]["diff"], T2["calibrated | finemi"]["ci"], True),
        ("Alpha + beta kernel, calibrated gain", B9["calibrated | finemi | alpha+beta"]["diff"], B9["calibrated | finemi | alpha+beta"]["ci"], False),
        ("Broadband removed, calibrated gain", BB["calibrated | finemi | removed"]["diff"], BB["calibrated | finemi | removed"]["ci"], False),
        ("Slow EEG, calibrated gain", C9["diff"], C9["ci"], False),
        ("Slow EEG, window + montage", H9["r_subject"], [H9["ci_low"], H9["ci_high"]], False),
        ("Band-power kernel, frozen", T9["hbci_to_finemi"]["mean_r"], T9["hbci_to_finemi"]["ci"], False),
        ("Alpha + beta kernel, frozen", B9["hbci_to_finemi | alpha+beta"]["mean_r"], B9["hbci_to_finemi | alpha+beta"]["ci"], False),
        ("Broadband removed, frozen", BB["hbci_to_finemi | removed"]["mean_r"], BB["hbci_to_finemi | removed"]["ci"], False),
        ("CBSI target, frozen", T2["hbci_to_finemi | cbsi"]["mean_r"], T2["hbci_to_finemi | cbsi"]["ci"], False)]
for n, f in (("plain", ""), ("detrend", "_detrend"), ("linked mastoids", "_lm"), ("Euclidean alignment", "_ea"), ("hardware emulation", "_hw")):
    v = RJ(f"external_finemi{f}_results.json")["hbci_to_finemi"]
    rows.append((f"Slow EEG, {n}", v["r_subject"], [v["ci_low"], v["ci_high"]], False))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.0, 4.3))
for i, (lab, m, c, nominal) in enumerate(rows):
    col = "#eb6834" if nominal else NEUTRAL
    ax.plot(c, [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(m, i, "o", ms=6, color=col, mec=SURF, mew=1.2)
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("hBCI -> FineMI: r or gain in r (95% CI)")
ax.grid(axis="y", visible=False)
fig.text(0.01, 0.005, "*nominal: does not survive correction for 16 cross-lab attempts", fontsize=6.5, color=INK2)
fig.subplots_adjust(bottom=0.2)
save(fig, "fig9_cross_lab")

# Fig 10 — does the personal delay repeat between halves?
def smooth10(c, w=5):
    c = np.asarray(c, float)
    return np.array([c[max(0, i - w // 2):i + w // 2 + 1].mean() for i in range(len(c))])
L10 = np.arange(-10, 21)
K10, S10 = shards("nvc_kernel_shard_*_of_*.json"), shards("nvc_snr_shard_*_of_*.json")
panels = [(COH[c][0].split(" (")[0], COH[c][1], [v for v in K10.values() if v["cohort"] == c]) for c in COH] + [("hBCI degraded to FineMI strength", NEUTRAL, [v for v in S10.values() if v["level"] == "finemi"])]
rng10 = np.random.default_rng(0)
fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.75), sharex=True, sharey=True)
for ax, (lab, col, people) in zip(axes, panels):
    dA = np.array([L10[np.argmax(smooth10(v["curve_A"]))] for v in people], float)
    dB = np.array([L10[np.argmax(smooth10(v["curve_B"]))] for v in people], float)
    ax.fill_between([-12, 22], [-14, 20], [-10, 24], color=GRID, alpha=0.7, linewidth=0)
    ax.scatter(dA + rng10.uniform(-0.25, 0.25, len(dA)), dB + rng10.uniform(-0.25, 0.25, len(dB)), s=15, color=col, edgecolors=SURF, linewidths=0.5)
    ax.set_title(f"{lab}\n{np.mean(np.abs(dA - dB) <= 2):.0%} within 2 s", fontsize=8)
    ax.set_xlim(-12, 22); ax.set_ylim(-12, 22)
    ax.set_xlabel("delay, first half (s)", fontsize=8)
axes[0].set_ylabel("delay, second half (s)", fontsize=8)
save(fig, "fig10_delay_halves")

# Fig 11 — sign flip by EEG band
B11, BB11 = RJ("nvc_bands_results.json"), RJ("nvc_broadband_results.json")
items = [("all features", B11, "all"), ("theta", B11, "theta"), ("alpha", B11, "alpha"), ("beta", B11, "beta"), ("gamma", B11, "gamma"), ("alpha + beta", B11, "alpha+beta"), ("broadband removed", BB11, "removed"), ("broadband only", BB11, "only")][::-1]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for i, (lab, src, key) in enumerate(items):
    for off, c in ((0.14, "mental_workload"), (-0.14, "finemi")):
        v = src[f"hbci_to_{c} | {key}"]
        ax.plot(v["ci"], [i + off] * 2, color=COH[c][1], lw=2, solid_capstyle="round")
        ax.plot(v["mean_r"], i + off, "o", ms=6, color=COH[c][1], mec=SURF, mew=1.2, label=COH[c][0] if i == 0 else None)
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.set_yticks(range(len(items)), [i[0] for i in items], fontsize=8)
ax.set_xlabel("frozen hBCI kernel: r with global HbO (95% CI)")
ax.grid(axis="y", visible=False)
ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "fig11_band_sign")

# Fig 12 — downstream decoding
D12 = RJ("downstream_repr_nlab0_results.json")
tasks = [("hbci_MI_left_right (primary)", "hBCI motor\nimagery"), ("hbci_MA_vs_rest", "hBCI mental\narithmetic"), ("mw_nback_0_2_3", "MW n-back"), ("finemi_8class", "FineMI\n8 classes")]
fig, ax = plt.subplots(figsize=(5.6, 3.1))
x = np.arange(4)
for j, (key, lab, col) in enumerate((("V", "virtual haemodynamics", "#2a78d6"), ("PC", "PCA", "#eb6834"), ("Rp", "random projection", NEUTRAL))):
    xs = x + (j - 1) * 0.27
    ax.bar(xs, [D12[t]["mean_acc"][key] for t, _ in tasks], 0.25, color=col, label=lab)
for i, (t, _) in enumerate(tasks):
    ax.plot([i - 0.42, i + 0.42], [D12[t]["chance"]] * 2, color=INK, lw=1.2, ls=":")
ax.plot([], [], color=INK, lw=1.2, ls=":", label="chance")
ax.set_xticks(x, [t[1] for t in tasks], fontsize=8)
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(axis="x", visible=False)
save(fig, "fig12_downstream")

# Fig 13 — which results survive correction
from scipy import stats as _st
c13, e13 = RJ("claim1_fresh_mental_workload_results.json"), RJ("external_claim2_results.json")["hbci_to_mental_workload"]
kr13, kt13, sm13, rel13 = RJ("nvc_kernel_results.json"), RJ("nvc_kernel_resid_results.json"), RJ("nvc_delay_smooth_results.json"), RJ("calib_reliability_results.json")
d13 = np.array([v["personal_plus_external"] - v["personal"] for v in RJ("nvc_transfer2_results.json")["calibrated | finemi"]["people"].values()])
s13 = np.random.default_rng(0).choice([-1, 1], size=(200000, len(d13)))
p83 = (np.sum((s13 * d13).mean(1) >= d13.mean()) + 1) / 200001
rho_p = lambda r, n: 2 * _st.t.sf(abs(r) * np.sqrt((n - 2) / (1 - r ** 2)), n - 2)
rows = [("Claim 1: shared space predicts fMRI", c13["fmri"]["joint"]["p_signflip"], 8), ("Claim 2: hBCI -> MW", e13["p_signflip"], 3), ("Claim 2: hBCI -> FineMI calibrated", p83, 16),
        ("Claim 3: hBCI kernel repeats", kr13["hbci"]["wilcoxon_p_greater"], 14), ("Claim 3: hBCI delay repeats", sm13["hbci | raw"]["p_agree"], 14),
        ("Claim 3: FineMI kernel, task removed", kt13["finemi"]["wilcoxon_p_greater"], 14), ("Claim 3: MW kernel, task removed", kt13["mental_workload"]["wilcoxon_p_greater"], 14), ("Claim 3: FineMI kernel, raw", kr13["finemi"]["wilcoxon_p_greater"], 14)]
rows += [(f"Claim 5: reliability, {COH[c][0].split(' (')[0]}", rho_p(rel13[c]["spearman_score_vs_minus_error"], rel13[c]["n"]), 4) for c in COH]
rows.sort(key=lambda r: r[1] * r[2], reverse=True)
fig, ax = plt.subplots(figsize=(6.0, 3.9))
for i, (lab, p, n) in enumerate(rows):
    v = np.log10(p / (0.05 / n))
    col = "#2a78d6" if v < 0 else "#eb6834"
    ax.plot([0, v], [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(v, i, "o", ms=6, color=col, mec=SURF, mew=1.2)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("log10 (p / corrected threshold); left of 0 survives")
ax.grid(axis="y", visible=False)
save(fig, "fig13_multiplicity")

# Fig 14 — reliability per person
R14 = RJ("calib_reliability_results.json")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, (c, (lab, col)) in zip(axes, COH.items()):
    P = list(R14[c]["people"].values())
    ax.scatter([p["cal_score"] for p in P], [p["test_r"] for p in P], s=18, color=col, edgecolors=SURF, linewidths=0.5)
    ax.set_title(f"{lab.split(' (')[0]}: ρ = {R14[c]['spearman_score_vs_minus_error']:.2f}", fontsize=8.5)
    ax.set_xlabel("score on 2 calibration runs", fontsize=8)
axes[0].set_ylabel("accuracy on later runs (r)", fontsize=8)
save(fig, "fig14_reliability_people")

# Fig 15 — confirmation in a third lab
F15, TR15, T15 = RJ("nvc_mendeley_founding.json"), RJ("nvc_mendeley_transfer.json")["frozen"], RJ("nvc_transfer_results.json")
R15 = {"Mental workload": RJ("replication_results.json")["R1"], "FineMI": RJ("replication_finemi_results.json")["R1"]}
rows15 = [(n, r["slow_lt0.5"]["r_subject"], r["slow_lt0.5"]["ci"], r["bandpower_1-45"]["r_subject"], r["bandpower_1-45"]["ci"]) for n, r in R15.items()]
rows15.append(("Third lab (P12)", F15["slow"]["mean_r"], F15["slow"]["ci"], F15["bandpower"]["mean_r"], F15["bandpower"]["ci"]))
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
ax = axes[0]
for i, (n, a, ca, b, cb) in enumerate(rows15):
    ax.plot([i - 0.12] * 2, ca, color="#2a78d6", lw=2)
    ax.plot(i - 0.12, a, "o", ms=6, color="#2a78d6", mec=SURF, mew=1.2, label="slow EEG (< 0.5 Hz)" if i == 0 else None)
    ax.plot([i + 0.12] * 2, cb, color=NEUTRAL, lw=2)
    ax.plot(i + 0.12, b, "s", ms=6, color=NEUTRAL, mec=SURF, mew=1.2, label="band power" if i == 0 else None)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(3), [r[0] for r in rows15], fontsize=8)
ax.set_ylabel("cross-person r with HbO (95% CI)")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(axis="x", visible=False)
ax = axes[1]
rows = [("Mental workload\n(same lab)", T15["hbci_to_mental_workload"]), ("FineMI", T15["hbci_to_finemi"]), ("Third lab (P12)", TR15)]
for i, (n, v) in enumerate(rows):
    col = "#2a78d6" if v["ci"][0] > 0 else "#eb6834"
    ax.plot([i, i], v["ci"], color=col, lw=2)
    ax.plot(i, v["mean_r"], "o", ms=7, color=col, mec=SURF, mew=1.2)
    ax.text(i + 0.1, v["mean_r"], f"{v['mean_r']:+.3f}", va="center", fontsize=7.5, color=INK)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(3), [r[0] for r in rows], fontsize=8)
ax.set_xlim(-0.4, 2.6)
ax.set_ylabel("frozen hBCI kernel: r (95% CI)")
ax.grid(axis="x", visible=False)
save(fig, "fig15_third_lab")

# Fig 16 — the slow-EEG advantage under two methods
M16 = RJ("nvc_method_check_results.json")["cohorts"]
N16 = (("hbci", "hBCI"), ("mental_workload", "Mental\nworkload"), ("finemi", "FineMI"), ("mendeley", "Third lab"))
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
for ax, k, title in zip(axes, ("continuous", "windowed"), ("Continuous recording (P12.1 method)", "30 s windows, not aligned to cues")):
    for i, (c, n) in enumerate(N16):
        a, b = M16[c][k]["slow"], M16[c][k]["bandpower"]
        ax.plot([i - 0.12] * 2, a["ci"], color="#2a78d6", lw=2)
        ax.plot(i - 0.12, a["mean_r"], "o", ms=6, color="#2a78d6", mec=SURF, mew=1.2, label="slow EEG (< 0.5 Hz)" if i == 0 else None)
        ax.plot([i + 0.12] * 2, b["ci"], color=NEUTRAL, lw=2)
        ax.plot(i + 0.12, b["mean_r"], "s", ms=6, color=NEUTRAL, mec=SURF, mew=1.2, label="band power" if i == 0 else None)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(4), [n for _, n in N16], fontsize=8)
    ax.set_title(title, fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("cross-person r with HbO (95% CI)")
axes[0].legend(fontsize=7.5, loc="upper right")
save(fig, "fig16_method_check")

# Fig 17 — stress tests of the claims that hold
C17, R17 = RJ("nvc_held_claim5.json"), RJ("calib_reliability_results.json")
O17, T17, E17 = RJ("nvc_held_overlap.json"), RJ("nvc_held_claim2.json"), RJ("external_claim2_results.json")
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
ax = axes[0]
for i, (c, n) in enumerate((("hbci", "hBCI"), ("mental_workload", "Mental\nworkload"), ("finemi", "FineMI"), ("mendeley", "Third lab"))):
    if c in R17:
        ax.plot([i - 0.12] * 2, R17[c]["ci95"], color=NEUTRAL, lw=2)
        ax.plot(i - 0.12, R17[c]["spearman_score_vs_minus_error"], "s", ms=6, color=NEUTRAL, mec=SURF, mew=1.2, label="cue-aligned" if i == 0 else None)
    q = C17[c]["raw"]
    ax.plot([i + 0.12] * 2, q["ci"], color="#2a78d6", lw=2)
    ax.plot(i + 0.12, q["rho"], "o", ms=6, color="#2a78d6", mec=SURF, mew=1.2, label="no cue alignment" if i == 0 else None)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(4), ["hBCI", "Mental\nworkload", "FineMI", "Third lab"], fontsize=8)
ax.set_ylabel("calibration vs later score (Spearman)", fontsize=8)
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(axis="x", visible=False)
ax = axes[1]
rows = [("Original", E17["hbci_to_mental_workload"]), ("Closest pairs\nremoved", O17["hbci_to_mw_excluded"]), ("No cue\nalignment", T17["hbci_to_mental_workload"])]
for i, (n, v) in enumerate(rows):
    ax.plot([i, i], [v["ci_low"], v["ci_high"]], color="#2a78d6", lw=2)
    ax.plot(i, v["r_subject"], "o", ms=7, color="#2a78d6", mec=SURF, mew=1.2)
    ax.text(i + 0.1, v["r_subject"], f"{v['r_subject']:+.3f}", va="center", fontsize=7.5, color=INK)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(3), [r[0] for r in rows], fontsize=8)
ax.set_xlim(-0.4, 2.6)
ax.set_ylabel("hBCI to mental workload: r (95% CI)", fontsize=8)
ax.grid(axis="x", visible=False)
save(fig, "fig17_stress_tests")

print("figures written to", OUT)
