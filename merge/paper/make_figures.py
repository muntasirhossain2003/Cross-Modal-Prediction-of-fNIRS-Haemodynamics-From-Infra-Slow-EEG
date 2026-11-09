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
    for f in sorted(glob.glob(str(A / pattern))):
        d.update(json.loads(Path(f).read_text()))
    return d


def save(fig, name):
    fig.tight_layout(rect=(0, 0.06, 1, 1) if name.startswith("fig4") else (0, 0, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# Fig 1 — where the information lives (RESULTS_SUMMARY, "The finding the paper is built on"; ridge, cross-person)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
slow = {"hbci": 0.197, "mental_workload": 0.135, "finemi": 0.203}
bp = {"hbci": 0.02, "mental_workload": 0.006, "finemi": 0.045}
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
tr = json.loads((A / "nvc_transfer_results.json").read_text())
bd = json.loads((A / "nvc_bands_results.json").read_text())
t2 = json.loads((A / "nvc_transfer2_results.json").read_text())
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
gain = {"hbci": (0.176, 0.135, 0.218, 0.103, 0.074, 0.136), "mental_workload": (0.059, 0.025, 0.099, 0.031, 0.013, 0.052),
        "finemi": (0.067, 0.033, 0.112, 0.041, 0.014, 0.079)}
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
rel = {"hbci": (0.68, 0.37, 0.86), "mental_workload": (0.56, 0.22, 0.78), "finemi": (0.90, 0.69, 0.97)}
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
print("figures written to", OUT)
