"""Draws Figure 1 (analysis pipeline) of the Biomedical Signal Processing and Control manuscript (bspc/Figure_1.pdf)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
OUT = Path(__file__).parent / "bspc"
INK, INK2, EDGE, FILL, BLUE, NEUTRAL = "#222222", "#52514e", "#9aa3ad", "#f5f6f8", "#2a78d6", "#a3a29c"

# Figure 1: full-page width is 190 mm, so text stays legible at print size.
COLS = [
    ("Recordings", 0.05, 1.5, [
        "Scalp EEG\n4 fNIRS and\n1 fMRI study",
        "fNIRS: HbO, HbR\n88 people, 3 labs",
        "fMRI: BOLD\n27 people"]),
    ("Signal processing", 1.9, 1.75, [
        "Infra-slow EEG\n< 0.5 Hz, 30 bins",
        "EEG band power\ntheta to gamma",
        "fNIRS target\nglobal, local HbO",
        "fMRI target\n5 motor ROIs"]),
    ("Models", 4.0, 1.75, [
        "Cross-person ridge",
        "Shared EEG space\nrank 32, two heads",
        "Frozen transfer\nbetween cohorts",
        "Two-run calibration\nEEG-only model",
        "Lagged ridge\n-10 to +20 s"]),
    ("Evaluation", 6.1, 1.55, [
        "Leave one person\nor run out",
        "Pairing nulls,\ncircular shifts",
        "Bootstrap CI\nover people",
        "Rules set first,\nBonferroni"]),
]
H = 3.7
fig, ax = plt.subplots(figsize=(7.48, H + 0.1))
ax.set_xlim(0, 7.7)
ax.set_ylim(0, H + 0.1)
ax.axis("off")
top = H - 0.42
for title, x, w, items in COLS:
    ax.text(x + w / 2, H - 0.12, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    gap = 0.1
    h = (top - 0.05 - gap * (len(items) - 1)) / len(items)
    for i, t in enumerate(items):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=FILL, ec=EDGE, lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.25)
mid = top / 2 + 0.03
for a, b in ((1.57, 1.88), (3.67, 3.98), (5.77, 6.08)):
    ax.add_patch(FancyArrowPatch((a, mid), (b, mid), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=BLUE))
fig.savefig(OUT / "Figure_1.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
