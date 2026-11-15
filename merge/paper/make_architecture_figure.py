"""Figure 3 of the BSPC manuscript: architecture of the proposed EEG-to-haemodynamics framework.

Panels: (a) overview, (b) infra-slow EEG encoder, (c) shared low-rank EEG space,
(d) two-run personal calibration and reliability score. Every block follows Section 2.2-2.3
of the manuscript (Eqs. 1-7); small traces are schematic, not data.

Usage: python make_architecture_figure.py [output_stem]
Writes <output_stem>.pdf and <output_stem>.png (default: bspc/Figure_3).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
                     "pdf.fonttype": 42})

INK, INK2, MUTE = "#111111", "#333333", "#7d7d7d"
FILL = {"feat": "#e6eef6", "ridge": "#f8e9dc", "basis": "#ece7f2", "calib": "#e5f0e4",
        "out": "#ffffff", "data": "#f2f2f2", "base": "#ffffff"}
EDGE = {"feat": "#2b2b2b", "ridge": "#2b2b2b", "basis": "#2b2b2b", "calib": "#2b2b2b",
        "out": "#111111", "data": "#2b2b2b", "base": MUTE}
PANEL_BG = "none"
LW = 0.6
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(7.2, 5.55))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.2)
ax.set_ylim(0, 5.55)
ax.axis("off")


def box(x, y, w, h, text="", kind="feat", fs=7.0, bold=False, dashed=False, va="center", sub=None):
    """Rounded module box with lower-left corner (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                fc=FILL[kind], ec=EDGE[kind], lw=LW, ls=(0, (3, 2)) if dashed else "-",
                                zorder=3))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fs, color=INK,
                fontweight="normal", zorder=4, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h * 0.64, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="normal", zorder=4)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=fs - 0.8, color=INK2,
                zorder=4)
    return (x, y, w, h)


def arrow(points, color=INK, lw=LW, dashed=False, head=True):
    """Polyline through points with an arrowhead on the last segment."""
    pts = np.asarray(points, float)
    ls = (0, (3, 2)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=lw, ls=ls,
                                mutation_scale=6, shrinkA=0, shrinkB=0))


def op(x, y, sym, r=0.065):
    """Operator circle: '+' sum, 'x' matrix product, 'C' concatenation."""
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=LW, zorder=4))
    label = {"+": "+", "x": r"$\times$", "C": "C"}[sym]
    ax.text(x, y - 0.004, label, ha="center", va="center", fontsize=6.6 if sym != "C" else 5.6,
            fontweight="bold", color=INK, zorder=5)


def shape(x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=6.0, color=INK2, zorder=5)


def panel(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#9a9a9a", lw=0.5, zorder=0))
    letter, rest = title.split(" ", 1)
    ax.text(x + 0.08, y + h - 0.12, letter, ha="left", va="center", fontsize=8.2, fontweight="bold",
            color=INK, zorder=5)
    ax.text(x + 0.3, y + h - 0.12, rest, ha="left", va="center", fontsize=8.2, color=INK, zorder=5)


def trace(x0, x1, yc, amp, kind="eeg", color=INK2, lw=0.6):
    t = np.linspace(0, 1, 200)
    if kind == "eeg":
        s = np.cumsum(rng.normal(0, 1, t.size)); s = (s - s.mean()) / (np.ptp(s) + 1e-9)
        s += 0.35 * np.sin(2 * np.pi * 9 * t) * 0.5
    elif kind == "slow":
        s = 0.5 * np.sin(2 * np.pi * 1.1 * t + 0.6)
    else:  # haemodynamic response
        s = np.exp(-((t - 0.45) / 0.18) ** 2) - 0.25 * np.exp(-((t - 0.8) / 0.12) ** 2) - 0.3
    ax.plot(x0 + t * (x1 - x0), yc + amp * s, color=color, lw=lw, zorder=4)


def head(cx, cy, r, sensors="eeg"):
    """Top view of a head (nose up) with EEG electrodes or fNIRS source-detector pairs."""
    for sx in (-1, 1):
        ax.add_patch(Ellipse((cx + sx * r * 0.98, cy), r * 0.24, r * 0.46, fc="white", ec=INK, lw=0.5,
                             zorder=4))
    ax.add_patch(Polygon([(cx - 0.17 * r, cy + 0.96 * r), (cx, cy + 1.2 * r), (cx + 0.17 * r, cy + 0.96 * r)],
                         closed=True, fc="white", ec=INK, lw=0.5, zorder=4))
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=INK, lw=0.6, zorder=5))
    if sensors == "eeg":
        pts = [(0, 0)] + [(0.47 * np.cos(a), 0.47 * np.sin(a)) for a in np.radians(np.arange(0, 360, 60) + 30)] \
            + [(0.8 * np.cos(a), 0.8 * np.sin(a)) for a in np.radians(np.arange(0, 360, 36) + 18)]
        for px, py in pts:
            ax.add_patch(Circle((cx + px * r, cy + py * r), 0.065 * r, fc="#4d4d4d", ec="none", zorder=6))
    else:
        for side in (-1, 1):
            src = [(side * 0.45, 0.3), (side * 0.45, -0.3)]
            det = [(side * 0.72, 0.0), (side * 0.2, 0.0)]
            for sx_, sy_ in src:
                for dx, dy in det:
                    ax.plot([cx + sx_ * r, cx + dx * r], [cy + sy_ * r, cy + dy * r], color="#9a9a9a", lw=0.4,
                            zorder=5.5)
            for sx_, sy_ in src:
                ax.add_patch(Rectangle((cx + sx_ * r - 0.07 * r, cy + sy_ * r - 0.07 * r), 0.14 * r, 0.14 * r,
                                       fc="#c8463d", ec="none", zorder=6))
            for dx, dy in det:
                ax.add_patch(Circle((cx + dx * r, cy + dy * r), 0.075 * r, fc="#2e6aa6", ec="none", zorder=6))


def scanner(cx, cy, w):
    """Front view of an MR scanner with its bore and patient table."""
    h = 0.8 * w
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="#e4e4e4", ec=INK, lw=0.5, zorder=4))
    ax.add_patch(Circle((cx, cy + 0.04 * w), 0.27 * w, fc="white", ec=INK, lw=0.5, zorder=5))
    ax.add_patch(Circle((cx, cy + 0.02 * w), 0.09 * w, fc="#f2d7c4", ec=INK, lw=0.4, zorder=6))
    ax.add_patch(Rectangle((cx - 0.42 * w, cy - 0.12 * w), 0.84 * w, 0.05 * w, fc="#8c8c8c", ec=INK, lw=0.4,
                           zorder=6))


def scatter_icon(x0, y0, w, h):
    """Tiny scatter of calibration score against later accuracy, with a fitted line."""
    ax.plot([x0, x0, x0 + w], [y0 + h, y0, y0], color=INK, lw=0.5, zorder=5)
    xs_ = np.array([0.12, 0.22, 0.3, 0.41, 0.5, 0.58, 0.7, 0.8, 0.9])
    ys_ = xs_ * 0.8 + 0.08 + np.array([0.06, -0.05, 0.08, -0.04, 0.05, -0.07, 0.04, -0.02, 0.03])
    ax.plot(x0 + np.array([0.06, 0.95]) * w, y0 + (np.array([0.06, 0.95]) * 0.8 + 0.08) * h, color="#c8463d",
            lw=0.6, zorder=5)
    for a, b in zip(xs_, ys_):
        ax.add_patch(Circle((x0 + a * w, y0 + b * h), 0.012, fc=INK2, ec="none", zorder=6))


def matrix_icon(x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=INK, lw=0.5, zorder=5))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.0, color=INK, zorder=6)


def bins(x0, x1, y0, h, n=12, color=EDGE["feat"]):
    vals = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(np.linspace(0.3, 4.2, n)))
    w = (x1 - x0) / n
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + i * w + w * 0.12, y0), w * 0.76, h * v, fc=color, ec="none", zorder=4))


# ---------------------------------------------------------------- (a) overview
panel(0.06, 0.62, 2.36, 4.88, "(a) Overview")

# input
box(0.22, 4.68, 1.08, 0.5, "", kind="data")
ax.text(0.76, 5.08, r"Scalp EEG window $\mathbf{x}$", ha="center", va="center", fontsize=7.0,
        color=INK, zorder=5)
head(0.43, 4.83, 0.12, "eeg")
for k in range(3):
    trace(0.64, 1.24, 4.93 - k * 0.09, 0.03, "eeg", lw=0.45)
shape(0.76, 4.58, r"$N_c\times6000$  (30 s, 200 Hz)", ha="center")

# encoder
arrow([(0.76, 4.5), (0.76, 4.3)])
box(0.22, 3.96, 1.08, 0.34, r"Infra-slow encoder  $\mathcal{E}$", kind="feat", bold=True,
    sub="fixed transform, panel (b)")
arrow([(0.76, 3.96), (0.76, 3.62)])
shape(0.82, 3.79, r"$\mathbf{F}\in\mathbb{R}^{n\times30N_c}$")

# shared space
box(0.22, 3.28, 1.08, 0.34, r"Shared EEG space  $U_K$", kind="basis", bold=True,
    sub=r"rank $K=32$, panel (c)")
shape(0.82, 3.12, r"$\mathbf{F}U_K\in\mathbb{R}^{n\times32}$")

# heads and outputs (right column of panel a)
box(1.5, 3.97, 0.42, 0.32, r"$\hat{T}_\mathrm{O}$", kind="ridge", fs=7, sub="fNIRS")
box(1.96, 3.97, 0.42, 0.32, r"$\hat{T}_\mathrm{M}$", kind="ridge", fs=7, sub="fMRI")
arrow([(1.3, 3.45), (1.71, 3.45), (1.71, 3.97)])
arrow([(1.71, 3.45), (2.17, 3.45), (2.17, 3.97)])

box(1.5, 4.62, 0.42, 0.56, "", kind="out")
head(1.71, 4.97, 0.1, "fnirs")
ax.text(1.71, 4.76, r"$\hat{\mathbf{Y}}_\mathrm{O}$" + "\nHbO, HbR", ha="center", va="center", fontsize=5.6,
        color=INK, zorder=5, linespacing=1.0)
box(1.96, 4.62, 0.42, 0.56, "", kind="out")
scanner(2.17, 4.98, 0.27)
ax.text(2.17, 4.76, r"$\hat{\mathbf{Y}}_\mathrm{M}$" + "\nBOLD", ha="center", va="center", fontsize=5.6,
        color=INK, zorder=5, linespacing=1.0)
arrow([(1.71, 4.29), (1.71, 4.62)])
arrow([(2.17, 4.29), (2.17, 4.62)])
shape(1.73, 4.45, r"$C\times30$")
shape(2.19, 4.45, r"$5\times15$")

# calibration branch: from F, not from the shared space
ax.text(0.24, 2.86, "per-person route", ha="left", va="center", fontsize=5.6, color=INK2, style="italic",
        zorder=5)
ax.text(2.36, 3.05, "cross-person route", ha="right", va="center", fontsize=5.6, color=INK2,
        style="italic", zorder=5)
ax.plot([0.1, 2.38], [2.96, 2.96], color=MUTE, lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
arrow([(0.76, 3.79), (0.14, 3.79), (0.14, 2.46), (0.22, 2.46)])
box(0.22, 2.28, 1.08, 0.36, "Two-run calibration", kind="calib", bold=True,
    sub=r"personal ridge $f_\mathrm{p}$, panel (d)")
box(1.5, 2.28, 0.88, 0.36, "Runs 1 and 2", kind="data", sub="paired EEG + fNIRS")
arrow([(1.5, 2.46), (1.3, 2.46)])

box(0.22, 1.4, 1.08, 0.5, "", kind="out")
ax.text(0.76, 1.76, "EEG-only HbO", ha="center", va="center", fontsize=6.3, color=INK,
        zorder=5)
ax.text(0.76, 1.55, r"$\hat{y}_\mathrm{pers}=\bar{y}_c+f_\mathrm{p}(z_c(\mathbf{x}))$", ha="center",
        va="center", fontsize=6.0, color=INK, zorder=5)
box(1.5, 1.4, 0.88, 0.5, "", kind="out")
ax.text(1.94, 1.76, "Reliability", ha="center", va="center", fontsize=6.3, color=INK,
        zorder=5)
scatter_icon(1.6, 1.47, 0.3, 0.2)
ax.text(1.98, 1.57, "predicts\naccuracy", ha="left", va="center", fontsize=5.8, color=INK2,
        zorder=5, linespacing=1.0)
arrow([(0.76, 2.28), (0.76, 1.9)])
arrow([(1.1, 2.28), (1.1, 2.08), (1.94, 2.08), (1.94, 1.9)])

# ridge note
box(0.22, 0.76, 2.16, 0.44, "", kind="base")
ax.text(1.3, 1.06, r"Every map is ridge:  $\hat{B}_\lambda=(F^{\top}F+\lambda I)^{-1}F^{\top}Y$",
        ha="center", va="center", fontsize=5.9, color=INK, zorder=5)
ax.text(1.3, 0.87, r"$\lambda^{\ast}$ by efficient leave-one-out, 13 values $10^{0}$–$10^{6}$",
        ha="center", va="center", fontsize=5.6, color=INK2, zorder=5)

# ---------------------------------------------------------------- (b) infra-slow encoder
panel(2.52, 4.0, 4.62, 1.5, "(b) Infra-slow EEG encoder")
yb, hb = 4.8, 0.4
box(2.62, yb, 0.42, hb, "", kind="data")
trace(2.66, 3.0, yb + 0.25, 0.06, "eeg", lw=0.45)
ax.text(2.83, yb + 0.09, r"$\mathbf{x}$", ha="center", va="center", fontsize=6.5, color=INK, zorder=5)
xs = [(3.18, 0.66, "Resample", "200 Hz; z-score\nper window"),
      (3.98, 0.84, "Low-pass 0.5 Hz", "Butterworth, 4th order,\nzero phase"),
      (4.96, 0.7, "Bin", r"30 means of 1 s" + "\nper channel"),
      (5.8, 0.62, "z-score", "over the person's\nown windows")]
prev = 3.04
for x, w, t, s in xs:
    arrow([(prev, yb + hb / 2), (x, yb + hb / 2)])
    box(x, yb, w, hb, "", kind="feat")
    ax.text(x + w / 2, yb + 0.29, t, ha="center", va="center", fontsize=6.2, color=INK,
            zorder=5)
    ax.text(x + w / 2, yb + 0.12, s, ha="center", va="center", fontsize=5.9, color=INK2, zorder=5,
            linespacing=1.0)
    prev = x + w
box(6.56, yb, 0.5, hb, "", kind="out")
ax.text(6.81, yb + 0.26, r"$\mathbf{F}$", ha="center", va="center", fontsize=7.5, color=INK, zorder=5)
ax.text(6.81, yb + 0.1, r"$30N_c$", ha="center", va="center", fontsize=5.4, color=INK2, zorder=5)
arrow([(prev, yb + hb / 2), (6.56, yb + hb / 2)])
shape(4.89, yb - 0.08, r"$N_c\times6000$", ha="center")
shape(5.73, yb - 0.08, r"$N_c\times30$", ha="center")
shape(6.49, yb - 0.08, r"$30N_c$", ha="center")

# band-power comparator row (dashed = not part of the proposed model)
yc2 = 4.12
arrow([(3.51, yb), (3.51, yc2 + 0.15), (3.98, yc2 + 0.15)], color=MUTE, dashed=True)
ax.text(2.64, yc2 + 0.15, "Band-power\ncomparator", ha="left", va="center", fontsize=6.2, color=INK2,
        zorder=5, linespacing=1.0)
box(3.98, yc2, 1.0, 0.3, "Band-pass, 5 bands", kind="base", dashed=True, fs=5.8, sub="1–45 Hz")
box(5.22, yc2, 0.98, 0.3, "Log power", kind="base", dashed=True, fs=5.8, sub="10 bins per window")
box(6.44, yc2, 0.62, 0.3, r"$\mathbf{F}_\mathrm{bp}$", kind="base", dashed=True, fs=6.5, sub=r"$50N_c$")
arrow([(4.98, yc2 + 0.15), (5.22, yc2 + 0.15)], color=MUTE, dashed=True)
arrow([(6.2, yc2 + 0.15), (6.44, yc2 + 0.15)], color=MUTE, dashed=True)

# ---------------------------------------------------------------- (c) shared low-rank space
panel(2.52, 2.2, 4.62, 1.7, "(c) Shared low-rank EEG space")
r1, r2, r3, hh = 3.36, 2.94, 2.34, 0.32
box(2.62, r1, 0.62, hh, r"$\mathbf{F}_\mathrm{O}$, $\mathbf{Y}_\mathrm{O}$", kind="data", fs=7.4,
    sub="fNIRS cohorts")
box(2.62, r2, 0.62, hh, r"$\mathbf{F}_\mathrm{M}$, $\mathbf{Y}_\mathrm{M}$", kind="data", fs=7.4,
    sub="fMRI cohort")
box(3.42, r1, 0.5, hh, r"ridge $\hat{B}_\mathrm{O}$", kind="ridge")
box(3.42, r2, 0.5, hh, r"ridge $\hat{B}_\mathrm{M}$", kind="ridge")
box(4.1, r1, 0.64, hh, r"$\hat{B}_\mathrm{O}\,/\,\Vert\hat{B}_\mathrm{O}\Vert_F$", kind="ridge", fs=7.4)
box(4.1, r2, 0.64, hh, r"$\hat{B}_\mathrm{M}\,/\,\Vert\hat{B}_\mathrm{M}\Vert_F$", kind="ridge", fs=7.4)
for r in (r1, r2):
    arrow([(3.24, r + hh / 2), (3.42, r + hh / 2)])
    arrow([(3.92, r + hh / 2), (4.1, r + hh / 2)])
ym = (r1 + r2 + hh) / 2
op(4.95, ym, "C")
arrow([(4.74, r1 + hh / 2), (4.95, r1 + hh / 2), (4.95, ym + 0.065)])
arrow([(4.74, r2 + hh / 2), (4.95, r2 + hh / 2), (4.95, ym - 0.065)])
box(5.18, ym - 0.17, 0.68, 0.34, "", kind="basis")
ax.text(5.52, ym + 0.105, "SVD", ha="center", va="center", fontsize=7.0, color=INK, zorder=5)
matrix_icon(5.25, ym - 0.145, 0.1, 0.19, r"$U$")
matrix_icon(5.41, ym - 0.105, 0.12, 0.12, r"$\Sigma$")
matrix_icon(5.59, ym - 0.085, 0.21, 0.1, r"$V^{\!\top}$")
box(6.08, ym - 0.17, 0.98, 0.34, r"$U_K$: first $K$ columns", kind="basis", sub=r"shared basis, $K=32$")
arrow([(5.015, ym), (5.18, ym)])
arrow([(5.86, ym), (6.08, ym)])

# head row
box(2.62, r3, 0.62, hh, r"$\mathbf{F}_m$", kind="data", fs=7.6, sub="new person")
op(3.67, r3 + hh / 2, "x")
arrow([(3.24, r3 + hh / 2), (3.605, r3 + hh / 2)])
arrow([(6.57, ym - 0.17), (6.57, 2.82), (3.67, 2.82), (3.67, r3 + hh / 2 + 0.065)])
box(3.96, r3, 0.98, hh, r"ridge head $\hat{T}_m$", kind="ridge", sub=r"$m\in\{\mathrm{O},\mathrm{M}\}$")
arrow([(3.735, r3 + hh / 2), (3.96, r3 + hh / 2)])
shape(3.75, r3 + hh / 2 + 0.14, r"$n\times32$")
box(5.18, r3, 1.88, hh, r"$\hat{\mathbf{Y}}_m=\mathbf{F}_m U_K\hat{T}_m$", kind="out", fs=7.6,
    sub=r"tested for non-inferiority, margin $-0.02$")
arrow([(4.94, r3 + hh / 2), (5.18, r3 + hh / 2)])

# ---------------------------------------------------------------- (d) calibration and reliability
panel(2.52, 0.62, 4.62, 1.48, "(d) Two-run personal calibration and reliability score")
ra, rb, rc, h4 = 1.6, 1.2, 0.72, 0.3
box(2.62, rb - 0.02, 0.62, 0.74, "", kind="data")
ax.text(2.93, rb + 0.6, "Runs 1, 2", ha="center", va="center", fontsize=6.2, color=INK,
        zorder=5)
trace(2.67, 3.19, rb + 0.4, 0.04, "eeg", lw=0.45)
trace(2.67, 3.19, rb + 0.17, 0.12, "hrf", color=EDGE["ridge"], lw=0.7)
ax.text(2.93, rb + 0.03, "EEG + HbO", ha="center", va="center", fontsize=5.8, color=INK2, zorder=5)

box(3.42, ra, 0.82, h4, r"Mean response $\bar{y}_c$", kind="calib", fs=5.9)
box(3.42, rb, 0.82, h4, r"Standardise $z_c(\mathbf{x})$", kind="calib", fs=5.9)
arrow([(3.24, ra + h4 / 2), (3.42, ra + h4 / 2)])
arrow([(3.24, rb + h4 / 2), (3.42, rb + h4 / 2)])
box(4.42, rb, 0.86, h4, r"Personal ridge $f_\mathrm{p}$", kind="ridge", fs=5.9, sub=r"on $y-\bar{y}_c$")
arrow([(4.24, rb + h4 / 2), (4.42, rb + h4 / 2)])
op(5.48, ra + h4 / 2, "+")
arrow([(4.24, ra + h4 / 2), (5.415, ra + h4 / 2)])
arrow([(5.28, rb + h4 / 2), (5.48, rb + h4 / 2), (5.48, ra + h4 / 2 - 0.065)])
box(5.7, ra, 1.36, h4, r"$\hat{y}_\mathrm{pers}=\bar{y}_c+f_\mathrm{p}(z_c(\mathbf{x}))$", kind="out",
    fs=5.9)
arrow([(5.545, ra + h4 / 2), (5.7, ra + h4 / 2)])
ax.text(6.38, ra - 0.05, r"gain = $r(\hat{y}_\mathrm{pers})-r(\bar{y}_c)$ on later runs",
        ha="center", va="center", fontsize=5.7, color=INK2, zorder=5)

box(3.42, rc, 0.82, h4, r"Population ridge $f_\mathrm{pop}$", kind="ridge", fs=5.5, sub="other people")
op(4.85, rc + h4 / 2, "+")
arrow([(4.24, rc + h4 / 2), (4.785, rc + h4 / 2)])
arrow([(4.85, rb), (4.85, rc + h4 / 2 + 0.065)])
ax.text(4.9, rb - 0.1, "0.5", ha="left", va="center", fontsize=5.8, color=INK2, zorder=5)
ax.text(4.6, rc + h4 / 2 + 0.06, "0.5", ha="center", va="center", fontsize=5.8, color=INK2, zorder=5)
box(5.06, rc, 0.88, h4, r"Hybrid $\hat{y}_\mathrm{hyb}$", kind="calib", fs=5.9,
    sub=r"$\bar{y}_c+0.5f_\mathrm{p}+0.5f_\mathrm{pop}$")
arrow([(4.915, rc + h4 / 2), (5.06, rc + h4 / 2)])
box(6.14, rc, 0.92, h4, r"Score $s$", kind="out", fs=5.9, bold=True, sub="cross-fitted, run 1 ↔ 2")
arrow([(5.94, rc + h4 / 2), (6.14, rc + h4 / 2)])
box(6.14, rb - 0.1, 0.92, h4, "Predicted accuracy", kind="out", fs=5.6, bold=True,
    sub=r"$\hat{a}s+\hat{b}$, fitted on other cohorts")
arrow([(6.6, rc + h4), (6.6, rb - 0.1)])

# ---------------------------------------------------------------- legend
ax.add_patch(Rectangle((0.06, 0.05), 7.08, 0.5, fc="white", ec="#9a9a9a", lw=0.5, zorder=0))
items = [("feat", "Fixed transform"), ("basis", "Shared basis (SVD)"), ("ridge", "Ridge map (closed form)"),
         ("calib", "Person-specific step"), ("out", "Output"), ("data", "Input data")]
x = 0.16
for kind, lab in items:
    ax.add_patch(FancyBboxPatch((x, 0.36), 0.2, 0.11, boxstyle="square,pad=0",
                                fc=FILL[kind], ec=EDGE[kind], lw=LW, zorder=3))
    ax.text(x + 0.26, 0.415, lab, ha="left", va="center", fontsize=5.8, color=INK, zorder=4)
    x += 0.33 + len(lab) * 0.04
ax.add_patch(FancyBboxPatch((x, 0.36), 0.2, 0.11, boxstyle="square,pad=0",
                            fc=FILL["base"], ec=MUTE, lw=LW, ls=(0, (3, 2)), zorder=3))
ax.text(x + 0.26, 0.415, "Comparator only", ha="left", va="center", fontsize=5.8, color=INK,
        zorder=4)
sy = 0.17
for xx, sym, lab in [(0.24, "+", "Weighted sum"), (1.3, "x", "Matrix product"), (2.4, "C", "Concatenation")]:
    op(xx, sy, sym, r=0.055)
    ax.text(xx + 0.1, sy, lab, ha="left", va="center", fontsize=5.8, color=INK, zorder=4)
ax.text(3.3, sy, r"$N_c$: EEG channels;  $n$: windows;  $C$: fNIRS channels;  "
        "HbO, HbR: oxy-, deoxyhaemoglobin", ha="left", va="center", fontsize=5.5,
        color=INK2, zorder=4)

stem = sys.argv[1] if len(sys.argv) > 1 else "bspc/Figure_3"
fig.savefig(stem + ".pdf", facecolor="white")
fig.savefig(stem + ".png", dpi=300, facecolor="white")
print("wrote", stem)
