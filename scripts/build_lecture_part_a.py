"""Builds lectures/Lecture_A_The_Brains_Two_Signals.pdf (teaching notes, Part A).

Figures are schematic illustrations drawn here (not project data) except where a
caption says the numbers come from the project. Run from the repo root:
    python scripts/build_lecture_part_a.py
"""
import subprocess
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "report_build" / "lecture_a"
FIG = BUILD / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUTDIR = ROOT / "lectures"
OUTDIR.mkdir(exist_ok=True)
PDF = OUTDIR / "Lecture_A_The_Brains_Two_Signals.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, VIOLET = "#e34948", "#4a3aa7"
GRAY, LIGHT, INK, MUTED, GRID = "#a8a6a0", "#e4e2dc", "#1a1a1a", "#55534f", "#e8e7e3"
HBO, HBR = RED, VIOLET  # conventional colours: oxygenated red, deoxygenated blue/violet
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 10, "svg.fonttype": "none",
    "axes.edgecolor": "#bdbbb5", "axes.labelcolor": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
})
W = 6.9


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f3f5f8", ec="#9aa7b8", fs=9.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, wrap=True)


def arrow(ax, p, q, color=MUTED, lw=1.4, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def hrf(t, peak=6.0, under=16.0, ratio=1 / 6):
    t = np.clip(t, 0, None)
    g1 = t ** (peak - 1) * np.exp(-t) / factorial(int(peak - 1))
    g2 = t ** (under - 1) * np.exp(-t) / factorial(int(under - 1))
    h = g1 - ratio * g2
    return h / h.max()


# ---------------------------------------------------------------- figures
def f_bigpicture():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.1, 1.5, 1.8, 1.0, "Neurons fire\n(brain activity)", fc="#fff4e5", ec=YELLOW, weight="bold")
    box(ax, 2.6, 2.75, 2.2, 0.9, "Electrical field\nspreads instantly", fc="#eaf2fc", ec=BLUE)
    box(ax, 2.6, 0.35, 2.2, 0.9, "Cells need energy\nblood flow increases", fc="#fdeceb", ec=RED)
    box(ax, 5.5, 2.75, 2.0, 0.9, "EEG\n(milliseconds)", fc=BLUE, ec=BLUE, color="white", weight="bold")
    box(ax, 5.5, 0.35, 2.0, 0.9, "fNIRS / fMRI\n(seconds later)", fc=RED, ec=RED, color="white", weight="bold")
    arrow(ax, (1.9, 2.2), (2.6, 3.2)); arrow(ax, (1.9, 1.8), (2.6, 0.8))
    arrow(ax, (4.8, 3.2), (5.5, 3.2)); arrow(ax, (4.8, 0.8), (5.5, 0.8))
    arrow(ax, (7.6, 3.1), (7.6, 0.95), color=ORANGE, lw=2.2, rad=-0.45)
    ax.text(8.25, 2.05, "Our project:\ncan we predict\nthe bottom signal\nfrom the top one?", fontsize=9.5,
            color=ORANGE, va="center", weight="bold")
    save(fig, "a1_bigpicture")


def f_coupling():
    fig, ax = plt.subplots(figsize=(W, 1.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.6); ax.axis("off")
    steps = ["1. Neurons\nfire", "2. They use up\nenergy (ATP)", "3. Helper cells\nrelease 'open\nup' chemicals",
             "4. Small\narteries widen", "5. Oxygen-rich\nblood floods in"]
    x = 0.05
    for i, s in enumerate(steps):
        box(ax, x, 0.3, 1.75, 1.15, s, fc="#fdeceb" if i == 4 else "#f3f5f8", ec=RED if i == 4 else "#9aa7b8", fs=8.6)
        if i < 4:
            arrow(ax, (x + 1.79, 0.87), (x + 2.0, 0.87))
        x += 2.0
    ax.text(0.05, 0.08, "time:  0 s", fontsize=8.5, color=MUTED)
    ax.text(9.95, 0.08, "≈ 1–2 s before flow starts rising, peak ≈ 5–6 s", fontsize=8.5, color=MUTED, ha="right")
    save(fig, "a2_coupling")


def f_overshoot():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.45})
    a = ax[0]
    cats = ["oxygen\nused", "oxygen\ndelivered"]
    a.bar([0, 1], [15, 40], color=[GRAY, RED], width=0.55)
    for x, v in zip([0, 1], [15, 40]):
        a.text(x, v + 1.2, f"+{v}%", ha="center", fontsize=9.5)
    a.set_xticks([0, 1]); a.set_xticklabels(cats); a.set_ylim(0, 48)
    a.set_ylabel("increase during activity")
    a.set_title("Supply overshoots demand"); a.grid(axis="x", visible=False)
    a = ax[1]
    labels = ["rest", "active"]
    o = np.array([60, 72]); r = np.array([40, 28])
    a.bar([0, 1], o, color=HBO, width=0.55, label="HbO (oxygen-carrying)")
    a.bar([0, 1], r, bottom=o, color=HBR, width=0.55, label="HbR (oxygen given away)")
    for x in [0, 1]:
        a.text(x, o[x] / 2, f"{o[x]}%", ha="center", va="center", color="white", fontsize=9)
        a.text(x, o[x] + r[x] / 2, f"{r[x]}%", ha="center", va="center", color="white", fontsize=9)
    a.set_xticks([0, 1]); a.set_xticklabels(labels); a.set_ylim(0, 112)
    a.set_title("So the blood in the area\nbecomes MORE oxygenated", fontsize=10.5)
    a.legend(loc="upper center", ncol=1, fontsize=8, bbox_to_anchor=(0.5, -0.18))
    a.grid(axis="x", visible=False); a.set_yticks([])
    save(fig, "a3_overshoot")


EL = {"FP1": (-0.3, 0.9), "FP2": (0.3, 0.9), "AF3": (-0.32, 0.72), "AF4": (0.32, 0.72),
      "F7": (-0.72, 0.56), "F3": (-0.4, 0.5), "FZ": (0, 0.47), "F4": (0.4, 0.5), "F8": (0.72, 0.56),
      "FC5": (-0.66, 0.26), "FC1": (-0.22, 0.24), "FC2": (0.22, 0.24), "FC6": (0.66, 0.26),
      "T7": (-0.9, 0), "C3": (-0.45, 0), "CZ": (0, 0), "C4": (0.45, 0), "T8": (0.9, 0),
      "CP5": (-0.66, -0.26), "CP1": (-0.22, -0.24), "CP2": (0.22, -0.24), "CP6": (0.66, -0.26),
      "P7": (-0.72, -0.56), "P3": (-0.4, -0.5), "PZ": (0, -0.47), "P4": (0.4, -0.5), "P8": (0.72, -0.56),
      "PO3": (-0.32, -0.72), "PO4": (0.32, -0.72), "O1": (-0.3, -0.9), "OZ": (0, -0.95), "O2": (0.3, -0.9)}


def f_eeg():
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.25})
    a = ax[0]
    a.set_xlim(-1.25, 1.25); a.set_ylim(-1.2, 1.3); a.set_aspect("equal"); a.axis("off")
    a.add_patch(Circle((0, 0), 1.05, fc="#fbfaf7", ec="#8a877f", lw=1.4))
    a.add_patch(Polygon([(-0.12, 1.03), (0, 1.2), (0.12, 1.03)], closed=True, fc="#fbfaf7", ec="#8a877f", lw=1.4))
    for sx in (-1, 1):
        a.add_patch(Rectangle((sx * 1.05 - (0.08 if sx > 0 else 0), -0.18), 0.08, 0.36, fc="#fbfaf7", ec="#8a877f"))
    motor = {"C3", "CZ", "C4"}
    for n, (x, y) in EL.items():
        hot = n in motor
        a.add_patch(Circle((x, y), 0.085, fc=BLUE if hot else "white", ec=BLUE if hot else "#9a978f", lw=1))
        a.text(x, y, n.title() if len(n) > 2 else n.replace("Z", "z"), ha="center", va="center",
               fontsize=5.2, color="white" if hot else INK, weight="bold" if hot else "normal")
    a.text(0, 1.25, "nose", ha="center", fontsize=8, color=MUTED)
    a.text(-1.22, -0.35, "left", fontsize=8, color=MUTED); a.text(1.22, -0.35, "right", fontsize=8, color=MUTED, ha="right")
    a.set_title("A  The 32 electrodes we use (top view)", fontsize=10)
    a = ax[1]
    fs = 200
    t = np.arange(0, 4, 1 / fs)
    rng = np.random.default_rng(3)
    amp = np.where(t < 2, 1.0, 0.35)
    amp = np.convolve(amp, np.ones(40) / 40, mode="same")
    x = amp * np.sin(2 * np.pi * 10.5 * t) + 0.25 * rng.standard_normal(len(t))
    a.plot(t, x, color=INK, lw=0.7)
    a.plot(t, amp * 1.15, color=BLUE, lw=2)
    a.axvspan(2, 4, color=BLUE, alpha=0.07)
    a.axvline(2, color=BLUE, lw=1, ls="--")
    a.text(0.05, 2.05, "rest:\nstrong mu rhythm", fontsize=8.5, color=MUTED)
    a.text(2.08, 2.05, "imagining right-hand squeeze:\nrhythm shrinks (ERD)", fontsize=8.5, color=BLUE)
    a.set_ylim(-1.9, 2.75); a.set_yticks([])
    a.set_xlabel("time (s)  —  EEG at electrode C3")
    a.set_title("B  What one electrode records", fontsize=10)
    save(fig, "a4_eeg")


def f_bands():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    t = np.linspace(0, 1, 2000)
    bands = [("Delta", 2, "0.5–4 Hz", "deep sleep"), ("Theta", 6, "4–8 Hz", "drowsiness, memory"),
             ("Alpha / mu", 10, "8–13 Hz", "relaxed, eyes closed; mu = resting motor cortex"),
             ("Beta", 20, "13–30 Hz", "active thinking, motor control"), ("Gamma", 40, "30+ Hz", "intense processing")]
    for i, (n, f, r, m) in enumerate(bands):
        y = (4 - i) * 1.0
        ax.plot(t, y + 0.33 * np.sin(2 * np.pi * f * t), color=[VIOLET, BLUE, AQUA, YELLOW, ORANGE][i], lw=1.2)
        ax.text(-0.02, y, f"{n}\n{r}", ha="right", va="center", fontsize=8.8, weight="bold")
        ax.text(1.02, y, m, ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 4.6)
    ax.set_yticks([]); ax.set_xlabel("one second of signal"); ax.grid(False)
    ax.spines["left"].set_visible(False)
    save(fig, "a5_bands")


def f_bandpower():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.3, "width_ratios": [1.4, 1]})
    fs = 200
    t = np.arange(0, 6, 1 / fs)
    rng = np.random.default_rng(5)
    mu = np.where((t > 2) & (t < 4.5), 0.35, 1.0)
    mu = np.convolve(mu, np.ones(60) / 60, "same")
    x = mu * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 21 * t) * mu ** 0.5 + 0.3 * rng.standard_normal(len(t))
    a = ax[0]
    a.plot(t, x, color=GRAY, lw=0.5, label="raw EEG (200 samples/s)")
    env = np.convolve(mu, np.ones(1), "same")
    step = 10
    a.plot(t[::step], (mu * 1.6)[::step], color=BLUE, lw=2.2, marker="o", ms=2.2, label="mu band power (20 values/s)")
    a.set_yticks([]); a.set_xlabel("time (s)")
    a.legend(loc="lower left", fontsize=8); a.set_ylim(-2.6, 2.4)
    a.set_title("Wiggles → 'how loud is each rhythm?'", fontsize=10)
    a = ax[1]
    names = ["delta", "theta", "mu/alpha", "beta"]
    rest = [0.5, 0.45, 1.0, 0.6]; act = [0.5, 0.45, 0.4, 0.35]
    xx = np.arange(4)
    a.bar(xx - 0.19, rest, 0.36, color=GRAY, label="rest")
    a.bar(xx + 0.19, act, 0.36, color=BLUE, label="imagining")
    a.set_xticks(xx); a.set_xticklabels(names, fontsize=8.5); a.set_yticks([])
    a.legend(fontsize=8); a.grid(axis="x", visible=False)
    a.set_title("Like a music equaliser", fontsize=10)
    save(fig, "a6_bandpower")


def f_spectra():
    nm = np.array([650, 700, 750, 760, 800, 850, 900, 950])
    hbo = np.array([368, 290, 518, 586, 816, 1058, 1198, 1204]) / 1000
    hbr = np.array([3750, 1795, 1405, 1548, 761, 691, 761, 642]) / 1000
    from scipy.interpolate import PchipInterpolator
    x = np.linspace(680, 950, 300)
    fig, a = plt.subplots(figsize=(W, 2.5))
    a.plot(x, PchipInterpolator(nm, hbo)(x), color=HBO, lw=2.2, label="HbO (oxygenated)")
    a.plot(x, PchipInterpolator(nm, hbr)(x), color=HBR, lw=2.2, label="HbR (deoxygenated)")
    for w, lab in [(760, "760 nm\nHbR absorbs more"), (850, "850 nm\nHbO absorbs more")]:
        a.axvline(w, color=MUTED, lw=1, ls="--")
        a.text(w + 3, 2.15, lab, fontsize=8.5, color=INK, va="top")
    a.annotate("isosbestic point (~800 nm):\nboth absorb equally", xy=(801, 0.78), xytext=(858, 0.12),
               fontsize=8.3, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    a.set_xlim(680, 950); a.set_ylim(0, 2.2)
    a.set_xlabel("wavelength of light (nm)"); a.set_ylabel("how strongly it absorbs")
    a.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    save(fig, "a7_spectra")


def f_banana():
    fig, a = plt.subplots(figsize=(W, 2.7))
    a.set_xlim(0, 10); a.set_ylim(-4.2, 0.9); a.axis("off")
    layers = [("scalp", 0, -0.6, "#f6dccb"), ("skull", -0.6, -1.4, "#efe8d8"), ("fluid (CSF)", -1.4, -1.75, "#dfeff6"),
              ("cortex (neurons)", -1.75, -3.0, "#f1e1ea"), ("white matter", -3.0, -4.2, "#f5f3ef")]
    for n, top, bot, c in layers:
        a.add_patch(Rectangle((0, bot), 10, top - bot, fc=c, ec="none"))
        a.text(9.9, (top + bot) / 2, n, ha="right", va="center", fontsize=8.5, color=MUTED)
    xs, xd = 2.0, 6.5
    xx = np.linspace(xs, xd, 100)
    depth = 2.35
    yy = -depth * np.sin(np.pi * (xx - xs) / (xd - xs)) ** 0.9
    a.fill_between(xx, yy * 1.18, yy * 0.62, color=RED, alpha=0.18)
    a.plot(xx, yy * 0.9, color=RED, lw=2)
    xsd = np.linspace(2.0, 2.8, 30)
    ys = -0.45 * np.sin(np.pi * (xsd - 2.0) / 0.8)
    a.plot(xsd, ys, color=YELLOW, lw=2)
    for x0, lab, c, ha, dx in [(xs, "light source", RED, "right", -0.25), (xd, "detector", INK, "center", 0),
                               (2.8, "short-channel detector (8 mm)", YELLOW, "left", 0.25)]:
        a.add_patch(Rectangle((x0 - 0.18, 0), 0.36, 0.28, fc=c, ec="none"))
        a.text(x0 + dx, 0.38, lab, ha=ha, va="bottom", fontsize=8.3)
    a.annotate("", xy=(xs, -3.95), xytext=(xd, -3.95), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    a.text((xs + xd) / 2, -4.15, "≈ 3 cm between source and detector", ha="center", fontsize=8.5)
    a.text(4.25, -2.55, "banana-shaped path reaches the cortex", ha="center", fontsize=8.5, color=RED)
    save(fig, "a8_banana")


def f_bold():
    t = np.linspace(0, 30, 600)
    h = hrf(t)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.3})
    a = ax[0]
    a.plot(t, h, color=HBO, lw=2.2, label="HbO (fNIRS)")
    a.plot(t, -0.3 * hrf(t - 1), color=HBR, lw=2.2, label="HbR (fNIRS)")
    a.axhline(0, color=MUTED, lw=0.8)
    a.axvline(0, color=ORANGE, lw=1.2, ls="--"); a.text(0.3, 1.02, "brief neural event", color=ORANGE, fontsize=8.5)
    a.set_xlabel("time after the event (s)"); a.set_yticks([0]); a.set_ylabel("change")
    a.legend(loc="center right"); a.set_title("What fNIRS sees", fontsize=10); a.set_ylim(-0.5, 1.15)
    a = ax[1]
    a.plot(t, h, color=AQUA, lw=2.2, label="BOLD (fMRI)")
    a.plot(t, -0.3 * hrf(t - 1), color=HBR, lw=1.4, ls=":", label="HbR")
    a.axhline(0, color=MUTED, lw=0.8)
    a.text(12, 0.6, "BOLD rises because\nHbR falls: it is roughly\nthe mirror image of HbR", fontsize=8.5, color=MUTED)
    a.set_xlabel("time after the event (s)"); a.set_yticks([0])
    a.legend(loc="upper right", bbox_to_anchor=(1, 0.86)); a.set_title("What fMRI sees", fontsize=10); a.set_ylim(-0.5, 1.15)
    save(fig, "a9_bold")


def f_hrf():
    t = np.linspace(-2, 30, 800)
    h = hrf(t)
    fig, a = plt.subplots(figsize=(W, 2.6))
    a.plot(t, h, color=HBO, lw=2.4)
    a.axhline(0, color=MUTED, lw=0.8)
    a.axvline(0, color=ORANGE, lw=1.3, ls="--")
    pk = t[np.argmax(h)]
    um = t[np.argmin(h)]
    a.annotate("1. Delay: almost\nnothing for the\nfirst 1–2 s", xy=(1.2, 0.03), xytext=(-7.8, 0.45), fontsize=8.6,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    a.annotate(f"2. Peak ≈ {pk:.0f} s after the event", xy=(pk, 1.0), xytext=(pk + 3.2, 0.95), fontsize=8.6,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    a.annotate("3. Return towards baseline", xy=(10.5, 0.25), xytext=(13, 0.52), fontsize=8.6,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    a.annotate("4. Undershoot: dips below\nbaseline, recovers by ~25–30 s", xy=(um, h.min()), xytext=(19.5, 0.3),
               fontsize=8.6, arrowprops=dict(arrowstyle="->", color=MUTED))
    a.text(0.25, -0.2, "event", color=ORANGE, fontsize=8.6)
    a.set_xlabel("time (s)"); a.set_ylabel("blood response"); a.set_yticks([0])
    a.set_ylim(-0.3, 1.15); a.set_xlim(-8, 30)
    save(fig, "a10_hrf")


def f_convolution():
    t = np.linspace(0, 40, 1000)
    events = [0, 3, 6]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.25}, sharey=True)
    a = ax[0]
    total = np.zeros_like(t)
    cols = [BLUE, AQUA, YELLOW]
    for e, c in zip(events, cols):
        r = hrf(t - e)
        total += r
        a.plot(t, r, color=c, lw=1.6)
        a.axvline(e, color=c, lw=1, ls="--")
    a.set_title("Three events, three responses", fontsize=10)
    a.set_xlabel("time (s)"); a.set_yticks([0]); a.axhline(0, color=MUTED, lw=0.8)
    a = ax[1]
    for e, c in zip(events, cols):
        a.plot(t, hrf(t - e), color=c, lw=1, alpha=0.5)
    a.plot(t, total, color=HBO, lw=2.4, label="what we actually measure")
    a.axhline(0, color=MUTED, lw=0.8)
    a.legend(loc="upper right"); a.set_xlabel("time (s)")
    a.set_title("They overlap and add up", fontsize=10)
    save(fig, "a11_convolution")


def f_noise_spectrum():
    f = np.logspace(-2.3, 0.5, 600)
    def bump(c, w, h):
        return h * np.exp(-0.5 * ((np.log10(f) - np.log10(c)) / w) ** 2)
    s = 0.25 * (f ** -0.5) / 7 + bump(0.033, 0.07, 0.9) + bump(0.1, 0.06, 0.75) + bump(0.27, 0.06, 0.55) + bump(1.15, 0.04, 0.85)
    fig, a = plt.subplots(figsize=(W, 2.5))
    a.axvspan(0.01, 0.2, color=BLUE, alpha=0.07)
    a.plot(f, s, color=INK, lw=1.6)
    a.set_xscale("log")
    for x, lab, c in [(0.033, "task\n(our trials)", BLUE), (0.1, "Mayer waves\n(blood pressure)", ORANGE),
                      (0.27, "breathing", ORANGE), (1.15, "heartbeat", ORANGE)]:
        a.text(x, s[np.argmin(abs(f - x))] + 0.06, lab, ha="center", fontsize=8.5, color=c)
    a.text(0.0105, 1.55, "band we keep: 0.01–0.2 Hz", fontsize=8.5, color=BLUE)
    a.set_ylim(0, 1.7); a.set_yticks([])
    a.set_xlabel("frequency (Hz) — how many cycles per second")
    a.set_ylabel("strength in the fNIRS signal")
    save(fig, "a12_noise")


def f_signatures():
    t = np.linspace(-2, 28, 500)
    h = hrf(t)
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1, 0.9]})
    a = ax[0]
    a.plot(t, h, color=HBO, lw=2, label="HbO"); a.plot(t, -0.35 * hrf(t - 1), color=HBR, lw=2, label="HbR")
    a.axhline(0, color=MUTED, lw=0.8); a.set_yticks([0]); a.set_xlabel("time (s)")
    a.set_title("Neural: opposite\ndirections", fontsize=10); a.legend(fontsize=8); a.set_ylim(-0.55, 1.15)
    a = ax[1]
    wave = 0.7 * np.sin(2 * np.pi * 0.1 * t) + 0.4 * h
    a.plot(t, wave, color=HBO, lw=2); a.plot(t, 0.75 * wave - 0.05, color=HBR, lw=2)
    a.axhline(0, color=MUTED, lw=0.8); a.set_yticks([0]); a.set_xlabel("time (s)")
    a.set_title("Systemic: same\ndirection", fontsize=10); a.set_ylim(-1.0, 1.15)
    a = ax[2]
    a.axhspan(-1, 0, color=BLUE, alpha=0.06); a.axhspan(0, 1, color=ORANGE, alpha=0.06)
    a.bar([0, 1], [-0.44, 0.92], color=[BLUE, ORANGE], width=0.55)
    for x, v in zip([0, 1], [-0.44, 0.92]):
        a.text(x, v + (0.06 if v > 0 else -0.13), f"{v:+.2f}", ha="center", fontsize=9)
    a.axhline(0, color=MUTED, lw=1)
    a.set_xticks([0, 1]); a.set_xticklabels(["TU Berlin", "HEFMI"]); a.set_ylim(-1, 1.1)
    a.set_ylabel("HbO–HbR correlation"); a.grid(axis="x", visible=False)
    a.set_title("Our two datasets", fontsize=10)
    save(fig, "a13_signatures")


def f_gsr():
    t = np.linspace(0, 40, 800)
    rng = np.random.default_rng(2)
    glob = 0.6 * np.sin(2 * np.pi * 0.1 * t + 0.4) + 0.25 * np.sin(2 * np.pi * 0.04 * t)
    resp = hrf(t - 5)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for a, n_active, title in [(ax[0], 1, "Response in ONE channel: GSR helps"),
                               (ax[1], 7, "Response in MOST channels: GSR erases it")]:
        X = np.array([glob + (resp if i < n_active else 0) + 0.04 * rng.standard_normal(len(t)) for i in range(8)])
        g = X.mean(0)
        beta = (X @ g) / (g @ g)
        clean = X - beta[:, None] * g
        a.plot(t, X[0], color=GRAY, lw=1, label="measured (brain + systemic wave)")
        a.plot(t, resp, color=INK, lw=1.4, ls="--", label="true brain response")
        a.plot(t, clean[0], color=BLUE if n_active == 1 else ORANGE, lw=2, label="after GSR")
        a.axhline(0, color=MUTED, lw=0.6)
        a.set_title(title, fontsize=10); a.set_xlabel("time (s)"); a.set_yticks([0])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h[:2] + [plt.Line2D([], [], color=BLUE, lw=2), plt.Line2D([], [], color=ORANGE, lw=2)],
               l[:2] + ["after GSR (recovered)", "after GSR (erased)"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.2), fontsize=8.3)
    save(fig, "a14_gsr")


def f_trial():
    t = np.linspace(-3, 30, 900)
    task = ((t >= 0) & (t < 10)).astype(float)
    k = hrf(np.linspace(0, 30, 900 // 33 * 30))
    dt = t[1] - t[0]
    kern = hrf(np.arange(0, 30, dt))
    hbo = np.convolve(task, kern)[:len(t)]
    hbo /= hbo.max()
    hbr = -0.3 * np.convolve(task, hrf(np.arange(0, 30, dt) - 1))[:len(t)] / np.convolve(task, kern)[:len(t)].max()
    mu = 1 - 0.6 * np.convolve(task, np.ones(int(0.4 / dt)) / int(0.4 / dt), "same")
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.3), sharex=True, gridspec_kw={"hspace": 0.25, "height_ratios": [1, 1.3]})
    for a in ax:
        a.axvspan(0, 10, color=YELLOW, alpha=0.1)
    ax[0].plot(t, mu, color=BLUE, lw=2.2)
    ax[0].set_ylabel("EEG mu power"); ax[0].set_yticks([])
    ax[0].text(0.3, 1.05, "drops within half a second", fontsize=8.5, color=BLUE)
    ax[0].text(-2.8, 0.45, "imagery\nperiod →", fontsize=8.5, color="#8a6a00")
    ax[0].set_ylim(0.25, 1.25)
    ax[1].plot(t, hbo, color=HBO, lw=2.2, label="HbO")
    ax[1].plot(t, hbr, color=HBR, lw=2.2, label="HbR")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_ylabel("fNIRS"); ax[1].set_yticks([0])
    ax[1].legend(loc="upper right")
    ax[1].annotate("rises slowly, peaks after\nthe imagery has ended", xy=(t[np.argmax(hbo)], 1.0), xytext=(15.5, 0.62),
                   fontsize=8.5, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax[1].set_xlabel("time from the cue (s)  —  one 30-second window, as used in our project")
    save(fig, "a15_trial")


for fn in [f_bigpicture, f_coupling, f_overshoot, f_eeg, f_bands, f_bandpower, f_spectra, f_banana,
           f_bold, f_hrf, f_convolution, f_noise_spectrum, f_signatures, f_gsr, f_trial]:
    fn()


# ---------------------------------------------------------------- helpers for the text
def fig(name, num, title, body):
    return f'<figure><img src="figures/{name}.svg"><figcaption><b>Figure {num}. {title}</b> {body}</figcaption></figure>'


def callout(kind, title, body):
    return f'<div class="call {kind}"><div class="ctitle">{title}</div>{body}</div>'


A = lambda t, b: callout("analogy", "Analogy · " + t, b)
E = lambda t, b: callout("example", "Example · " + t, b)
P = lambda b: callout("project", "In our project", b)
K = lambda b: callout("key", "Key idea", b)


def Q(items):
    li = "".join(f"<li>{q}</li>" for q in items)
    return f'<div class="call think"><div class="ctitle">Pause and think</div><ol class="qs">{li}</ol><div class="hint">Try to answer before reading on. Answers are at the end.</div></div>'


# ---------------------------------------------------------------- document
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Lecture A: The Brain's Two Signals</title>
<style>
@page {{ size: A4; margin: 22mm 21mm 20mm 21mm;
  @bottom-center {{ content: counter(page); font-family: Calibri, sans-serif; font-size: 9pt; color: #777; }}
  @top-right {{ content: "NeuroVascNet course · Lecture A"; font-family: Calibri, sans-serif; font-size: 8.5pt; color: #999; }} }}
@page :first {{ @top-right {{ content: none; }} @bottom-center {{ content: none; }} }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: Cambria, Georgia, serif; font-size: 10.6pt; line-height: 1.5; color: #1a1a1a; margin: 0; }}
h1, h2, h3, .ctitle, figcaption, table, .kicker, .meta, .hint {{ font-family: Calibri, "Segoe UI", sans-serif; }}
.kicker {{ font-size: 10pt; letter-spacing: .08em; text-transform: uppercase; color: {BLUE}; font-weight: 700; margin-top: 6mm; }}
h1 {{ font-size: 25pt; line-height: 1.12; margin: 2pt 0 4pt; }}
.subtitle {{ font-size: 12.5pt; color: #444; font-style: italic; margin-bottom: 10pt; }}
.meta {{ font-size: 9.5pt; color: #555; border-top: 1.5px solid #222; border-bottom: .6px solid #bbb; padding: 5pt 0; margin-bottom: 12pt; }}
h2 {{ font-size: 16pt; margin: 20pt 0 6pt; padding-bottom: 2pt; border-bottom: .8px solid #ccc; break-after: avoid; }}
h3 {{ font-size: 12pt; margin: 13pt 0 3pt; break-after: avoid; color: #222; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }}
ul, ol {{ margin: 0 0 7pt; padding-left: 17pt; }} li {{ margin-bottom: 2.5pt; }}
figure {{ margin: 8pt 0 12pt; break-inside: avoid; }} figure img {{ width: 100%; display: block; }}
figcaption {{ font-size: 9pt; line-height: 1.35; color: #333; margin-top: 4pt; }}
.call {{ border-left: 3.5px solid; padding: 6pt 11pt 3pt; margin: 8pt 0 11pt; break-inside: avoid; border-radius: 0 4px 4px 0; }}
.call p:last-child {{ margin-bottom: 4pt; }}
.ctitle {{ font-weight: 700; font-size: 9.5pt; letter-spacing: .03em; text-transform: uppercase; margin-bottom: 3pt; }}
.analogy {{ background: #fff7e8; border-color: {YELLOW}; }} .analogy .ctitle {{ color: #8a5a00; }}
.example {{ background: #eef6f1; border-color: {AQUA}; }} .example .ctitle {{ color: #0f6b4a; }}
.project {{ background: #eef3fb; border-color: {BLUE}; }} .project .ctitle {{ color: #1d5aa6; }}
.key {{ background: #f6f2fb; border-color: {VIOLET}; }} .key .ctitle {{ color: {VIOLET}; }}
.think {{ background: #f6f6f4; border-color: #8a877f; }} .think .ctitle {{ color: #444; }}
.hint {{ font-size: 8.5pt; color: #777; margin: 2pt 0 4pt; }}
.objectives {{ background: #f4f6f9; padding: 8pt 12pt 3pt; margin-bottom: 10pt; break-inside: avoid; }}
.objectives h3 {{ margin-top: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; line-height: 1.35; margin: 4pt 0 12pt; break-inside: avoid; }}
thead th {{ border-top: 1.3px solid #222; border-bottom: .7px solid #222; text-align: left; padding: 3pt 6pt; }}
td {{ padding: 3pt 6pt; vertical-align: top; border-bottom: .4px solid #e2e2e2; }}
tbody tr:last-child td {{ border-bottom: 1.3px solid #222; }}
.tcap {{ font-family: Calibri, sans-serif; font-size: 9pt; color: #333; margin-top: 6pt; }}
.formula {{ text-align: center; font-size: 11.5pt; margin: 6pt 0 8pt; }}
.answers li {{ margin-bottom: 5pt; }}
</style></head><body>

<div class="kicker">NeuroVascNet course · Lecture A</div>
<h1>The Brain's Two Signals</h1>
<div class="subtitle">Neurons, blood flow, and the three machines that measure them: EEG, fNIRS and fMRI</div>
<div class="meta">Part A of 7 · No background assumed · About 90 minutes of reading</div>

<div class="objectives">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>explain why active brain tissue receives <i>more</i> oxygen than it uses, and why that matters for every measurement in this project;</li>
<li>describe what EEG, fNIRS and fMRI each measure, how fast they are, and what they cannot see;</li>
<li>sketch the haemodynamic response and explain why it arrives about 6 seconds late;</li>
<li>tell a genuine brain response apart from body noise, and explain why one of our datasets failed that test.</li>
</ul>
</div>

<h2>0. The whole project in one picture</h2>
<p>Before any detail, here is the question this entire project asks. When a group of brain cells becomes active, two things happen. First, the cells produce tiny electrical currents, and these can be picked up at the scalp almost instantly. Second, the cells use up energy, and the body responds by sending them more blood. That blood arrives several seconds later. We have one machine that listens to the electricity (EEG) and two machines that watch the blood (fNIRS and fMRI).</p>
{fig("a1_bigpicture", 1, "One brain event, two signals.", "The same neural activity produces a fast electrical signal and a slow blood-flow signal. The project asks whether the slow one can be predicted from the fast one alone.")}
<p>Why would anyone want this? EEG is cheap, portable and fast. fMRI costs millions and needs a hospital; fNIRS is cheaper but still bulky. If we could take a cheap EEG recording and <i>predict</i> what the blood is doing, we would get some of the information of the expensive machines for the price of the cheap one.</p>
{A("lightning and thunder", "<p>Lightning and thunder come from the same event. You see the flash immediately, and the sound arrives a few seconds later. If you have seen enough storms, you can watch the flash and predict roughly when the thunder will come and how loud it will be. EEG is the lightning: fast. The blood response is the thunder: slow and delayed. Our project tries to learn to predict the thunder from the lightning. Keep this picture in mind; we will come back to it several times, including to explain why the delay is different for different people, just as thunder takes longer to reach you the further away you stand.</p>")}

<h2>1. Neurons and neurovascular coupling</h2>
<h3>1.1 Neurons in two minutes</h3>
<p>Your brain contains about 86 billion nerve cells called <b>neurons</b>. A neuron is a cell that passes messages using electricity and chemistry. It receives signals from other neurons through thousands of contact points called <b>synapses</b>, adds them up, and if the total is strong enough it "fires": it sends a brief electrical pulse down a long cable (the axon) to the next neurons. A single pulse lasts about one millisecond (a thousandth of a second).</p>
<p>Neurons are organised into regions with different jobs. The one that matters most for us is the <b>motor cortex</b>, a strip running across the top of the brain from ear to ear, which controls movement. The left motor cortex controls the right side of the body, and the right controls the left. This crossing-over will matter when we look at electrode positions.</p>

<h3>1.2 The brain's energy problem</h3>
<p>Firing is expensive. Every pulse requires the neuron to pump charged particles back across its membrane afterwards, and that pumping burns energy in the form of a molecule called ATP. To make ATP, cells need glucose (sugar) and oxygen. The brain is about 2% of your body weight but uses about 20% of your oxygen and energy.</p>
<p>Here is the problem: the brain has almost no energy stores. Muscles keep reserves of sugar; the brain does not. It depends on a continuous supply delivered by the blood, and when one region suddenly works harder, that region needs extra supply within seconds.</p>
{A("a city without warehouses", "<p>Imagine a city where no shop keeps stock. Every loaf of bread sold must be delivered by truck at the moment it is needed. When one neighbourhood suddenly gets busy, the traffic system has to notice and send more trucks there, quickly. The roads are blood vessels, the trucks are red blood cells carrying oxygen, and the traffic-control system is neurovascular coupling.</p>")}

<h3>1.3 Neurovascular coupling: how the brain orders more blood</h3>
<p><b>Neurovascular coupling</b> is the name for the link between neural activity (neuro-) and blood flow (vascular). The chain of events is shown in Figure 2. When neurons fire, they and their neighbouring support cells (called astrocytes) release chemical messengers such as nitric oxide and potassium. These messengers tell the muscle around nearby small arteries to relax. The arteries widen, resistance falls, and more blood flows into the area.</p>
{fig("a2_coupling", 2, "The coupling chain.", "From neural firing to increased blood supply. The whole chain takes seconds, which is why blood-based signals always lag behind electrical ones.")}
<p>The important point is that steps 3 to 5 are slow compared to step 1. Electrical activity happens in milliseconds; blood vessels widen over seconds. This difference in speed is the origin of the delay at the heart of this project.</p>

<h3>1.4 The overshoot: why active tissue gets <i>more</i> oxygen than it needs</h3>
<p>You might expect active brain tissue to become <i>short</i> of oxygen, because it is using more. The opposite happens. When blood flow increases, it increases by much more than the tissue actually uses. A typical figure is that oxygen consumption rises by around 15% while blood flow rises by around 40% (the exact numbers vary). The fresh blood arrives so generously that much of its oxygen passes through unused.</p>
{fig("a3_overshoot", 3, "Supply overshoots demand.", "Illustrative numbers. Because delivery rises much more than use, the blood in an active region ends up carrying a higher share of oxygen than at rest.")}
{A("watering the whole garden bed", "<p>One plant in a bed is thirsty. Instead of giving that plant a cup of water, the gardener turns on the hose over the entire bed. The thirsty plant gets its water, but the soil ends up wetter than before. Active brain tissue is the thirsty plant; the extra blood is the hose; the soil is the blood in the region, which ends up richer in oxygen than it was at rest.</p>")}
<p>Haemoglobin is the protein in red blood cells that carries oxygen. It exists in two forms that we will meet constantly:</p>
<ul>
<li><b>HbO</b> (oxygenated haemoglobin, also written HbO<sub>2</sub> or oxy-Hb): haemoglobin carrying oxygen.</li>
<li><b>HbR</b> (deoxygenated or "reduced" haemoglobin, also written HHb or deoxy-Hb): haemoglobin that has given its oxygen away.</li>
</ul>
{K("<p>When a brain region becomes active, <b>HbO goes up and HbR goes down</b>. They move in opposite directions. This is the fingerprint of a genuine neural response, and it is the single most useful fact in this lecture. We will use it in Section 6 to show that one of our datasets was mostly measuring something else.</p>")}

<h3>1.5 Why this matters for the project</h3>
<p>All three hemodynamic measures in our project — HbO and HbR from fNIRS, and BOLD from fMRI — are measurements of this blood response. None of them sees neurons directly. They see the <i>consequence</i> of neural activity, filtered through blood vessels. EEG, by contrast, sees the electrical activity itself. The project therefore links two different physical processes, joined by neurovascular coupling.</p>
{Q(["Why does an active brain region end up with <i>more</i> oxygenated blood rather than less?",
    "If a region becomes active, which way do HbO and HbR move?",
    "Which step in Figure 2 is the slow one, and what does that imply for timing?"])}

<h2>2. EEG: listening to the brain's electricity</h2>
<h3>2.1 Where the signal comes from</h3>
<p><b>Electroencephalography (EEG)</b> measures voltage differences on the scalp using metal electrodes. The voltages are tiny, typically 10 to 100 microvolts (millionths of a volt). A single neuron is far too weak to detect from outside the head. What EEG picks up is the combined activity of millions of neurons in the outer layer of the brain (the cortex) that are arranged in parallel and active at the same moment. When they are active together, their small currents add up into a field strong enough to pass through the skull and reach the scalp.</p>
{A("the stadium microphone", "<p>Put a microphone outside a football stadium. You cannot hear any one conversation inside. But when 50,000 fans sing the same chant together, you hear it clearly. EEG works the same way: it hears only what large groups of neurons do <i>together</i>. Individual neurons, or groups doing different things, cancel out into background hiss. This is why EEG is sensitive to rhythms — repeated, synchronised activity — and poor at telling exactly where something happens.</p>")}

<h3>2.2 Electrodes and the 10-20 system</h3>
<p>To compare recordings across people and studies, electrodes are placed at standard positions called the <b>10-20 system</b> (named after the 10% and 20% spacing between positions, measured from landmarks on the skull). Each position has a name made of a letter and a number:</p>
<ul>
<li>The letter is the brain region underneath: <b>F</b>rontal, <b>C</b>entral, <b>P</b>arietal, <b>O</b>ccipital, <b>T</b>emporal. Combinations such as FC sit between two regions.</li>
<li>Odd numbers are on the left, even numbers on the right, and <b>z</b> means the midline.</li>
</ul>
<p>So <b>C3</b> sits over the left motor cortex and <b>C4</b> over the right. Because of the crossing-over mentioned in Section 1.1, imagining a movement of the <i>right</i> hand mainly changes the signal at C3.</p>
{fig("a4_eeg", 4, "Electrodes, and what one of them records.", "(A) The 32 standard positions shared by all our datasets; blue marks the motor electrodes C3, Cz and C4. (B) Illustrative EEG at C3. At rest there is a clear rhythm of about 10 Hz. When the person imagines squeezing their right hand (from 2 s), the rhythm becomes much weaker. The blue line traces the rhythm's strength.")}
{P("<p>All four datasets were reduced to the same <b>32 electrodes</b> shown in Figure 4A, matched by name, so that one model can read EEG from any of them. We tested using only the 7 motor electrodes instead, and prediction got worse (0.157 vs 0.228 in an early experiment), so all 32 are kept.</p>")}

<h3>2.3 Sampling rate: how often we take a measurement</h3>
<p>EEG is recorded as a series of snapshots. The number of snapshots per second is the <b>sampling rate</b>, measured in hertz (Hz). Our project resamples all EEG to <b>200 Hz</b>: 200 voltage values per second from each electrode.</p>
{A("film frames", "<p>A film is a series of still frames shown quickly, typically 24 per second. If you film a spinning wheel with too few frames per second, it can look as if it is spinning backwards. Sampling a signal works the same way: to capture a rhythm properly you need at least two samples per cycle. At 200 Hz we can faithfully capture rhythms up to 100 Hz, which covers everything EEG cares about.</p>")}
{E("how big is one trial?", "<p>Our project cuts recordings into <b>30-second windows</b>. One window from one electrode at 200 Hz is 30 × 200 = 6,000 numbers. With 32 electrodes, one window is a table of 32 × 6,000 = 192,000 numbers. That is the raw input the model sees for a single trial.</p>")}

<h3>2.4 Brain rhythms and frequency bands</h3>
<p>EEG is full of rhythms — oscillations that repeat at characteristic speeds. The speed of a rhythm is its <b>frequency</b>: the number of complete cycles per second. Researchers group frequencies into named bands, each loosely linked to brain states.</p>
{fig("a5_bands", 5, "The classic EEG frequency bands.", "Each line shows one second of a pure rhythm in that band. Faster rhythms have more cycles per second. The labels on the right are typical associations, not strict rules.")}
{E("closing your eyes", "<p>The most famous EEG effect: if a person relaxes and closes their eyes, a strong alpha rhythm (about 10 Hz) appears at the back of the head. Open the eyes and it shrinks. You could see this with a cheap EEG headset in a classroom. The visual brain 'idles' in sync when it has nothing to process, and breaks out of sync when it starts working.</p>")}

<h3>2.5 Motor imagery and event-related desynchronisation</h3>
<p>Most of our datasets use <b>motor imagery</b>: the participant is asked to <i>imagine</i> a movement, such as squeezing their right hand, without actually moving. Imagined movement activates much of the same motor cortex as real movement, so it produces a measurable brain response without muscle noise.</p>
<p>What does EEG show during motor imagery? Something that surprises most people: the rhythm over the motor cortex gets <b>weaker</b>. The relevant rhythms are the <b>mu</b> rhythm (8–13 Hz, the motor cortex's version of alpha) and the <b>beta</b> rhythm (13–30 Hz). Their drop in power is called <b>event-related desynchronisation (ERD)</b>, and you can see it in Figure 4B.</p>
{A("the humming crowd", "<p>A crowd waiting quietly might hum the same note together, producing a loud, steady tone. When an announcement starts and everyone turns to discuss it with their neighbours, the humming stops: each person is now doing something different. The room is busier than before, yet the shared tone is gone. Resting motor cortex 'hums' together; when it starts working, neurons pursue individual tasks and the shared rhythm fades. <b>Less rhythm means more work.</b></p>")}

<h3>2.6 Band power: turning wiggles into features</h3>
<p>Raw EEG is a fast, noisy wiggle. For many purposes, including ours, the useful information is not the exact wiggle but <i>how strong each rhythm is</i> over time. This is called <b>band power</b>: for each band, we filter the signal to keep just that band and measure its strength in short steps.</p>
{fig("a6_bandpower", 6, "From raw EEG to band power.", "(A) Illustrative raw EEG (grey) and the strength of its mu rhythm over time (blue). The dip is motor imagery. (B) Band power works like the bars of a music equaliser: one number per band, updated continuously.")}
{P("<p>The model's input is band power, not raw EEG. Each of the 32 electrodes is split into <b>4 bands</b>, giving 32 × 4 = <b>128 channels</b>, and each channel is updated <b>20 times per second</b>. A 30-second window therefore becomes 128 × 600 numbers. This is what Topic 14 in Part D will cover in detail.</p>")}

<h3>2.7 Artifacts: things that are not the brain</h3>
<p>Anything that produces voltage near the electrodes ends up in the EEG. These unwanted signals are called <b>artifacts</b>, and they are often far larger than brain activity:</p>
<ul>
<li><b>Eye blinks and eye movements</b>: the eye is electrically charged, and a blink produces a large slow wave at the front electrodes.</li>
<li><b>Muscle activity</b>: clenching the jaw or tensing the neck produces fast, spiky noise.</li>
<li><b>Mains electricity ("line noise")</b>: power cables hum at 50 Hz (or 60 Hz in some countries). Our pipeline removes it with a <b>notch filter</b> at 50 Hz, which cuts out that one frequency.</li>
<li><b>Inside an MRI scanner</b>: enormous artifacts from the scanner itself. Section 4.4 covers these.</li>
</ul>

<h3>2.8 Strengths and weaknesses</h3>
<p>EEG's great strength is <b>time</b>: it follows the brain millisecond by millisecond. Its great weakness is <b>space</b>: because the skull smears the electrical field, one electrode picks up activity from a wide area, and EEG cannot see deep brain structures well. It is also cheap, silent, portable and harmless.</p>
{Q(["Why can EEG detect a chant-like rhythm but not a single neuron?",
    "Which electrode would you watch if someone imagines moving their <i>left</i> hand?",
    "During motor imagery, does mu power over the motor cortex go up or down? Why is that not a contradiction?",
    "At 200 Hz, how many numbers does one electrode produce in a 30-second window?"])}

<h2>3. fNIRS: shining light through the skull</h2>
<h3>3.1 The basic idea</h3>
<p>Hold a torch against your fingers in a dark room and they glow red. Light passes through skin, flesh and even thin bone, especially red and near-infrared light. <b>Functional near-infrared spectroscopy (fNIRS)</b> uses this. It shines near-infrared light into the head at one point, collects whatever light comes back out a few centimetres away, and measures how much was absorbed on the way. Because haemoglobin is one of the main absorbers, changes in absorption tell us about changes in blood.</p>

<h3>3.2 Two colours to tell two kinds of haemoglobin apart</h3>
<p>Between about 650 and 950 nanometres (nm) there is an "optical window": skin, bone and water absorb relatively little, while haemoglobin absorbs a lot. Crucially, HbO and HbR absorb different colours differently (Figure 7). HbR absorbs more strongly below about 800 nm, and HbO absorbs more strongly above it.</p>
{fig("a7_spectra", 7, "HbO and HbR absorb near-infrared light differently.", "Approximate absorption curves. fNIRS devices usually use two wavelengths on either side of the crossing point; the TU Berlin device used 760 nm and 850 nm.")}
{A("two inks, two coloured glasses", "<p>Suppose a glass of water contains a mix of red ink and blue ink, and you want to know how much of each. Looking through one coloured filter mostly tells you about one ink; looking through a second filter mostly tells you about the other. With two readings you can solve for two unknowns. fNIRS does exactly this with two wavelengths of light and two kinds of haemoglobin.</p>")}

<h3>3.3 The banana-shaped path</h3>
<p>Light entering the head does not travel in a straight line. It scatters in every direction, like light in fog. The photons that happen to reach a detector a few centimetres away have, on average, followed a curved, banana-shaped route that dips into the tissue (Figure 8). With the source and detector about 3 cm apart, that banana reaches roughly 1–1.5 cm deep, just enough to touch the surface of the cortex. Each source–detector pair is called a <b>channel</b>.</p>
{fig("a8_banana", 8, "How fNIRS light travels.", "Schematic cross-section of the head. Light from the source scatters through scalp and skull; light reaching a detector about 3 cm away has sampled the cortex. A detector only 8 mm away (yellow) sees only the scalp, which is useful for removing scalp blood flow (Section 6.4).")}
{K("<p>Every fNIRS channel sees the scalp as well as the brain. The light must pass through skin, which has its own blood supply, on the way in and on the way out. This built-in contamination is the main weakness of fNIRS.</p>")}

<h3>3.4 From light to haemoglobin: the modified Beer–Lambert law</h3>
<p>The instrument measures light intensity; we want haemoglobin concentrations. The bridge is the <b>modified Beer–Lambert law</b>, which says that the change in how much light is absorbed is proportional to the change in concentration of the absorber:</p>
<div class="formula">change in absorption = (how strongly it absorbs) × (change in concentration) × (path length)</div>
<p>Do this at two wavelengths and you have two equations with two unknowns, the change in HbO and the change in HbR. The "path length" term is tricky, because scattering makes light travel much further than the straight distance between source and detector. A correction factor called the <b>differential pathlength factor (DPF)</b>, typically around 6, accounts for this. It depends on the tissue, so anything that changes the head's optical properties, such as a brain haemorrhage, changes it too.</p>
{E("reading the light during activity", "<p>A region becomes active. HbR falls and HbO rises. At 760 nm, where HbR is the stronger absorber, less HbR means <i>more</i> light gets through. At 850 nm, where HbO dominates, more HbO means <i>less</i> light gets through. So the detector sees 760 nm brighten and 850 nm dim at the same time. Solving the two equations turns that pattern back into 'HbO up, HbR down'.</p>")}
{P("<p>The TU Berlin dataset gave us the <b>raw light intensities at 760 and 850 nm</b>, and we converted them ourselves with a controlled Beer–Lambert step. HEFMI came already converted. Units are a trap here: concentrations can be reported in mol/L, mmol/L or µmol/L, factors of 1,000 apart. Our audit found HEFMI and TU Berlin values differing by about <b>2,600 times</b>, which is why the pipeline harmonises units before comparing datasets. The ICH clinical result (Part F) needs care for the same reason as the DPF: haemorrhage can change how light travels.</p>")}

<h3>3.5 Strengths and weaknesses</h3>
<p>fNIRS measures <b>both</b> HbO and HbR, is portable, quiet, tolerant of some movement, and much cheaper than fMRI. It samples quickly (our data: 10 times per second), though the blood signal itself is slow. Its weaknesses: it sees only the outer centimetre or so of cortex, its spatial detail is coarse (a few centimetres), and every channel mixes brain signal with scalp blood flow.</p>
{Q(["Why does fNIRS need two wavelengths of light rather than one?",
    "If HbR falls, does more or less 760 nm light reach the detector?",
    "Why would a detector placed only 8 mm from the source be useful?"])}

<h2>4. fMRI: watching blood with a magnet</h2>
<h3>4.1 The BOLD signal</h3>
<p><b>Functional magnetic resonance imaging (fMRI)</b> uses a very strong magnet and radio waves to build images of the brain, typically one full 3D image every 2 seconds. It does not measure blood directly. It relies on a lucky physical fact: <b>HbR is slightly magnetic</b> (paramagnetic), while HbO is not. HbR disturbs the magnetic field around blood vessels and makes the MRI signal from that tissue a little weaker.</p>
<p>Now recall the overshoot from Section 1.4. When a region becomes active, fresh blood washes HbR out. Less HbR means less magnetic disturbance, so the MRI signal gets slightly <i>stronger</i>. This signal is called <b>BOLD</b>, for Blood-Oxygen-Level-Dependent. The change is small, usually around 0.5–2%.</p>
{A("tiny magnets that blur the picture", "<p>Think of HbR molecules as tiny magnets scattered through the tissue, each slightly smudging the scanner's view. Activity floods the area with fresh blood and sweeps many of those magnets away, so the view of that patch becomes a little clearer and brighter. BOLD is the scanner noticing that patch getting brighter.</p>")}
{fig("a9_bold", 9, "fNIRS and fMRI see the same event differently.", "Schematic responses to one brief neural event. fNIRS measures HbO and HbR separately; fMRI's BOLD signal behaves roughly like an upside-down HbR.")}

<h3>4.2 Voxels, TR and regions of interest</h3>
<p>An fMRI image is made of small cubes called <b>voxels</b>, typically 2–3 mm on each side, so one scan contains tens of thousands of measurements covering the whole brain, including deep structures fNIRS cannot reach. The time between scans is the <b>TR</b> (repetition time), commonly 2 seconds, which is a sampling rate of 0.5 Hz. To make the data manageable, voxels are often averaged within anatomical areas called <b>regions of interest (ROIs)</b>, defined by a standard brain atlas.</p>
{P("<p>Maisha's pipeline summarises each fMRI scan as <b>48 ROIs</b> from the Harvard–Oxford atlas, sampled every 2 s. A 30-second window therefore holds only <b>15 time points</b> per ROI, compared with 300 for fNIRS at 10 Hz.</p>")}

<h3>4.3 EEG inside the scanner</h3>
<p>Two of our datasets (ds002336 and ds002739) recorded EEG and fMRI <i>at the same time</i>, which is what makes them useful for learning the link between the two. But the scanner is hostile to EEG. Every time it takes an image, it switches strong magnetic gradients on and off, and these induce huge voltages in the EEG wires — the <b>gradient artifact</b>. In ds002336 the raw EEG reached about 8,200 µV, while real brain signals are around 10–100 µV. There is also the <b>ballistocardiogram</b>: with each heartbeat the head and electrodes move slightly in the magnetic field, adding a pulse-shaped artifact.</p>
{A("whispering next to a jackhammer", "<p>Imagine recording someone whispering next to a jackhammer. Hopeless? Not if the jackhammer strikes in a perfectly regular rhythm. Record many strikes, average them to get the exact sound of one strike, then subtract that template every time a strike happens. What remains is the whisper. This is <b>average artifact subtraction (AAS)</b>, and it works because the scanner's artifact repeats exactly with every image.</p>")}
{P("<p>We implemented our own AAS correction for ds002336, bringing the EEG from about 8,200 µV of artifact down to a clean ~16 µV. ds002739 came already corrected. One ds002336 recording (subject xp102, motor localiser) turned out to be completely flat and unusable.</p>")}

<h3>4.4 The three machines side by side</h3>
<div class="tcap"><b>Table 1.</b> EEG, fNIRS and fMRI compared.</div>
<table><thead><tr><th>Property</th><th>EEG</th><th>fNIRS</th><th>fMRI (BOLD)</th></tr></thead><tbody>
<tr><td>What it measures</td><td>electrical activity of neurons</td><td>HbO and HbR in blood</td><td>HbR's magnetic effect</td></tr>
<tr><td>Speed of the signal</td><td>milliseconds</td><td>seconds (blood response)</td><td>seconds (blood response)</td></tr>
<tr><td>Sampling in our data</td><td>200 Hz</td><td>10 Hz</td><td>0.5 Hz (one image every 2 s)</td></tr>
<tr><td>Spatial detail</td><td>poor (several cm)</td><td>moderate (2–3 cm)</td><td>excellent (2–3 mm)</td></tr>
<tr><td>Depth</td><td>mostly surface</td><td>outer ~1.5 cm only</td><td>whole brain</td></tr>
<tr><td>Portability and cost</td><td>wearable, cheap</td><td>wearable, moderate</td><td>hospital scanner, very expensive</td></tr>
<tr><td>Main contamination</td><td>eye, muscle, line noise</td><td>scalp blood, body physiology</td><td>head motion, physiology</td></tr>
</tbody></table>
{Q(["Why does BOLD go <i>up</i> when a region becomes active, even though fMRI is sensitive to HbR?",
    "Why is EEG recorded inside an MRI scanner so badly contaminated, and why can AAS fix it?",
    "How many fMRI time points does a 30-second window contain at a TR of 2 s?"])}

<h2>5. The haemodynamic response</h2>
<h3>5.1 Its shape</h3>
<p>If a brief burst of neural activity happens at time zero, the blood signal follows a characteristic curve called the <b>haemodynamic response function (HRF)</b> (Figure 10). It has four phases: a delay of 1–2 seconds in which almost nothing happens; a rise to a peak about 5–6 seconds after the event; a fall back towards baseline; and a shallow undershoot below baseline that recovers over 20–30 seconds. One brief moment of activity therefore leaves a trace lasting around half a minute.</p>
{fig("a10_hrf", 10, "The haemodynamic response function.", "The standard shape of the blood response to a single brief neural event, as used in fMRI analysis. HbO and BOLD follow this shape; HbR follows it upside down and slightly later.")}
{A("striking a heavy bell", "<p>Hit a large church bell with a hammer. The hammer strike lasts a split second, but the bell takes a moment to swell to full volume and then rings on long after. The strike is the neural event; the ringing is the blood response. If you only listened to the ringing, you would still know roughly when the bell was struck — but your guess would be blurred and late.</p>")}

<h3>5.2 The neurovascular delay</h3>
<p>The time from neural event to blood response is the <b>neurovascular delay</b>. It differs between people and brain regions, and it changes with age, health and medication. Going back to the storm analogy, it is like the gap between lightning and thunder: always present, but longer for some observers than others.</p>
{P("<p>We measured this delay directly from the data, with no model, by finding the time shift at which EEG power and the HbO signal line up best. The average was <b>6.4 seconds</b>, people differed by about <b>1.1 seconds</b>, and each person's delay was fairly consistent between sessions (a test–retest agreement of 0.53, Part F). That consistency suggests the delay is a genuine personal trait — and also that a model trained on some people will be slightly mistimed for others.</p>")}

<h3>5.3 Responses add up</h3>
<p>Real brain activity is not one isolated event. When events come close together, their blood responses overlap, and to a good approximation they simply add (Figure 11). In mathematics this operation, smearing each moment of activity with the HRF and adding all the copies, is called <b>convolution</b>. You will see this word in papers; it just means "blur with the HRF and add".</p>
{fig("a11_convolution", 11, "Overlapping responses add together.", "(A) Three brief events 3 seconds apart, each producing its own HRF. (B) What the instrument records is their sum: one larger, longer response.")}
<div class="formula">blood signal(t) ≈ HRF ⊛ neural activity(t) + noise</div>
<p>This formula is the hidden assumption behind the entire project. If EEG tells us about neural activity, and we know the HRF, then the blood signal should be predictable. The project tests how true that is in practice — and the noise term turns out to be the hard part.</p>

<h3>5.4 A single trial, start to finish</h3>
<p>Figure 12 puts the pieces together for one motor-imagery trial: the cue appears, the person imagines squeezing their hand for 10 seconds, then rests.</p>
{fig("a15_trial", 12, "One trial seen by EEG and fNIRS.", "Schematic. The EEG mu rhythm drops within half a second of the cue and recovers as soon as imagery stops. HbO climbs slowly, peaks after imagery has already ended, and takes the rest of the window to settle. This is why our project uses 30-second windows: shorter windows would cut the blood response off.")}
{Q(["Roughly how long after a brief event does the blood response peak?",
    "Why does one second of neural activity affect the blood signal for about 30 seconds?",
    "In Figure 12, which signal would you use to know exactly <i>when</i> imagery started, and why?"])}

<h2>6. Systemic noise: when the blood isn't about the brain</h2>
<h3>6.1 The body is noisy</h3>
<p>Blood flow in the head changes for many reasons that have nothing to do with neurons. These are called <b>systemic</b> signals, because they come from the whole circulatory system rather than local brain activity:</p>
<ul>
<li><b>Heartbeat</b> — about 1 Hz (60 beats per minute).</li>
<li><b>Breathing</b> — about 0.2–0.3 Hz (12–18 breaths per minute).</li>
<li><b>Mayer waves</b> — slow oscillations in blood pressure around 0.1 Hz (one every 10 seconds).</li>
<li><b>Very slow drifts</b> — posture, temperature, alertness.</li>
<li><b>Scalp blood flow</b> — skin blood vessels change too, and every fNIRS channel sees them (Section 3.3).</li>
</ul>

<h3>6.2 Why filtering cannot remove everything</h3>
<p>The natural first defence is filtering: keep only the frequencies where the brain response lives and throw away the rest. Our pipeline keeps fNIRS between 0.01 and 0.2 Hz, which removes heartbeat and most breathing. But look at Figure 13: Mayer waves sit right inside the band we keep, close to the frequency of the task itself. No filter can separate two signals that share the same frequencies.</p>
{fig("a12_noise", 13, "Where the noise lives.", "Schematic frequency content of an fNIRS signal. The shaded band is what our filter keeps. Heartbeat and breathing are removed, but Mayer waves overlap with the task response and survive.")}
<p>There is a worse case still. Doing a task can itself change the body: concentrating raises heart rate and blood pressure a little, <i>in time with the task</i>. This <b>task-evoked systemic change</b> starts and stops exactly when the brain response does, so it looks like a brain response in every way except one.</p>

<h3>6.3 Telling neural from systemic: the fingerprint</h3>
<p>That one way is the fingerprint from Section 1.4. A genuine neural response comes from the oxygen overshoot, so <b>HbO rises and HbR falls</b>: they are negatively correlated. A systemic change, such as a rise in blood pressure, pushes more blood of every kind into the tissue, so <b>HbO and HbR rise and fall together</b>: they are positively correlated. A neural response is also local, while a systemic one appears in every channel at once.</p>
{fig("a13_signatures", 14, "Neural and systemic signals leave different fingerprints.", "(A) A neural response: HbO and HbR move in opposite directions. (B) A systemic signal: they move together. (C) Real values from our project: TU Berlin has the neural fingerprint; HEFMI has a strongly systemic one.")}
{P("<p>This test is how the project discovered its biggest problem. In <b>TU Berlin</b> the HbO–HbR correlation is <b>−0.44</b>: the neural fingerprint. In <b>HEFMI</b>, which was originally our main training set, it is <b>+0.92</b>: almost perfectly systemic. HEFMI's target was also very unreliable from one half of the trials to the other (split-half reliability 0.08, a concept covered in Part E). A model cannot learn to predict brain responses from a signal that is mostly heartbeat and blood pressure, which explains much of the early 'prediction is near zero' result. Part F tells that story in full.</p>")}

<h3>6.4 Global signal regression, and how it backfired</h3>
<p>One popular fix for systemic noise is <b>global signal regression (GSR)</b>. Systemic signals appear in every channel, so you average all channels to estimate the global signal, then subtract it from each channel. Whatever is common to all channels disappears.</p>
{A("the party microphones", "<p>You place eight microphones around a noisy party to record one person speaking near microphone 3. The party chatter reaches every microphone, so you average all eight and subtract the average from microphone 3 — the chatter cancels and the speaker remains. Now suppose the speaker is using a loudspeaker that reaches every microphone. The speaker is now part of the 'common' signal, and subtracting the average removes the speaker along with the chatter.</p>")}
{fig("a14_gsr", 15, "Global signal regression works for local responses and fails for broad ones.", "Simulation with eight channels sharing a systemic wave. (A) When only one channel responds, GSR removes the wave and recovers the response. (B) When most channels respond, the response becomes part of the global average, and GSR removes it too.")}
{P("<p>A motor-imagery response can be quite broad, covering many fNIRS channels. When the audit measured it, GSR was removing about <b>83% of the evoked HbO response</b> on TU Berlin — exactly the failure in Figure 15B. Both later pipelines therefore switched GSR <b>off</b>. Maisha's pipeline confirmed this independently: the part GSR removed was not uniform across channels, so it contained real, spatially structured signal.</p>")}
<p>The better fix is <b>short-separation regression</b>. The short channel in Figure 8 sees only the scalp, so it measures the contamination directly; subtracting it removes scalp and systemic signal without touching the brain response. It requires hardware with short channels and access to raw data, which is one reason the choice of dataset mattered so much.</p>

<h3>6.5 The lesson for the whole project</h3>
<p>Before asking whether a model can predict a signal, ask whether the signal is the thing you think it is. The fingerprint test and the reliability check take minutes and require no machine learning, yet they explained months of disappointing results. This is the theme of the whole course: <b>data quality is checked first, and models second</b>.</p>
{Q(["Why can't a frequency filter remove Mayer waves?",
    "A dataset shows HbO and HbR rising together during every trial. What would you suspect?",
    "When does GSR help and when does it hurt?",
    "Why is a short-separation channel a better fix than GSR?"])}

<h2>Summary</h2>
<ul>
<li>Active neurons need energy; the brain responds by over-supplying fresh blood. As a result <b>HbO rises and HbR falls</b> — the fingerprint of real brain activity.</li>
<li><b>EEG</b> measures the electrical activity of large synchronised groups of neurons: very fast, spatially blurry. During motor imagery the <b>mu and beta rhythms weaken</b> over the motor cortex (ERD). The model reads EEG as <b>band power</b>.</li>
<li><b>fNIRS</b> shines two wavelengths of near-infrared light through the scalp and uses the Beer–Lambert law to estimate HbO and HbR in the outer cortex. Every channel also sees scalp blood.</li>
<li><b>fMRI</b> detects HbR's magnetic effect across the whole brain; <b>BOLD</b> behaves roughly like upside-down HbR. EEG recorded inside the scanner needs heavy artifact correction (AAS).</li>
<li>The blood response follows the <b>HRF</b>: it peaks about 5–6 s after an event and lasts about 30 s; responses to nearby events add up (<b>convolution</b>). In our data the measured delay is about <b>6.4 s</b> and differs between people.</li>
<li><b>Systemic noise</b> from heart, breathing and blood pressure can mimic a brain response. The HbO/HbR sign separates the two: TU Berlin passes (−0.44), HEFMI fails (+0.92). <b>GSR</b> removed most of the real response and was switched off.</li>
</ul>

<h3>Coming next: Lecture B</h3>
<p>Now that you know what the signals are, Lecture B states the research question precisely: what the system is supposed to output, why it must work from EEG alone, and why the absence of a dataset with the same people in fNIRS and fMRI shapes the whole design.</p>

<h2>Glossary</h2>
<table><thead><tr><th style="width:24%">Term</th><th>Meaning</th></tr></thead><tbody>
<tr><td>Neurovascular coupling</td><td>The link by which neural activity causes a local increase in blood flow.</td></tr>
<tr><td>HbO / HbR</td><td>Oxygenated / deoxygenated haemoglobin. Activity makes HbO rise and HbR fall.</td></tr>
<tr><td>EEG</td><td>Electroencephalography: scalp voltages from synchronised neural activity.</td></tr>
<tr><td>10-20 system</td><td>Standard electrode positions; letters give the region, odd = left, even = right, z = midline.</td></tr>
<tr><td>Sampling rate</td><td>Measurements per second, in hertz (Hz).</td></tr>
<tr><td>Frequency band</td><td>A range of rhythm speeds (delta, theta, alpha/mu, beta, gamma).</td></tr>
<tr><td>Motor imagery</td><td>Imagining a movement without performing it.</td></tr>
<tr><td>ERD</td><td>Event-related desynchronisation: a drop in rhythm power when a region becomes active.</td></tr>
<tr><td>Band power</td><td>The strength of one frequency band over time; the model's EEG input.</td></tr>
<tr><td>fNIRS</td><td>Functional near-infrared spectroscopy: measures HbO and HbR with light.</td></tr>
<tr><td>Channel (fNIRS)</td><td>One source–detector pair.</td></tr>
<tr><td>Beer–Lambert law</td><td>Converts changes in light absorption into changes in concentration.</td></tr>
<tr><td>DPF</td><td>Differential pathlength factor: corrects for light travelling further than the straight distance.</td></tr>
<tr><td>fMRI / BOLD</td><td>Magnetic resonance imaging of the blood-oxygen-dependent signal.</td></tr>
<tr><td>Voxel / ROI / TR</td><td>3D pixel / region of interest / time between fMRI images.</td></tr>
<tr><td>Gradient artifact / AAS</td><td>Scanner-induced EEG noise / removing it by subtracting an averaged template.</td></tr>
<tr><td>HRF</td><td>Haemodynamic response function: the blood response to a brief event.</td></tr>
<tr><td>Neurovascular delay</td><td>Time from neural event to blood response (≈ 6.4 s in our data).</td></tr>
<tr><td>Convolution</td><td>Blurring a signal with the HRF and adding overlapping copies.</td></tr>
<tr><td>Systemic signal</td><td>Blood changes from heart, breathing, blood pressure or scalp, not local brain activity.</td></tr>
<tr><td>Mayer waves</td><td>Blood-pressure oscillations near 0.1 Hz.</td></tr>
<tr><td>GSR</td><td>Global signal regression: subtracting the all-channel average.</td></tr>
<tr><td>Short-separation channel</td><td>An fNIRS channel ~8 mm long that sees only the scalp.</td></tr>
</tbody></table>

<h2>Answers to "Pause and think"</h2>
<h3>Section 1</h3>
<ol class="answers">
<li>Because blood flow increases much more than oxygen use. The extra fresh blood brings more oxygen than the tissue consumes, so the share of oxygenated blood in the area rises.</li>
<li>HbO goes up and HbR goes down.</li>
<li>Steps 3–5, the widening of blood vessels. They take seconds while firing takes milliseconds, so blood signals always lag behind electrical ones.</li>
</ol>
<h3>Section 2</h3>
<ol class="answers">
<li>EEG only detects activity that millions of neurons produce together; a single neuron's field is far too weak and is cancelled out by everything else, like one voice in a stadium.</li>
<li>C4, over the right motor cortex, because each side of the brain controls the opposite side of the body.</li>
<li>It goes down (ERD). It is not a contradiction because the rhythm reflects neurons idling <i>in sync</i>; when they start working on individual tasks the shared rhythm fades even though activity has increased.</li>
<li>30 × 200 = 6,000.</li>
</ol>
<h3>Section 3</h3>
<ol class="answers">
<li>There are two unknowns (HbO and HbR), and each wavelength gives one equation. Two wavelengths, absorbed differently by the two forms, give two equations that can be solved.</li>
<li>More. HbR is the stronger absorber at 760 nm, so less HbR means less absorption and more light reaching the detector.</li>
<li>It sees only the scalp, so it measures the scalp and systemic contamination directly, which can then be subtracted from the normal channels.</li>
</ol>
<h3>Section 4</h3>
<ol class="answers">
<li>Activity washes HbR out of the region. HbR weakens the MRI signal, so having less of it makes the signal stronger.</li>
<li>The scanner's switching magnetic gradients induce huge voltages in the EEG wires. AAS works because the artifact repeats almost identically with every image, so an averaged template can be subtracted.</li>
<li>30 ÷ 2 = 15 time points.</li>
</ol>
<h3>Section 5</h3>
<ol class="answers">
<li>About 5–6 seconds (6.4 s on average in our data).</li>
<li>Because the HRF itself lasts about 30 seconds: the blood takes time to rise, fall and recover from its undershoot.</li>
<li>EEG, because it responds within half a second of the cue; the blood signal is too slow and blurred to time the start precisely.</li>
</ol>
<h3>Section 6</h3>
<ol class="answers">
<li>Mayer waves occur near 0.1 Hz, inside the same frequency band as the task response, so any filter that keeps the response also keeps them.</li>
<li>A systemic signal (for example, blood pressure rising with the task) rather than a neural response.</li>
<li>It helps when the brain response is local and the noise is global. It hurts when the brain response is broad, because the response then becomes part of the global average and is subtracted away.</li>
<li>The short channel measures the actual scalp contamination at that location, so it removes noise without assuming the brain response is absent from the average.</li>
</ol>

</body></html>"""

(BUILD / "lecture_a.html").write_text(html, encoding="utf-8")
tmp = BUILD / "lecture_a.pdf"
tmp.unlink(missing_ok=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={tmp}", (BUILD / "lecture_a.html").as_uri()], check=True, capture_output=True)
if not tmp.exists():
    raise SystemExit("Chrome did not produce a PDF")
try:
    PDF.write_bytes(tmp.read_bytes())
    print("wrote", PDF)
except PermissionError:
    print(f"{PDF.name} is open in another program; new version left at {tmp}")
