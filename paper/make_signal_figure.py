"""Real-data figure for the BSPC manuscript: what the signals look like at each step, for one hBCI person.

Person 20 is used because their two-run calibration gain is the median of the 29 hBCI people
(results/calibration_results.json), so the example is typical rather than the best case.
Panels: (a) z-scored EEG of one cue-locked window, (b) one channel split into the 0.5-45 Hz band a
conventional filter keeps and the infra-slow band below 0.5 Hz it removes, with the 1 s bin means used as
features, (c) EEG power spectrum of this person, (d) measured global HbO and HbR of the same window,
(e) measured global HbO against the EEG-only personal prediction (Eq. 4) on a held-out run, calibrated on
two other runs of the same task, with the calibration mean response for comparison.

Usage: python make_signal_figure.py [output_stem]   (default: bspc/Figure_signals)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]   # repository root

CACHE = Path(str(REPO / "pipeline/neurovascnet_merge_v1_cache/train_fnirs_a2133f5d6a71ec34"))
PERSON, FS, ALPHAS = 20, 200.0, np.logspace(0, 6, 13)
SHOW_CH = ["FCC3h", "Cz", "FCC4h", "Pz"]
FOCUS_CH = "Cz"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
HBO, HBR, PRED, BASE = "#c8463d", "#2e6aa6", "#2a78d6", "#8a8984"
TASK = "#efeeea"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
                     "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def bins(x, n):
    return x.reshape(*x.shape[:-1], n, -1).mean(-1)


def slow(x):
    return sps.sosfiltfilt(sps.butter(4, 0.5, btype="low", fs=FS, output="sos"), x, axis=-1)


def zs(F, ref):
    return (F - ref.mean(0)) / (ref.std(0) + 1e-8)


def trial_r(p, t):
    p, t = p - p.mean(), t - t.mean()
    return float(p @ t / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-12))


# ------------------------------------------------------------------ data
sid = np.load(CACHE / "subject_ids.npy")
rec = np.load(CACHE / "rec_ids.npy")
tid = np.load(CACHE / "task_ids.npy")
import json
ch = json.loads((CACHE / "meta.json").read_text())["info"]["ch_names"]
m = np.flatnonzero(sid == PERSON)
eeg = np.asarray(np.load(CACHE / "eeg.npy", mmap_mode="r")[m], np.float64)       # (n, 30, 6000), z-scored
hemo = np.asarray(np.load(CACHE / "hemo.npy", mmap_mode="r")[m], np.float64)     # (n, 72, 300), 10 Hz
rec_p, tid_p = rec[m], tid[m]
nh = hemo.shape[1] // 2                                                          # first half HbO

# calibration on MI runs 0 and 2, test on MI run 4 (same task, later run)
cal = np.isin(rec_p, [f"s{PERSON:02d}/MI_run0", f"s{PERSON:02d}/MI_run2"])
test = rec_p == f"s{PERSON:02d}/MI_run4"
F = bins(slow(eeg), 30).reshape(len(eeg), -1)
Y = bins(hemo, 30).reshape(len(hemo), -1)
base = Y[cal].mean(0)
pers = RidgeCV(alphas=ALPHAS).fit(zs(F[cal], F[cal]), Y[cal] - base)
P = base + pers.predict(zs(F[test], F[cal]))
glob = lambda Z: Z.reshape(len(Z), 2 * nh, 30)[:, :nh].mean(1)                   # global HbO, 30 bins
Pg, Tg, Bg = glob(P), glob(Y[test]), glob(np.broadcast_to(base, Y[test].shape))
r_pers = np.array([trial_r(a, b) for a, b in zip(Pg, Tg)])
r_base = np.array([trial_r(a, b) for a, b in zip(Bg, Tg)])
print(f"test trials {test.sum()}, mean trial r personal {r_pers.mean():.3f}, mean response {r_base.mean():.3f}")

# example window for (a)-(d): the test trial whose personal r is the median
te_idx = np.flatnonzero(test)
k_med = int(np.argsort(r_pers)[len(r_pers) // 2])
w = te_idx[k_med]
t_eeg = np.arange(eeg.shape[-1]) / FS - 2.0
t_hem = np.arange(hemo.shape[-1]) / 10.0 - 2.0
t_bin = np.arange(30) + 0.5 - 2.0

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(7.2, 6.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.62, wspace=0.26,
                      left=0.075, right=0.985, top=0.955, bottom=0.075)


def task_shade(ax):
    ax.axvspan(0, 10, color=TASK, zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)


# (a) raw EEG, four channels
ax = fig.add_subplot(gs[0, 0])
task_shade(ax)
for i, c in enumerate(SHOW_CH):
    x = eeg[w, ch.index(c)]
    ax.plot(t_eeg, x / 6 - i, color=INK, lw=0.35)
    ax.text(-2.4, -i, c, ha="right", va="center", fontsize=7, color=INK2)
ax.set_yticks([]); ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.set_xlim(-2, 28); ax.set_xlabel("time from cue (s)")
ax.set_title("(a)  Scalp EEG, one 30 s window", pad=4)

# (b) one channel split into the two bands
ax = fig.add_subplot(gs[0, 1])
task_shade(ax)
x = eeg[w, ch.index(FOCUS_CH)]
hi = sps.sosfiltfilt(sps.butter(4, [0.5, 45], btype="band", fs=FS, output="sos"), x)
lo = slow(x)
off = 3.4
ax.plot(t_eeg, hi + off, color=INK2, lw=0.3)
ax.plot(t_eeg, lo, color=PRED, lw=1.1)
ax.step(np.r_[t_bin - 0.5, t_bin[-1] + 0.5], np.r_[bins(lo, 30), bins(lo, 30)[-1]], where="post",
        color=INK, lw=0.6)
ax.text(10.4, off + hi.max() + 0.15, "0.5–45 Hz: kept by a standard filter", ha="left", va="bottom",
        fontsize=6.6, color=INK2)
ax.text(10.4, lo.min() - 0.3, "below 0.5 Hz: removed by it\nsteps: 1 s bin means (features)",
        ha="left", va="top", fontsize=6.6, color=PRED, linespacing=1.1)
ax.set_xlim(-2, 28); ax.set_xlabel("time from cue (s)"); ax.set_ylabel(f"{FOCUS_CH} (z)")
ax.set_yticks([])
ax.set_title(f"(b)  One channel ({FOCUS_CH}) split into two bands", pad=4)
ax.set_ylim(lo.min() - 2.0, off + hi.max() + 0.9)

# (c) spectrum of this person
ax = fig.add_subplot(gs[1, 0])
f, pxx = sps.welch(eeg, fs=FS, nperseg=6000, axis=-1)
pm = pxx.mean((0, 1))
keep = (f > 0) & (f <= 45)
ax.axvspan(f[keep][0], 0.5, color="#dbe8f5", lw=0, zorder=0)
ax.loglog(f[keep], pm[keep], color=INK, lw=0.9)
ax.axvline(0.5, color=PRED, lw=0.8, ls=(0, (3, 2)))
ax.text(0.13, pm[keep].min() * 3, "removed by a\n0.5 Hz high-pass", ha="center", va="bottom",
        fontsize=6.6, color=PRED)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (z$^2$/Hz)")
ax.set_title(f"(c)  EEG spectrum, person {PERSON}, all channels", pad=4)
ax.grid(which="minor", visible=False)

# (d) haemodynamics of the same window
ax = fig.add_subplot(gs[1, 1])
task_shade(ax)
ax.plot(t_hem, hemo[w, :nh].mean(0), color=HBO, lw=1.1, label="HbO")
ax.plot(t_hem, hemo[w, nh:].mean(0), color=HBR, lw=1.1, label="HbR")
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xlim(-2, 28); ax.set_xlabel("time from cue (s)"); ax.set_ylabel("channel mean (a.u.)")
ax.set_title("(d)  fNIRS, same window, mean of 36 channels", pad=4)
ax.legend(loc="upper right", ncol=2, handlelength=1.4)

# (e) measured vs predicted over consecutive held-out trials
ax = fig.add_subplot(gs[2, :])
n_show = 6
start = max(0, min(k_med - 2, len(te_idx) - n_show))
seg = range(start, start + n_show)
for j, k in enumerate(seg):
    xx = j * 31 + np.arange(30)
    ax.axvspan(j * 31 + 2, j * 31 + 12, color=TASK, lw=0, zorder=0)
    ax.plot(xx, Tg[k], color=INK, lw=1.2, label="measured HbO" if j == 0 else None)
    ax.plot(xx, Pg[k], color=PRED, lw=1.2, label="EEG-only prediction" if j == 0 else None)
    ax.plot(xx, Bg[k], color=BASE, lw=0.9, ls=(0, (3, 2)),
            label="calibration mean response" if j == 0 else None)
    ax.annotate(f"r = {r_pers[k]:.2f}", xy=(j * 31 + 15, 1.0), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=6.8, color=PRED)
ax.set_xticks([j * 31 + 15 for j in range(n_show)])
ax.set_xticklabels([f"trial {k + 1}" for k in seg])
ax.set_xlim(-1, n_show * 31)
ax.set_ylabel("global HbO (a.u.)")
ax.set_title("(e)  Later run, not used for fitting: measured HbO and EEG-only prediction after two "
             "calibration runs", pad=14)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, handlelength=1.8)
ax.grid(axis="x", visible=False)

stem = sys.argv[1] if len(sys.argv) > 1 else "bspc/Figure_signals"
fig.savefig(stem + ".pdf", facecolor=SURF)
fig.savefig(stem + ".png", dpi=250, facecolor=SURF)
print("wrote", stem)
