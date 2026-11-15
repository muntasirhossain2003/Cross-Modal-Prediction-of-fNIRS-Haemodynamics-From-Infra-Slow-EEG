"""Sanity checks for P2.3 (hBCI only, first 10 people for speed):
1. injected shift: delaying the NIRS by +5 s must move the measured lag by +5 s
2. flipped offset: aligning with the wrong offset sign must weaken the coupling
3. marker-locked averages: when do task-evoked slow EEG (GFP) and HbO peak?
"""
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps

sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
src = (HERE / "continuous_lag.py").read_text()
src = src[:src.index("rng = np.random.default_rng(SEED)\nres = {}")]       # definitions only
ns = {"__name__": "lag", "__file__": str(HERE / "continuous_lag.py")}
exec(src, ns)

people = []
for pid, pairs in ns["hbci_people"]():
    people.append((pid, pairs))
    if len(people) == 10:
        break
xc, M = ns["xcorr"], ns["MAXLAG"]


def lag_of(pairs, shift=0):
    curve = np.mean([xc(a, np.roll(b, shift)[:len(b)] if shift == 0 else
                        np.concatenate([np.full(shift, b[0]), b[:-shift]])) for a, b in pairs], 0)
    return int(np.argmax(curve)) - M, float(curve.max())


print("1. injected +5 s NIRS delay")
for pid, pairs in people:
    l0, _ = lag_of(pairs)
    l5, _ = lag_of(pairs, 5)
    print(f"   {pid}: lag {l0:+d} -> {l5:+d} (expected {l0 + 5:+d})")

print("2. offset sign flipped (wrong alignment)")
_runs, beer_lambert, split_wl, run_pair = ns["_runs"], ns["beer_lambert"], ns["split_wl"], ns["run_pair"]
HB = ns["HBCI"]
for s in range(1, 6):
    eeg_dir, nirs_dir = HB / "EEG data", HB / "NRI data" / "NIRS_01-29"
    ed = next(g / f"subject {s:02d}" / "with occular artifact" for g in sorted(eeg_dir.glob("EEG_*"))
              if (g / f"subject {s:02d}" / "with occular artifact").exists())
    er_all, em_all = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
    nr_all, nm_all = _runs(nirs_dir / f"subject {s:02d}" / "cnt.mat", "cnt"), _runs(nirs_dir / f"subject {s:02d}" / "mrk.mat", "mrk")
    out = {}
    for sign in (1, -1):
        pairs = []
        for r in range(min(er_all.size, nr_all.size)):
            er, nr, em, nm = er_all[r], nr_all[r], em_all[r], nm_all[r]
            clab = [str(c) for c in np.atleast_1d(er.clab)]
            ex = np.asarray(er.x, float)[:, [i for i, c in enumerate(clab) if c not in ("VEOG", "HEOG")]].T
            lo, hi = split_wl(np.asarray(nr.x, float), nr.clab)
            hbo, _ = beer_lambert(lo, hi)
            off = float(np.atleast_1d(nm.time)[0] - np.atleast_1d(em.time)[0]) / 1000.0
            pr = run_pair(ex, float(np.atleast_1d(er.fs).ravel()[0]), np.asarray(hbo, float),
                          float(np.atleast_1d(nr.fs).ravel()[0]), sign * off)
            if pr is not None:
                pairs.append(pr)
        out[sign] = lag_of(pairs) if pairs else (None, float("nan"))
    off_print = float(np.atleast_1d(nm_all[0].time)[0] - np.atleast_1d(em_all[0].time)[0]) / 1000.0
    print(f"   s{s:02d} (offset {off_print:+.1f} s): correct lag {out[1][0]} peak |r| {out[1][1]:.3f} | "
          f"flipped lag {out[-1][0]} peak |r| {out[-1][1]:.3f}")

print("3. marker-locked averages (hBCI cache, all trials)")
C = Path(r"G:\research_bio_medical\merge\neurovascnet_merge_v1_cache\train_fnirs_a2133f5d6a71ec34")
eeg = np.load(C / "eeg.npy", mmap_mode="r")
hemo = np.load(C / "hemo.npy")
tid = np.load(C / "task_ids.npy")
sos = sps.butter(4, 0.1, btype="low", fs=200.0, output="sos")
t_e = np.arange(30) - 2 + 0.5
for task, name in ((0, "MI"), (1, "MA")):
    idx = np.flatnonzero(tid == task)
    acc = np.zeros((eeg.shape[1], 30))
    for i in range(0, len(idx), 200):
        x = sps.sosfiltfilt(sos, np.asarray(eeg[idx[i:i + 200]], np.float64), axis=-1)
        acc += x.reshape(len(x), x.shape[1], 30, 200).mean(-1).sum(0)
    erp = acc / len(idx)
    erp -= erp[:, :2].mean(1, keepdims=True)
    gfp = np.sqrt((erp ** 2).mean(0))
    hbo = hemo[idx, :36].mean((0, 1)).reshape(30, 10).mean(-1)
    print(f"   {name}: HbO peak at {t_e[np.argmax(hbo)]:+.1f} s | slow-EEG GFP peak at {t_e[np.argmax(gfp)]:+.1f} s")
    print("      HbO  ", " ".join(f"{v:+.3f}" for v in hbo[::3]))
    print("      GFP  ", " ".join(f"{v:.3f}" for v in gfp[::3]))
