"""FINAL_PLAN P2.6: task-evoked timing of slow EEG vs HbO (epoch caches, hBCI + mental workload)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
from pathlib import Path

import numpy as np
import scipy.signal as sps
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]   # repository root

ROOT = Path(str(REPO / "pipeline/neurovascnet_merge_v1_cache"))
COHORTS = {"hbci": ROOT / "train_fnirs_a2133f5d6a71ec34", "mental_workload": ROOT / "replication_tuberlin_mw"}
OUT = Path(__file__).with_name("evoked_timing_results.json")
FS_EEG, FS, TMIN, NSURR, SEED = 200.0, 10.0, -2.0, 200, 1337
SOS = sps.butter(4, 0.1, btype="low", fs=FS, output="sos")
T = np.arange(300) / FS + TMIN
BASE = T < 0
WIN = (T >= 0) & (T <= 25)


def eeg_10hz(eeg, idx, chunk=200):
    out = []
    for i in range(0, len(idx), chunk):
        x = np.asarray(eeg[idx[i:i + chunk]], np.float64)
        out.append(x.reshape(len(x), x.shape[1], 300, int(FS_EEG / FS)).mean(-1))
    return np.concatenate(out)


def gfp(avg):
    x = sps.sosfiltfilt(SOS, avg, axis=-1)
    x = x - x[:, BASE].mean(1, keepdims=True)
    return np.sqrt((x ** 2).mean(0))


def run_cohort(cache, rng):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    hemo = np.load(cache / "hemo.npy", mmap_mode="r")
    sid, tid = np.load(cache / "subject_ids.npy"), np.load(cache / "task_ids.npy")
    people = {}
    for s in np.unique(sid):
        rows = []
        for t in np.unique(tid[sid == s]):
            idx = np.flatnonzero((sid == s) & (tid == t))
            if len(idx) < 10:
                continue
            E = eeg_10hz(eeg, idx)
            g = gfp(E.mean(0))
            g_lat, g_peak = T[WIN][np.argmax(g[WIN])], g[WIN].max()
            null = np.array([gfp((rng.choice([-1.0, 1.0], len(idx))[:, None, None] * E).mean(0))[WIN].max()
                             for _ in range(NSURR)])
            h = np.asarray(hemo[idx, :36], np.float64).mean((0, 1))
            h = h - h[BASE].mean()
            h_lat = T[WIN][np.argmax(h[WIN])]
            rows.append({"task": int(t), "n_trials": len(idx), "gfp_peak_s": float(g_lat), "hbo_peak_s": float(h_lat),
                         "delay_s": float(h_lat - g_lat), "gfp_peak": float(g_peak),
                         "null95": float(np.percentile(null, 95)), "evoked_present": bool(g_peak > np.percentile(null, 95))})
        if not rows:
            continue
        people[int(s)] = {"tasks": rows, "delay_s": float(np.mean([r["delay_s"] for r in rows])),
                          "evoked_present": bool(np.mean([r["evoked_present"] for r in rows]) >= 0.5)}
        p = people[int(s)]
        print(f"  s{int(s):02d}: " + " | ".join(f"task {r['task']} GFP {r['gfp_peak_s']:+.1f} s HbO {r['hbo_peak_s']:+.1f} s "
                                                 f"{'evoked' if r['evoked_present'] else 'no evoked'}" for r in rows)
              + f" -> delay {p['delay_s']:+.1f} s", flush=True)
    present = [p["delay_s"] for p in people.values() if p["evoked_present"]]
    allp = [p["delay_s"] for p in people.values()]

    def test(v):
        v = np.asarray(v)
        if len(v) < 5 or not np.any(v != 0):
            return {"n": len(v), "median_s": float(np.median(v)) if len(v) else None, "wilcoxon_p": None}
        return {"n": len(v), "median_s": float(np.median(v)), "iqr_s": [float(x) for x in np.percentile(v, [25, 75])],
                "wilcoxon_p": float(wilcoxon(v).pvalue), "n_positive": int((v > 0).sum())}

    r = {"people": people, "n_people": len(people), "n_evoked_present": len(present),
         "primary_evoked_present": test(present), "all_people": test(allp)}
    pr = r["primary_evoked_present"]
    ok = pr["wilcoxon_p"] is not None and pr["median_s"] > 0 and pr["wilcoxon_p"] < 0.05
    r["verdict"] = "evoked EEG-first" if ok else "not shown"
    return r


rng = np.random.default_rng(SEED)
res = {}
for name, cache in COHORTS.items():
    print(name, flush=True)
    res[name] = run_cohort(cache, rng)
    r = res[name]
    print(f"== {name}: evoked present in {r['n_evoked_present']}/{r['n_people']}; primary {r['primary_evoked_present']}; "
          f"all {r['all_people']} -> {r['verdict']}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
print("DONE")
