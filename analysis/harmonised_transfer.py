"""FINAL_PLAN P6.1 (Claim 2): frozen slow-EEG ridge across labs with matched trial window and matched NIRS montage."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
from pathlib import Path

import nbformat
import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]   # repository root

NB = Path(str(REPO / "pipeline/CrossModal_InfraSlowEEG_fNIRS_fMRI.ipynb"))
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
TAG = os.environ.get("FINEMI_TAG", "")                   # "" = P4 native cache (primary), "_hw" = P5.2 cache
CACHES = {"hbci": ROOT / "train_fnirs_a2133f5d6a71ec34", "mental_workload": ROOT / "replication_tuberlin_mw",
          "finemi": ROOT / ("replication_finemi" + TAG)}
OUT = Path(__file__).with_name(f"harmonised_transfer{TAG}_results.json")
FS, NWIN = 200.0, 20                                      # 20 x 1 s bins = -2..18 s
ALPHAS = np.logspace(0, 6, 13)
HBCI_LEFT_MOTOR = np.arange(12, 24)                       # C5CP5 ... C1CP1 in the hBCI file order

nb = nbformat.read(NB, as_version=4)
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12] + [nb.cells[19]]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)
subject_level_test, verdict_text, interpolation_matrix = ns["subject_level_test"], ns["verdict_text"], ns["interpolation_matrix"]


def _bins(x, nb_):
    return x.reshape(*x.shape[:-1], nb_, -1).mean(-1)


def load(name):
    cache = CACHES[name]
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    F = np.concatenate([_bins(sps.sosfiltfilt(sos, np.asarray(eeg[i:i + 120], np.float64), axis=-1), 30)
                        for i in range(0, len(eeg), 120)])
    hemo = _bins(np.load(cache / "hemo.npy").astype(np.float64), 30)          # (n, 2k, 30)
    k_ = hemo.shape[1] // 2
    meta = json.loads((cache / "meta.json").read_text())
    return {"F": F, "hbo": hemo[:, :k_], "hbr": hemo[:, k_:],
            "ch": meta["info"]["ch_names"] if "info" in meta else meta["ch_names"],
            "sid": np.load(cache / "subject_ids.npy"), "tid": np.load(cache / "task_ids.npy")}


def view(d, window, motor):
    """Features (n, ch, t) and pooled target (n, 2*t) for one harmonisation setting."""
    sl = slice(0, NWIN) if window else slice(None)
    idx = HBCI_LEFT_MOTOR if (motor and d["hbo"].shape[1] == 36 and d is D["hbci"]) else slice(None)
    Y = np.concatenate([d["hbo"][:, idx].mean(1)[:, sl], d["hbr"][:, idx].mean(1)[:, sl]], 1)
    return d["F"][..., sl], Y


def zs_person(F, sid):
    F = F.reshape(len(F), -1).copy()
    for s in np.unique(sid):
        m = sid == s
        F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
    return F


def demean(Y, g):
    Y = Y.copy()
    for k in np.unique(g):
        m = g == k
        Y[m] -= Y[m].mean(0)
    return Y


D = {k: load(k) for k in CACHES}
res = {}
RUNS = [("hbci", "finemi", True, True, "primary: window + montage"),
        ("hbci", "finemi", True, False, "window only"),
        ("hbci", "finemi", False, True, "montage only"),
        ("mental_workload", "finemi", True, False, "MW window only"),
        ("finemi", "hbci", True, True, "reverse, window + montage"),
        ("hbci", "mental_workload", True, True, "within lab reference, window + montage")]
for src, dst, win, mot, label in RUNS:
    a, b = D[src], D[dst]
    Fa, Ya = view(a, win, mot)
    Fb, Yb = view(b, win, mot)
    W = interpolation_matrix(b["ch"], a["ch"])
    Fb = np.einsum("ij,njt->nit", W, Fb)
    mdl = RidgeCV(alphas=ALPHAS).fit(zs_person(Fa, a["sid"]), demean(Ya, a["sid"] * 1000 + a["tid"]))
    g = b["sid"].astype(np.int64) * 1000 + b["tid"]
    P = demean(mdl.predict(zs_person(Fb, b["sid"])), g)
    T = demean(Yb, g)
    st = subject_level_test(P, T, b["sid"])
    st["verdict"] = verdict_text(st)
    st["pass"] = bool(st["verdict"] == "detectable")
    h = Yb.shape[1] // 2
    st["r_HbO"] = subject_level_test(P[:, :h], T[:, :h], b["sid"])["r_subject"]
    st["r_HbR"] = subject_level_test(P[:, h:], T[:, h:], b["sid"])["r_subject"]
    res[f"{src}_to_{dst} | {label}"] = st
    print(f"{src} -> {dst} [{label}]: r {st['r_subject']:+.4f} [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] "
          f"p_sf {st['p_signflip']:.4f} ({st['n_positive']}/{st['n_subjects']}) HbO {st['r_HbO']:+.4f} "
          f"HbR {st['r_HbR']:+.4f} -> {st['verdict']}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
print("DONE")
