"""FINAL_PLAN P3.2 (Claim 2): slow-EEG ridge frozen on hBCI, tested on mental workload (and the reverse)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
from pathlib import Path

import nbformat
import numpy as np
import scipy.signal as sps
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]   # repository root

NB = Path(str(REPO / "pipeline/Neurovascnet_marge_v1.ipynb"))
ROOT = NB.parent / "neurovascnet_merge_v1_cache"
CACHES = {"hbci": ROOT / "train_fnirs_a2133f5d6a71ec34", "mental_workload": ROOT / "replication_tuberlin_mw"}
OUT = Path(__file__).with_name("external_claim2_class_results.json")
FS, SEED = 200.0, 1337
ALPHAS = np.logspace(0, 6, 13)

nb = nbformat.read(NB, as_version=4)
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12] + [nb.cells[19]]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)
subject_level_test, verdict_text, interpolation_matrix = ns["subject_level_test"], ns["verdict_text"], ns["interpolation_matrix"]
print("practical_r =", ns["CFG"].practical_r)


def _bins(x, nb_):
    return x.reshape(*x.shape[:-1], nb_, -1).mean(-1)


def load(name):
    cache = CACHES[name]
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    sos = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    F = np.concatenate([_bins(sps.sosfiltfilt(sos, np.asarray(eeg[i:i + 120], np.float64), axis=-1), 30)
                        for i in range(0, len(eeg), 120)])                  # (n, ch, 30)
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    Y = _bins(np.stack([hemo[:, :36].mean(1), hemo[:, 36:].mean(1)], 1), 30).reshape(len(hemo), -1)
    meta = json.loads((cache / "meta.json").read_text())
    chans = meta["info"]["ch_names"] if "info" in meta else meta["ch_names"]
    return {"F": F, "Y": Y, "ch": chans, "sid": np.load(cache / "subject_ids.npy"),
            "tid": np.load(cache / "task_ids.npy"), "rec": np.load(cache / "rec_ids.npy"),
            "cls": np.unique(np.load(cache / "labels.npy"), return_inverse=True)[1]}


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


def trial_corr(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    return (P * T).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-12)


D = {k: load(k) for k in CACHES}
for k, d in D.items():
    print(k, d["F"].shape, len(d["ch"]), "channels")

res = {}
for src, dst in (("hbci", "mental_workload"), ("mental_workload", "hbci")):
    a, b = D[src], D[dst]
    W = interpolation_matrix(b["ch"], a["ch"])                       # dst montage -> src positions
    assert W is not None, "interpolation matrix unavailable"
    Fb = np.einsum("ij,njt->nit", W, b["F"])
    Fa_z, Fb_z = zs_person(a["F"], a["sid"]), zs_person(Fb, b["sid"])
    Ya = demean(a["Y"], (a["sid"] * 1000 + a["tid"]) * 100 + a["cls"])                   # train on within-person-task variation
    mdl = RidgeCV(alphas=ALPHAS).fit(Fa_z, Ya)
    g = (b["sid"].astype(np.int64) * 1000 + b["tid"]) * 100 + b["cls"]
    P = demean(mdl.predict(Fb_z), g)
    T = demean(b["Y"], g)
    st = subject_level_test(P, T, b["sid"])
    st["verdict"] = verdict_text(st)
    st["alpha"] = float(mdl.alpha_)
    st["claim2_pass"] = bool(st["verdict"] == "detectable")
    # per-chromophore, descriptive
    for nm, sl in (("HbO", slice(0, 30)), ("HbR", slice(30, 60))):
        st[f"r_{nm}"] = subject_level_test(P[:, sl], T[:, sl], b["sid"])["r_subject"]
    res[f"{src}_to_{dst}"] = st
    print(f"{src} -> {dst}: r_subject {st['r_subject']:+.4f} [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] "
          f"null {st['null_mean']:+.4f} p_sf {st['p_signflip']:.4f} ({st['n_positive']}/{st['n_subjects']}) "
          f"HbO {st['r_HbO']:+.4f} HbR {st['r_HbR']:+.4f} -> {st['verdict']} | Claim 2 pass: {st['claim2_pass']}",
          flush=True)

    OUT.write_text(json.dumps(res, indent=1))
print("DONE")
