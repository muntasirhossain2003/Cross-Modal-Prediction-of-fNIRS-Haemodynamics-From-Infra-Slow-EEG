"""FINAL_PLAN P3.4 (Claim 4): does EEG-predicted ("virtual") hemodynamics help an EEG-only classification task?"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import json
from pathlib import Path

import numpy as np
import scipy.signal as sps
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"G:\research_bio_medical\merge\neurovascnet_merge_v1_cache")
CACHES = {"hbci": ROOT / "train_fnirs_a2133f5d6a71ec34", "mental_workload": ROOT / "replication_tuberlin_mw"}
OUT = Path(__file__).with_name("downstream_claim4_results.json")
FS, SEED, TMIN = 200.0, 1337, -2.0
ALPHAS = np.logspace(0, 6, 13)
WIN = slice(int(5 - TMIN), int(15 - TMIN))           # 1 s bins covering 5-15 s after the cue


def _bins(x, nb):
    return x.reshape(*x.shape[:-1], nb, -1).mean(-1)


def load(cache):
    eeg = np.load(cache / "eeg.npy", mmap_mode="r")
    slow = sps.butter(4, 0.5, btype="low", fs=FS, output="sos")
    bands = [sps.butter(4, b, btype="band", fs=FS, output="sos") for b in ((8, 13), (13, 30))]
    a, b = int(-TMIN * FS), int((10 - TMIN) * FS)
    F, BP = [], []
    for i in range(0, len(eeg), 120):
        x = np.asarray(eeg[i:i + 120], np.float64)
        F.append(_bins(sps.sosfiltfilt(slow, x, axis=-1), 30).reshape(len(x), -1))
        BP.append(np.concatenate([np.log(sps.sosfiltfilt(s, x, axis=-1)[..., a:b].var(-1) + 1e-12) for s in bands], 1))
    hemo = np.load(cache / "hemo.npy").astype(np.float64)
    g = np.repeat(np.stack([hemo[:, :36].mean(1), hemo[:, 36:].mean(1)], 1), 36, axis=1)
    Y = _bins(hemo - g, 30)                                   # local target (n, 72, 30)
    return {"F": np.concatenate(F), "BP": np.concatenate(BP), "Y": Y, "sid": np.load(cache / "subject_ids.npy"),
            "tid": np.load(cache / "task_ids.npy"), "lab": np.load(cache / "labels.npy")}


def zs_person(X, sid):
    X = X.copy()
    for s in np.unique(sid):
        m = sid == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return X


def demean(Y, g):
    Y = Y.copy()
    for k in np.unique(g):
        m = g == k
        Y[m] -= Y[m].mean(0)
    return Y


def virtual(d, Fz, train_idx, apply_idx, shuffle, rng):
    Y = demean(d["Y"].reshape(len(d["Y"]), -1), d["sid"].astype(np.int64) * 1000 + d["tid"])
    Ytr = Y[train_idx].copy()
    if shuffle:
        for s in np.unique(d["sid"][train_idx]):
            m = np.flatnonzero(d["sid"][train_idx] == s)
            Ytr[m] = Ytr[rng.permutation(m)]
    P = RidgeCV(alphas=ALPHAS).fit(Fz[train_idx], Ytr).predict(Fz[apply_idx])
    return P.reshape(len(apply_idx), 72, 30)[..., WIN].mean(-1)


def virtual_oof(d, Fz, people_idx, shuffle, rng):
    """Out-of-fold virtual hemodynamics for the classifier's training people (inner person folds)."""
    out = np.zeros((len(people_idx), 72))
    subs = np.unique(d["sid"][people_idx])
    for hold in np.array_split(np.random.default_rng(SEED).permutation(subs), 3):
        te = np.isin(d["sid"][people_idx], hold)
        out[te] = virtual(d, Fz, people_idx[~te], people_idx[te], shuffle, rng)
    return out


def boot_ci(v, rng):
    v = np.asarray(v)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def run_task(d, mask, name, rng):
    Fz = zs_person(d["F"], d["sid"])
    BPz = zs_person(d["BP"], d["sid"])
    proj = rng.standard_normal((Fz.shape[1], 72)) / np.sqrt(Fz.shape[1])
    task_idx = np.flatnonzero(mask)
    y = np.unique(d["lab"][task_idx], return_inverse=True)[1]
    subs = np.unique(d["sid"][task_idx])
    acc = {k: {} for k in ("A", "B", "S", "R", "hemo_only")}
    for hold in np.array_split(np.random.default_rng(SEED).permutation(subs), 3):
        te_t = np.isin(d["sid"][task_idx], hold)
        tr_people = np.flatnonzero(~np.isin(d["sid"], hold))            # all trials of training people
        tr_i, te_i = task_idx[~te_t], task_idx[te_t]
        pos = {v: i for i, v in enumerate(tr_people)}
        feats = {}
        for key, shuffle in (("B", False), ("S", True)):
            oof = virtual_oof(d, Fz, tr_people, shuffle, rng)
            v_tr = oof[[pos[i] for i in tr_i]]
            v_te = virtual(d, Fz, tr_people, te_i, shuffle, rng)
            feats[key] = (np.hstack([BPz[tr_i], v_tr]), np.hstack([BPz[te_i], v_te]))
            if key == "B":
                feats["hemo_only"] = (v_tr, v_te)
        feats["A"] = (BPz[tr_i], BPz[te_i])
        feats["R"] = (np.hstack([BPz[tr_i], Fz[tr_i] @ proj]), np.hstack([BPz[te_i], Fz[te_i] @ proj]))
        for key, (Xtr, Xte) in feats.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, y[~te_t])
            hit = clf.predict(Xte) == y[te_t]
            for s in hold:
                acc[key][int(s)] = float(hit[d["sid"][te_i] == s].mean())
    people = sorted(acc["A"])
    mean = {k: float(np.mean([acc[k][s] for s in people])) for k in acc}
    diffs = {}
    for other in ("A", "S", "R"):
        v = [acc["B"][s] - acc[other][s] for s in people]
        diffs[f"B_minus_{other}"] = {"mean": float(np.mean(v)), "ci": boot_ci(v, rng)}
    ok = (diffs["B_minus_A"]["mean"] >= 0.02 and diffs["B_minus_A"]["ci"][0] > 0
          and diffs["B_minus_S"]["ci"][0] > 0 and diffs["B_minus_R"]["ci"][0] > 0)
    print(f"{name}: " + ", ".join(f"{k} {v:.4f}" for k, v in mean.items()) + " | " +
          ", ".join(f"{k} {v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" for k, v in diffs.items()) +
          f" -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"mean_acc": mean, "diffs": diffs, "pass": bool(ok), "chance": 1 / len(np.unique(y)), "per_person": acc}


rng = np.random.default_rng(SEED)
res = {}
h = load(CACHES["hbci"])
res["hbci_MI_left_right (primary)"] = run_task(h, h["tid"] == 0, "hBCI MI left vs right (primary)", rng)
res["hbci_MA_vs_rest"] = run_task(h, h["tid"] == 1, "hBCI MA vs rest", rng)
OUT.write_text(json.dumps(res, indent=1))
m = load(CACHES["mental_workload"])
res["mw_nback_0_2_3"] = run_task(m, m["tid"] == 0, "MW n-back 0/2/3", rng)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
