"""FINAL_PLAN P4: FineMI loader -> epoch cache (notebook helpers) + continuous 1 Hz pairs for the lag test."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
import time
import warnings
from pathlib import Path

import mne
import nbformat
import numpy as np
import scipy.signal as sps

REPO = Path(__file__).resolve().parents[1]   # repository root

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
NB = Path(str(REPO / "pipeline/Neurovascnet_marge_v1.ipynb"))
ROOT = Path(str(REPO / "dataset/Multi-joint Upper-Limb EEG+FNIR/FineMI/FineMI"))
REREF = os.environ.get("REREF", "")                       # "lm" = linked mastoids (P4.B)
HWHP = os.environ.get("HWHP", "")                         # "1" = emulate BrainAmp 10 s first-order high-pass (P5.2)
CACHE = Path(str(REPO / "pipeline/neurovascnet_merge_v1_cache/replication_finemi")
             + ("_hw" if HWHP else "") + ("_lm" if REREF else ""))
FS = 200.0
EOG = ("HEO", "VEO")
DROP = {"M1", "M2", "CB1", "CB2", "EKG", "EMG"} | set(EOG)
GAP_S, MAXLAG, MINSHIFT = 40.0, 20, 60

nb = nbformat.read(NB, as_version=4)
nb.cells[4].source = nb.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)
cfg = ns["CFG"]
resample_to, zscore_trial, baseline_correct, bandpass = (ns[k] for k in ("resample_to", "zscore_trial", "baseline_correct", "bandpass"))
n_base = int(cfg.baseline_s * cfg.hemo_fs)
MONT = {k.upper(): k for k in mne.channels.make_standard_montage("standard_1005").ch_names}


def to_1hz(x, fs, lo, hi):
    sos = (sps.butter(4, hi, btype="low", fs=fs, output="sos") if lo is None
           else sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos"))
    y = sps.sosfiltfilt(sos, x, axis=-1)
    step = int(round(fs))
    n = y.shape[-1] // step
    return y[:, :n * step].reshape(y.shape[0], n, step).mean(-1)


def events(raw):
    a = raw.annotations
    keep = [i for i, d in enumerate(a.description) if d.replace(".0", "").strip().isdigit()]
    return np.asarray(a.onset)[keep], [int(float(a.description[i])) for i in keep]


def read_block(s, name):
    e = mne.io.read_raw_cnt(ROOT / f"subject{s}" / "EEG" / f"{name}.cnt", preload=True)
    te, le = events(e)
    eeg_ch = [c for c in e.ch_names if c not in DROP]
    eog_ch = [c for c in EOG if c in e.ch_names]
    e.pick(eeg_ch + eog_ch + (["M1", "M2"] if REREF else []))
    e.resample(FS, npad="auto")
    x = e.get_data()
    if REREF:
        x = x[:-2] - x[-2:].mean(0, keepdims=True)
    if HWHP:                                              # causal first-order high-pass, time constant 10 s
        bh, ah = sps.butter(1, 1.0 / (2 * np.pi * 10.0), btype="high", fs=FS)
        x = sps.lfilter(bh, ah, x, axis=-1, zi=sps.lfilter_zi(bh, ah)[None, :] * x[:, :1])[0]
    x = x.astype(np.float32)
    n = mne.io.read_raw_nirx(ROOT / f"subject{s}" / "fNIRS" / name, preload=True)
    tn, ln = events(n)
    hb = mne.preprocessing.nirs.beer_lambert_law(mne.preprocessing.nirs.optical_density(n), ppf=6.0)
    names = hb.ch_names
    hbo = hb.get_data(picks=[i for i, c in enumerate(names) if c.endswith("hbo")])
    hbr = hb.get_data(picks=[i for i, c in enumerate(names) if c.endswith("hbr")])
    return {"x": x, "eeg_ch": eeg_ch, "n_eeg": len(eeg_ch), "te": te, "le": le, "tn": tn, "ln": ln,
            "hbo": hbo, "hbr": hbr, "n_fs": float(n.info["sfreq"])}


eeg_l, eog_l, hemo_l, sid_l, rec_l, lab_l, audit, cont = [], [], [], [], [], [], [], {}
ch_names = None
t0 = time.time()
subjects = sorted(int(p.name[7:]) for p in ROOT.glob("subject*") if p.is_dir())
TMP = CACHE / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
for s in subjects:
    pairs = []
    eeg_l = []
    for name in sorted(p.stem for p in (ROOT / f"subject{s}" / "EEG").glob("*.cnt")):
        try:
            b = read_block(s, name)
        except Exception as exc:
            audit.append((s, name, f"unreadable: {type(exc).__name__} {exc}"[:200])); continue
        if b["le"] != b["ln"]:
            audit.append((s, name, f"event sequences differ ({len(b['le'])} vs {len(b['ln'])})")); continue
        canon = [MONT.get(c.upper(), c) for c in b["eeg_ch"]]
        ch_names = ch_names or canon
        if canon != ch_names:
            audit.append((s, name, "channel set differs")); continue
        te, tn = b["te"], b["tn"]
        run = np.concatenate([[0], np.cumsum(np.diff(te) > GAP_S)])
        off = tn - te
        hbo_f, hbr_f = bandpass(b["hbo"], b["n_fs"], .01, .2), bandpass(b["hbr"], b["n_fs"], .01, .2)
        n_ok = 0
        for k in range(len(te)):
            a_e, b_e = int(round((te[k] + cfg.tmin_s) * FS)), int(round((te[k] + cfg.tmax_s) * FS))
            a_n, b_n = int(round((tn[k] + cfg.tmin_s) * b["n_fs"])), int(round((tn[k] + cfg.tmax_s) * b["n_fs"]))
            if a_e < 0 or b_e > b["x"].shape[1] or a_n < 0 or b_n > hbo_f.shape[1]:
                continue
            seg = resample_to(b["x"][:, a_e:b_e].astype(np.float64), cfg.eeg_samples)
            eeg_l.append(zscore_trial(seg[:b["n_eeg"]]).astype(np.float32))
            eog_l.append(zscore_trial(seg[b["n_eeg"]:]).astype(np.float32))
            so = baseline_correct(resample_to(hbo_f[:, a_n:b_n], cfg.hemo_samples), n_base)
            sr = baseline_correct(resample_to(hbr_f[:, a_n:b_n], cfg.hemo_samples), n_base)
            hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
            sid_l.append(s); rec_l.append(f"s{s:02d}/{name}_run{run[k]}"); lab_l.append(b["le"][k])
            n_ok += 1
        # continuous 1 Hz pairs, one per run, as the P2.3/P3.3 loaders
        for r in np.unique(run):
            m = run == r
            if np.ptp(off[m]) > 1.0:
                audit.append((s, f"{name}_run{r}", f"offset drifts {np.ptp(off[m]):.2f} s")); continue
            o = float(np.median(off[m]))
            a, bb = max(0.0, te[m].min() - 20.0), te[m].max() + 60.0
            ia, ib = int(a * FS), min(b["x"].shape[1], int(bb * FS))
            na, nb_ = max(0, int((a + o) * b["n_fs"])), min(b["hbo"].shape[1], int((bb + o) * b["n_fs"]))
            n_s = min((ib - ia) / FS, (nb_ - na) / b["n_fs"])
            E = to_1hz(b["x"][:b["n_eeg"], ia:ia + int(n_s * FS)].astype(np.float64), FS, None, 0.1)
            H = to_1hz(b["hbo"][:, na:na + int(n_s * b["n_fs"])], b["n_fs"], 0.01, 0.1)
            L = min(E.shape[1], H.shape[1])
            if L >= 3 * MAXLAG + MINSHIFT:
                pairs.append((E[:, :L].T.astype(np.float32), H[:, :L].T.astype(np.float32)))
        audit.append((s, name, f"ok {n_ok} trials, {run.max() + 1} runs, offset {np.median(off):+.2f} s"))
        del b
    cont[s] = pairs
    if eeg_l:
        np.save(TMP / f"eeg_s{s:02d}.npy", np.stack(eeg_l))
    print(f"subject{s}: " + "; ".join(a[2] for a in audit if a[0] == s) + f" | {len(pairs)} continuous runs "
          f"| {(time.time() - t0) / 60:.1f} min", flush=True)

CACHE.mkdir(parents=True, exist_ok=True)
labs = np.asarray(lab_l)
parts = sorted(TMP.glob("eeg_s*.npy"))
total = sum(np.load(p, mmap_mode="r").shape[0] for p in parts)
assert total == len(sid_l), (total, len(sid_l))
first = np.load(parts[0], mmap_mode="r")
E = np.lib.format.open_memmap(CACHE / "eeg.npy", mode="w+", dtype=np.float32, shape=(total,) + first.shape[1:])
i = 0
for p in parts:
    a = np.load(p, mmap_mode="r"); E[i:i + len(a)] = a; i += len(a)
    del a
E.flush(); del E, first
import gc; gc.collect()
for p in parts:
    try:
        p.unlink()
    except PermissionError:
        print("could not delete", p)
arrays = {"eog": np.stack(eog_l), "hemo": np.stack(hemo_l), "subject_ids": np.asarray(sid_l),
          "task_ids": np.zeros(len(sid_l), int), "rec_ids": np.asarray(rec_l), "labels": labs}
for k, v in arrays.items():
    np.save(CACHE / f"{k}.npy", v)
np.savez(CACHE / "continuous.npz", **{f"s{s}_r{i}_{part}": arr for s, ps in cont.items()
                                     for i, pr in enumerate(ps) for part, arr in zip("EH", pr)})
(CACHE / "meta.json").write_text(json.dumps({"ch_names": ch_names, "audit": audit, "n_hbo": int(hemo_l[0].shape[0] // 2)},
                                            indent=1, default=str))
print(f"DONE {len(sid_l)} trials, {len(set(sid_l))} people, EEG {np.load(CACHE / 'eeg.npy', mmap_mode='r').shape}, hemo {arrays['hemo'].shape}")
