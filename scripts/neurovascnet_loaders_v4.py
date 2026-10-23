"""TU Berlin hBCI paired EEG+fNIRS loader — the v4 primary fNIRS supervision set.

Replaces HEFMI, which the data-quality audit showed to be systemic-dominated
(HbO/HbR = +0.898, target split-half reliability 0.127 after the v3 pipeline).
TU Berlin hBCI measures 0.528 reliability with true SCI 0.902 — a ~4x higher
prediction ceiling before any model change. See NeuroVascNet_Data_Quality_Audit.md
sections 2.0-2.0c.

Layout
------
    EEG data/EEG_01-05/subject 01/with occular artifact/{cnt,mrk}.mat
        6 runs alternating MI / MA, 200 Hz, 32 series = 30 EEG + VEOG + HEOG
    NRI data/NIRS_01-29/subject 01/{cnt,mrk}.mat
        6 runs in the same order, 10 Hz, 72 series = 36 SD pairs x 2 wavelengths
        RAW intensities at 760/850 nm -> we do our own Beer-Lambert

Only the MI (motor imagery) runs are used. 29 subjects x 3 runs x 20 events.

Output schema matches the notebook contract:
    eeg          [N, 30, 6000]   30 s @ 200 Hz, z-scored per trial
    hemo         [N, 72, 300]    36 HbO + 36 HbR, 30 s @ 10 Hz, micromolar
    labels       [N]             trial class
    subject_ids  [N]
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io as sio
from scipy import signal as sps

warnings.filterwarnings("ignore")

ROOT = Path("dataset/TU Berlin hBCI EEG+FNRI")
EEG_ROOT = ROOT / "EEG data"
NIRS_ROOT = ROOT / "NRI data" / "NIRS_01-29"
CACHE_DIR = Path("neurovascnet_outputs/cache_v4")

# Beer-Lambert constants (Prahl / Wray), cm^-1 / (mol/L)
EXT = {760: {"HbO": 1486.59, "HbR": 3843.71},
       850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}
SD_CM = 3.0

EOG_CHANNELS = ("VEOG", "HEOG")


@dataclass(frozen=True)
class LoaderV4Config:
    tmin_s: float = -2.0          # 2 s pre-stimulus baseline
    tmax_s: float = 28.0          # -> 30 s total window
    eeg_fs: float = 200.0
    hemo_fs: float = 10.0
    eeg_samples: int = 6000
    hemo_samples: int = 300
    baseline_s: float = 2.0
    n_val_subjects: int = 6       # held-out subjects for validation
    apply_gsr: bool = False       # audit: GSR removes ~83% of the evoked response
    task: str = "MI"              # "MI" (motor imagery) or "MA" (mental arithmetic)


# ---------------------------------------------------------------- helpers
def _runs(path: Path, key_hint: str | None = None):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    key = key_hint if key_hint in keys else keys[0]
    return np.atleast_1d(m[key]).ravel()


def _bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def _beer_lambert(lo_wl: np.ndarray, hi_wl: np.ndarray):
    """(ch, T) raw intensities -> HbO, HbR in micromolar."""
    def od(x):
        base = np.median(x, axis=-1, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        return -np.log(np.clip(x / base, 1e-9, None))

    a1, a2 = od(lo_wl), od(hi_wl)
    L1, L2 = SD_CM * DPF[760], SD_CM * DPF[850]
    E = np.array([[EXT[760]["HbO"] * L1, EXT[760]["HbR"] * L1],
                  [EXT[850]["HbO"] * L2, EXT[850]["HbR"] * L2]])
    Ei = np.linalg.inv(E)
    return ((Ei[0, 0] * a1 + Ei[0, 1] * a2) * 1e6,
            (Ei[1, 0] * a1 + Ei[1, 1] * a2) * 1e6)


def _split_wavelengths(x: np.ndarray, clab):
    """(T, 72) -> (lo, hi) each (36, T), matched by source-detector pair name."""
    names = [str(c) for c in np.atleast_1d(clab)]
    lo = {n.replace("lowWL", ""): i for i, n in enumerate(names) if "lowWL" in n}
    hi = {n.replace("highWL", ""): i for i, n in enumerate(names) if "highWL" in n}
    pairs = sorted(set(lo) & set(hi))
    return (np.stack([x[:, lo[p]] for p in pairs]),
            np.stack([x[:, hi[p]] for p in pairs]), pairs)


def _regress_global(x):
    g = x.mean(axis=0, keepdims=True)
    gc = g - g.mean(axis=-1, keepdims=True)
    bc = x - x.mean(axis=-1, keepdims=True)
    beta = (bc * gc).sum(-1, keepdims=True) / ((gc ** 2).sum(-1, keepdims=True) + 1e-12)
    return bc - beta * gc


def _eeg_subject_dir(subject_num: int) -> Path | None:
    for group in sorted(EEG_ROOT.glob("EEG_*")):
        d = group / f"subject {subject_num:02d}" / "with occular artifact"
        if d.exists():
            return d
    return None


def _resample(x, n):
    return x if x.shape[-1] == n else sps.resample(x, n, axis=-1)


# ---------------------------------------------------------------- loader
def load_tuberlin_hbci(split: Literal["train", "val", "all"] = "train",
                       cfg: LoaderV4Config = LoaderV4Config(),
                       subjects: range | None = None) -> dict:
    subjects = subjects or range(1, 30)
    all_subj = list(subjects)
    val_subj = set(all_subj[-cfg.n_val_subjects:])
    if split == "train":
        keep = [s for s in all_subj if s not in val_subj]
    elif split == "val":
        keep = [s for s in all_subj if s in val_subj]
    else:
        keep = all_subj

    eeg_out, hemo_out, lab_out, sid_out, audit = [], [], [], [], []
    n_base = int(cfg.baseline_s * cfg.hemo_fs)

    for s in keep:
        eeg_dir = _eeg_subject_dir(s)
        nirs_dir = NIRS_ROOT / f"subject {s:02d}"
        if eeg_dir is None or not nirs_dir.exists():
            audit.append({"subject": s, "status": "missing directory", "trials": 0})
            continue
        try:
            e_runs = _runs(eeg_dir / "cnt.mat", "cnt")
            e_mrks = _runs(eeg_dir / "mrk.mat", "mrk")
            n_runs = _runs(nirs_dir / "cnt.mat", "cnt")
            n_mrks = _runs(nirs_dir / "mrk.mat", "mrk")
        except Exception as exc:
            audit.append({"subject": s, "status": f"read fail: {exc}", "trials": 0})
            continue

        n_trials_subj = 0
        for r in range(min(e_runs.size, n_runs.size)):
            er, nr = e_runs[r], n_runs[r]
            if not str(getattr(er, "title", "")).upper().startswith(cfg.task):
                continue

            # ---- EEG ----
            e_fs = float(np.atleast_1d(er.fs).ravel()[0])
            e_clab = [str(c) for c in np.atleast_1d(er.clab)]
            keep_ch = [i for i, c in enumerate(e_clab) if c not in EOG_CHANNELS]
            ex = np.asarray(er.x, dtype=np.float64)[:, keep_ch].T      # (ch, T)

            # ---- fNIRS: raw wavelengths -> HbO/HbR ----
            n_fs = float(np.atleast_1d(nr.fs).ravel()[0])
            lo, hi, _ = _split_wavelengths(np.asarray(nr.x, dtype=np.float64), nr.clab)
            hbo, hbr = _beer_lambert(lo, hi)
            hbo = _bandpass(hbo, n_fs, 0.01, 0.2)
            hbr = _bandpass(hbr, n_fs, 0.01, 0.2)
            if cfg.apply_gsr:
                hbo, hbr = _regress_global(hbo), _regress_global(hbr)

            # ---- events ----
            em = e_mrks[r] if r < e_mrks.size else None
            nm = n_mrks[r] if r < n_mrks.size else None
            if em is None or nm is None:
                continue
            e_on = np.atleast_1d(np.asarray(em.time, dtype=float)) / 1000.0
            n_on = np.atleast_1d(np.asarray(nm.time, dtype=float)) / 1000.0
            y = np.asarray(em.y)
            labels = np.argmax(y, axis=0) if y.ndim == 2 else np.asarray(y).ravel()

            m = min(e_on.size, n_on.size, labels.size)
            for k in range(m):
                a_e = int(round((e_on[k] + cfg.tmin_s) * e_fs))
                b_e = int(round((e_on[k] + cfg.tmax_s) * e_fs))
                a_n = int(round((n_on[k] + cfg.tmin_s) * n_fs))
                b_n = int(round((n_on[k] + cfg.tmax_s) * n_fs))
                if a_e < 0 or b_e > ex.shape[1] or a_n < 0 or b_n > hbo.shape[1]:
                    continue

                seg_e = _resample(ex[:, a_e:b_e], cfg.eeg_samples)
                mu = seg_e.mean(-1, keepdims=True)
                sd = seg_e.std(-1, keepdims=True) + 1e-6
                seg_e = ((seg_e - mu) / sd).astype(np.float32)

                so = _resample(hbo[:, a_n:b_n], cfg.hemo_samples)
                sr = _resample(hbr[:, a_n:b_n], cfg.hemo_samples)
                so = so - so[:, :n_base].mean(-1, keepdims=True)
                sr = sr - sr[:, :n_base].mean(-1, keepdims=True)

                eeg_out.append(seg_e)
                hemo_out.append(np.concatenate([so, sr], 0).astype(np.float32))
                lab_out.append(int(labels[k]))
                sid_out.append(s)
                n_trials_subj += 1

        audit.append({"subject": s, "status": "ok" if n_trials_subj else "no trials",
                      "trials": n_trials_subj})

    if not eeg_out:
        raise RuntimeError(f"hBCI loader produced no samples for split={split}")

    return {
        "eeg": np.stack(eeg_out),
        "hemo": np.stack(hemo_out),
        "labels": np.asarray(lab_out, dtype=np.int64),
        "subject_ids": np.asarray(sid_out, dtype=np.int64),
        "true_delays": np.zeros(len(eeg_out), dtype=np.float32),
        "audit": np.asarray(audit, dtype=object),
    }


def build_cache(cfg: LoaderV4Config = LoaderV4Config(), force: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.task}{'_gsr' if cfg.apply_gsr else ''}"
    out = {}
    for split in ("train", "val"):
        p = CACHE_DIR / f"hbci_{split}_{tag}.npz"
        if p.exists() and not force:
            print(f"{p.name}: cached")
            out[split] = p
            continue
        d = load_tuberlin_hbci(split, cfg)
        np.savez_compressed(
            p, eeg=d["eeg"], hemo=d["hemo"], labels=d["labels"],
            subject_ids=d["subject_ids"], true_delays=d["true_delays"])
        print(f"{p.name}: {d['eeg'].shape[0]} trials, "
              f"{np.unique(d['subject_ids']).size} subjects, "
              f"eeg {d['eeg'].shape}, hemo {d['hemo'].shape}")
        out[split] = p
    return out


if __name__ == "__main__":
    build_cache()
