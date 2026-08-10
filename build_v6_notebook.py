"""Emit NeuroVascNet_v6.ipynb -- the 4-dataset design, built properly.

Structure (this is the point -- v5/Local grew organically and the order was a mess):

    1  SETTINGS      four paths + run mode, nothing else to edit
    2  SETUP         imports, device
    3  LOADERS       ALL FOUR datasets loaded here, together, up front
    4  DATA SUMMARY  one table: shapes, subjects, reliability, ceiling
    5  METRICS       correlation, permutation null, no-EEG baseline, demeaning
    6  MODEL         ONE shared encoder + per-modality heads
    7  TRAINING      joint: fNIRS and fMRI batches interleaved into one encoder
    8  INTERNAL      cross-subject CV on the training sets
    9  EXTERNAL      the two held-out datasets
   10  SUMMARY

Design, from the blueprint:
    TRAIN     fNIRS  HEFMI            ->|
    TRAIN     fMRI   ds002336         ->|  one shared EEG encoder
    EXTERNAL  fNIRS  TU Berlin hBCI   <-|  never seen in training
    EXTERNAL  fMRI   ds002739         <-|

Usage:  python build_v6_notebook.py
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

# ===========================================================================
md(r"""
# NeuroVascNet — one shared encoder, two modalities

**Edit cell 1. Run all.**

One EEG encoder is trained with **both** fNIRS and fMRI supervision, then
deployed on **EEG only**. Two datasets train it; two entirely separate datasets
validate it.

| Role | Dataset | Modality | Target |
|---|---|---|---|
| **TRAIN** | HEFMI-ICH | EEG–fNIRS | 90 HbO + 90 HbR = 180 ch |
| **TRAIN** | OpenNeuro ds002336 | EEG–fMRI | 48 Harvard–Oxford ROI |
| **EXTERNAL** | TU Berlin hBCI | EEG–fNIRS | HbO/HbR, different montage |
| **EXTERNAL** | OpenNeuro ds002739 | EEG–fMRI | 48 Harvard–Oxford ROI |

### How the two modalities share one encoder

EEG channel counts differ (32 / 63 / 30 / 63) and so do the targets (180 vs 48).
So the network is:

```
EEG --> per-dataset input adapter (1x1 conv, n_ch -> 64)
    --> SHARED TRUNK              <- the thing that learns
    --> per-modality head         fNIRS head (180) | fMRI head (48)
```

Only the trunk is shared, and only the trunk is what "cross-modal" means. The
adapters and heads exist so different montages can reach it.

### How external validation compares different montages

- **fMRI** — both datasets use the same Harvard–Oxford 48-ROI atlas, so
  prediction and target are compared **ROI by ROI**.
- **fNIRS** — HEFMI's 90 optode pairs and hBCI's 36 have no published crosswalk,
  so channel-by-channel comparison is not defensible. Instead both are pooled to
  **mean HbO and mean HbR** and compared on those two time courses. Lower
  resolution, but an honest comparison rather than an invented mapping.
""")

# ===========================================================================
md("## 1 · Settings")

co(r'''
# =============================================================================
#                          >>>  EDIT THIS CELL  <<<
# =============================================================================

# =============================================================================
#                     >>>  EVERY DATASET HAS ITS OWN PATH  <<<
#
#   Edit the full path for each dataset below. They are independent: changing
#   one does not affect the others. Set any path to None to switch it off.
#   The comment above each line says which folder must exist inside it.
# =============================================================================
# ---- THE TWO fNIRS COHORTS ---------------------------------------------------
# One trains the fNIRS head, the other validates it. FNIRS_TRAIN below decides
# which way round. Both are motor imagery EEG+fNIRS and the people never
# overlap, so either direction is a legitimate experiment.

# HEFMI-ICH, 37 people (17 control, 20 haemorrhage).   must contain  epo_data/
PATH_FNIRS_HEFMI            = r"G:\research_bio_medical\dataset\HEFMI-ICH EEG+FNRI"

# TU Berlin hBCI, 29 people, motor imagery + arithmetic. must contain  EEG data/
PATH_FNIRS_HBCI             = r"G:\research_bio_medical\dataset\TU Berlin hBCI EEG+FNRI"

# ---- WHICH fNIRS COHORT TRAINS? ----------------------------------------------
#   "HEFMI"  HEFMI trains, hBCI validates.  The original 4-dataset blueprint.
#   "hBCI"   hBCI trains, HEFMI validates.  What the data audit argues for:
#            HEFMI's fNIRS target reproduces itself poorly (split-half 0.207)
#            and its HbO/HbR correlation is POSITIVE, meaning the signal is
#            dominated by systemic blood flow rather than an evoked response.
#            hBCI scores 0.440 and negative on the same two measures.
# Everything else is unchanged, so the two runs are directly comparable.
# The clinical analysis always uses HEFMI, whichever way this is set.
FNIRS_TRAIN = "HEFMI"

# hBCI recorded motor imagery AND mental arithmetic from the same 29 people.
# True pools both, roughly doubling trials per person. A person keeps one
# subject id across the two tasks, so cross-subject splits stay honest.
HBCI_POOL_MI_MA = True

# ---- MEMORY LIMIT ------------------------------------------------------------
# Each EEG trial is 30 s at 200 Hz = 6000 samples. ds002739 has ~450 stimulus
# events per run, so 24 subjects x 2 runs is ~21,400 trials -> 27.8 GB in ONE
# array, which does not fit in 32 GB of RAM.
#
# This caps the trials taken per fMRI run, sampled EVENLY across the run so the
# whole session is still covered. It also balances the two modalities: without
# it the fMRI trials outnumber the fNIRS trials roughly 6:1 and the shared
# encoder is dominated by fMRI.
#
# 100 per run  ~=  4,800 trials  ~=  6.7 GB   (fits, and is what runs by default)
# None         ->  every trial. Needs ~64 GB of RAM.
MAX_FMRI_TRIALS_PER_RUN = 100

# ---- fMRI DATA (roles are fixed) ---------------------------------------------
# fMRI supervision. ds002336, 10 subjects. Contains "derivatives/".
PATH_TRAIN_FMRI_DS002336     = r"G:\research_bio_medical\dataset\OpenNeuro ds002336"

# External fMRI, never trained on. ds002739, 24 subjects. Contains "sub-01/".
PATH_EXTERNAL_FMRI_DS002739  = r"G:\research_bio_medical\dataset\External fMRI validation OpenNeuro ds002739"

# ---- CLINICAL DATA (section 16) ----------------------------------------------
# The SAME HEFMI folder as above, deliberately not written out a second time --
# two copies of one path drift apart the moment you edit one of them.
#
# It keeps its own name because section 16 asks a different question: 17 healthy
# controls against 20 haemorrhage patients, measured directly from the
# recordings with no model involved. Set it to None to skip that section, or to
# another folder to point it elsewhere, without disturbing the training set.
PATH_CLINICAL_HEFMI = PATH_FNIRS_HEFMI

# =============================================================================
#                        >>>  CHOOSE HOW TO RUN  <<<
#
#   True   SMOKE TEST -- few subjects, 2 epochs, ~10 min.
#                        Proves every cell runs. NUMBERS ARE NOISY.
#   False  FULL RUN    -- all subjects, 30 epochs, 1000 permutations.
#                        Hours. These are the numbers you report.
# =============================================================================
QUICK_TEST = True
# =============================================================================
''')

# ===========================================================================
md("## 2 · Setup")

co(r'''
import sys, subprocess

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

for _mod, _pkg in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                   ("torch", "torch"), ("sklearn", "scikit-learn"),
                   ("nibabel", "nibabel"), ("nilearn", "nilearn"), ("mne", "mne")]:
    try:
        __import__(_mod)
    except ImportError:
        print(f"installing {_pkg} ...")
        _pip(_pkg)

import os, re, json, math, time, random, warnings
from dataclasses import dataclass, asdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps
from scipy import stats
import torch
import torch.nn as nn

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz          # numpy < 2 compatibility

SEED = 1337


def set_determinism(seed=SEED):
    """Put every random source into a known state.

    Called once at import and again before each training run, so a rerun on a
    different machine follows the same sequence. Seeding numpy and torch alone
    is not enough: Python's own `random` drives some library code, cuDNN picks
    convolution algorithms by timing unless told not to, and cuBLAS needs a
    workspace setting before the first CUDA call for reproducible reductions.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False       # benchmark picks by timing


def seeded_loader(ds, batch_size, seed=SEED, shuffle=True):
    """DataLoader whose shuffle order depends only on `seed`.

    Without an explicit generator the shuffle draws from the global torch RNG,
    so the batch order silently depends on how many random numbers were drawn
    earlier in the notebook -- run cells in a different order and the training
    data arrives in a different sequence.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        generator=g if shuffle else None, num_workers=0)


set_determinism()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    tmin_s: float = -2.0
    tmax_s: float = 28.0
    eeg_fs: float = 200.0
    hemo_fs: float = 10.0
    eeg_samples: int = 6000        # 30 s @ 200 Hz
    hemo_samples: int = 300        # 30 s @ 10 Hz
    fmri_vols: int = 15            # 30 s @ TR 2 s
    baseline_s: float = 2.0
    n_folds: int = 5
    n_perm: int = 1000
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    apply_gsr: bool = False        # audited: GSR removes 83% of the evoked response
    # loss weights
    w_main: float = 1.0            # pearson (shape)
    w_hetero: float = 0.1          # gaussian NLL -> uncertainty
    w_smooth: float = 0.05

CFG = Config()
if QUICK_TEST:
    CFG = Config(**{**asdict(CFG), "epochs": 2, "n_perm": 200, "n_folds": 3})

OUT = Path("neurovascnet_v6_outputs"); OUT.mkdir(exist_ok=True)

print(f"numpy {np.__version__} | torch {torch.__version__} | device {DEVICE}")
print()
if QUICK_TEST:
    print("+==============================================================+")
    print("|                     RUNNING: SMOKE TEST                      |")
    print("|  Few subjects, 2 epochs. Proves the notebook runs.           |")
    print("|  THE NUMBERS ARE NOISY -- do not report them.                |")
    print("|  Set QUICK_TEST = False in cell 1 for the real run.          |")
    print("+==============================================================+")
else:
    print("+==============================================================+")
    print("|                       RUNNING: FULL RUN                      |")
    print("|  All subjects, 30 epochs, 1000 permutations.                 |")
    print("|  These are the numbers you report.                           |")
    print("+==============================================================+")
''')

co(r'''
# ---- resolve the dataset paths -----------------------------------------------
# FNIRS_TRAIN decides which cohort trains and which validates. Each entry also
# records WHICH LOADER to use, so the roles can swap without the loading code
# having to guess what it is being handed.
_HEFMI = (PATH_FNIRS_HEFMI, "epo_data", "HEFMI",  "hefmi")
_HBCI  = (PATH_FNIRS_HBCI,  "EEG data", "hBCI",   "hbci")

if FNIRS_TRAIN == "HEFMI":
    _tr, _ex = _HEFMI, _HBCI
elif FNIRS_TRAIN == "hBCI":
    _tr, _ex = _HBCI, _HEFMI
else:
    raise ValueError(f'FNIRS_TRAIN must be "HEFMI" or "hBCI", got {FNIRS_TRAIN!r}')

ROLES = {
    "train_fnirs":    (_tr[0], _tr[1], f"TRAIN    fNIRS  {_tr[2]}"),
    "train_fmri":     (PATH_TRAIN_FMRI_DS002336,    "derivatives", "TRAIN    fMRI   ds002336"),
    "external_fnirs": (_ex[0], _ex[1], f"EXTERNAL fNIRS  {_ex[2]}"),
    "external_fmri":  (PATH_EXTERNAL_FMRI_DS002739, "sub-01",      "EXTERNAL fMRI   ds002739"),
    "clinical":       (PATH_CLINICAL_HEFMI,         "epo_data",    "CLINICAL        ICH vs controls"),
}
FNIRS_LOADER = {"train_fnirs": _tr[3], "external_fnirs": _ex[3]}
print(f"fNIRS configuration: {_tr[2]} trains, {_ex[2]} validates\n")

# A training set and its external partner must never be the same folder.
for _a, _b, _m in (("train_fnirs", "external_fnirs", "fNIRS"),
                   ("train_fmri", "external_fmri", "fMRI")):
    _pa, _pb = ROLES[_a][0], ROLES[_b][0]
    if _pa and _pb and Path(_pa).resolve() == Path(_pb).resolve():
        raise ValueError(f"{_a} and {_b} are the same folder -- the {_m} external "
                         "validation would be testing on training data.")

PATHS = {}
for _key, (_raw, _marker, _label) in ROLES.items():
    if not _raw:
        print(f"  [off ] {_label}")
        PATHS[_key] = None
        continue
    _p = Path(_raw)
    if not _p.exists():
        print(f"  [BAD ] {_label}   path does not exist: {_p}")
        PATHS[_key] = None
    elif not (_p / _marker).exists():
        print(f"  [BAD ] {_label}   no '{_marker}/' inside: {_p}")
        PATHS[_key] = None
    else:
        print(f"  [ok  ] {_label}")
        PATHS[_key] = _p

if not (PATHS["train_fnirs"] or PATHS["train_fmri"]):
    _lines = []
    for _k, (_r, _m, _l) in ROLES.items():
        _lines.append(f"      {_l}\n         path   {_r}\n         needs  {_m}/")
    raise FileNotFoundError(
        "No TRAINING dataset resolved -- nothing to train on.\n\n"
        "  Edit the PATH_* lines in the settings cell. Each one is a full path\n"
        "  to that dataset's folder, and each must contain the folder named\n"
        "  below:\n\n" + "\n".join(_lines))

print(f"\noutputs -> {OUT.resolve()}")

# A fingerprint of the environment. If two machines produce different numbers,
# comparing these lines is the fastest way to find out why.
print("\nenvironment")
print(f"  python  {sys.version.split()[0]}")
print(f"  numpy   {np.__version__}   scipy {__import__('scipy').__version__}")
print(f"  torch   {torch.__version__}   device {DEVICE}")
if DEVICE == "cuda":
    print(f"  gpu     {torch.cuda.get_device_name(0)}")
print(f"  seed    {SEED}   cudnn.deterministic={torch.backends.cudnn.deterministic}"
      f"  benchmark={torch.backends.cudnn.benchmark}")
''')

# ===========================================================================
md(r"""
## 3 · Loaders — all four datasets, in one place

Every loader returns the **same dict**, so the rest of the notebook never cares
which dataset it is holding:

```python
{"eeg": (n, ch, 6000), "hemo": (n, targets, T), "subject_ids": (n,),
 "labels": (n,), "modality": "fnirs" | "fmri", "name": str}
```

Shared preprocessing: 30 s windows, EEG z-scored per trial, haemodynamic signals
baseline-corrected on the 2 s before stimulus. GSR is **off** — the audit
measured that it removes 83% of the evoked response along with the noise.
""")

co(r'''
# ---- shared signal helpers ---------------------------------------------------
EXT_COEF = {760: {"HbO": 1486.59, "HbR": 3843.71},
            850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}
SD_CM = 3.0


def bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def resample_to(x, n):
    return x if x.shape[-1] == n else sps.resample(x, n, axis=-1)


# --- EEG montage crosswalk ----------------------------------------------------
# HEFMI and hBCI use different montages. HEFMI has C3/C4/FC5/CP5; hBCI uses the
# 10-5 intermediate positions (FCC3h, CCP3h ...) and has NO C3 or C4 at all.
# Only 12 labels are shared and not one of them is a motor electrode, so matching
# by name would hand the encoder exactly the wrong channels. Matching by POSITION
# in the array is worse still -- it pairs electrodes that are nowhere near each
# other. Both montages do have known 3-D coordinates, so the honest route is to
# interpolate one onto the other's positions with spherical splines, which is
# what recovers a genuine C3 estimate from FCC3h and CCP3h.

HEFMI_CH = ["FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
            "C4", "T7", "T8", "FC2", "FC5", "FC6", "Pz", "CP5",
            "PO3", "PO4", "Oz", "FP2", "FP1", "FZ", "F3", "F4",
            "F7", "F8", "P3", "P4", "P7", "P8", "O1", "O2"]


def _canon(name):
    """MATLAB-cased labels -> the casing MNE's standard_1005 uses."""
    n = str(name).strip()
    up = n.upper()
    fixed = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz",
             "PZ": "Pz", "OZ": "Oz", "FPZ": "Fpz"}
    return fixed.get(up, n)


def electrode_positions(names):
    """3-D positions for electrode labels.

    Returns None when the labels are missing or any one is unknown -- the fMRI
    datasets carry no electrode names at all, so callers must handle None.
    """
    if not names:
        return None
    try:
        import mne
        pos = mne.channels.make_standard_montage(
            "standard_1005").get_positions()["ch_pos"]
        return np.array([pos[_canon(c)] for c in names])
    except (KeyError, ImportError):
        return None


def interpolation_matrix(src_names, dst_names):
    """Spherical-spline matrix mapping src electrodes onto dst positions.

    None whenever the crosswalk cannot be built -- missing names, an unknown
    label, or MNE unavailable. Callers fall back and say so in the output.
    """
    ps, pdst = electrode_positions(src_names), electrode_positions(dst_names)
    if ps is None or pdst is None:
        return None
    try:
        from mne.channels.interpolation import _make_interpolation_matrix
        r = np.linalg.norm(ps, axis=1).mean()      # project to a unit sphere
        return _make_interpolation_matrix(ps / r, pdst / r)
    except Exception:
        return None


def subsample_onsets(onsets, cap=None):
    """Keep at most `cap` onsets, spread EVENLY over the run.

    Taking the first N would sample only the start of the session and inherit
    whatever drift or habituation lives there.
    """
    cap = MAX_FMRI_TRIALS_PER_RUN if cap is None else cap
    onsets = np.asarray(onsets)
    if not cap or len(onsets) <= cap:
        return onsets
    idx = np.unique(np.linspace(0, len(onsets) - 1, cap).round().astype(int))
    return onsets[idx]


def stack_free(arrays, n_ch=None):
    """np.stack that does not hold two full copies at once.

    np.stack allocates the whole output while the input list is still alive,
    so peak memory is 2x the result. These arrays run to gigabytes, so instead
    preallocate once and drop each input as it is copied in.
    """
    if not arrays:
        return None
    a0 = arrays[0]
    rows = a0.shape[0] if n_ch is None else n_ch
    out = np.empty((len(arrays), rows) + a0.shape[1:], dtype=np.float32)
    for i in range(len(arrays)):
        out[i] = arrays[i][:rows]
        arrays[i] = None          # free as we go
    arrays.clear()
    return out


def zscore_trial(seg):
    return ((seg - seg.mean(-1, keepdims=True)) /
            (seg.std(-1, keepdims=True) + 1e-6)).astype(np.float32)


def baseline_correct(seg, n_base):
    return seg - seg[:, :n_base].mean(-1, keepdims=True)


def beer_lambert(lo_wl, hi_wl):
    """Raw 760/850 nm intensities -> HbO, HbR in micromolar."""
    def od(x):
        base = np.median(x, axis=-1, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        return -np.log(np.clip(x / base, 1e-9, None))
    a1, a2 = od(lo_wl), od(hi_wl)
    L1, L2 = SD_CM * DPF[760], SD_CM * DPF[850]
    E = np.array([[EXT_COEF[760]["HbO"] * L1, EXT_COEF[760]["HbR"] * L1],
                  [EXT_COEF[850]["HbO"] * L2, EXT_COEF[850]["HbR"] * L2]])
    Ei = np.linalg.inv(E)
    return ((Ei[0, 0] * a1 + Ei[0, 1] * a2) * 1e6,
            (Ei[1, 0] * a1 + Ei[1, 1] * a2) * 1e6)


def pool_hbo_hbr(hemo):
    """(n, 2k, T) -> (n, 2, T): mean HbO and mean HbR.

    The only montage-independent way to compare two fNIRS datasets whose optode
    layouts have no published crosswalk. Used for external validation ONLY.
    """
    k = hemo.shape[1] // 2
    return np.stack([hemo[:, :k].mean(1), hemo[:, k:].mean(1)], axis=1)
''')

co(r'''
# ---- 1/4  TRAIN fNIRS: HEFMI -------------------------------------------------
# EEG channel names live in code/EpoDataMaking/MakeDatasetFromRaw.m, because the
# .mat stores them as a MATLAB string object SciPy decodes to an opaque blob.
HEFMI_EEG_CHANNELS = [
    "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
    "C4",  "T7",  "T8",  "FC2", "FC5", "FC6", "Pz", "CP5",
    "PO3", "PO4", "Oz",  "FP2", "FP1", "FZ",  "F3", "F4",
    "F7",  "F8",  "P3",  "P4",  "P7",  "P8",  "O1", "O2",
]


def read_probe(root):
    """fNIRS optode geometry from any .nirs file (SD struct)."""
    hits = sorted(Path(root).rglob("*.nirs"))
    if not hits:
        return None
    try:
        SD = sio.loadmat(hits[0], squeeze_me=True, struct_as_record=False)["SD"]
    except Exception:
        return None
    ml = np.asarray(SD.MeasList)
    return {"sources": int(np.asarray(SD.SrcPos).shape[0]),
            "detectors": int(np.asarray(SD.DetPos).shape[0]),
            "pairs": len({(int(a), int(b)) for a, b in ml[:, :2]}),
            "wavelengths": sorted({int(w) for w in np.atleast_1d(SD.Lambda).ravel()}),
            "unit": str(getattr(SD, "SpatialUnit", "?"))}


def load_hefmi(root, cfg=CFG, max_people=None):
    n_base = int(cfg.baseline_s * cfg.hemo_fs)
    eeg_l, hemo_l, lab_l, sid_l, audit = [], [], [], [], []
    people = {}
    for group in ("Normals", "Patients"):
        gdir = root / "epo_data" / group / "EEG"
        if not gdir.exists():
            continue
        for sd in sorted(p for p in gdir.iterdir() if p.is_dir()):
            person = f"{group}_{sd.name}"        # Normals/subject1 != Patients/subject1
            if max_people and person not in people and len(people) >= max_people:
                continue
            for ef in sorted(sd.glob("*_epo.mat")):
                nf = root / "epo_data" / group / "NIRS" / sd.name / ef.name
                if not nf.exists():
                    audit.append({"person": person, "file": ef.name,
                                  "status": "no fNIRS partner", "trials": 0}); continue
                try:
                    e = sio.loadmat(ef, squeeze_me=True, struct_as_record=False)["epo"]
                    n = sio.loadmat(nf, squeeze_me=True, struct_as_record=False)["epo"]
                    X = np.asarray(e.X, float)
                    oxy, dxy = np.asarray(n.oxy.X, float), np.asarray(n.doxy.X, float)
                    y = np.asarray(e.y).ravel()
                except Exception as exc:
                    audit.append({"person": person, "file": ef.name,
                                  "status": f"read fail: {type(exc).__name__}",
                                  "trials": 0}); continue
                sid = people.setdefault(person, len(people))
                k = min(X.shape[2], oxy.shape[2], dxy.shape[2], len(y))
                n_ok = 0
                for t in range(k):
                    eeg_l.append(zscore_trial(resample_to(X[:, :, t].T, cfg.eeg_samples)))
                    so = baseline_correct(resample_to(oxy[:, :, t].T, cfg.hemo_samples), n_base)
                    sr = baseline_correct(resample_to(dxy[:, :, t].T, cfg.hemo_samples), n_base)
                    hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
                    lab_l.append(int(y[t])); sid_l.append(sid); n_ok += 1
                audit.append({"person": person, "file": ef.name,
                              "status": "ok" if n_ok else "no trials", "trials": n_ok})
    if not eeg_l:
        return None
    return {"name": "HEFMI", "modality": "fnirs",
            "ch_names": list(HEFMI_CH),
            "eeg": np.stack(eeg_l), "hemo": np.stack(hemo_l),
            "labels": np.asarray(lab_l, np.int64),
            "subject_ids": np.asarray(sid_l, np.int64),
            "people": {v: k for k, v in people.items()},
            "audit": pd.DataFrame(audit)}
''')

co(r'''
# ---- 2/4  EXTERNAL fNIRS: TU Berlin hBCI -------------------------------------
EOG = ("VEOG", "HEOG")


def _runs(path, key=None):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    return np.atleast_1d(m[key if key in keys else keys[0]]).ravel()


def _split_wavelengths(x, clab):
    names = [str(c) for c in np.atleast_1d(clab)]
    lo = {n.replace("lowWL", ""): i for i, n in enumerate(names) if "lowWL" in n}
    hi = {n.replace("highWL", ""): i for i, n in enumerate(names) if "highWL" in n}
    pairs = sorted(set(lo) & set(hi))
    return (np.stack([x[:, lo[p]] for p in pairs]),
            np.stack([x[:, hi[p]] for p in pairs]))


def load_hbci(root, cfg=CFG, subjects=range(1, 30), task="MI"):
    eeg_dir, nirs_dir = root / "EEG data", root / "NRI data" / "NIRS_01-29"
    n_base = int(cfg.baseline_s * cfg.hemo_fs)
    eeg_l, hemo_l, lab_l, sid_l, audit = [], [], [], [], []

    def subj_dir(n):
        for g in sorted(eeg_dir.glob("EEG_*")):
            d = g / f"subject {n:02d}" / "with occular artifact"
            if d.exists():
                return d
        return None

    for s in subjects:
        ed, nd = subj_dir(s), nirs_dir / f"subject {s:02d}"
        if ed is None or not nd.exists():
            audit.append({"subject": s, "status": "missing", "trials": 0}); continue
        try:
            er_all, em_all = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
            nr_all, nm_all = _runs(nd / "cnt.mat", "cnt"), _runs(nd / "mrk.mat", "mrk")
        except Exception as exc:
            audit.append({"subject": s, "status": f"read fail: {exc}", "trials": 0}); continue
        n_ok = 0
        for r in range(min(er_all.size, nr_all.size)):
            er, nr = er_all[r], nr_all[r]
            if not str(getattr(er, "title", "")).upper().startswith(task):
                continue
            e_fs = float(np.atleast_1d(er.fs).ravel()[0])
            clab = [str(c) for c in np.atleast_1d(er.clab)]
            keep = [i for i, c in enumerate(clab) if c not in EOG]
            ch_names = [clab[i] for i in keep]
            ex = np.asarray(er.x, float)[:, keep].T
            n_fs = float(np.atleast_1d(nr.fs).ravel()[0])
            lo, hi = _split_wavelengths(np.asarray(nr.x, float), nr.clab)
            hbo, hbr = beer_lambert(lo, hi)
            hbo, hbr = bandpass(hbo, n_fs, .01, .2), bandpass(hbr, n_fs, .01, .2)
            em = em_all[r] if r < em_all.size else None
            nm = nm_all[r] if r < nm_all.size else None
            if em is None or nm is None:
                continue
            e_on = np.atleast_1d(np.asarray(em.time, float)) / 1000.0
            n_on = np.atleast_1d(np.asarray(nm.time, float)) / 1000.0
            yy = np.asarray(em.y)
            labels = np.argmax(yy, 0) if yy.ndim == 2 else np.asarray(yy).ravel()
            for k in range(min(e_on.size, n_on.size, labels.size)):
                a_e, b_e = (int(round((e_on[k] + cfg.tmin_s) * e_fs)),
                            int(round((e_on[k] + cfg.tmax_s) * e_fs)))
                a_n, b_n = (int(round((n_on[k] + cfg.tmin_s) * n_fs)),
                            int(round((n_on[k] + cfg.tmax_s) * n_fs)))
                if a_e < 0 or b_e > ex.shape[1] or a_n < 0 or b_n > hbo.shape[1]:
                    continue
                eeg_l.append(zscore_trial(resample_to(ex[:, a_e:b_e], cfg.eeg_samples)))
                so = baseline_correct(resample_to(hbo[:, a_n:b_n], cfg.hemo_samples), n_base)
                sr = baseline_correct(resample_to(hbr[:, a_n:b_n], cfg.hemo_samples), n_base)
                hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
                lab_l.append(int(labels[k])); sid_l.append(int(s)); n_ok += 1
        audit.append({"subject": s, "status": "ok" if n_ok else "no trials", "trials": n_ok})
    if not eeg_l:
        return None
    return {"name": "TU Berlin hBCI", "modality": "fnirs",
            "ch_names": ch_names,
            "eeg": np.stack(eeg_l), "hemo": np.stack(hemo_l),
            "labels": np.asarray(lab_l, np.int64),
            "subject_ids": np.asarray(sid_l, np.int64),
            "audit": pd.DataFrame(audit)}
''')

co(r'''
# ---- 3/4 and 4/4  fMRI: ds002336 (train) and ds002739 (external) -------------
# Both use the SAME Harvard-Oxford 48-ROI atlas, so their targets are directly
# comparable ROI by ROI -- which is what makes the fMRI external arm valid.
TR_S = 2.0
_MASKER = None


def _ho_masker():
    global _MASKER
    if _MASKER is None:
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        _MASKER = (NiftiLabelsMasker(labels_img=atlas.maps, standardize=False, verbose=0),
                   [str(x) for x in atlas.labels[1:]])
    return _MASKER


def _roi(bold):
    """ROI timeseries aligned to the FULL 48-label atlas.

    NiftiLabelsMasker silently drops regions with no voxels in a given subject's
    field of view, so different subjects come back with different column counts.
    That breaks stacking, and worse, it would misalign ds002336 against ds002739
    in the external comparison. Map every returned column back to its atlas
    index and leave missing regions as zeros, so column j is ALWAYS region j.
    """
    masker, labels = _ho_masker()
    ts = masker.fit_transform(str(bold))
    kept = [int(x) for x in getattr(masker, "labels_", [])]
    kept = [k for k in kept if k != 0]                 # 0 is background
    full = np.zeros((ts.shape[0], len(labels)), dtype=float)
    if len(kept) == ts.shape[1]:
        for j, lab in enumerate(kept):
            if 1 <= lab <= len(labels):
                full[:, lab - 1] = ts[:, j]
    elif ts.shape[1] == len(labels):
        full = ts                                       # already aligned
    else:
        full[:, :ts.shape[1]] = ts[:, :len(labels)]     # last resort, still fixed width
    return full, labels


def _detrend(ts):
    """Remove cubic drift. Scanner drift alone inflates apparent reliability."""
    n = ts.shape[0]
    t = np.linspace(-1, 1, n)
    X = np.column_stack([np.ones(n), t, t ** 2, t ** 3])
    beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
    return ts - X @ beta


def load_ds002336(root, cfg=CFG, max_subjects=None):
    """EEG-fMRI aligned by R128 scanner triggers: volume k IS EEG sample R128[k]."""
    import mne
    subs = sorted(d.name for d in root.glob("sub-*") if d.is_dir())[:max_subjects]
    tasks = ("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF")
    eeg_l, hemo_l, sid_l, audit = [], [], [], []
    for si, sub in enumerate(subs):
        for task in tasks:
            vhdr = root / "derivatives" / sub / "eeg_pp" / f"{sub}_task-{task}_eeg_pp.vhdr"
            bolds = sorted(root.rglob(f"{sub}_task-{task}*_bold.nii.gz"))
            ev = root / f"task-{task}_events.tsv"
            if not (vhdr.exists() and bolds and ev.exists()):
                continue
            try:
                raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
                eeg = raw.get_data(picks=mne.pick_types(raw.info, eeg=True)) * 1e6
                fs = float(raw.info["sfreq"])
                trig = np.array([o for o, d in zip(raw.annotations.onset,
                                                   raw.annotations.description)
                                 if "R128" in str(d)])
                trig = (trig * fs).astype(int)
                if trig.size < cfg.fmri_vols:
                    raise ValueError(f"only {trig.size} R128 triggers")
                ts = _detrend(_roi(bolds[0])[0])
                df = pd.read_csv(ev, sep="\t")
                if "trial_type" in df.columns:
                    df = df[~df.trial_type.astype(str).str.lower().str.contains("rest")]
                n_ok = 0
                for o in subsample_onsets(
                        pd.to_numeric(df["onset"], errors="coerce").dropna().values):
                    v = int(round(o / TR_S))
                    if v < 0 or v + cfg.fmri_vols > ts.shape[0] or v >= trig.size:
                        continue
                    s0 = trig[v]
                    if s0 + cfg.eeg_samples > eeg.shape[1]:
                        continue
                    seg_h = ts[v:v + cfg.fmri_vols].T
                    hemo_l.append((seg_h - seg_h[:, :1]).astype(np.float32))
                    eeg_l.append(zscore_trial(eeg[:, s0:s0 + cfg.eeg_samples]))
                    sid_l.append(si); n_ok += 1
                audit.append({"subject": sub, "task": task, "trials": n_ok,
                              "triggers": int(trig.size), "status": "ok" if n_ok else "none"})
            except Exception as exc:
                audit.append({"subject": sub, "task": task, "trials": 0,
                              "status": f"skip: {type(exc).__name__}"})
    if not eeg_l:
        return None
    nc = min(e.shape[0] for e in eeg_l)
    return {"name": "ds002336", "modality": "fmri",
            "eeg": stack_free(eeg_l, nc), "hemo": stack_free(hemo_l),
            "labels": np.zeros(len(sid_l), np.int64),
            "subject_ids": np.asarray(sid_l, np.int64),
            "audit": pd.DataFrame(audit)}


def load_ds002739(root, cfg=CFG, max_subjects=None):
    """EEG-fMRI with per-run events.tsv (no R128 in this dataset)."""
    subs = sorted(d.name for d in root.glob("sub-*") if d.is_dir())[:max_subjects]
    eeg_l, hemo_l, sid_l, audit = [], [], [], []
    for si, sub in enumerate(subs):
        for run in ("01", "02"):
            mat = root / sub / "EEG" / f"EEG_data_{sub}_run-{run}.mat"
            bold = root / sub / "func" / f"{sub}_task-main_run-{run}_bold.nii.gz"
            ev = root / sub / "func" / f"{sub}_task-main_run-{run}_events.tsv"
            if not (mat.exists() and bold.exists() and ev.exists()):
                continue
            try:
                m = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)["EEGdata"]
                eeg = np.asarray(m.Y, float)
                fs = float(np.asarray(m.fd).ravel()[0])
                ts = _detrend(_roi(bold)[0])
                df = pd.read_csv(ev, sep="\t")
                if "trial_type" in df.columns:
                    df = df[df.trial_type.astype(str).str.contains("stim", case=False)]
                n_ok = 0
                for o in subsample_onsets(
                        pd.to_numeric(df["onset"], errors="coerce").dropna().values):
                    v, s0 = int(round(o / TR_S)), int(round(o * fs))
                    if v < 0 or v + cfg.fmri_vols > ts.shape[0]:
                        continue
                    if s0 < 0 or s0 + cfg.eeg_samples > eeg.shape[1]:
                        continue
                    seg_h = ts[v:v + cfg.fmri_vols].T
                    hemo_l.append((seg_h - seg_h[:, :1]).astype(np.float32))
                    eeg_l.append(zscore_trial(eeg[:, s0:s0 + cfg.eeg_samples]))
                    sid_l.append(si); n_ok += 1
                audit.append({"subject": sub, "run": run, "trials": n_ok,
                              "status": "ok" if n_ok else "none"})
            except Exception as exc:
                audit.append({"subject": sub, "run": run, "trials": 0,
                              "status": f"skip: {type(exc).__name__}"})
    if not eeg_l:
        return None
    nc = min(e.shape[0] for e in eeg_l)
    return {"name": "ds002739", "modality": "fmri",
            "eeg": stack_free(eeg_l, nc), "hemo": stack_free(hemo_l),
            "labels": np.zeros(len(sid_l), np.int64),
            "subject_ids": np.asarray(sid_l, np.int64),
            "audit": pd.DataFrame(audit)}
''')

# ===========================================================================
md("### Load all four now")

co(r'''
LIMIT_PEOPLE = 6 if QUICK_TEST else None
LIMIT_FMRI   = 3 if QUICK_TEST else None

DATA = {}
t0 = time.time()

def load_fnirs(role):
    """Load whichever fNIRS cohort FNIRS_TRAIN assigned to this role."""
    path, which = PATHS[role], FNIRS_LOADER[role]
    print(f"loading {role.replace('_', ' ').upper()} ({which}) ...", flush=True)
    if which == "hefmi":
        d = load_hefmi(path, max_people=LIMIT_PEOPLE)
        probe = read_probe(path)
        if probe:
            print(f"   montage: {probe['sources']} sources, {probe['detectors']} "
                  f"detectors, {probe['pairs']} S-D pairs, "
                  f"{probe['wavelengths']} nm ({probe['unit']})")
        return d
    subs = range(1, 7) if QUICK_TEST else range(1, 30)
    if not HBCI_POOL_MI_MA:
        return load_hbci(path, subjects=subs, task="MI")

    # hBCI recorded two paradigms from the SAME 29 people: motor imagery and
    # mental arithmetic. Pooling roughly doubles the trials per person.
    #
    # The subject id is the real subject number in both, so a person keeps ONE
    # id across the two tasks and cross-subject CV still splits by person. This
    # is deliberate: giving MA its own ids would make 58 "subjects" out of 29
    # people and put the same person in train and test.
    mi = load_hbci(path, subjects=subs, task="MI")
    ma = load_hbci(path, subjects=subs, task="MA")
    if mi is None:
        return ma
    if ma is None:
        return mi
    if mi["eeg"].shape[1:] != ma["eeg"].shape[1:] or \
       mi["hemo"].shape[1:] != ma["hemo"].shape[1:]:
        print("   MI and MA shapes differ -- using MI only")
        return mi

    # keep the two paradigms' labels distinct; nothing should silently treat
    # "left hand" and "mental arithmetic" as the same class
    off = int(mi["labels"].max()) + 1
    d = dict(mi)
    d["eeg"] = np.concatenate([mi["eeg"], ma["eeg"]], 0)
    d["hemo"] = np.concatenate([mi["hemo"], ma["hemo"]], 0)
    d["labels"] = np.concatenate([mi["labels"], ma["labels"] + off], 0)
    d["subject_ids"] = np.concatenate([mi["subject_ids"], ma["subject_ids"]], 0)
    d["task_ids"] = np.concatenate([np.zeros(len(mi["eeg"]), np.int64),
                                    np.ones(len(ma["eeg"]), np.int64)], 0)
    d["name"] = "TU Berlin hBCI (MI+MA)"
    n_people = len(np.unique(d["subject_ids"]))
    print(f"   pooled MI ({len(mi['eeg'])} trials) + MA ({len(ma['eeg'])} trials) "
          f"= {len(d['eeg'])} trials from {n_people} people")
    return d


if PATHS["train_fnirs"]:
    DATA["train_fnirs"] = load_fnirs("train_fnirs")

if PATHS["train_fmri"]:
    print("loading TRAIN fMRI (ds002336) ...", flush=True)
    DATA["train_fmri"] = load_ds002336(PATHS["train_fmri"], max_subjects=LIMIT_FMRI)

if PATHS["external_fnirs"]:
    DATA["external_fnirs"] = load_fnirs("external_fnirs")

if PATHS["external_fmri"]:
    print("loading EXTERNAL fMRI (ds002739) ...", flush=True)
    DATA["external_fmri"] = load_ds002739(PATHS["external_fmri"], max_subjects=LIMIT_FMRI)

DATA = {k: v for k, v in DATA.items() if v is not None}
print(f"\nloaded {len(DATA)} datasets in {(time.time()-t0)/60:.1f} min")
''')

# ===========================================================================
md("## 4 · Data summary")

co(r'''
def split_half_reliability(hemo, sids, n_rep=50):
    """Does the target reproduce itself? Caps any model at sqrt(reliability)."""
    rng = np.random.default_rng(SEED)
    out = []
    for s in np.unique(sids):
        X = hemo[sids == s]
        if len(X) < 4:
            continue
        rs = []
        for _ in range(n_rep):
            p = rng.permutation(len(X))
            a = X[p[:len(X) // 2]].mean(0).ravel()
            b = X[p[len(X) // 2:]].mean(0).ravel()
            if a.std() > 0 and b.std() > 0:
                rs.append(np.corrcoef(a, b)[0, 1])
        if rs:
            out.append(np.mean(rs))
    return float(np.mean(out)) if out else float("nan")


def hbo_hbr_sign(hemo):
    """Negative = physiological. Positive = systemic (heartbeat, scalp flow)."""
    k = hemo.shape[1] // 2
    return float(np.nanmedian([np.corrcoef(t[:k].mean(0), t[k:].mean(0))[0, 1]
                               for t in hemo]))


rows = []
for role, d in DATA.items():
    rel = split_half_reliability(d["hemo"], d["subject_ids"])
    rows.append({
        "role": role, "dataset": d["name"], "modality": d["modality"],
        "subjects": int(len(np.unique(d["subject_ids"]))),
        "trials": int(len(d["eeg"])),
        "eeg_ch": int(d["eeg"].shape[1]),
        "targets": int(d["hemo"].shape[1]),
        "reliability": round(rel, 4),
        "ceiling": round(math.sqrt(max(rel, 0)), 4),
        "hbo_hbr": (round(hbo_hbr_sign(d["hemo"]), 3)
                    if d["modality"] == "fnirs" else None),
    })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT / "data_summary.csv", index=False)
display(summary_df)

# Keep each set's own ceiling so later sections can express a correlation as a
# fraction of what is achievable ON THAT SET. Comparing a raw r measured on a
# noisy target against one measured on a clean target is meaningless.
TARGET_QUALITY = {r["role"]: {"dataset": r["dataset"],
                              "reliability": r["reliability"],
                              "ceiling": r["ceiling"],
                              "hbo_hbr": r["hbo_hbr"]} for r in rows}

print("reliability = does the target reproduce itself; ceiling = sqrt(reliability)")
print("hbo_hbr     = must be NEGATIVE for a real evoked response")
for r in rows:
    if r["hbo_hbr"] is not None and r["hbo_hbr"] > 0:
        print(f"  !! {r['dataset']}: HbO/HbR is {r['hbo_hbr']:+.3f} -- "
              "systemic-dominated, not brain-driven")
''')

# ===========================================================================
md(r"""
## 5 · Metrics

**The no-EEG baseline is the number that matters.** Trials from one person share
a mean, so a model that learns only *"which subject is this"* scores well while
knowing nothing about the brain. Subject-demeaning drives that baseline to
exactly zero; whatever survives is the EEG contribution.
""")

co(r'''
def _center(x):
    x = x.reshape(len(x), -1).astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def mean_corr(pred, targ):
    return float((_center(pred) * _center(targ)).sum(1).mean())


def perm_null(pred, targ, n_perm=None):
    """Exact permutation null in one matmul: G holds every pairing already,
    so each shuffle is a gather rather than a recomputation."""
    n_perm = n_perm or CFG.n_perm
    rng = np.random.default_rng(SEED)
    G = _center(pred) @ _center(targ).T
    obs = float(np.diag(G).mean())
    idx = np.arange(len(G))
    null = np.array([G[idx, rng.permutation(len(G))].mean() for _ in range(n_perm)])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1))


def constant_baseline(train_targ, test_targ):
    """Ignores EEG entirely: predicts the training mean for every trial."""
    return mean_corr(np.repeat(train_targ.mean(0, keepdims=True), len(test_targ), 0),
                     test_targ)


def demean_by_subject(y, sids):
    out = y.astype(np.float64).copy()
    for s in np.unique(sids):
        m = sids == s
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def strata(d, mask=None):
    """Grouping used for demeaning: subject, or subject x task when pooled.

    Demeaning by subject alone removes "which person is this". It does NOT
    remove "which task is this". With MI and MA pooled, the two paradigms have
    different mean haemodynamic responses, so a model that only classifies the
    task still scores a correlation -- the same artefact as the no-EEG baseline,
    one level up. Grouping by subject AND task removes that route.
    """
    sids = d["subject_ids"] if mask is None else d["subject_ids"][mask]
    tids = d.get("task_ids")
    if tids is None:
        return np.asarray(sids)
    tids = tids if mask is None else tids[mask]
    return np.asarray(sids).astype(np.int64) * 1000 + np.asarray(tids).astype(np.int64)


def subset(d, mask):
    """Slice every per-trial array in a dataset together.

    Spreading a `{**d, "eeg": ...}` literal across call sites silently kept
    task_ids and labels at full length while eeg was sliced, so any later use
    of them was misaligned.
    """
    out = dict(d)
    n = len(d["eeg"])
    for k, v in d.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[mask]
    return out
''')

# ===========================================================================
md(r"""
## 6 · Model — one shared trunk, two heads

```
EEG (n_ch, 6000)
      |
      v  per-dataset adapter        1x1 conv, n_ch -> width
      |                             (only because montages differ)
      v  SHARED TRUNK               <-- the cross-modal part
      |
      +--> fNIRS head  -> (180, 300)   mean + log-variance
      +--> fMRI  head  -> (48, 15)     mean + log-variance
```

Only the trunk is shared. A held-out subject gets the **average** subject
embedding, so cross-subject evaluation cannot leak identity.

The log-variance head is what the abstention layer reads, and it is trained by
the Gaussian NLL term in the loss.
""")

co(r'''
class SharedEncoder(nn.Module):
    """Per-dataset input adapters -> one shared convolutional trunk."""

    def __init__(self, in_channels: dict, width=64, emb=8, n_subjects=64):
        super().__init__()
        self.adapters = nn.ModuleDict(
            {k: nn.Conv1d(c, width, 1) for k, c in in_channels.items()})
        self.trunk = nn.Sequential(
            nn.Conv1d(width, width, 25, 5, 12), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 15, 4, 7),  nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 9, 3, 4),   nn.BatchNorm1d(width), nn.GELU(),
            nn.AdaptiveAvgPool1d(16))
        self.subj = nn.Embedding(n_subjects, emb)
        nn.init.zeros_(self.subj.weight)
        self.out_dim = width * 16 + emb

    def forward(self, x, dataset_key, sid=None):
        h = self.trunk(self.adapters[dataset_key](x)).flatten(1)
        e = (self.subj(sid) if sid is not None
             else self.subj.weight.mean(0).expand(len(x), -1))
        return torch.cat([h, e], 1)


class HemoHead(nn.Module):
    """One modality's output: mean and log-variance over (channels, time)."""

    def __init__(self, in_dim, n_ch, n_t, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.3))
        self.mean = nn.Linear(hidden, n_ch * n_t)
        self.logvar = nn.Linear(hidden, n_ch * n_t)
        nn.init.zeros_(self.logvar.weight); nn.init.zeros_(self.logvar.bias)
        self.shape = (n_ch, n_t)

    def forward(self, z):
        h = self.body(z)
        return (self.mean(h).view(len(z), *self.shape),
                self.logvar(h).view(len(z), *self.shape).clamp(-8, 8))


class NeuroVascNet(nn.Module):
    def __init__(self, in_channels: dict, head_shapes: dict, n_subjects=64):
        super().__init__()
        self.encoder = SharedEncoder(in_channels, n_subjects=n_subjects)
        self.heads = nn.ModuleDict(
            {m: HemoHead(self.encoder.out_dim, c, t) for m, (c, t) in head_shapes.items()})

    def forward(self, x, dataset_key, modality, sid=None):
        return self.heads[modality](self.encoder(x, dataset_key, sid))


# ---- loss --------------------------------------------------------------------
def pearson_loss(pred, target, eps=1e-6):
    """Scale-invariant: targets SHAPE, not amplitude. Amplitude here is dominated
    by systemic drift, which is exactly what we do not want to fit."""
    p = pred - pred.mean(-1, keepdim=True)
    t = target - target.mean(-1, keepdim=True)
    p = p / (p.norm(dim=-1, keepdim=True) + eps)
    t = t / (t.norm(dim=-1, keepdim=True) + eps)
    return (1.0 - (p * t).sum(-1)).mean()


def hetero_loss(mean, target, logvar):
    return (torch.exp(-logvar) * (mean - target) ** 2 + logvar).mean()


def smooth_loss(pred):
    d = pred[..., 1:] - pred[..., :-1]
    return (d ** 2).mean()


def compute_loss(mean, logvar, target, cfg=CFG):
    terms = {"pearson": pearson_loss(mean, target),
             "hetero": hetero_loss(mean, target, logvar),
             "smooth": smooth_loss(mean)}
    total = (cfg.w_main * terms["pearson"] + cfg.w_hetero * terms["hetero"]
             + cfg.w_smooth * terms["smooth"])
    return total, {k: float(v) for k, v in terms.items()}
''')

# ===========================================================================
md(r"""
## 7 · Joint training — both modalities into one encoder

Each step draws a batch from **one** dataset and routes it through that dataset's
adapter and its modality's head. Batches alternate, so the shared trunk receives
gradients from fNIRS and fMRI supervision in the same run.

`train_joint(...)` takes a list of datasets — pass one to get a single-modality
model, pass both to get the cross-modal one. **That comparison is claim 1.**
""")

co(r'''
def make_model(datasets, n_subjects=None):
    in_ch = {d["name"]: int(d["eeg"].shape[1]) for d in datasets}
    heads = {}
    for d in datasets:
        heads.setdefault(d["modality"], (int(d["hemo"].shape[1]), int(d["hemo"].shape[2])))
    n_subj = n_subjects or max(int(d["subject_ids"].max()) for d in datasets) + 1
    return NeuroVascNet(in_ch, heads, n_subjects=n_subj).to(DEVICE)


def train_joint(datasets, cfg=CFG, holdout=None, verbose=True, seed=SEED,
                use_adapter=True):
    """Train one encoder on any number of datasets.

    holdout: {dataset_name: array_of_subject_ids} kept out of training.
    seed:    re-seeded per call, so a run does not depend on how many random
             numbers earlier cells happened to consume.
    use_adapter: whether the per-subject embedding receives subject ids during
             training. False makes it dead weight, which is the control needed
             to tell subject adaptation apart from cross-modal supervision.
    """
    set_determinism(seed)
    model = make_model(datasets)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loaders = {}
    for d in datasets:
        m = np.ones(len(d["eeg"]), bool)
        if holdout and d["name"] in holdout:
            m = ~np.isin(d["subject_ids"], holdout[d["name"]])
        ds = torch.utils.data.TensorDataset(
            torch.tensor(d["eeg"][m], dtype=torch.float32),
            torch.tensor(d["hemo"][m], dtype=torch.float32),
            torch.tensor(d["subject_ids"][m], dtype=torch.long))
        loaders[d["name"]] = (seeded_loader(ds, cfg.batch_size), d["modality"])

    # An epoch is as long as the LARGEST dataset; smaller ones cycle. Draining
    # each loader once instead would end every epoch with only the biggest
    # dataset still feeding the trunk -- 3480 fNIRS trials against 384 fMRI is
    # 55 steps versus 6, so the last 49 steps of each epoch would be
    # single-modality and the "joint" model would drift back to one modality.
    #
    # Each step also sums BOTH modalities into ONE backward pass. Stepping the
    # optimiser separately per modality is alternating fine-tuning: the second
    # step partly undoes the first, which is interference, not joint training.
    steps_per_epoch = max(len(dl) for dl, _ in loaders.values())
    model.train()
    for ep in range(cfg.epochs):
        iters = {k: iter(dl) for k, (dl, _) in loaders.items()}
        agg, steps = {}, 0
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            total = 0.0
            for key, (dl, modality) in loaders.items():
                try:
                    be, bh, bs = next(iters[key])
                except StopIteration:
                    iters[key] = iter(dl)                # cycle the smaller set
                    be, bh, bs = next(iters[key])
                _sid = bs.to(DEVICE) if use_adapter else None
                mean, logvar = model(be.to(DEVICE), key, modality, _sid)
                loss, parts = compute_loss(mean, logvar, bh.to(DEVICE), cfg)
                total = total + loss
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
            total.backward(); opt.step()
            steps += 1
        if verbose and (ep + 1) % max(1, cfg.epochs // 3) == 0:
            print(f"   epoch {ep+1:3d}  " +
                  "  ".join(f"{k}={v/max(steps,1):.4f}" for k, v in agg.items()))
    return model


@torch.no_grad()
def predict(model, d, want_var=False):
    """Held-out prediction: sid=None -> average embedding, so no identity leak."""
    model.eval()
    means, lvs = [], []
    for i in range(0, len(d["eeg"]), 256):
        b = torch.tensor(d["eeg"][i:i + 256], dtype=torch.float32).to(DEVICE)
        m, lv = model(b, d["name"], d["modality"], None)
        means.append(m.cpu().numpy()); lvs.append(lv.cpu().numpy())
    return (np.concatenate(means), np.concatenate(lvs)) if want_var else np.concatenate(means)
''')

# ===========================================================================
md("## 8 · Internal evaluation — cross-subject on the training sets")

co(r'''
def cross_subject_cv(datasets, cfg=CFG):
    """Subject-wise folds. Every fold trains a fresh JOINT model."""
    primary = datasets[0]
    subs = np.unique(primary["subject_ids"])
    folds = np.array_split(np.random.default_rng(SEED).permutation(subs),
                           min(cfg.n_folds, len(subs)))
    rows, P, T = [], [], []
    for fi, te_subs in enumerate(folds):
        model = train_joint(datasets, cfg, holdout={primary["name"]: te_subs},
                            verbose=False)
        te = np.isin(primary["subject_ids"], te_subs)
        sub = subset(primary, te)
        pred = predict(model, sub)
        r_raw = mean_corr(pred, sub["hemo"])
        r_const = constant_baseline(primary["hemo"][~te], sub["hemo"])
        pd_ = demean_by_subject(pred, sub["subject_ids"])
        td_ = demean_by_subject(sub["hemo"], sub["subject_ids"])
        # and again removing task means, which subject demeaning leaves behind
        g = strata(sub)
        pdt = demean_by_subject(pred, g)
        tdt = demean_by_subject(sub["hemo"], g)
        r_task = mean_corr(pdt, tdt)
        rows.append({"fold": fi, "test_subjects": len(te_subs),
                     "trials": int(te.sum()), "r_raw": r_raw,
                     "r_no_eeg_baseline": r_const,
                     "r_demeaned": mean_corr(pd_, td_),
                     "r_demeaned_subject_x_task": r_task})
        P.append(pdt); T.append(tdt)
        print(f"  fold {fi}: raw {r_raw:+.4f} | no-EEG {r_const:+.4f} | "
              f"demeaned {rows[-1]['r_demeaned']:+.4f} | "
              f"subject x task {r_task:+.4f}")
    obs, p = perm_null(np.concatenate(P), np.concatenate(T))
    return pd.DataFrame(rows), {"r_demeaned": obs, "p": p}


TRAIN_SETS = [DATA[k] for k in ("train_fnirs", "train_fmri") if k in DATA]
print(f"training on: {', '.join(d['name'] + ' (' + d['modality'] + ')' for d in TRAIN_SETS)}")
print()
cv_tables, cv_summaries = {}, {}
for d in TRAIN_SETS:
    print(f"--- cross-subject CV, evaluating on {d['name']}")
    others = [x for x in TRAIN_SETS if x is not d]
    tbl, summ = cross_subject_cv([d] + others)
    cv_tables[d["name"]], cv_summaries[d["name"]] = tbl, summ
    print(f"    demeaned r = {summ['r_demeaned']:+.4f}   p = {summ['p']:.4f}\n")
    tbl.to_csv(OUT / f"cv_{d['name'].replace(' ', '_')}.csv", index=False)
''')

# ===========================================================================
md(r"""
## 9 · CLAIM 1 — does cross-modal supervision beat single-modality?

This is the thesis, tested directly. Same trunk, same budget, same evaluation —
the **only** difference is whether the encoder also received fMRI supervision.

`train_joint([fnirs])` vs `train_joint([fnirs, fmri])`. Several seeds; the
confidence interval must exclude zero **and** every seed must agree.
""")

co(r'''
def claim1_cross_modal(cfg=CFG):
    if len(TRAIN_SETS) < 2:
        print("need both a fNIRS and an fMRI training set -- skipped")
        return None, None
    primary = TRAIN_SETS[0]
    subs = np.unique(primary["subject_ids"])
    hold = subs[-max(2, len(subs) // 5):]
    te = np.isin(primary["subject_ids"], hold)
    sub = subset(primary, te)
    td = demean_by_subject(sub["hemo"], sub["subject_ids"])

    rows = []
    for seed in range(2 if QUICK_TEST else 3):
        # SAME seed for both arms, so the pair differs only by supervision
        m_joint = train_joint(TRAIN_SETS, cfg, holdout={primary["name"]: hold},
                              verbose=False, seed=SEED + seed)
        m_single = train_joint([primary], cfg, holdout={primary["name"]: hold},
                               verbose=False, seed=SEED + seed)
        pj = demean_by_subject(predict(m_joint, sub), sub["subject_ids"])
        ps = demean_by_subject(predict(m_single, sub), sub["subject_ids"])

        # Score EACH HELD-OUT SUBJECT separately. Pooling every subject into one
        # correlation throws away the pairing: the two arms saw the same people,
        # so the informative quantity is the per-person difference, not the
        # difference of two pooled averages.
        for s_id in hold:
            m = sub["subject_ids"] == s_id
            if m.sum() < 4:
                continue
            r_j, r_s = mean_corr(pj[m], td[m]), mean_corr(ps[m], td[m])
            rows.append({"seed": seed, "subject": int(s_id), "trials": int(m.sum()),
                         "cross_modal": r_j, "single_modality": r_s,
                         "diff": r_j - r_s})
        _sd = pd.DataFrame(rows)
        _sd = _sd[_sd.seed == seed]
        print(f"  seed {seed}: cross-modal {_sd.cross_modal.mean():+.4f} | "
              f"single {_sd.single_modality.mean():+.4f} | "
              f"mean per-subject diff {_sd['diff'].mean():+.4f}")

    df = pd.DataFrame(rows)

    # Average the seeds within each subject first, so seed noise does not get
    # counted as extra evidence. n is then the number of PEOPLE.
    per_subj = df.groupby("subject", as_index=False)["diff"].mean()
    d = per_subj["diff"].values
    n = len(d)

    # subject bootstrap CI -- resample PEOPLE, not trials
    rng = np.random.default_rng(SEED)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # paired sign-flip null: if supervision does nothing, the sign of each
    # person's difference is arbitrary
    flips = rng.choice([-1.0, 1.0], size=(10000, n))
    null = (flips * d).mean(1)
    p_val = float((np.sum(np.abs(null) >= abs(d.mean())) + 1) / (len(null) + 1))

    n_pos = int((d > 0).sum())
    holds = bool(n > 1 and lo > 0 and p_val < 0.05)

    print(f"\n  SUBJECT-LEVEL (n = {n} people, seeds averaged within subject)")
    print(f"  mean per-subject difference {d.mean():+.4f}")
    print(f"  95% CI (subject bootstrap)  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  sign-flip p                 {p_val:.4f}")
    print(f"  subjects favouring cross-modal  {n_pos}/{n}")
    print(f"  VERDICT: {'HOLDS' if holds else 'NOT CONFIRMED'}")

    # the pooled view kept alongside, so the two are never confused
    pooled = df.groupby("seed")[["cross_modal", "single_modality"]].mean()
    pooled_diff = (pooled.cross_modal - pooled.single_modality).values
    print(f"  (pooled-by-seed difference, n = {len(pooled_diff)} seeds: "
          f"{pooled_diff.mean():+.4f})")

    return df, {"unit": "subject", "n_subjects": n,
                "mean_diff": float(d.mean()), "ci_low": float(lo),
                "ci_high": float(hi), "p_signflip": p_val,
                "n_subjects_positive": n_pos,
                "pooled_by_seed_diff": float(pooled_diff.mean()),
                "verdict": "HOLDS" if holds else "NOT CONFIRMED"}


claim1_df, claim1 = claim1_cross_modal()
if claim1_df is not None:
    claim1_df.to_csv(OUT / "claim1_cross_modal.csv", index=False)
''')

# ===========================================================================
md(r"""
## 10 · Protocol ablation — where inflated numbers come from

Each row removes one evaluation shortcut. Nothing about the model or data
changes, only how it is scored. The drop is inflation attributable to protocol.
""")

co(r'''
def protocol_ablation(d, cfg=CFG):
    subs = np.unique(d["subject_ids"])
    rng = np.random.default_rng(SEED)

    # 1. trial-level split -- the same person appears on both sides
    idx = rng.permutation(len(d["eeg"])); cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    leak = {**d, "eeg": d["eeg"][tr], "hemo": d["hemo"][tr],
            "subject_ids": d["subject_ids"][tr]}
    m = train_joint([leak], cfg, verbose=False)
    sub = subset(d, te)
    r_leak = mean_corr(predict(m, sub), sub["hemo"])

    tbl = cv_tables[d["name"]]
    rows = [
        {"protocol": "1. trial-level split (subject on both sides)", "r": r_leak},
        {"protocol": "2. + subject-wise CV", "r": float(tbl.r_raw.mean())},
        {"protocol": "3. + no-EEG baseline made explicit",
         "r": float(tbl.r_no_eeg_baseline.mean())},
        {"protocol": "4. + subject demeaning  <-- HONEST",
         "r": float(cv_summaries[d["name"]]["r_demeaned"])},
    ]
    return pd.DataFrame(rows)


if TRAIN_SETS:
    abl = protocol_ablation(TRAIN_SETS[0])
    display(abl)
    abl.to_csv(OUT / "protocol_ablation.csv", index=False)
    infl = abl.r.iloc[0] / abl.r.iloc[-1] if abl.r.iloc[-1] > 0 else float("nan")
    print(f"\ninflation from protocol alone: {infl:.2f}x "
          f"({abl.r.iloc[0]:+.4f} -> {abl.r.iloc[-1]:+.4f})")
''')

# ===========================================================================
md(r"""
## 11 · CLAIM 5 — reliability gating and calibration

**Gate** — MC-dropout disagreement, plus the model's own predicted variance.
A working gate makes accuracy rise as coverage falls.

**Calibration** — ranking and calibration are different properties. ECE measures
the gap between stated confidence and actual accuracy; Platt scaling fits a
logistic map from confidence to probability on one half and scores it on the
other, so it cannot overfit.

Ranking and calibration are reported separately. `corr(uncertainty, error)`
answers whether the gate ORDERS trials correctly; ECE answers whether its
numbers are on the right SCALE. A gate can do the first and fail the second.
""")

co(r'''
@torch.no_grad()
def mc_dropout(model, d, n_pass=20):
    model.train()                       # keep dropout active
    preds = []
    for _ in range(n_pass):
        out = []
        for i in range(0, len(d["eeg"]), 256):
            b = torch.tensor(d["eeg"][i:i + 256], dtype=torch.float32).to(DEVICE)
            out.append(model(b, d["name"], d["modality"], None)[0].cpu().numpy())
        preds.append(np.concatenate(out))
    preds = np.stack(preds)
    conf_mc = -preds.std(0).reshape(len(d["eeg"]), -1).mean(1)
    _, lv = predict(model, d, want_var=True)
    return preds.mean(0), conf_mc, -lv.reshape(len(d["eeg"]), -1).mean(1)


def expected_calibration_error(prob, correct, n_bins=10):
    """ECE over probabilities that are ALREADY in [0, 1].

    This deliberately does not rescale its input. An earlier version min-max
    normalised here, which silently cancelled any temperature applied by the
    caller -- (c/T - lo/T)/((hi-lo)/T) is just (c-lo)/(hi-lo) -- so calibration
    could never change the score. Callers must pass calibrated probabilities.
    """
    p, correct = np.asarray(prob, float), np.asarray(correct, float)
    if p.max() - p.min() < 1e-12:
        return float("nan"), pd.DataFrame()
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == n_bins - 1 else p < edges[i + 1])
        if m.sum() == 0:
            continue
        stated, actual = float(p[m].mean()), float(correct[m].mean())
        ece += (m.sum() / len(p)) * abs(stated - actual)
        rows.append({"bin": i, "n": int(m.sum()), "stated": round(stated, 4),
                     "actual": round(actual, 4), "gap": round(stated - actual, 4)})
    return float(ece), pd.DataFrame(rows)


def raw_to_prob(conf):
    """Uncalibrated confidence -> a probability, with no fitting.

    The baseline any calibration has to beat. Rank order is preserved, so this
    changes only the SCALE, which is exactly what calibration is meant to fix.
    """
    conf = np.asarray(conf, float)
    lo, hi = conf.min(), conf.max()
    return np.full_like(conf, 0.5) if hi - lo < 1e-12 else (conf - lo) / (hi - lo)


def fit_platt(conf, correct):
    """Platt scaling: fit p = sigmoid(a * conf + b) on held-in data.

    Replaces the old temperature search, which could not work: ECE renormalised
    its input, so dividing by T cancelled out and every T scored identically.
    """
    from sklearn.linear_model import LogisticRegression
    conf = np.asarray(conf, float).reshape(-1, 1)
    y = np.asarray(correct, float).ravel()
    if len(np.unique(y)) < 2:
        return lambda c: raw_to_prob(c)
    z = (conf - conf.mean()) / (conf.std() + 1e-12)   # condition the fit
    lr = LogisticRegression(solver="lbfgs", C=1e3).fit(z, y)
    mu, sd = float(conf.mean()), float(conf.std() + 1e-12)

    def apply(c):
        zc = (np.asarray(c, float).reshape(-1, 1) - mu) / sd
        return lr.predict_proba(zc)[:, 1]
    return apply


gate = calibration = None
if TRAIN_SETS:
    d0 = TRAIN_SETS[0]
    subs = np.unique(d0["subject_ids"]); hold = subs[-max(2, len(subs) // 5):]
    te = np.isin(d0["subject_ids"], hold)
    sub = subset(d0, te)
    mdl = train_joint(TRAIN_SETS, CFG, holdout={d0["name"]: hold}, verbose=False)
    mean_pred, conf, conf_var = mc_dropout(mdl, sub)
    pd_ = demean_by_subject(mean_pred, sub["subject_ids"])
    td_ = demean_by_subject(sub["hemo"], sub["subject_ids"])

    order = np.argsort(-conf)
    cov_rows = []
    for c in (1.0, .8, .6, .4, .2):
        k = max(int(len(order) * c), 8)
        cov_rows.append({"coverage": c, "n_kept": k,
                         "r": mean_corr(pd_[order[:k]], td_[order[:k]])})
    cov = pd.DataFrame(cov_rows); display(cov)
    cov.to_csv(OUT / "coverage_curve.csv", index=False)
    mono = bool(np.all(np.diff(cov.r.values) >= -1e-3))
    print(f"accuracy rises as coverage falls: {mono}"
          f"  -> {'gate is informative' if mono else 'gate is NOT informative'}")

    # Per-trial error. A correlation computed from ONE trial is dominated by
    # noise, and a confidence score cannot rank a quantity that is mostly noise
    # -- so a null ranking result would say more about this metric than about
    # the gate. Normalised squared error is the stable estimator; the
    # correlation version is kept alongside to show the difference.
    _num = ((pd_ - td_) ** 2).reshape(len(pd_), -1).mean(1)
    _den = (td_ ** 2).reshape(len(td_), -1).mean(1) + 1e-12
    err = _num / _den
    err_corr = np.array([1.0 - mean_corr(pd_[i:i+1], td_[i:i+1])
                         for i in range(len(pd_))])
    correct = (err < np.median(err)).astype(float)

    # Does stated uncertainty track actual error? This is the RANKING question,
    # and it is separate from calibration -- a gate can order trials correctly
    # while its numbers are on the wrong scale. Reported as Spearman so it
    # matches the corr_unc_err column of the earlier analysis.
    def _rank(e):
        ok = np.isfinite(e) & np.isfinite(conf)
        return (float(stats.spearmanr(-conf[ok], e[ok]).correlation)
                if ok.sum() > 8 else float("nan"))

    corr_unc_err = _rank(err)                  # normalised squared error
    corr_unc_err_r = _rank(err_corr)           # single-trial correlation error
    print(f"corr(uncertainty, error) = {corr_unc_err:+.4f}   [normalised MSE]")
    print(f"                           {corr_unc_err_r:+.4f}   "
          "[single-trial correlation -- noisier]")
    print("  -> " + ("uncertainty tracks error" if corr_unc_err > 0.1
                     else "no useful ranking"))
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(conf))
    fit, hold_ix = perm[:len(perm)//2], perm[len(perm)//2:]
    # Uncalibrated baseline vs Platt scaling FIT ON ONE HALF, scored on the other.
    ece0, _ = expected_calibration_error(raw_to_prob(conf[hold_ix]), correct[hold_ix])
    platt = fit_platt(conf[fit], correct[fit])
    p_cal = platt(conf[hold_ix])
    ece1, bins = expected_calibration_error(p_cal, correct[hold_ix])
    print(f"\nECE {ece0:.4f} -> {ece1:.4f} (Platt scaling, held out)")
    print("  -> " + ("calibration improved the gate" if ece1 < ece0
                     else "gate RANKS but does not CALIBRATE"))
    bins.to_csv(OUT / "calibration_bins.csv", index=False)
    gate = {"informative": mono}
    calibration = {"method": "Platt scaling", "ece_before": round(ece0, 4),
                   "ece_after": round(ece1, 4), "improved": bool(ece1 < ece0),
                   "corr_unc_err": (round(corr_unc_err, 4)
                                    if np.isfinite(corr_unc_err) else None)}

    # CLAIM 5 needs BOTH halves: the gate must rank AND the confidence must be
    # honest. Ranking alone is not a usable confidence number.
    c5 = "HOLDS" if (mono and ece1 <= ece0) else ("PARTIAL" if mono else "FAILS")
    print(f"\nCLAIM 5 VERDICT: {c5}")
    if c5 == "PARTIAL":
        print("  the gate RANKS trials correctly but its confidence is not calibrated")
    elif c5 == "FAILS":
        print("  accuracy does not improve as low-confidence trials are dropped")
    claim5 = {"ranks": bool(mono), "calibrates": bool(ece1 <= ece0),
              "corr_unc_err": (round(corr_unc_err, 4)
                               if np.isfinite(corr_unc_err) else None),
              "corr_unc_err_trialcorr": (round(corr_unc_err_r, 4)
                                         if np.isfinite(corr_unc_err_r) else None),
              "ece_before": round(ece0, 4), "ece_after": round(ece1, 4),
              "method": "Platt scaling, fit/score split", "verdict": c5}
else:
    claim5 = None
''')

# ===========================================================================
md(r"""
## 12 · Within-subject probe — the other ceiling

Train and test *inside* the same person. Not a usable system — it needs paired
data from every new patient — but it bounds what cross-subject could reach.

If within-subject is also near zero the **target** is the bottleneck. If it is
much higher, the signal exists but does not generalise **across people**, which
is a different problem with a different fix.
""")

co(r'''
def bandpower_features(eeg, fs=CFG.eeg_fs):
    feats = []
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
        x = bandpass(eeg.astype(np.float64), fs, lo, hi)
        feats.append(np.log(np.mean(x ** 2, axis=-1) + 1e-12))
    return np.concatenate(feats, axis=1)


def within_subject_probe(d, min_trials=20, test_frac=0.3):
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(SEED)
    X = bandpower_features(d["eeg"]); Y = d["hemo"].reshape(len(d["hemo"]), -1)
    rows = []
    for s in np.unique(d["subject_ids"]):
        m = np.where(d["subject_ids"] == s)[0]
        if len(m) < min_trials:
            continue
        idx = rng.permutation(m); cut = int(len(idx) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        pred = Ridge(alpha=10.0).fit(X[tr], Y[tr]).predict(X[te])
        rows.append({"subject": int(s), "n_test": len(te), "r": mean_corr(pred, Y[te])})
    return pd.DataFrame(rows)


within_subject = None
if TRAIN_SETS:
    wsp = within_subject_probe(TRAIN_SETS[0])
    if len(wsp):
        display(wsp); wsp.to_csv(OUT / "within_subject_probe.csv", index=False)
        w = float(wsp.r.mean())
        c = cv_summaries[TRAIN_SETS[0]["name"]]["r_demeaned"]
        print(f"\nwithin-subject r = {w:+.4f}   cross-subject r = {c:+.4f}")
        if w > 2 * abs(c):
            print("  -> the signal EXISTS but does not generalise across people.")
        elif w < 0.05:
            print("  -> the TARGET is the bottleneck; more subjects will not help.")
        within_subject = {"within_r": round(w, 4), "cross_r": round(float(c), 4)}
''')

# ===========================================================================
md(r"""
## 13 · CLAIM 3 — does the delay module measure physiology?

The delay is produced as `dmin + (dmax - dmin) * sigmoid(...)`, so it is
**forced** inside whatever window we choose. Landing in 2–8 s therefore proves
nothing. This trains the same module under two windows: if the answer follows
the window instead of holding at a fixed number of seconds, it is reporting our
prior, not the biology.
""")

co(r'''
class DelayHead(nn.Module):
    """Estimated neurovascular delay.

    bounded=True reproduces the original: a sigmoid squashed into [dmin, dmax],
    so the output cannot leave the window whatever the data says. Reporting
    "the delays are physiological" from that is circular.

    bounded=False lets the head output any real number. The physiological band
    then becomes something the delays can either land in or miss, which is what
    makes it a measurement.
    """

    def __init__(self, in_dim, dmin, dmax, bounded=True):
        super().__init__()
        self.dmin, self.dmax, self.bounded = dmin, dmax, bounded
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, z):
        raw = self.net(z).squeeze(-1)
        if not self.bounded:
            return raw
        return self.dmin + (self.dmax - self.dmin) * torch.sigmoid(raw)


def claim3_delay(d, dmin, dmax, cfg=CFG, bounded=True):
    subs = np.unique(d["subject_ids"]); hold = subs[-max(2, len(subs) // 5):]
    model = train_joint([d], cfg, holdout={d["name"]: hold}, verbose=False)
    head = DelayHead(model.encoder.out_dim, dmin, dmax, bounded).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    tr = ~np.isin(d["subject_ids"], hold)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(d["eeg"][tr], dtype=torch.float32),
        torch.tensor(d["hemo"][tr], dtype=torch.float32),
        torch.tensor(d["subject_ids"][tr], dtype=torch.long))
    dl = seeded_loader(ds, cfg.batch_size)
    fs = cfg.hemo_fs if d["modality"] == "fnirs" else 1.0 / TR_S
    for _ in range(cfg.epochs):
        for be, bh, bs in dl:
            with torch.no_grad():
                z = model.encoder(be.to(DEVICE), d["name"], bs.to(DEVICE))
            delay = head(z)
            # supervise toward the empirical lag, CLAMPED to the same window --
            # this clamp is the second half of the circularity
            with torch.no_grad():
                p = model.heads[d["modality"]](z)[0]
                pc = p - p.mean(-1, keepdim=True); tc = bh.to(DEVICE)
                tc = tc - tc.mean(-1, keepdim=True)
                best_r = torch.full((len(p),), -2.0, device=DEVICE)
                best_l = torch.zeros(len(p), device=DEVICE)
                # when free, search the whole window: capping the search at dmax
                # is a third bound on top of the sigmoid and the clamp
                _max_lag = (int(dmax * fs) if bounded
                            else max(int(dmax * fs), pc.shape[-1] - 4))
                for lag in range(0, _max_lag + 1):
                    a, b = (pc, tc) if lag == 0 else (pc[..., :-lag], tc[..., lag:])
                    if a.shape[-1] < 4:
                        break
                    r = (a * b).sum(-1).mean(-1)
                    upd = r > best_r
                    best_r = torch.where(upd, r, best_r)
                    best_l = torch.where(upd, torch.full_like(best_l, lag / fs), best_l)
                tgt = best_l.clamp(dmin, dmax) if bounded else best_l
            opt.zero_grad(); nn.functional.mse_loss(delay, tgt).backward(); opt.step()

    te = np.isin(d["subject_ids"], hold)
    per_subject, all_d = [], []
    with torch.no_grad():
        for s in np.unique(d["subject_ids"][te]):
            m = (d["subject_ids"] == s)
            b = torch.tensor(d["eeg"][m], dtype=torch.float32).to(DEVICE)
            z = model.encoder(b, d["name"], None)
            dv = head(z).cpu().numpy()
            all_d.append(dv); per_subject.append(float(dv.mean()))
    all_d = np.concatenate(all_d)
    return {"window": f"[{dmin:.0f},{dmax:.0f}] s", "mean_s": float(all_d.mean()),
            "between_subject_sd": float(np.std(per_subject)),
            "frac_of_range": float((all_d.mean() - dmin) / (dmax - dmin))}


claim3 = None
if TRAIN_SETS:
    # BOUNDED: the original. Two different windows, to show the answer tracks
    # whichever bound it was given.
    r3 = [claim3_delay(TRAIN_SETS[0], 2.0, 8.0, bounded=True),
          claim3_delay(TRAIN_SETS[0], 0.0, 20.0, bounded=True)]
    c3 = pd.DataFrame(r3); display(c3); c3.to_csv(OUT / "claim3_delay.csv", index=False)
    moved = abs(r3[1]["mean_s"] - r3[0]["mean_s"])
    frac_moved = abs(r3[1]["frac_of_range"] - r3[0]["frac_of_range"])
    collapsed = r3[0]["between_subject_sd"] < 0.5
    print(f"\nBOUNDED head:")
    print(f"  widening the window moved the delay by {moved:.2f} s,")
    print(f"  but its position within the range moved only {frac_moved:.3f}")
    print(f"  between-subject SD = {r3[0]['between_subject_sd']:.3f} s")
    if moved > 1.0:
        print("  -> the answer follows the BOUND we chose, not the physiology")

    # FREE: no sigmoid, no clamp, lag search over the whole window. Landing in
    # the physiological band is now something the data can fail to do.
    free = claim3_delay(TRAIN_SETS[0], 2.0, 8.0, bounded=False)
    pd.DataFrame([free]).to_csv(OUT / "claim3_delay_free.csv", index=False)
    in_band = 2.0 <= free["mean_s"] <= 8.0
    varies = free["between_subject_sd"] >= 0.5
    print(f"\nFREE head (unbounded, full lag search, no clamp):")
    print(f"  mean delay          {free['mean_s']:.2f} s"
          f"   {'INSIDE' if in_band else 'OUTSIDE'} the 2-8 s physiological band")
    print(f"  between-subject SD  {free['between_subject_sd']:.3f} s"
          f"   {'varies between people' if varies else 'collapsed to a constant'}")

    # The claim needs BOTH: physiologically plausible AND subject-varying,
    # established without the window being imposed in advance.
    verdict = "HOLDS" if (in_band and varies) else "FAILS"
    print(f"\nVERDICT: {verdict}")
    if not in_band:
        print("  freed of the bound, the delay leaves the physiological range")
    if not varies:
        print("  delays barely differ between people -> a constant, not a marker")
    claim3 = {"bounded_sec_moved": round(moved, 3),
              "bounded_frac_moved": round(frac_moved, 3),
              "bounded_between_subject_sd": round(r3[0]["between_subject_sd"], 4),
              "free_mean_s": round(free["mean_s"], 3),
              "free_between_subject_sd": round(free["between_subject_sd"], 4),
              "free_in_physiological_band": bool(in_band),
              "verdict": verdict}
''')

# ===========================================================================
md(r"""
## 14 · CLAIM 4 — does the encoder help an EEG-only task?

Freeze the trunk, fit a linear probe on the task labels, compare against three
controls. **This does not need accurate prediction** — a weak predictor can
still induce a useful representation. Losing to raw band-power means the encoder
is not yet worth using.
""")

co(r'''
@torch.no_grad()
def embed(model, d):
    model.eval()
    out = []
    for i in range(0, len(d["eeg"]), 256):
        b = torch.tensor(d["eeg"][i:i + 256], dtype=torch.float32).to(DEVICE)
        out.append(model.encoder(b, d["name"], None).cpu().numpy())
    return np.concatenate(out)


def probe_accuracy(X, y, groups, n_splits=3):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    subs = np.unique(groups)
    folds = np.array_split(np.random.default_rng(SEED).permutation(subs),
                           min(n_splits, len(subs)))
    accs = []
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
    return (float(np.mean(accs)), float(np.std(accs))) if accs else (float("nan"), 0.0)


claim4 = None
if TRAIN_SETS and len(np.unique(TRAIN_SETS[0]["labels"])) >= 2:
    d0 = TRAIN_SETS[0]; y, g = d0["labels"], d0["subject_ids"]
    rows = []
    m_sup = train_joint(TRAIN_SETS, CFG, verbose=False)
    a, s_ = probe_accuracy(embed(m_sup, d0), y, g)
    rows.append({"encoder": "cross-modal supervised", "accuracy": a, "sd": s_})

    perm = np.random.default_rng(SEED).permutation(len(d0["hemo"]))
    shuf = {**d0, "hemo": d0["hemo"][perm]}
    a, s_ = probe_accuracy(embed(train_joint([shuf], CFG, verbose=False), d0), y, g)
    rows.append({"encoder": "SHUFFLED-target control", "accuracy": a, "sd": s_})

    torch.manual_seed(SEED)
    a, s_ = probe_accuracy(embed(make_model(TRAIN_SETS), d0), y, g)
    rows.append({"encoder": "random init (untrained)", "accuracy": a, "sd": s_})

    a, s_ = probe_accuracy(bandpower_features(d0["eeg"]), y, g)
    rows.append({"encoder": "raw band-power", "accuracy": a, "sd": s_})

    c4 = pd.DataFrame(rows); display(c4)
    c4.to_csv(OUT / "claim4_downstream.csv", index=False)
    acc = dict(zip(c4.encoder, c4.accuracy))
    sup = acc["cross-modal supervised"]
    v = {k: sup - acc[k] for k in acc if k != "cross-modal supervised"}
    for k, val in v.items():
        print(f"  vs {k:26s} {val:+.4f}")
    if all(x > 0 for x in v.values()):
        verdict = "HOLDS"
    elif v["SHUFFLED-target control"] > 0 and v["random init (untrained)"] > 0:
        verdict = "PARTIAL"
    else:
        verdict = "FAILS"
    print(f"VERDICT: {verdict}")
    if verdict == "PARTIAL":
        print("  beats shuffled and random -> it learns something real")
        print("  loses to raw band-power   -> not yet worth using")
    claim4 = {"supervised": round(sup, 4),
              **{f"vs_{k.split()[0].lower()}": round(x, 4) for k, x in v.items()},
              "verdict": verdict}
else:
    print("no usable task labels -- claim 4 skipped")
''')

# ===========================================================================
md(r"""
## 15 · Clinical comparison — altered coupling in ICH

HEFMI holds 17 healthy controls and 20 intracerebral-haemorrhage patients under
an identical protocol. **This needs no accurate prediction** — the coupling
descriptors are measured directly.

Artefact controls are reported alongside: patients move more and haemorrhage
alters skull optics, so if those differ the coupling difference may be
acquisition rather than physiology.
""")

co(r'''
CLIN_EEG_FS, CLIN_NIRS_FS = 256.0, 11.0    # native rates in the HEFMI epo files
CLIN_BAND = (13.0, 30.0)                   # beta -- sensorimotor
CLIN_MAX_LAG_S = 12.0                      # HbO cannot lead the EEG


def beta_envelope(x, fs, band, out_fs):
    """Band-power envelope of EEG, resampled to the fNIRS rate."""
    e = np.abs(sps.hilbert(bandpass(x, fs, *band), axis=-1)) ** 2
    return sps.resample(e, max(int(round(e.shape[-1] / fs * out_fs)), 4), axis=-1)


def xcorr_lag(a, b, fs, max_lag_s=CLIN_MAX_LAG_S):
    """Peak correlation of a vs b, and the lag in seconds where it peaks.

    Only NON-NEGATIVE lags are searched: the haemodynamic response follows the
    electrical one. Allowing negative lags would let the fit claim blood flow
    preceded the neural activity that caused it.
    """
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best_r, best_l = -2.0, 0.0
    for lag in range(0, min(int(max_lag_s * fs), n - 4) + 1):
        x, y = a[:n - lag], b[lag:]
        if len(x) < 4:
            break
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, lag / fs
    return (best_r, best_l) if best_r > -2 else (np.nan, np.nan)


def load_clinical(root, groups=("Normals", "Patients")):
    """Per-recording coupling descriptors for the clinical comparison.

    A separate loader from load_hefmi(): that one produces trials for TRAINING,
    this one produces one row per recording with the descriptors the group
    comparison needs, plus the artefact controls that decide whether any
    difference is physiology or acquisition.

    Identity note: Normals/subject1 and Patients/subject1 are DIFFERENT people,
    and every person's files are named 1_epo.mat inside their own folder, so the
    group MUST be part of the key. Splitting on filename would merge two people.
    """
    rows = []
    for group in groups:
        gdir = root / "epo_data" / group / "EEG"
        if not gdir.exists():
            continue
        for sd in sorted(p for p in gdir.iterdir() if p.is_dir()):
            for ef in sorted(sd.glob("*_epo.mat")):
                nf = root / "epo_data" / group / "NIRS" / sd.name / ef.name
                if not nf.exists():
                    continue
                try:
                    e = sio.loadmat(ef, squeeze_me=True, struct_as_record=False)["epo"]
                    n = sio.loadmat(nf, squeeze_me=True, struct_as_record=False)["epo"]
                    X = np.asarray(e.X, float)
                    oxy = np.asarray(n.oxy.X, float)
                    dxy = np.asarray(n.doxy.X, float)
                except Exception:
                    continue
                k = min(X.shape[2], oxy.shape[2])
                hbo = np.transpose(oxy[:, :, :k], (2, 1, 0))
                hbr = np.transpose(dxy[:, :, :k], (2, 1, 0))
                eeg = np.transpose(X[:, :, :k], (2, 1, 0))
                cc = [np.corrcoef(hbo[t].ravel(), hbr[t].ravel())[0, 1] for t in range(k)]
                dd = np.diff(hbo.reshape(k, -1), axis=-1)

                # --- neurovascular coupling: beta envelope vs HbO -------------
                # Beta (13-30 Hz) is the sensorimotor band. Its power envelope is
                # the electrical event; HbO is the haemodynamic answer. Positive
                # lag means HbO trails the EEG, the physiological direction.
                rs, ls = [], []
                for t in range(k):
                    env = beta_envelope(eeg[t], CLIN_EEG_FS, CLIN_BAND, CLIN_NIRS_FS)
                    r, lag = xcorr_lag(env.mean(0), hbo[t].mean(0), CLIN_NIRS_FS)
                    if np.isfinite(r):
                        rs.append(r); ls.append(lag)

                # --- target reliability: split-half over trials ---------------
                rng = np.random.default_rng(0)
                rel = []
                for _ in range(30):
                    p = rng.permutation(k)
                    a_ = hbo[p[:k // 2]].mean(0).ravel()
                    b_ = hbo[p[k // 2:]].mean(0).ravel()
                    if a_.std() > 0 and b_.std() > 0:
                        rel.append(np.corrcoef(a_, b_)[0, 1])

                rows.append({
                    "person": f"{group}_{sd.name}", "group": group,
                    "recording": ef.name,
                    # primary measures -- ALL of them are reported, including the
                    # ones that come out null. Reporting only hbo_hbr because it
                    # reached significance would be selective reporting.
                    "coupling_r": float(np.mean(rs)) if rs else np.nan,
                    "coupling_lag_s": float(np.mean(ls)) if ls else np.nan,
                    "reliability": float(np.mean(rel)) if rel else np.nan,
                    "hbo_hbr": float(np.nanmedian(cc)),
                    # artefact controls -- patients move more, and haemorrhage
                    # alters skull optics, so these must be reported alongside
                    "motion_frac": float((np.abs(dd) > 5 * np.nanstd(dd)).mean()),
                    "hbo_amp": float(np.percentile(np.abs(hbo), 99.5)),
                    "eeg_amp": float(np.percentile(np.abs(eeg), 99.5)),
                    "n_trials": k})
    return pd.DataFrame(rows) if rows else None


clinical = None
CLINICAL_ROOT = PATHS.get("clinical")

if CLINICAL_ROOT is None:
    print("clinical dataset not set (PATH_CLINICAL_HEFMI) -- section skipped")
else:
    print(f"clinical data: {CLINICAL_ROOT}")
    cdf = load_clinical(CLINICAL_ROOT)
    if cdf is None:
        print("  no readable recordings found")
    else:
        # collapse sessions -> one row per PERSON, so n is people not recordings
        per = cdf.groupby(["person", "group"], as_index=False).mean(numeric_only=True)
        ctrl = per[per.group == "Normals"]
        pat = per[per.group == "Patients"]
        print(f"  {len(cdf)} recordings -> {len(per)} people "
              f"({len(ctrl)} controls, {len(pat)} patients)")

        # All four primary measures are tested and all four are reported. Three
        # of them come out null; dropping those and keeping only the significant
        # one would turn four tests into an unstated multiple-comparison problem.
        MEASURES = [("coupling_r", True), ("coupling_lag_s", True),
                    ("reliability", True), ("hbo_hbr", True),
                    ("motion_frac", False), ("hbo_amp", False),
                    ("eeg_amp", False), ("n_trials", False)]
        res = []
        for name, primary in MEASURES:
            x, y2 = ctrl[name].dropna().values, pat[name].dropna().values
            if len(x) < 3 or len(y2) < 3:
                continue
            pv = stats.mannwhitneyu(x, y2, alternative="two-sided").pvalue
            res.append({"measure": name, "kind": "primary" if primary else "artefact control",
                        "controls": round(float(x.mean()), 4),
                        "patients": round(float(y2.mean()), 4),
                        "p": round(float(pv), 5), "sig": "*" if pv < .05 else ""})
        res = pd.DataFrame(res)
        per.to_csv(OUT / "clinical_per_person.csv", index=False)
        cdf.to_csv(OUT / "clinical_per_recording.csv", index=False)

        # Four primary tests means four chances at a false positive. Holm-
        # Bonferroni over the PRIMARY measures only -- the artefact controls are
        # not hypotheses, they are checks, so correcting them would make it
        # harder to detect a confound, which is backwards.
        _p = res[res.kind == "primary"].sort_values("p")
        _m, _adj, _run = len(_p), {}, 0.0
        for _rank, (_, _r) in enumerate(_p.iterrows()):
            _run = max(_run, min(1.0, _r["p"] * (_m - _rank)))
            _adj[_r["measure"]] = round(_run, 5)
        res["p_holm"] = res.measure.map(_adj)
        res["survives_correction"] = res.p_holm.notna() & (res.p_holm < .05)
        display(res)
        res.to_csv(OUT / "clinical_comparison.csv", index=False)

        prim_sig = res[(res.kind == "primary") & (res.p_holm < .05)]
        ctrl_sig = res[(res.kind == "artefact control") & (res.p < .05)]
        print(f"\n{_m} primary measures tested; p_holm is the corrected p-value.")
        if len(prim_sig) == 0:
            print("No coupling measure separates the groups after correction.")
        else:
            print("Group difference in: " + ", ".join(prim_sig.measure))
            if len(ctrl_sig):
                print("BUT artefact controls ALSO differ: " + ", ".join(ctrl_sig.measure))
                print("-> the difference may reflect acquisition, not physiology.")
            else:
                print("Artefact controls do NOT differ -> the difference is not")
                print("explained by movement or signal amplitude.")
        clinical = {r["measure"]: {"controls": r["controls"], "patients": r["patients"],
                                   "p": r["p"],
                                   "p_holm": (None if pd.isna(r["p_holm"])
                                              else r["p_holm"]),
                                   "kind": r["kind"]}
                    for _, r in res.iterrows()}
''')

# ===========================================================================
md(r"""
## 16 · CLAIM 2 — external validation on held-out cohorts

Train once on the training sets, then evaluate **once** on data from people the
model has never seen. No tuning against these.

- **fMRI** — same 48-ROI atlas both sides, so compared ROI by ROI.
- **fNIRS** — different montages with no published crosswalk, so both sides are
  pooled to mean HbO / mean HbR. Coarser, but honest.
""")

co(r'''
if not TRAIN_SETS:
    print("no training data -- skipping external validation")
    external = {}
else:
    print("training the final joint model on all training data ...")
    final_model = train_joint(TRAIN_SETS, CFG)
    external = {}

    for role, key in (("external_fnirs", "fnirs"), ("external_fmri", "fmri")):
        if role not in DATA:
            print(f"\n{role}: not loaded -- skipped")
            continue
        ext = DATA[role]
        train_same = next((d for d in TRAIN_SETS if d["modality"] == key), None)
        if train_same is None:
            print(f"\n{ext['name']}: no {key} training set -- cannot validate")
            continue

        # Get the external montage onto the training montage. Interpolating by
        # POSITION lets the model use the adapter it actually trained; the
        # fallback below invents a channel correspondence and is much weaker.
        M = interpolation_matrix(ext.get("ch_names"), train_same.get("ch_names"))
        if M is not None:
            ext_in = dict(ext)
            ext_in["eeg"] = np.einsum("ij,njt->nit", M,
                                      ext["eeg"].astype(np.float64)).astype(np.float32)
            ext_in["name"] = train_same["name"]      # use the TRAINED adapter
            align = (f"spherical-spline interpolation, "
                     f"{len(ext['ch_names'])} -> {len(train_same['ch_names'])} ch")
        else:
            # No electrode names (the fMRI sets carry none), so fall back to
            # copying the trained adapter and pairing channels by index. That
            # correspondence is arbitrary and will understate transfer.
            with torch.no_grad():
                src = final_model.encoder.adapters[train_same["name"]]
                new = nn.Conv1d(ext["eeg"].shape[1], src.weight.shape[0], 1).to(DEVICE)
                n = min(ext["eeg"].shape[1], src.weight.shape[1])
                new.weight.zero_(); new.weight[:, :n] = src.weight[:, :n]
                new.bias.copy_(src.bias)
                final_model.encoder.adapters[ext["name"]] = new
            ext_in = ext
            align = "channels paired BY INDEX (no electrode names) -- weak"
        print(f"  montage      {align}")

        pred = predict(final_model, ext_in)
        targ = ext["hemo"]
        if key == "fnirs" and pred.shape[1] != targ.shape[1]:
            # different montages -> compare pooled HbO/HbR, never invented channels
            pred, targ = pool_hbo_hbr(pred), pool_hbo_hbr(targ)
            note = "pooled to mean HbO / mean HbR (montages differ)"
        else:
            note = "ROI by ROI (same atlas)"

        g_ext = strata(ext)
        pd_ = demean_by_subject(pred, g_ext)
        td_ = demean_by_subject(targ, g_ext)
        r_ext, p_ext = perm_null(pd_, td_)
        r_int = cv_summaries.get(train_same["name"], {}).get("r_demeaned", float("nan"))
        drop = (r_int - r_ext) / abs(r_int) * 100 if r_int and np.isfinite(r_int) else float("nan")

        # The two cohorts have different target quality, so a raw r is not
        # comparable across them. Express each as a fraction of the ceiling of
        # the set it was measured on, and say so when that ceiling is low --
        # otherwise a bad TARGET reads as a model that failed to transfer.
        q_ext = TARGET_QUALITY.get(role, {})
        q_int = TARGET_QUALITY.get(train_same["role"] if "role" in train_same
                                   else f"train_{key}", {})
        c_ext, c_int = q_ext.get("ceiling"), q_int.get("ceiling")
        pct_ext = r_ext / c_ext * 100 if c_ext else float("nan")
        pct_int = r_int / c_int * 100 if c_int and np.isfinite(r_int) else float("nan")

        print(f"\n=== EXTERNAL {key.upper()}: {ext['name']} ===")
        print(f"  comparison    {note}")
        print(f"  internal  r = {r_int:+.4f}   ({train_same['name']}, "
              f"ceiling {c_int}, {pct_int:.1f}% of it)")
        print(f"  EXTERNAL  r = {r_ext:+.4f}   p = {p_ext:.4f}   "
              f"(ceiling {c_ext}, {pct_ext:.1f}% of it)")
        print(f"  drop        = {drop:.0f}% of the internal correlation")
        print(f"  VERDICT: {'HOLDS' if p_ext < 0.05 else 'FAILS'} -- "
              + ("survives a new cohort" if p_ext < 0.05
                 else "does not retain significance on unseen people"))
        if p_ext >= 0.05 and c_ext is not None and c_ext < 0.5:
            print(f"  CAUTION: {ext['name']} has a low ceiling ({c_ext}). A null")
            print("  result here is ambiguous -- the target may be too noisy to")
            print("  measure transfer at all. Do not read this as the encoder")
            print("  failing to generalise.")
        if q_ext.get("hbo_hbr") is not None and q_ext["hbo_hbr"] > 0:
            print(f"  CAUTION: HbO/HbR is {q_ext['hbo_hbr']:+.3f} on this set "
                  "(should be negative);")
            print("  its fNIRS is systemic-dominated, not an evoked response.")
        external[ext["name"]] = {"modality": key, "comparison": note,
                                 "montage_alignment": align,
                                 "internal_r": r_int, "external_r": r_ext,
                                 "p": p_ext, "drop_pct": drop,
                                 "external_ceiling": c_ext,
                                 "pct_of_external_ceiling": round(pct_ext, 2),
                                 "subjects": int(len(np.unique(ext["subject_ids"]))),
                                 "trials": int(len(ext["eeg"]))}
    if external:
        pd.DataFrame(external).T.to_csv(OUT / "external_validation.csv")

# CLAIM 2 holds only if the effect survives on people the model never saw.
if external:
    _sig = {k: (v["p"] < 0.05) for k, v in external.items()}
    if all(_sig.values()):
        c2 = "HOLDS"
    elif any(_sig.values()):
        c2 = "PARTIAL"
    else:
        c2 = "FAILS"
    print(f"\nCLAIM 2 VERDICT: {c2}")
    for k, v in external.items():
        print(f"  {k:18s} r = {v['external_r']:+.4f}  p = {v['p']:.4f}  "
              f"({'significant' if v['p'] < 0.05 else 'not significant'})")
    if c2 == "FAILS":
        print("  the effect does not retain significance on unseen cohorts")
    claim2 = {"per_dataset": {k: {"r": round(v["external_r"], 4),
                                  "p": round(v["p"], 5)} for k, v in external.items()},
              "verdict": c2}
else:
    claim2 = None
''')

# ===========================================================================
md("## 17 · Summary")

co(r'''
# ---- the five claims, in one place ------------------------------------------
CLAIMS = [
    (1, "cross-modal supervision beats single-modality", claim1),
    (2, "transfers to held-out cohorts with a bounded drop", claim2),
    (3, "recovers physiological, subject-varying delays", claim3),
    (4, "improves a downstream EEG-only task", claim4),
    (5, "reliability gating tracks real error", claim5),
]
print("=" * 66)
print("SCORECARD -- the five claims")
print("=" * 66)
score = {}
for num, text, obj in CLAIMS:
    v = (obj or {}).get("verdict", "NOT RUN")
    mark = {"HOLDS": "[HOLDS ]", "PARTIAL": "[PARTIAL]",
            "FAILS": "[FAILS ]", "NOT CONFIRMED": "[NOT CONF]"}.get(v, "[NOT RUN]")
    print(f"  {mark}  claim {num}: {text}")
    score[f"claim{num}"] = v
n_hold = sum(1 for v in score.values() if v == "HOLDS")
print("=" * 66)
print(f"{n_hold} of 5 hold"
      + ("   (SMOKE TEST -- not reportable)" if QUICK_TEST else ""))
print()

result = {
    "mode": "SMOKE TEST" if QUICK_TEST else "FULL RUN",
    "scorecard": score,
    "device": DEVICE,
    "datasets": {r: {"name": d["name"], "modality": d["modality"],
                     "subjects": int(len(np.unique(d["subject_ids"]))),
                     "trials": int(len(d["eeg"]))} for r, d in DATA.items()},
    "internal": {k: {"r_demeaned": round(v["r_demeaned"], 4), "p": round(v["p"], 5)}
                 for k, v in cv_summaries.items()},
    "claim1": claim1,
    "claim2": claim2,
    "claim3": claim3,
    "claim4": claim4,
    "claim5": claim5,
    "gate": gate,
    "calibration": calibration,
    "within_subject": within_subject,
    "clinical": clinical,
    "external": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                     for kk, vv in v.items()} for k, v in external.items()},
}
print(json.dumps(result, indent=2))
json.dump(result, open(OUT / "summary.json", "w"), indent=2)
print(f"\noutputs -> {OUT.resolve()}")

if QUICK_TEST:
    print("\n" + "!" * 62)
    print("SMOKE TEST ONLY -- these numbers are noisy and must not be reported.")
    print("Set QUICK_TEST = False in cell 1 for the real run.")
    print("!" * 62)
''')

nbf.write(nb, "NeuroVascNet_v6.ipynb")
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"wrote NeuroVascNet_v6.ipynb ({len(nb.cells)} cells, {n_code} code)")
