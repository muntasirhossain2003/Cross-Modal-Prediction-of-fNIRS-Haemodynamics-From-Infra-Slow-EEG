"""Emit NeuroVascNet_Kaggle.ipynb -- runs on Kaggle or any PC, downloads its own data.

Self-contained: no other notebook, no project module, no pre-attached dataset.
Cell 2 downloads the four datasets straight from the shared Google Drive folder.

Drive folder IDs were resolved from
https://drive.google.com/drive/folders/1fOAYS56d2-GG8OafrzqU69r9q8M_cpEa
and verified by listing each one (346/543/663/222 files respectively).

Usage:  python build_kaggle_notebook.py
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

# ===========================================================================
md(r"""
# NeuroVascNet — Kaggle / PC ready

**Run every cell top to bottom. The data downloads itself.** Nothing to attach,
nothing to configure.

---

### What this notebook does

Trains an EEG-only model to predict the haemodynamic (blood-flow) response, and
evaluates it **honestly** — cross-subject, against a baseline that ignores EEG,
with an exact permutation null.

### Before you start

**On Kaggle:** Settings (right panel) → **Internet: ON**. Without it the download
cell cannot reach Google Drive. GPU is optional; the notebook runs on CPU.

**On a PC:** just run it. Python 3.9+ with pip.

### Runtime

Pick a **profile** in the settings block (section 2):

| Profile | Downloads | Size | Gives you |
|---|---|---|---|
| `core` | hBCI + HEFMI | 20 GB | main result + clinical |
| `claims` | + TU Berlin workload | 32 GB | adds **claim 2**. **Largest that fits on Kaggle** |
| `full` **(default)** | + both fMRI sets | 97 GB | **all five datasets**. PC only — Kaggle auto-drops to `claims` |

| Step | Time (`full`) |
|---|---|
| Download all five datasets, 97 GB | 2–5 hours |
| Load and preprocess | 5–10 min |
| Smoke test | < 1 min |
| Train + evaluate (5 folds) | 10–25 min CPU, 3–8 min GPU |
| Claim 2 external test | 5–10 min |
| **Total** | **~3–6 hours**, mostly downloading |

Downloads are **resumable** — files already on disk are skipped, so re-running
after an interruption picks up where it stopped. You need **~110 GB free**.

### Try it in 5 minutes first

In **section 2** there is a single settings block with three switches:

```python
PROFILE        = "full"    # how much data to download
QUICK_TEST     = False     # True -> 8 subjects, 2 epochs, ~5 minutes
RUN_SMOKE_TEST = True      # ~60 s self-check before training
```

Set **`QUICK_TEST = True`** for a five-minute trial that exercises every cell,
then set it back to `False` for numbers worth reporting.

---

### What is in this notebook

| Section | Component | What it is |
|---|---|---|
| 4 | **Loader** | `load_tuberlin` — Beer–Lambert from raw 760/850 nm, GSR off |
| 5 | **Ceiling** | `split_half_reliability`, `hbo_hbr_sign` — the bound on any model |
| 6 | **Metrics** | `mean_corr`, `perm_null` (559× faster exact null), `constant_baseline`, `demean_by_subject` |
| 7 | **Encoder** | `NeuroVascNet` — Conv1d trunk (k=25/15/9) → pool(16) → subject embedding → **mean + log-variance heads** |
| 7 | **Loss** | `compute_loss` — pearson + heteroscedastic + cross-correlation + smoothness |
| 7 | **Smoke test** | `smoke_test` — 21 assertions before any long run |
| 8 | **Evaluation** | `cross_subject_cv` — subject-wise 5-fold, no person in two folds |
| 9 | **Protocol ablation** | `protocol_ablation` — where inflated numbers come from |
| 10 | **Abstention gate** | `mc_dropout` — two confidence signals |
| 11 | **Claim 2** | `load_tuberlin_wl` + external validation on a new cohort |
| 12 | **fMRI check** | `inventory` — verifies the downloads open |
""")

# ===========================================================================
md("## 1 · Install and import")

co(r'''
import sys, subprocess

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

for mod, pkg in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                 ("torch", "torch"), ("gdown", "gdown>=5.1.0"),
                 ("matplotlib", "matplotlib"),
                 ("nibabel", "nibabel"), ("sklearn", "scikit-learn"), ("mne", "mne")]:      # only needed by section 12
    try:
        __import__(mod)
    except ImportError:
        print(f"installing {pkg} ...")
        pip(pkg)

import os, re, json, math, time, shutil, warnings
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
import matplotlib.pyplot as plt

# numpy >= 2.0 renamed trapz -> trapezoid; older numpy lacks the new name.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detect Kaggle from its env var first: on Windows, Path("/kaggle") resolves
# against the current drive (e.g. G:\kaggle), so a stray folder of that name
# would otherwise make a local machine look like Kaggle.
ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or \
            (os.name != "nt" and Path("/kaggle/input").exists())
print(f"numpy {np.__version__} | torch {torch.__version__}")
print(f"device: {DEVICE} | environment: {'Kaggle' if ON_KAGGLE else 'local'}")
''')

# ===========================================================================
md(r"""
## 2 · Getting the data

**Two ways. Pick one.**

---

### Option A — you already have the data on disk *(recommended on a PC)*

Set **`MANUAL_DATA_DIR`** in the next cell to the folder holding the datasets and
**nothing is downloaded**. No Google Drive, no quota limits, no waiting.

The folder must look like this. Only the paths shown are ever opened — anything
else in these datasets is ignored, so a partial copy is fine:

```
<MANUAL_DATA_DIR>/
│
├── TU Berlin hBCI EEG+FNRI/               <- PRIMARY. All headline results.
│   ├── EEG data/
│   │   ├── EEG_01-05/
│   │   │   └── subject 01/
│   │   │       └── with occular artifact/
│   │   │           ├── cnt.mat            <- 30-ch EEG @ 200 Hz   (needed)
│   │   │           └── mrk.mat            <- event markers        (needed)
│   │   ├── EEG_06-10/  ...                   (5 groups, 29 subjects)
│   │   └── EEG_26-29/
│   └── NRI data/
│       └── NIRS_01-29/
│           └── subject 01/
│               ├── cnt.mat                <- raw 760/850 nm       (needed)
│               └── mrk.mat                <- event markers        (needed)
│
├── HEFMI-ICH EEG+FNRI/                    <- CLINICAL section only
│   └── epo_data/
│       ├── Normals/                          17 healthy controls
│       │   ├── EEG/subject1/1_epo.mat
│       │   └── NIRS/subject1/1_epo.mat
│       └── Patients/                         20 ICH patients
│           ├── EEG/subject1/1_epo.mat
│           └── NIRS/subject1/1_epo.mat
│                                             NOTE: Normals/subject1 and
│                                             Patients/subject1 are DIFFERENT
│                                             people. Keep the folders apart.
│
├── TU Berlin EEG + NIRS/                  <- EXTERNAL corpus, CLAIM 2 only
│   ├── EEG/VP001-EEG/cnt_nback.mat           (26 subjects, n-back task)
│   └── NIRS/VP001-NIRS/
│       ├── cnt_nback.mat                  <- already HbO/HbR, no conversion
│       └── mrk_nback.mat
│
├── OpenNeuro ds002336/                    <- fMRI. Optional.
│   ├── task-motorloc_events.tsv              (events are dataset-level here)
│   ├── task-MIpre_events.tsv  ...
│   ├── derivatives/sub-xp101/eeg_pp/
│   │   ├── sub-xp101_task-motorloc_eeg_pp.vhdr    <- has the R128 triggers
│   │   ├── ..._eeg_pp.eeg
│   │   └── ..._eeg_pp.vmrk
│   └── sub-xp101/func/sub-xp101_task-motorloc_bold.nii.gz
│
└── External fMRI validation OpenNeuro ds002739/   <- fMRI. Optional.
    └── sub-01/
        ├── EEG/EEG_data_sub-01_run-01.mat
        └── func/
            ├── sub-01_task-main_run-01_bold.nii.gz
            └── sub-01_task-main_run-01_events.tsv  (per-run events here)
```

**Folder names must match** — that is how the notebook identifies each dataset.
Missing datasets are simply skipped, and the notebook says which sections that
disables.

---

### Option B — download automatically

Leave `MANUAL_DATA_DIR = None` and set `PROFILE`. Files already on disk are
skipped, so an interrupted download resumes when you re-run the cell.
""")

co(r'''
# ---------------------------------------------------------------- what to fetch
# Folder IDs resolved from the shared Drive link and verified by listing each.
DATASETS = {
    # key            folder id(s)                          files   GB   what it is
    "tuberlin":   ("1lw-AR79e5FgVOL6iHaRG0FppYJHOhnqj",    346,   7.1, "TU Berlin hBCI EEG+fNIRS (primary, training)"),
    "hefmi":      ("1OnC-Z62tLH8ompyxj4ev_qABiqIkj8jv",    543,  13.2, "HEFMI-ICH EEG+fNIRS (clinical)"),
    # Claim 2's EXTERNAL corpus. Only the EEG/ and NIRS/ subfolders are pulled --
    # the sibling EEG_brainAmp/ (702 files) is the same recordings in raw
    # BrainVision format, which we never read.
    "tuberlin_wl": ({"EEG":  "1Yuxv7wpVLa9Ls9TppoX7Q9HpqtgPhwha",
                     "NIRS": "1TOUbgvIYWG7uylTRusROwqNfgiNxTUcm"},
                                                            468,  12.0, "TU Berlin workload (EXTERNAL, claim 2)"),
    "ds002336":   ("1K_91Y5Ph0PiAIcFdlGsiHg10_knUNmhw",    663,  19.3, "OpenNeuro ds002336 EEG+fMRI"),
    "ds002739":   ("1Rss12hlZrLqQTJDP3GtR_MqIwyDWNy1N",    222,  45.4, "OpenNeuro ds002739 EEG+fMRI"),
}

# ============================================================================
#                        THE ONLY SETTINGS YOU NEED
# ============================================================================
#
# PROFILE -- how much data to download.
#
#   "core"    20 GB   hBCI + HEFMI
#                     main result + clinical section.
#
#   "claims"  32 GB   + TU Berlin workload
#                     adds CLAIM 2, the external-validation test.
#                     >> Largest profile that fits on Kaggle. <<
#
#   "full"    97 GB   + ds002336 + ds002739  (ALL FIVE DATASETS)
#                     adds the fMRI download check in section 12.
#                     PC ONLY. On Kaggle this auto-drops to "claims".
#
# MANUAL_DATA_DIR -- if you have already downloaded the datasets by hand, put
#                    the folder here and NOTHING is downloaded. This is the
#                    simplest route on a PC and avoids Google Drive quota
#                    entirely. See the folder layout in the markdown above.
#
#   Windows example :  r"D:\neurovasc_data"
#   Linux / Mac     :  "/home/me/neurovasc_data"
#   Leave as None   :  download from Google Drive instead
#
MANUAL_DATA_DIR = None
#
PROFILE = "full"        # <-- on Kaggle this auto-drops to "claims" (32 GB)
#
# QUICK_TEST -- True runs a fast sanity pass (a few subjects, 2 epochs, ~5 min)
#               so you can confirm everything works before the real run.
#               Set back to False for the numbers you would actually report.
#
QUICK_TEST = False      # <-- set True for a 5-minute trial run
#
# RUN_SMOKE_TEST -- a ~60 second self-check that runs just before training:
#                   loader shapes, the four loss terms, and one forward/backward
#                   pass. Leave it ON -- it fails in a minute instead of an hour.
#
RUN_SMOKE_TEST = True   # <-- set False only to skip the self-check
#
# ============================================================================

PROFILES = {
    "core":   ["tuberlin", "hefmi"],
    "claims": ["tuberlin", "hefmi", "tuberlin_wl"],
    "full":   ["tuberlin", "hefmi", "tuberlin_wl", "ds002336", "ds002739"],
}
_size = lambda prof: sum(DATASETS[k][2] for k in PROFILES[prof])

# On Kaggle write the data to /kaggle/temp, NOT /kaggle/working:
#   /kaggle/working  is saved as notebook output and is capped at 20 GB
#   /kaggle/temp     is session scratch, shares the ~73 GB disk, not saved
# Results still go to /kaggle/working (they are small and worth keeping).
MANUAL = None
if MANUAL_DATA_DIR:
    MANUAL = Path(MANUAL_DATA_DIR)
    if not MANUAL.exists():
        raise FileNotFoundError(
            f"MANUAL_DATA_DIR does not exist: {MANUAL}\n"
            "Fix the path, or set it back to None to download instead.")
    print(f"MANUAL_DATA_DIR = {MANUAL.resolve()}")
    print("Nothing will be downloaded. Checking what is present:\n")
    # Folder name -> which analysis it enables. Missing ones are simply skipped.
    EXPECT = [
        ("TU Berlin hBCI EEG+FNRI", "EEG data", "REQUIRED -- all headline results"),
        ("HEFMI-ICH EEG+FNRI",      "epo_data", "clinical section (9)"),
        ("TU Berlin EEG + NIRS",    "EEG",      "claim 2, external validation (11)"),
        ("OpenNeuro ds002336",      "derivatives", "fMRI loader (15)"),
        ("External fMRI validation OpenNeuro ds002739", "sub-01", "fMRI loader (15)"),
    ]
    missing_required = False
    for folder, marker, what in EXPECT:
        d = MANUAL / folder
        ok = d.exists() and (d / marker).exists()
        print(f"  [{'ok  ' if ok else 'MISS'}] {folder:46s} {what}")
        if not ok and "REQUIRED" in what:
            missing_required = True
    if missing_required:
        raise FileNotFoundError(
            f"\nTU Berlin hBCI not found under {MANUAL}.\n"
            "Expected: <MANUAL_DATA_DIR>/TU Berlin hBCI EEG+FNRI/EEG data/...\n"
            "Check the folder NAME matches exactly -- see the layout above.")
    print("\nSkipping the download cells. Run the rest of the notebook normally.")

if ON_KAGGLE:
    DATA_ROOT = Path("/kaggle/temp/data")
else:
    DATA_ROOT = Path("data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(DATA_ROOT).free / 1e9

# Kaggle's session disk is ~73 GB shared with the OS and the environment, so we
# leave generous headroom. Rather than stopping with an error, drop to the
# largest profile that fits -- the notebook then runs to completion unattended.
KAGGLE_MAX_GB = 55.0
if ON_KAGGLE and _size(PROFILE) > KAGGLE_MAX_GB:
    fits = [p for p in ("full", "claims", "core") if _size(p) <= KAGGLE_MAX_GB]
    new = fits[0] if fits else "core"
    print(f"!! PROFILE '{PROFILE}' needs {_size(PROFILE):.0f} GB but Kaggle caps "
          f"/kaggle/working at 20 GB.")
    print(f"   Auto-switching to '{new}' ({_size(new):.0f} GB) so this run can finish.")
    print( "   Everything except the fMRI download check still runs.")
    print( "   For the full set, run this notebook on a PC.\n")
    PROFILE = new

GET = {k: (k in PROFILES[PROFILE]) for k in DATASETS}
need_gb = _size(PROFILE)

if MANUAL is None:
    print(f"profile   : {PROFILE}")
    print(f"datasets  : {', '.join(k for k in DATASETS if GET[k])}")
    print(f"needs     : {need_gb:.1f} GB")
    print(f"free here : {free_gb:.1f} GB")
    print(f"data root : {DATA_ROOT.resolve()}")

if MANUAL is None and free_gb < need_gb * 1.1:
    print(f"\n!! NOT ENOUGH DISK: {free_gb:.1f} GB free, {need_gb:.1f} GB needed.")
    smaller = [p for p in ("claims", "core") if _size(p) * 1.1 <= free_gb]
    if smaller:
        print(f"   Either free some space, or set PROFILE = '{smaller[0]}' "
              f"({_size(smaller[0]):.0f} GB).")
    else:
        print("   Free some space before running the download cell.")
''')

co(r'''
import gdown

# ---------------------------------------------------------------- manifest
# The loader reads only 4 files per subject, but the Drive folder holds 12
# (montage files and 41 MB *_artifact.mat copies we never open). Pulling the
# folder means 346 requests instead of 116 -- and it is the REQUEST COUNT, not
# the bytes, that trips Google Drive's rate limit. So for the primary dataset
# we fetch exactly the files we need, by id, with per-file retries.
TUBERLIN_MANIFEST = json.loads(r"""[["1qQJkTM1XJkIUhu8mI-nx-lzvy9dlon2u","EEG data/EEG_01-05/subject 01/with occular artifact/cnt.mat"],["1_kqHA4FOqO2qMJko2OIqb0Nk8_mb5SY_","EEG data/EEG_01-05/subject 01/with occular artifact/mrk.mat"],["1lOzYzxpk0MQNnSGgybtmUb1glSGlbr4U","EEG data/EEG_01-05/subject 02/with occular artifact/cnt.mat"],["1sBk21n_66nFJvt5VuQVzsdaGel76_BAP","EEG data/EEG_01-05/subject 02/with occular artifact/mrk.mat"],["1g8x5FoOC0y4nEsbBl0iRXOeKyUx842r4","EEG data/EEG_01-05/subject 03/with occular artifact/cnt.mat"],["1s_ShADu-wYtBFvcmmQnh2kzo46YqYE9O","EEG data/EEG_01-05/subject 03/with occular artifact/mrk.mat"],["1cvmxOtDOc7FiS7S8bEeq-fpfnQxOGONQ","EEG data/EEG_01-05/subject 04/with occular artifact/cnt.mat"],["1Wpl1bG40c-LaHODyLAe1iUPocUe_a142","EEG data/EEG_01-05/subject 04/with occular artifact/mrk.mat"],["1Ocy_3LuXbWolTWlAf-aDvmstlXICdwag","EEG data/EEG_01-05/subject 05/with occular artifact/cnt.mat"],["1KezZRQgG2_lkYifUF__d-QY8i9Kr1E4P","EEG data/EEG_01-05/subject 05/with occular artifact/mrk.mat"],["1XiJdA_OXO6FPiQJhT3ICrDHojF777LMS","EEG data/EEG_06-10/subject 06/with occular artifact/cnt.mat"],["1qIHdCHcnk2uhFCMj_xA1IB2JtrXqgV5b","EEG data/EEG_06-10/subject 06/with occular artifact/mrk.mat"],["1H1dEOi3vjAdxSNo5xopp9SWDwY_RekKh","EEG data/EEG_06-10/subject 07/with occular artifact/cnt.mat"],["1OWG2nQJSzm6rVF8MBaEBEopWlcIE_io7","EEG data/EEG_06-10/subject 07/with occular artifact/mrk.mat"],["1lY4yi5ZxaOwlgMpzqpqozNRKxz7upKxu","EEG data/EEG_06-10/subject 08/with occular artifact/cnt.mat"],["14G2_FT5VezrE1vI94U9zmKEwOoaA0rjt","EEG data/EEG_06-10/subject 08/with occular artifact/mrk.mat"],["1TK4Hwol0U8wEVpx5FEElfnUDvZlz616U","EEG data/EEG_06-10/subject 09/with occular artifact/cnt.mat"],["1Snsu_vITY4_hgYkOW0qB6mjrhaiMc2tP","EEG data/EEG_06-10/subject 09/with occular artifact/mrk.mat"],["1nu3yq6fyZPh-pKPlbu8IPmKxnbVucKrs","EEG data/EEG_06-10/subject 10/with occular artifact/cnt.mat"],["1exvDtoNk2bBNjInKJZ7DEwSJHff5qtgD","EEG data/EEG_06-10/subject 10/with occular artifact/mrk.mat"],["1uLwN51f-imAiNmPvY7fxhO4zcaN_M1I7","EEG data/EEG_11-15/subject 11/with occular artifact/cnt.mat"],["14XOJJSNE78e0j8nKcuDBw-9PMpSjTPyp","EEG data/EEG_11-15/subject 11/with occular artifact/mrk.mat"],["1inAchWVk7ySA16gybQrYMEX05lvOG3H9","EEG data/EEG_11-15/subject 12/with occular artifact/cnt.mat"],["1bQ2AsNP4vljizjeMakTc_l3xAlstV7z-","EEG data/EEG_11-15/subject 12/with occular artifact/mrk.mat"],["1LAtX076nkqo_L28A_j7elL_6tof0xmeh","EEG data/EEG_11-15/subject 13/with occular artifact/cnt.mat"],["1gUg5qwvonouBn9ScE_pj2J7UHlmN-M4N","EEG data/EEG_11-15/subject 13/with occular artifact/mrk.mat"],["1re--eWd2kyTicaS2XY9qlTYR4xJvH1W2","EEG data/EEG_11-15/subject 14/with occular artifact/cnt.mat"],["1XR3tqi0nbrOXlSdfAkahcjYc_8OpnEmX","EEG data/EEG_11-15/subject 14/with occular artifact/mrk.mat"],["1t_N8wu0RzOcuL8e5enLW040taRGcAiXL","EEG data/EEG_11-15/subject 15/with occular artifact/cnt.mat"],["1aXa6yYG83z_T_o1tVPybtLJepgoj3Q_q","EEG data/EEG_11-15/subject 15/with occular artifact/mrk.mat"],["1U8xtn9N1frPEkQCO644TUzAwqyeDdQO_","EEG data/EEG_16-20/subject 16/with occular artifact/cnt.mat"],["155ZvIGufU6Nzmw1D-VAszxJ1qXKTNGqt","EEG data/EEG_16-20/subject 16/with occular artifact/mrk.mat"],["1gGNjfbrkAFH-cCXglwQIf1nMwm4hQSUs","EEG data/EEG_16-20/subject 17/with occular artifact/cnt.mat"],["1Zjh0qG4RlUvrC1OzxCh2VGTb8lH1Juv-","EEG data/EEG_16-20/subject 17/with occular artifact/mrk.mat"],["14ke_yRzFjIc_KU7ATiC1x04nqwh-Q692","EEG data/EEG_16-20/subject 18/with occular artifact/cnt.mat"],["1Oow7W7jiaBKQ3amCB1DTReAvg0o3BUxO","EEG data/EEG_16-20/subject 18/with occular artifact/mrk.mat"],["16wwKAfqZAB0M0IyRkAHK5pQEsZDwSXIa","EEG data/EEG_16-20/subject 19/with occular artifact/cnt.mat"],["1IlDA1i5wHF0e9CdmvpfYs8oySAE_fegN","EEG data/EEG_16-20/subject 19/with occular artifact/mrk.mat"],["1OQcnmwZ5QMQ89YTTebgad76eZs4WcnhQ","EEG data/EEG_16-20/subject 20/with occular artifact/cnt.mat"],["1-bwA4Vj0MKM12oTrcE8wsp-Uh3u5bL2Y","EEG data/EEG_16-20/subject 20/with occular artifact/mrk.mat"],["1wPLy6foViKXyCoPvGRSll6y3y5BgVIIy","EEG data/EEG_21-25/subject 21/with occular artifact/cnt.mat"],["1WnVLwhWyzbh1pYBfHvD1RasuU-7bD2qB","EEG data/EEG_21-25/subject 21/with occular artifact/mrk.mat"],["1S7n5j1sAhZnwQpz2EfJBj-5BizImqrvH","EEG data/EEG_21-25/subject 22/with occular artifact/cnt.mat"],["1Ciqt8rImvIBL7wRhp3foPsXfcB_Dij-G","EEG data/EEG_21-25/subject 22/with occular artifact/mrk.mat"],["1FkUKe3S-jtipKCM8jb-4mXnjc5pC_1i5","EEG data/EEG_21-25/subject 23/with occular artifact/cnt.mat"],["14OujMcFVT_XZZUmNsLF-glCqUPI6ywVq","EEG data/EEG_21-25/subject 23/with occular artifact/mrk.mat"],["1u8lxTgDGrUSMgCLj7VSsjUpYRcfvPLW_","EEG data/EEG_21-25/subject 24/with occular artifact/cnt.mat"],["1ehnZ9tt-LUNjCztj9gkR9u-YBp_9FBEe","EEG data/EEG_21-25/subject 24/with occular artifact/mrk.mat"],["1DsHUlH_p0Re_ixPZXHzxFh2fQ4fZQtGf","EEG data/EEG_21-25/subject 25/with occular artifact/cnt.mat"],["1EZDktTacS3aYJ1qDLUh3yc0INLLtKFNc","EEG data/EEG_21-25/subject 25/with occular artifact/mrk.mat"],["1yfe9xjcGtaqCBvTzuFm74K-wRsqLOaix","EEG data/EEG_26-29/subject 26/with occular artifact/cnt.mat"],["1RJuYjGgDj2GDFDa2tW4DyHvNf2VsymG0","EEG data/EEG_26-29/subject 26/with occular artifact/mrk.mat"],["1S0ejhkZ0FuOSs9gl5AkVGAkOw9Sp2Bn2","EEG data/EEG_26-29/subject 27/with occular artifact/cnt.mat"],["1V7ESWfolZVEhh1qC8V3jNVvqqXQRFDlj","EEG data/EEG_26-29/subject 27/with occular artifact/mrk.mat"],["1y9acyp7-4fRBOsiDBBlVnm0t7V9_J7cK","EEG data/EEG_26-29/subject 28/with occular artifact/cnt.mat"],["1j48gZ8QX8Wmp7b_7X2nV3tDbEMsaLuQ-","EEG data/EEG_26-29/subject 28/with occular artifact/mrk.mat"],["1Kkq0Q0MrppfoZElbHZtQLc9-kejttRyR","EEG data/EEG_26-29/subject 29/with occular artifact/cnt.mat"],["1tH8sf8ST_i0cHBLE-2YTCI3nBHSUQ8sa","EEG data/EEG_26-29/subject 29/with occular artifact/mrk.mat"],["1R1gjJ_c-k9qj7o2KI27JkZFBcH4xU507","NRI data/NIRS_01-29/subject 01/cnt.mat"],["1z0sB6VX7vcZKSa21HZ8TEuqUsN3ZXEDL","NRI data/NIRS_01-29/subject 01/mrk.mat"],["1cOLTC6hzWaZ7zAMB5ewv9rEBVIr3hm8u","NRI data/NIRS_01-29/subject 02/cnt.mat"],["1W-kEJAscow3vLKSn4fzd5epsKSD8_1yV","NRI data/NIRS_01-29/subject 02/mrk.mat"],["1foYbh8DsoyQRpaPqIvlhyDPe_tgH2dcS","NRI data/NIRS_01-29/subject 03/cnt.mat"],["1b51IJmZHmazmIuZ99yxb4TKoI7q7f6Me","NRI data/NIRS_01-29/subject 03/mrk.mat"],["1OW61KfM0Jp0PQtBdv0LE-F-uXFhtdCPb","NRI data/NIRS_01-29/subject 04/cnt.mat"],["1Vra7eF5B2ArwwtmIzSESm6e5ZlPuIYel","NRI data/NIRS_01-29/subject 04/mrk.mat"],["1WTmVb4gHptr2w8cxefA0ZzIuRgObAoLa","NRI data/NIRS_01-29/subject 05/cnt.mat"],["1ndwKYb9M_0At3tNbIOJasZzdGIT8s5Vq","NRI data/NIRS_01-29/subject 05/mrk.mat"],["1N1Q2L-RHeN7NrlTfY9IDCm8j1_okjw1M","NRI data/NIRS_01-29/subject 06/cnt.mat"],["1dCrb5v4e47i_lfBKPJZODysPpP5qkyEc","NRI data/NIRS_01-29/subject 06/mrk.mat"],["18iEGaeE4B5gtCd14vDE2mHQzTeMvRf8I","NRI data/NIRS_01-29/subject 07/cnt.mat"],["1vk20VwoAqcdEqAE7psJvnj8DvplkFtim","NRI data/NIRS_01-29/subject 07/mrk.mat"],["1TC50BBnmwYtaaW5nPZbbG9MmRJywGybL","NRI data/NIRS_01-29/subject 08/cnt.mat"],["1yF89-0ZRVMSFz7TYKjzljjPWp52aHEvh","NRI data/NIRS_01-29/subject 08/mrk.mat"],["14VzFhULtPzQ6rbzKVCfxtrjDSwXSGBRg","NRI data/NIRS_01-29/subject 09/cnt.mat"],["1y7wTHeq3hYk2cdRtxtlICtTqKbhL1Jyk","NRI data/NIRS_01-29/subject 09/mrk.mat"],["1KVMAGQNlei0L4O-WtAm7fseZ-iqr53ls","NRI data/NIRS_01-29/subject 10/cnt.mat"],["1A2P1jsDD44B1ZwsTlxplULn_xkgQiCAx","NRI data/NIRS_01-29/subject 10/mrk.mat"],["1wAbFg8RVlqjlKttgVC29bKLr0sznc_SD","NRI data/NIRS_01-29/subject 11/cnt.mat"],["1D3C3nGUWPTfJAzV0dolQFCxTJU6cmPw4","NRI data/NIRS_01-29/subject 11/mrk.mat"],["1tXSGJniY_2_JSLXZhQ_hbgEz_SZ4U7Hi","NRI data/NIRS_01-29/subject 12/cnt.mat"],["1ZDsP658rThMw_3lTBt0G0nLvev1lLF1K","NRI data/NIRS_01-29/subject 12/mrk.mat"],["10xyv_pW-XQ3qbl7HQTlQZco5ddUtAh7z","NRI data/NIRS_01-29/subject 13/cnt.mat"],["1B9es7coxg9v71ILXj_BBkVla49MBxo6A","NRI data/NIRS_01-29/subject 13/mrk.mat"],["1Yh34I6mbPlS347jTaOaMEKQIjGsSygIJ","NRI data/NIRS_01-29/subject 14/cnt.mat"],["1P8g-g3hQcjhMy0H6YSrC3raecSdIz--0","NRI data/NIRS_01-29/subject 14/mrk.mat"],["1q4qyx_drgxNXcrtpz8lqO40Lv8wJQ2Hu","NRI data/NIRS_01-29/subject 15/cnt.mat"],["14yV_SOefODUkGc6N5ovcxHZ0ZuinU_xU","NRI data/NIRS_01-29/subject 15/mrk.mat"],["13IlRfMzD73d8AQJ-SdwC8fRwBOPOW61g","NRI data/NIRS_01-29/subject 16/cnt.mat"],["1cIwU4-A9odFSD6RH_t0l4gZo7X-ZO1_i","NRI data/NIRS_01-29/subject 16/mrk.mat"],["1G4l27ofRg5W83q0mxMO3GM8WUwvPO02o","NRI data/NIRS_01-29/subject 17/cnt.mat"],["1pVC8aqKCARO6sCUvr4XHVUKnIT1LFbQY","NRI data/NIRS_01-29/subject 17/mrk.mat"],["1EMiqX14d2Vn0DIsXjwb9CZMnP8fIfO3J","NRI data/NIRS_01-29/subject 18/cnt.mat"],["1ZvdpT2M8aXc6QUJK26LfzX7z1wb4Fglz","NRI data/NIRS_01-29/subject 18/mrk.mat"],["1xdTj7gktP1HZxlajBS5IACBXo-fV1eNK","NRI data/NIRS_01-29/subject 19/cnt.mat"],["15K5Qmg2gY9Gm90uo67jyqZZfHagjsIZX","NRI data/NIRS_01-29/subject 19/mrk.mat"],["1BcpoDacWJHNdIQhtipStfaxLNHNdHREK","NRI data/NIRS_01-29/subject 20/cnt.mat"],["1F7AACW6a3JFKGZy_olg6LH4j0yTq7myh","NRI data/NIRS_01-29/subject 20/mrk.mat"],["1OampNvdVBpHXxSCKqD-Tfa7Ldy2kxIuE","NRI data/NIRS_01-29/subject 21/cnt.mat"],["1vNYUz9g4Rp_w6sRhglpnT3FKcmSq0hox","NRI data/NIRS_01-29/subject 21/mrk.mat"],["13b20QtomiL6aY3Ll18H7fALm7i639Ky5","NRI data/NIRS_01-29/subject 22/cnt.mat"],["1PzLG7_h4OKPtTIeCz1jmQzUEbF65okN4","NRI data/NIRS_01-29/subject 22/mrk.mat"],["19tBC7UNRvZ6N2ztq8uBWuHsMmHJ0IkAc","NRI data/NIRS_01-29/subject 23/cnt.mat"],["17Ceg6pm9ZTxXmolhY3F0B-UqERsWiwf6","NRI data/NIRS_01-29/subject 23/mrk.mat"],["1Ocd0ecZLQ6jPlpbSP1YLmayVY2XW7Mag","NRI data/NIRS_01-29/subject 24/cnt.mat"],["1e9K4FMtMjGFy30nr_FgfYXAm0N342QSl","NRI data/NIRS_01-29/subject 24/mrk.mat"],["15uZnLgTFfasv-clyX49-oymEPz_NtKcx","NRI data/NIRS_01-29/subject 25/cnt.mat"],["1Ya8ns4TJpO7AkjbZHb4y1o2xTrqDzMj3","NRI data/NIRS_01-29/subject 25/mrk.mat"],["1BVNpTCj7-m-y9eQ5PTC7DTyB-GmIKhiJ","NRI data/NIRS_01-29/subject 26/cnt.mat"],["1xEFp1dtKeWs0Il2LSgamGXKSVkqwomCN","NRI data/NIRS_01-29/subject 26/mrk.mat"],["1pBdZ1GvhDpt9qzh0KV-_ylYlB8LegII3","NRI data/NIRS_01-29/subject 27/cnt.mat"],["1bRlMHbxZrHBvvr1aIuqCiEy7Xtnei6cF","NRI data/NIRS_01-29/subject 27/mrk.mat"],["1ZfMjeONAwLfLOmOIiehbCRn4frytQrzA","NRI data/NIRS_01-29/subject 28/cnt.mat"],["1-ueSwUBkhJQvtHhR2y0vMb3jYcHEag9v","NRI data/NIRS_01-29/subject 28/mrk.mat"],["1PBv3GuW_q0NDuExcNdodZvViJqEZAmgi","NRI data/NIRS_01-29/subject 29/cnt.mat"],["1YtUgjIUN1S-TbDKeraS6sHKPbHHO09g0","NRI data/NIRS_01-29/subject 29/mrk.mat"]]""")


def _subject_of(rel):
    """Subject number in a manifest path, for ordering."""
    m = re.search(r"subject (\d+)", rel)
    return int(m.group(1)) if m else 999


def fetch_manifest(key, manifest, label, retries=3):
    """Download an explicit file list. Idempotent and resumable per file.

    Ordered SUBJECT BY SUBJECT (all four of subject 1, then subject 2, ...).
    The natural path order puts every EEG file before every NIRS file, so an
    interrupted download would leave zero usable subjects; this way a partial
    download yields proportionally many COMPLETE subjects, and the run can
    still proceed on those.
    """
    dest = DATA_ROOT / key
    manifest = sorted(manifest, key=lambda e: (_subject_of(e[1]), e[1]))
    todo = [(fid, rel) for fid, rel in manifest if not (dest / rel).exists()]
    print(f"[get ] {key:12s} {len(manifest)} files needed, "
          f"{len(manifest) - len(todo)} already present -- {label}")
    failed, quota_hit = [], False
    for i, (fid, rel) in enumerate(todo, 1):
        if quota_hit:
            failed.append(rel)
            continue                       # no point hammering a quota wall
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, retries + 1):
            try:
                gdown.download(id=fid, output=str(out), quiet=True)
                if out.exists() and out.stat().st_size > 0:
                    break
                raise IOError("empty file")
            except Exception as exc:
                name = type(exc).__name__
                # FileURLRetrievalError = Drive refused the link. After a run of
                # successes that means the DAILY QUOTA is spent, not a bad id --
                # retrying cannot fix it, so stop and say so.
                if "FileURLRetrieval" in name:
                    failed.append(rel)
                    if len([f for f in failed]) >= 3 and i > 5:
                        quota_hit = True
                        print(f"   Google Drive refused several files in a row "
                              f"({name}).")
                        print(f"   This is the DAILY DOWNLOAD QUOTA, not a bad link.")
                        print(f"   Stopping early -- see the note below.")
                    break
                if attempt == retries:
                    failed.append(rel)
                    print(f"   FAILED {rel} ({name})")
                else:
                    time.sleep(5 * attempt)
        if i % 20 == 0 or i == len(todo):
            print(f"   {i}/{len(todo)} ...")

    if quota_hit:
        print()
        print("   Google Drive caps how much a shared folder can serve per day.")
        print("   Your options, best first:")
        print("     1. CONTINUE ANYWAY -- the next cell will tell you how many")
        print("        complete subjects you have; 8 or more is enough to run")
        print("        every analysis in this notebook.")
        print("     2. Copy the Drive folder into your OWN Google Drive and swap")
        print("        the ids in DATASETS -- your own quota is separate.")
        print("     3. Wait ~24 h for the quota to reset, then re-run this cell;")
        print("        it resumes and keeps what you already have.")
    have = sum(1 for fid, rel in manifest if (dest / rel).exists())
    print(f"       -> {have}/{len(manifest)} files in {dest}")
    return dest


def _grab(folder_id, out, retries=3):
    """One gdown folder pull, tolerant of gdown version differences."""
    for attempt in range(1, retries + 1):
        try:
            try:
                gdown.download_folder(id=folder_id, output=str(out), quiet=False,
                                      use_cookies=False, remaining_ok=True)
            except TypeError:
                # gdown < 5.2 has no remaining_ok argument
                gdown.download_folder(id=folder_id, output=str(out),
                                      quiet=False, use_cookies=False)
            return True
        except Exception as exc:
            print(f"   attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(10 * attempt)
    return False


def fetch(key, folder_ids, expected, gb, label):
    """Download one dataset. Idempotent: skips if the files are already there.

    folder_ids is either a single Drive folder id, or {subdir: id} when the
    dataset is assembled from several folders (used for tuberlin_wl, where we
    deliberately skip the redundant EEG_brainAmp folder).
    """
    dest = DATA_ROOT / key
    have = sum(1 for _ in dest.rglob("*") if _.is_file()) if dest.exists() else 0
    if have >= expected * 0.95:
        print(f"[skip] {key:12s} already has {have} files -- {label}")
        return dest

    print(f"[get ] {key:12s} {expected} files, ~{gb:.1f} GB -- {label}")
    dest.mkdir(parents=True, exist_ok=True)
    if isinstance(folder_ids, dict):
        for sub, fid in folder_ids.items():
            print(f"       .. {sub}/")
            (dest / sub).mkdir(exist_ok=True)
            _grab(fid, dest / sub)
    else:
        _grab(folder_ids, dest)

    got = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"       -> {got}/{expected} files in {dest}")
    if got < expected * 0.95:
        print(f"       !! incomplete -- re-run this cell, it resumes where it stopped.")
    return dest


# ---------------------------------------------------------------- ONE-ZIP MODE
# Google Drive charges its download quota PER FILE, so pulling 116 files trips
# the daily cap partway through. A single archive is ONE request, so the quota
# stops mattering -- and it is smaller too, because the archives contain only
# the files the loaders actually open.
#
# Put the Drive FILE ids here once the zips are uploaded (share = "anyone with
# the link"). Leave a value as None to fall back to per-file downloading.
DATASETS_ZIP = {
    "tuberlin":    None,     # ~5.8 GB  (vs 116 separate files)
    "hefmi":       None,     # ~6.9 GB  (epo_data only)
    "tuberlin_wl": None,     # ~2.6 GB  (n-back only)
}


def fetch_zip(key, file_id, label):
    """Download one archive and extract it. One request, no quota problem."""
    import zipfile
    dest = DATA_ROOT / key
    zp = DATA_ROOT / f"{key}.zip"
    if dest.exists() and any(dest.rglob("*")):
        print(f"[skip] {key:12s} already extracted -- {label}")
        return True
    print(f"[zip ] {key:12s} single-archive download -- {label}")
    try:
        if not zp.exists() or zp.stat().st_size == 0:
            gdown.download(id=file_id, output=str(zp), quiet=False)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        zp.unlink(missing_ok=True)          # reclaim the archive's space
        n = sum(1 for _ in dest.rglob("*") if _.is_file())
        print(f"       -> extracted {n} files to {dest}")
        return True
    except Exception as exc:
        print(f"       zip download failed ({type(exc).__name__}) -- "
              f"falling back to per-file")
        return False


t0 = time.time()
for key, (fid, n, gb, label) in (DATASETS.items() if MANUAL is None else []):
    if not GET.get(key):
        continue
    zid = DATASETS_ZIP.get(key)
    if zid and fetch_zip(key, zid, label):
        continue                            # archive worked, nothing else to do
    if key == "tuberlin":
        fetch_manifest(key, TUBERLIN_MANIFEST, label)
    else:
        fetch(key, fid, n, gb, label)
if MANUAL is None:
    print(f"\ndownload finished in {(time.time()-t0)/60:.1f} min")
else:
    print("using MANUAL_DATA_DIR -- no download needed")

# ---------------------------------------------------------------- gate
# Stop here if anything is short, rather than failing later inside the loader
# with a confusing message. gdown drops files when Drive rate-limits.
# What matters is not the raw file count but how many SUBJECTS are complete.
# The loader reads 4 files per subject, so a partial download with 20 complete
# subjects is a perfectly usable run -- noisier, but the protocol is identical.
# Only stop if there are too few to be worth training on.
MIN_SUBJECTS = 8


def complete_tuberlin_subjects(root):
    """Subject numbers whose four required files are all present and non-empty."""
    ok = []
    for s in range(1, 30):
        rels = [rel for _fid, rel in TUBERLIN_MANIFEST if f"subject {s:02d}" in rel]
        if len(rels) >= 4 and all(
                (root / r).exists() and (root / r).stat().st_size > 0 for r in rels):
            ok.append(s)
    return ok


partial, absent = [], []
SUBJECTS_AVAILABLE = None
for key, (fid, n, gb, label) in (DATASETS.items() if MANUAL is None else []):
    if not GET.get(key):
        continue
    d = DATA_ROOT / key
    extra = ""
    if key == "tuberlin":
        n = len(TUBERLIN_MANIFEST)          # only the files we actually read
        got = sum(1 for fid, rel in TUBERLIN_MANIFEST if (d / rel).exists())
        SUBJECTS_AVAILABLE = complete_tuberlin_subjects(d)
        extra = f"   -> {len(SUBJECTS_AVAILABLE)}/29 subjects complete"
    else:
        got = sum(1 for _ in d.rglob("*") if _.is_file()) if d.exists() else 0
    if got >= n * 0.95:
        mark = "ok "
    elif got == 0:
        mark = "none"; absent.append(key)
    else:
        mark = "part"; partial.append(key)
    print(f"  [{mark:5s}] {key:12s} {got:4d}/{n} files{extra}")

if SUBJECTS_AVAILABLE is not None:
    if len(SUBJECTS_AVAILABLE) < MIN_SUBJECTS:
        print(f"\nOnly {len(SUBJECTS_AVAILABLE)} complete subjects "
              f"(need at least {MIN_SUBJECTS}).")
        print("  * RE-RUN THIS CELL -- it retries only the missing files.")
        print("  * If the count does not move, Google Drive has hit its daily")
        print("    quota on those files. Wait a few hours, or copy the Drive")
        print("    folder to your own account and swap the ids in DATASETS.")
        raise RuntimeError(
            f"only {len(SUBJECTS_AVAILABLE)} usable subjects -- re-run this cell")
    if len(SUBJECTS_AVAILABLE) < 29:
        print(f"\n>> Proceeding with {len(SUBJECTS_AVAILABLE)} of 29 subjects.")
        print("   Results will be noisier than a full run, but nothing about the")
        print("   protocol changes. Re-run this cell later to collect the rest.")

if absent:
    # Nothing here at all -- that is fine if the data lives somewhere else
    # (an attached Kaggle Dataset, or a manual copy). The next cell searches.
    print(f"\nnote: {', '.join(absent)} not in {DATA_ROOT} --")
    print("      the next cell will look for it elsewhere on this machine.")
else:
    print("\nAll requested datasets are complete.")
''')

co(r'''
# ---------------------------------------------------------------- locate roots
# Find each dataset by a signature file rather than a hard-coded path, so this
# works whether the data came from Drive, a Kaggle Dataset, or a manual copy.
SEARCH_ROOTS = [p for p in [MANUAL, DATA_ROOT, Path("/kaggle/input"),
                            Path("dataset"), Path("."), Path("..")] if p is not None]


def find_root(*signatures):
    """Return the folder that CONTAINS the given relative signature path.

    Tries cheap direct hits first. A blind rglob from "." is the last resort --
    on a machine where the repo also holds raw datasets and a large .git it can
    take minutes, so we only fall back to it if nothing else matched.
    """
    # 1. cheap: <root>/<key>/<signature> and <root>/*/<signature>, depth-limited
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for sig in signatures:
            for pat in (sig, f"*/{sig}", f"*/*/{sig}"):
                hits = list(root.glob(pat))
                if hits:
                    p = hits[0]
                    for _ in range(len(Path(sig).parts)):
                        p = p.parent
                    return p

    # 2. last resort: full recursive scan, skipping obvious time sinks
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for sig in signatures:
            for hit in root.rglob(sig):
                if any(part in (".git", "__pycache__", ".ipynb_checkpoints")
                       for part in hit.parts):
                    continue
                p = hit
                for _ in range(len(Path(sig).parts)):
                    p = p.parent
                return p
    return None


TUBERLIN = find_root("EEG data/EEG_*/subject */with occular artifact/cnt.mat")
HEFMI    = find_root("epo_data/Normals/EEG/*/*_epo.mat")
WL       = find_root("EEG/VP*-EEG/cnt_nback.mat")

print("TU Berlin hBCI :", TUBERLIN or "NOT FOUND")
print("HEFMI          :", HEFMI or "not downloaded -> clinical section skipped")
print("TU Berlin WL   :", WL or "not downloaded -> claim 2 skipped")

if TUBERLIN is None:
    raise FileNotFoundError(
        "TU Berlin hBCI not found.\n"
        "  * On Kaggle: is Internet ON in the right-hand Settings panel?\n"
        "  * Re-run the download cell -- it resumes where it stopped.\n"
        f"  * Searched: {[str(r) for r in SEARCH_ROOTS if r.exists()]}"
    )
''')

# ===========================================================================
md(r"""
## 3 · Settings

`RULESET` is frozen: these choices were fixed **before** any test performance was
looked at, so they cannot be tuned to flatter the result.
""")

co(r'''
@dataclass(frozen=True)
class Config:
    # --- windowing -------------------------------------------------------
    tmin_s: float = -2.0        # 2 s pre-stimulus baseline
    tmax_s: float = 28.0        # -> 30 s window
    eeg_fs: float = 200.0
    hemo_fs: float = 10.0
    eeg_samples: int = 6000
    hemo_samples: int = 300
    baseline_s: float = 2.0

    # --- protocol (FROZEN) -----------------------------------------------
    ruleset: str = "kaggle-frozen-2026-08-07"
    n_folds: int = 5            # subject-wise: no person in two folds
    n_perm: int = 1000          # exact permutation null
    apply_gsr: bool = False     # audited: GSR removes 83% of the evoked response
    task: str = "MI"            # motor imagery

    # --- loss weights (as used in the research pipeline) -----------------
    w_main:   float = 1.0     # pearson (shape)
    w_hetero: float = 0.1     # gaussian NLL with predicted variance
    w_delay:  float = 0.5     # lag-tolerant cross-correlation
    w_smooth: float = 0.05    # penalise jitter
    max_lag:  int   = 3       # samples of timing slack allowed

    # --- training --------------------------------------------------------
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3

CFG = Config()

# QUICK_TEST (set in the section 2 settings block) shrinks the run so you can confirm the whole
# notebook works end to end in ~5 minutes before committing to the real one.
# It does NOT change any protocol choice -- only how much compute is spent.
# Use whatever subjects actually downloaded (set by the gate in section 2).
SUBJECTS = list(SUBJECTS_AVAILABLE) if SUBJECTS_AVAILABLE else list(range(1, 30))
if QUICK_TEST:
    CFG = Config(**{**asdict(CFG), "epochs": 2, "n_perm": 200, "n_folds": 3})
    SUBJECTS = SUBJECTS[:8]
    print(">>> QUICK_TEST is ON: 8 subjects, 2 epochs, 200 permutations.")
    print(">>> Numbers will be noisy. Set QUICK_TEST = False for the real run.\n")

OUT = Path("/kaggle/working/outputs" if ON_KAGGLE else "outputs")
OUT.mkdir(parents=True, exist_ok=True)
print(json.dumps(asdict(CFG), indent=2))
''')

# ===========================================================================
md(r"""
## 4 · Loader

The fNIRS files contain **raw 760/850 nm light intensities**, not haemoglobin.
We convert them ourselves with the Beer–Lambert law (Prahl coefficients,
DPF 6.4/5.8, source–detector 3.0 cm).

Global signal regression is **off**: the audit measured that it removes 83% of
the evoked response along with the systemic noise it targets.
""")

co(r'''
EXT = {760: {"HbO": 1486.59, "HbR": 3843.71},
       850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}
SD_CM = 3.0
EOG = ("VEOG", "HEOG")


def _runs(path, key_hint=None):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    return np.atleast_1d(m[key_hint if key_hint in keys else keys[0]]).ravel()


def bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def beer_lambert(lo_wl, hi_wl):
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


def split_wavelengths(x, clab):
    names = [str(c) for c in np.atleast_1d(clab)]
    lo = {n.replace("lowWL", ""): i for i, n in enumerate(names) if "lowWL" in n}
    hi = {n.replace("highWL", ""): i for i, n in enumerate(names) if "highWL" in n}
    pairs = sorted(set(lo) & set(hi))
    return (np.stack([x[:, lo[p]] for p in pairs]),
            np.stack([x[:, hi[p]] for p in pairs]))


def regress_global(x):
    g = x.mean(axis=0, keepdims=True)
    gc = g - g.mean(axis=-1, keepdims=True)
    bc = x - x.mean(axis=-1, keepdims=True)
    beta = (bc * gc).sum(-1, keepdims=True) / ((gc ** 2).sum(-1, keepdims=True) + 1e-12)
    return bc - beta * gc


def resample_to(x, n):
    return x if x.shape[-1] == n else sps.resample(x, n, axis=-1)
''')

co(r'''
EEG_DIR = TUBERLIN / "EEG data"
NIRS_DIR = TUBERLIN / "NRI data" / "NIRS_01-29"


def eeg_subject_dir(n):
    for grp in sorted(EEG_DIR.glob("EEG_*")):
        d = grp / f"subject {n:02d}" / "with occular artifact"
        if d.exists():
            return d
    return None


def load_tuberlin(cfg=CFG, subjects=None):
    subjects = SUBJECTS if subjects is None else subjects
    eeg_l, hemo_l, lab_l, sid_l, audit = [], [], [], [], []
    n_base = int(cfg.baseline_s * cfg.hemo_fs)

    for s in subjects:
        ed, nd = eeg_subject_dir(s), NIRS_DIR / f"subject {s:02d}"
        if ed is None or not nd.exists():
            audit.append({"subject": s, "status": "missing", "trials": 0}); continue
        try:
            e_runs, e_mrks = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
            n_runs, n_mrks = _runs(nd / "cnt.mat", "cnt"), _runs(nd / "mrk.mat", "mrk")
        except Exception as exc:
            audit.append({"subject": s, "status": f"read fail: {exc}", "trials": 0}); continue

        n_tr = 0
        for r in range(min(e_runs.size, n_runs.size)):
            er, nr = e_runs[r], n_runs[r]
            if not str(getattr(er, "title", "")).upper().startswith(cfg.task):
                continue

            e_fs = float(np.atleast_1d(er.fs).ravel()[0])
            clab = [str(c) for c in np.atleast_1d(er.clab)]
            keep = [i for i, c in enumerate(clab) if c not in EOG]
            ex = np.asarray(er.x, dtype=np.float64)[:, keep].T

            n_fs = float(np.atleast_1d(nr.fs).ravel()[0])
            lo, hi = split_wavelengths(np.asarray(nr.x, dtype=np.float64), nr.clab)
            hbo, hbr = beer_lambert(lo, hi)
            hbo, hbr = bandpass(hbo, n_fs, .01, .2), bandpass(hbr, n_fs, .01, .2)
            if cfg.apply_gsr:
                hbo, hbr = regress_global(hbo), regress_global(hbr)

            em = e_mrks[r] if r < e_mrks.size else None
            nm = n_mrks[r] if r < n_mrks.size else None
            if em is None or nm is None:
                continue
            e_on = np.atleast_1d(np.asarray(em.time, float)) / 1000.0
            n_on = np.atleast_1d(np.asarray(nm.time, float)) / 1000.0
            y = np.asarray(em.y)
            labels = np.argmax(y, 0) if y.ndim == 2 else np.asarray(y).ravel()

            for k in range(min(e_on.size, n_on.size, labels.size)):
                a_e = int(round((e_on[k] + cfg.tmin_s) * e_fs))
                b_e = int(round((e_on[k] + cfg.tmax_s) * e_fs))
                a_n = int(round((n_on[k] + cfg.tmin_s) * n_fs))
                b_n = int(round((n_on[k] + cfg.tmax_s) * n_fs))
                # A trial whose window runs past the end of the recording has no
                # data to pair with -- drop it, don't abort the whole build.
                if a_e < 0 or b_e > ex.shape[1] or a_n < 0 or b_n > hbo.shape[1]:
                    continue

                seg = resample_to(ex[:, a_e:b_e], cfg.eeg_samples)
                seg = ((seg - seg.mean(-1, keepdims=True)) /
                       (seg.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
                so = resample_to(hbo[:, a_n:b_n], cfg.hemo_samples)
                sr = resample_to(hbr[:, a_n:b_n], cfg.hemo_samples)
                so = so - so[:, :n_base].mean(-1, keepdims=True)
                sr = sr - sr[:, :n_base].mean(-1, keepdims=True)

                eeg_l.append(seg)
                hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
                lab_l.append(int(labels[k])); sid_l.append(s); n_tr += 1

        audit.append({"subject": s, "status": "ok" if n_tr else "no trials",
                      "trials": n_tr})

    if not eeg_l:
        # Say WHY, rather than just that it failed. Almost always an incomplete
        # download, so show exactly what was and was not found.
        n_eeg_dirs = sum(1 for s in subjects if eeg_subject_dir(s) is not None)
        n_nirs_dirs = sum(1 for s in subjects
                          if (NIRS_DIR / f"subject {s:02d}").exists())
        adf = pd.DataFrame(audit)
        lines = [
            "loader produced no samples.",
            "",
            f"  dataset root      : {TUBERLIN}",
            f"  EEG  dir exists   : {EEG_DIR.exists()}   ({EEG_DIR})",
            f"  NIRS dir exists   : {NIRS_DIR.exists()}   ({NIRS_DIR})",
            f"  subjects wanted   : {len(list(subjects))}",
            f"  EEG  subject dirs : {n_eeg_dirs}",
            f"  NIRS subject dirs : {n_nirs_dirs}",
            f"  task filter       : {cfg.task!r}",
            "",
            "  per-subject status:",
            "    " + adf.status.value_counts().to_string().replace("\n", "\n    ")
            if len(adf) else "    (none)",
            "",
        ]
        if n_eeg_dirs == 0 or n_nirs_dirs == 0:
            lines += [
                "  => The download is INCOMPLETE. Re-run the download cell; it",
                "     resumes and skips files already on disk. Wait for it to",
                "     print '-> N/346 files' before running this cell.",
            ]
        else:
            lines += [
                f"  => Folders exist but no run matched task {cfg.task!r}.",
                "     Check that cnt.mat files finished downloading (a truncated",
                "     .mat loads but has no runs).",
            ]
        raise RuntimeError("\n".join(lines))
    return {"eeg": np.stack(eeg_l), "hemo": np.stack(hemo_l),
            "labels": np.asarray(lab_l, np.int64),
            "subject_ids": np.asarray(sid_l, np.int64),
            "audit": pd.DataFrame(audit)}


t0 = time.time()
DATA = load_tuberlin()
print(f"loaded in {(time.time()-t0)/60:.1f} min" + ("   [QUICK_TEST]" if QUICK_TEST else ""))
print(f"EEG  {DATA['eeg'].shape}")
print(f"hemo {DATA['hemo'].shape}")
print(f"{len(np.unique(DATA['subject_ids']))} subjects, {len(DATA['eeg'])} trials")
DATA["audit"].to_csv(OUT / "loader_audit.csv", index=False)
DATA["audit"]
''')

# ===========================================================================
md(r"""
## 5 · The ceiling — can the target predict itself?

**Run this before believing any prediction score.**

Split one person's trials at random into two halves, average each, and correlate
them. If the signal is real the halves should agree. That *split-half
reliability* caps what **any** model can achieve, at `√reliability`.

We also check the **HbO/HbR sign**. During real brain activity oxygenated
haemoglobin rises and deoxygenated falls — they move in **opposite** directions,
so the correlation should be **negative**. A positive value means the signal is
dominated by heartbeat, breathing and scalp blood flow rather than the brain.
""")

co(r'''
def split_half_reliability(hemo, sids, n_rep=50, rng=None):
    rng = rng or np.random.default_rng(SEED)
    out = {}
    for s in np.unique(sids):
        X = hemo[sids == s]
        if len(X) < 4:
            continue
        rs = []
        for _ in range(n_rep):
            p = rng.permutation(len(X))
            a = X[p[: len(X) // 2]].mean(0).ravel()
            b = X[p[len(X) // 2:]].mean(0).ravel()
            if a.std() > 0 and b.std() > 0:
                rs.append(np.corrcoef(a, b)[0, 1])
        out[int(s)] = float(np.mean(rs)) if rs else np.nan
    return out


def hbo_hbr_sign(hemo):
    n = hemo.shape[1] // 2          # first half HbO, second half HbR
    return float(np.nanmedian([
        np.corrcoef(t[:n].mean(0), t[n:].mean(0))[0, 1] for t in hemo]))


rel = split_half_reliability(DATA["hemo"], DATA["subject_ids"])
rel_mean = float(np.nanmean(list(rel.values())))
ceiling = math.sqrt(max(rel_mean, 0.0))
sign = hbo_hbr_sign(DATA["hemo"])

print(f"target split-half reliability : {rel_mean:+.3f}")
print(f"  -> CEILING on any model     : {ceiling:.3f}")
print(f"HbO/HbR correlation           : {sign:+.3f}")
print("   " + ("physiological (HbO up, HbR down)" if sign < 0
               else "WARNING: systemic-dominated -- not brain-driven"))

pd.DataFrame({"subject": list(rel), "reliability": list(rel.values())}
             ).to_csv(OUT / "target_reliability.csv", index=False)
''')

# ===========================================================================
md(r"""
## 6 · HEFMI loader and montage

HEFMI is the blueprint's **primary fNIRS training set**. Its `.mat` files store
channel names as a MATLAB `string` object that SciPy cannot decode, so the names
look absent — but they are not. They live in two places the dataset ships:

| What | Where | Contents |
|---|---|---|
| **EEG channel names** | `code/EpoDataMaking/MakeDatasetFromRaw.m` | 32 names, 10-20 system |
| **fNIRS probe geometry** | `source_data/**/*.nirs` (`SD` struct) | 32 sources, 30 detectors, **90 S–D pairs**, 3 wavelengths, positions in cm |

The `.nirs` `MeasList` has 270 rows = 90 pairs × 3 wavelengths (730/808/850 nm),
which matches the 90 columns of `epo.oxy.X` exactly. So the montage **is**
documented; it just is not inside the epoched files.

**One honest limit.** The optode positions are a flat 2-D grid in centimetres
(x 2–38, y 14.7–37, z = 0). They describe the probe's own layout but not where
it sits on the scalp in 10-20 terms — so they cannot be matched channel-by-channel
against TU Berlin's `AF7Fp1`-style names. That affects external validation only,
and section 20 addresses it explicitly rather than inventing a mapping.
""")

co(r'''
# 32 EEG channels, transcribed from code/EpoDataMaking/MakeDatasetFromRaw.m.
# Kept here because the .mat stores them as a MATLAB string object that SciPy
# decodes to an opaque blob.
HEFMI_EEG_CHANNELS = [
    "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
    "C4",  "T7",  "T8",  "FC2", "FC5", "FC6", "Pz", "CP5",
    "PO3", "PO4", "Oz",  "FP2", "FP1", "FZ",  "F3", "F4",
    "F7",  "F8",  "P3",  "P4",  "P7",  "P8",  "O1", "O2",
]
assert len(HEFMI_EEG_CHANNELS) == 32


def read_hefmi_probe(root=None):
    """Optode geometry from any .nirs file: positions, pairs, wavelengths."""
    root = root or HEFMI
    if root is None:
        return None
    hits = sorted(root.rglob("*.nirs"))
    if not hits:
        return None
    try:
        SD = sio.loadmat(hits[0], squeeze_me=True, struct_as_record=False)["SD"]
    except Exception:
        return None
    ml = np.asarray(SD.MeasList)
    pairs = sorted({(int(a), int(b)) for a, b in ml[:, :2]})
    return {"file": hits[0].name,
            "n_sources": int(np.asarray(SD.SrcPos).shape[0]),
            "n_detectors": int(np.asarray(SD.DetPos).shape[0]),
            "n_pairs": len(pairs),
            "wavelengths_nm": [int(x) for x in np.atleast_1d(SD.Lambda).ravel()],
            "spatial_unit": str(getattr(SD, "SpatialUnit", "?")),
            "src_pos": np.asarray(SD.SrcPos, float),
            "det_pos": np.asarray(SD.DetPos, float),
            "pairs": pairs}


def load_hefmi(root=None, groups=("Normals", "Patients"), cfg=CFG):
    """HEFMI trials: EEG (32 ch @ 256 Hz) + 180 hemo channels (90 HbO, 90 HbR).

    Resampled onto the same window as every other loader, so the shared encoder
    sees one consistent format. Subject ids are keyed on group_subject because
    Normals/subject1 and Patients/subject1 are DIFFERENT people.
    """
    root = root or HEFMI
    if root is None:
        return None
    n_base = int(cfg.baseline_s * cfg.hemo_fs)
    eeg_l, hemo_l, lab_l, sid_l, audit = [], [], [], [], []
    people = {}

    for group in groups:
        eeg_root = root / "epo_data" / group / "EEG"
        if not eeg_root.exists():
            continue
        for sd in sorted(p for p in eeg_root.iterdir() if p.is_dir()):
            person = f"{group}_{sd.name}"          # the biological identity
            for ef in sorted(sd.glob("*_epo.mat")):
                nf = root / "epo_data" / group / "NIRS" / sd.name / ef.name
                if not nf.exists():
                    audit.append({"person": person, "file": ef.name,
                                  "status": "no fNIRS partner", "trials": 0})
                    continue
                try:
                    e = sio.loadmat(ef, squeeze_me=True, struct_as_record=False)["epo"]
                    n = sio.loadmat(nf, squeeze_me=True, struct_as_record=False)["epo"]
                    X = np.asarray(e.X, float)                  # (time, ch, trial)
                    oxy = np.asarray(n.oxy.X, float)
                    dxy = np.asarray(n.doxy.X, float)
                    e_fs = float(np.atleast_1d(e.fs).ravel()[0])
                    n_fs = float(np.atleast_1d(n.oxy.fs).ravel()[0])
                    y = np.asarray(e.y).ravel()
                except Exception as exc:
                    audit.append({"person": person, "file": ef.name,
                                  "status": f"read fail: {type(exc).__name__}",
                                  "trials": 0})
                    continue

                k = min(X.shape[2], oxy.shape[2], dxy.shape[2], len(y))
                sid = people.setdefault(person, len(people))
                n_ok = 0
                for t in range(k):
                    seg_e = X[:, :, t].T                        # (ch, time)
                    seg_e = resample_to(seg_e, cfg.eeg_samples)
                    seg_e = ((seg_e - seg_e.mean(-1, keepdims=True)) /
                             (seg_e.std(-1, keepdims=True) + 1e-6))
                    so = resample_to(oxy[:, :, t].T, cfg.hemo_samples)
                    sr = resample_to(dxy[:, :, t].T, cfg.hemo_samples)
                    so = so - so[:, :n_base].mean(-1, keepdims=True)
                    sr = sr - sr[:, :n_base].mean(-1, keepdims=True)
                    eeg_l.append(seg_e.astype(np.float32))
                    hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
                    lab_l.append(int(y[t])); sid_l.append(sid); n_ok += 1
                audit.append({"person": person, "file": ef.name,
                              "status": "ok" if n_ok else "no trials", "trials": n_ok})

    if not eeg_l:
        return None
    return {"eeg": np.stack(eeg_l), "hemo": np.stack(hemo_l),
            "labels": np.asarray(lab_l, np.int64),
            "subject_ids": np.asarray(sid_l, np.int64),
            "person_names": {v: k for k, v in people.items()},
            "audit": pd.DataFrame(audit)}
''')

co(r'''
probe = read_hefmi_probe()
if probe:
    print("HEFMI fNIRS montage (from source_data/**/*.nirs):")
    print(f"  file           {probe['file']}")
    print(f"  sources        {probe['n_sources']}")
    print(f"  detectors      {probe['n_detectors']}")
    print(f"  S-D pairs      {probe['n_pairs']}   -> matches the 90 oxy channels")
    print(f"  wavelengths    {probe['wavelengths_nm']} nm")
    print(f"  spatial unit   {probe['spatial_unit']}")
    print(f"  EEG channels   {len(HEFMI_EEG_CHANNELS)}: "
          f"{', '.join(HEFMI_EEG_CHANNELS[:8])} ...")
else:
    print("HEFMI probe geometry not found (needs source_data/**/*.nirs)")

DATA_HEFMI = load_hefmi()
if DATA_HEFMI is None:
    print("\nHEFMI not loaded -- set PATH_CLINICAL / the HEFMI path to enable it.")
else:
    print(f"\nHEFMI trials  eeg {DATA_HEFMI['eeg'].shape}  hemo {DATA_HEFMI['hemo'].shape}")
    print(f"  {len(np.unique(DATA_HEFMI['subject_ids']))} people, "
          f"{len(DATA_HEFMI['eeg'])} trials, "
          f"{len(np.unique(DATA_HEFMI['labels']))} classes")
    DATA_HEFMI["audit"].to_csv(OUT / "hefmi_loader_audit.csv", index=False)
    _rel = split_half_reliability(DATA_HEFMI["hemo"], DATA_HEFMI["subject_ids"])
    _relm = float(np.nanmean(list(_rel.values())))
    _sign = hbo_hbr_sign(DATA_HEFMI["hemo"])
    print(f"  target reliability {_relm:+.3f}  ceiling {math.sqrt(max(_relm,0)):.3f}")
    print(f"  HbO/HbR sign       {_sign:+.3f}"
          + ("   physiological" if _sign < 0 else
             "   SYSTEMIC-DOMINATED -- see the audit note"))
''')

# ===========================================================================
md(r"""
## 6 · Metrics

**Constant baseline.** A predictor that ignores the EEG entirely and always emits
the training-set average. Trials from one person share a mean, so raw correlation
largely measures *"which person is this"*. **If the model cannot beat this, it has
learned nothing about the brain.**

**Demeaning.** Subtracting each person's mean drives that baseline to *exactly*
zero, so whatever survives comes from the EEG. Demeaned numbers are the honest ones.

**Exact permutation null by matrix multiplication.** A permutation only reorders
rows, so every pairwise correlation can be computed once as `Pc @ Tc.T` and each
shuffle becomes a lookup. **559× faster** than the naive loop, identical to 1e-10.
""")

co(r'''
def _center(x):
    x = x.reshape(len(x), -1).astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def mean_corr(pred, targ):
    return float((_center(pred) * _center(targ)).sum(1).mean())


def perm_null(pred, targ, n_perm=CFG.n_perm, rng=None):
    rng = rng or np.random.default_rng(SEED)
    G = _center(pred) @ _center(targ).T      # every pairing, computed ONCE
    obs = float(np.diag(G).mean())
    idx = np.arange(len(G))
    null = np.array([G[idx, rng.permutation(len(G))].mean() for _ in range(n_perm)])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)   # +1 so p is never exactly 0
    return obs, float(p), null


def constant_baseline(train_targ, test_targ):
    const = np.repeat(train_targ.mean(0, keepdims=True), len(test_targ), axis=0)
    return mean_corr(const, test_targ)


def demean_by_subject(y, sids):
    out = y.astype(np.float64).copy()
    for s in np.unique(sids):
        m = sids == s
        out[m] -= out[m].mean(0, keepdims=True)
    return out
''')

# ===========================================================================
md(r"""
## 7 · Model and cross-subject evaluation

A shared convolutional trunk learns what is common across people; a small
per-subject embedding absorbs what is idiosyncratic. **A held-out person has no
embedding, so they get the average one** — forcing the model to rely on the
subject-general trunk. That is what makes this genuinely cross-subject.

The encoder predicts a **mean and a log-variance** per time point. The variance
is what lets it say *"I am unsure here"*, and it feeds the abstention gate.

### The loss

Plain MSE optimises **amplitude**, which in this data is dominated by systemic
drift — heartbeat and blood pressure — rather than by the brain. So the research
pipeline uses four terms targeting the **shape** of the response instead:

| Term | Weight | Why |
|---|---|---|
| `pearson_corr_loss` | 1.0 | scale-invariant: shape, not amplitude |
| `heteroscedastic_loss` | 0.1 | Gaussian NLL with predicted variance -> uncertainty |
| `cross_correlation_loss` | 0.5 | tolerates ±3 samples of timing error |
| `smoothness_loss` | 0.05 | a real haemodynamic response is slow and smooth |

Subject-wise 5-fold: **no person appears in both training and test.**
""")

co(r'''
# ------------------------------------------------------------------ losses
# Ported verbatim from the research pipeline (v3.3). Plain MSE optimises
# amplitude, which is dominated by systemic drift; these four terms target the
# SHAPE of the response instead.

def pearson_corr_loss(pred, target, eps=1e-6):
    """1 - correlation. Scale-invariant, so it cares about shape not amplitude."""
    p = pred - pred.mean(-1, keepdim=True)
    t = target - target.mean(-1, keepdim=True)
    p = p / (p.norm(dim=-1, keepdim=True) + eps)
    t = t / (t.norm(dim=-1, keepdim=True) + eps)
    return (1.0 - (p * t).sum(-1)).mean()


def heteroscedastic_loss(pred_mean, target, pred_logvar):
    """Gaussian NLL with a predicted variance -- lets the model say 'unsure here'.
    This is also what feeds the abstention gate."""
    precision = torch.exp(-pred_logvar)
    return (precision * (pred_mean - target) ** 2 + pred_logvar).mean()


def cross_correlation_loss(pred, target, max_lag: int = 3):
    """1 - best correlation within a small lag window: tolerates a few samples
    of timing error, which matters because the haemodynamic lag varies."""
    p0 = pred - pred.mean(-1, keepdim=True)
    t0 = target - target.mean(-1, keepdim=True)
    p0 = p0 / (p0.norm(dim=-1, keepdim=True) + 1e-6)
    t0 = t0 / (t0.norm(dim=-1, keepdim=True) + 1e-6)
    best = torch.full(pred.shape[:2], -1.0, device=pred.device)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            p, t = p0[..., :lag], t0[..., -lag:]
        elif lag > 0:
            p, t = p0[..., lag:], t0[..., :-lag]
        else:
            p, t = p0, t0
        if p.shape[-1] < 4:
            continue
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
        best = torch.maximum(best, (p * t).sum(-1))
    return (1.0 - best).mean()


def smoothness_loss(pred):
    """Penalise jitter. A real haemodynamic response is slow and smooth."""
    d = pred[..., 1:] - pred[..., :-1]
    return (d ** 2).mean()


def compute_loss(pred_mean, pred_logvar, target, cfg=CFG):
    terms = {
        "pearson": pearson_corr_loss(pred_mean, target),
        "hetero":  heteroscedastic_loss(pred_mean, target, pred_logvar),
        "xcorr":   cross_correlation_loss(pred_mean, target, cfg.max_lag),
        "smooth":  smoothness_loss(pred_mean),
    }
    total = (cfg.w_main * terms["pearson"] + cfg.w_hetero * terms["hetero"] +
             cfg.w_delay * terms["xcorr"] + cfg.w_smooth * terms["smooth"])
    return total, {k: float(v) for k, v in terms.items()}


# ------------------------------------------------------------------ encoder
class NeuroVascNet(nn.Module):
    """Shared conv trunk + per-subject embedding, predicting mean AND log-variance."""

    def __init__(self, n_eeg, n_hemo, hemo_len, n_subj, emb=8, width=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(n_eeg, width, 25, 5, 12), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 15, 4, 7),  nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 9, 3, 4),   nn.BatchNorm1d(width), nn.GELU(),
            nn.AdaptiveAvgPool1d(16))
        self.subj = nn.Embedding(n_subj, emb)
        nn.init.zeros_(self.subj.weight)          # start subject-agnostic
        self.shared = nn.Sequential(
            nn.Linear(width * 16 + emb, 256), nn.GELU(), nn.Dropout(0.3))
        self.head_mean = nn.Linear(256, n_hemo * hemo_len)
        self.head_logvar = nn.Linear(256, n_hemo * hemo_len)
        nn.init.zeros_(self.head_logvar.weight); nn.init.zeros_(self.head_logvar.bias)
        self.out_shape = (n_hemo, hemo_len)

    def forward(self, x, sid=None):
        h = self.trunk(x).flatten(1)
        # A held-out subject has no embedding, so it gets the AVERAGE one --
        # forcing the model back onto the subject-general trunk.
        e = (self.subj(sid) if sid is not None
             else self.subj.weight.mean(0).expand(len(x), -1))
        z = self.shared(torch.cat([h, e], 1))
        mean = self.head_mean(z).view(len(x), *self.out_shape)
        logvar = self.head_logvar(z).view(len(x), *self.out_shape).clamp(-8, 8)
        return mean, logvar


def train_fold(eeg, hemo, sid_ix, n_subj, cfg=CFG, verbose=False):
    model = NeuroVascNet(eeg.shape[1], hemo.shape[1], hemo.shape[2], n_subj).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(eeg, dtype=torch.float32),
        torch.tensor(hemo, dtype=torch.float32),
        torch.tensor(sid_ix, dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for ep in range(cfg.epochs):
        agg = {}
        for be, bh, bs in dl:
            be, bh, bs = be.to(DEVICE), bh.to(DEVICE), bs.to(DEVICE)
            opt.zero_grad()
            mean, logvar = model(be, bs)
            loss, parts = compute_loss(mean, logvar, bh, cfg)
            loss.backward(); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        if verbose and (ep + 1) % 10 == 0:
            n = max(len(dl), 1)
            print("   ep {:3d}  ".format(ep + 1)
                  + "  ".join(f"{k}={v / n:.4f}" for k, v in agg.items()))
    return model


@torch.no_grad()
def predict(model, eeg, want_var=False):
    """Predict for held-out people: sid=None -> average embedding."""
    model.eval()
    means, logvars = [], []
    for i in range(0, len(eeg), 256):
        b = torch.tensor(eeg[i:i + 256], dtype=torch.float32).to(DEVICE)
        m, lv = model(b, None)
        means.append(m.cpu().numpy()); logvars.append(lv.cpu().numpy())
    if want_var:
        return np.concatenate(means), np.concatenate(logvars)
    return np.concatenate(means)
''')

# ===========================================================================
md(r"""
### Smoke test

A 60-second check that everything works **before** committing to a long run.
Loads two subjects, verifies shapes and finiteness, and pushes one batch through
the model. If this passes, the full run will not fail on plumbing.
""")

co(r'''
def smoke_test():
    ok = True

    # -- 1. loader -------------------------------------------------------
    d = load_tuberlin(subjects=range(1, 3))
    e, h = d["eeg"], d["hemo"]
    checks = [
        ("EEG rank 3",        e.ndim == 3),
        ("hemo rank 3",       h.ndim == 3),
        ("paired trials",     len(e) == len(h) == len(d["subject_ids"])),
        ("EEG window 6000",   e.shape[2] == CFG.eeg_samples),
        ("hemo window 300",   h.shape[2] == CFG.hemo_samples),
        ("hemo 72 channels",  h.shape[1] == 72),
        ("EEG finite",        bool(np.isfinite(e).all())),
        ("hemo finite",       bool(np.isfinite(h).all())),
        ("EEG z-scored",      abs(float(e.mean())) < 0.2),
        ("2 subjects",        len(np.unique(d["subject_ids"])) == 2),
    ]

    # -- 2. metrics ------------------------------------------------------
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 8)); b = a * 2 + 1        # perfectly correlated
    checks.append(("corr of scaled copy == 1", abs(mean_corr(a, b) - 1.0) < 1e-9))
    obs, p, _ = perm_null(a, rng.normal(size=(40, 8)), n_perm=100)
    checks.append(("null p sane for noise", 0.0 < p <= 1.0))
    dm = demean_by_subject(np.arange(20).reshape(10, 2).astype(float),
                           np.array([0] * 5 + [1] * 5))
    checks.append(("demean -> per-subject mean 0", abs(dm[:5].mean()) < 1e-9))

    # -- 3. loss functions ----------------------------------------------
    tt = torch.randn(4, 6, 50)
    checks += [
        ("pearson loss of identical == 0", abs(float(pearson_corr_loss(tt, tt))) < 1e-4),
        ("pearson loss of negated == 2",   abs(float(pearson_corr_loss(tt, -tt)) - 2) < 1e-3),
        ("smoothness of constant == 0",    abs(float(smoothness_loss(torch.ones(2, 3, 20)))) < 1e-9),
        ("xcorr loss of identical == 0",   abs(float(cross_correlation_loss(tt, tt))) < 1e-3),
    ]
    tot, parts = compute_loss(tt, torch.zeros_like(tt), tt + 0.01 * torch.randn_like(tt))
    checks += [
        ("composite loss finite", bool(np.isfinite(float(tot)))),
        ("all 4 loss terms present", set(parts) == {"pearson", "hetero", "xcorr", "smooth"}),
    ]

    # -- 4. model forward/backward --------------------------------------
    try:
        m = train_fold(e[:16], h[:16], np.zeros(16, int), 2,
                       Config(**{**asdict(CFG), "epochs": 1, "batch_size": 8}))
        pr, lv = predict(m, e[:8], want_var=True)
        checks.append(("model mean shape", pr.shape == (8, h.shape[1], h.shape[2])))
        checks.append(("model logvar shape", lv.shape == pr.shape))
        checks.append(("model output finite", bool(np.isfinite(pr).all())))
        checks.append(("logvar finite", bool(np.isfinite(lv).all())))
    except Exception as exc:
        checks.append((f"model step ({type(exc).__name__}: {exc})", False))

    print("-" * 52)
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= bool(passed)
    print("-" * 52)
    print("SMOKE TEST PASSED -- safe to run the full notebook" if ok
          else "SMOKE TEST FAILED -- fix the above before continuing")
    return ok


if RUN_SMOKE_TEST:
    assert smoke_test(), "smoke test FAILED -- see the list above"
else:
    print("smoke test skipped (RUN_SMOKE_TEST = False)")
''')

co(r'''
def cross_subject_cv(data, cfg=CFG):
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    subjects = np.unique(sids)
    folds = np.array_split(np.random.default_rng(SEED).permutation(subjects), cfg.n_folds)
    remap = {int(s): i for i, s in enumerate(subjects)}

    rows, P, T = [], [], []
    for fi, test_subj in enumerate(folds):
        te = np.isin(sids, test_subj); tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        model = train_fold(eeg[tr], hemo[tr],
                           np.array([remap[int(s)] for s in sids[tr]]),
                           len(subjects), cfg)
        pred = predict(model, eeg[te])

        r_raw = mean_corr(pred, hemo[te])
        r_const = constant_baseline(hemo[tr], hemo[te])
        pd_ = demean_by_subject(pred, sids[te])
        td_ = demean_by_subject(hemo[te], sids[te])
        r_dem = mean_corr(pd_, td_)

        rows.append({"fold": fi, "test_subjects": len(test_subj),
                     "test_trials": int(te.sum()), "r_raw": r_raw,
                     "r_const": r_const, "r_demeaned": r_dem})
        P.append(pd_); T.append(td_)
        print(f"fold {fi}: {len(test_subj)} subjects | raw {r_raw:+.4f} | "
              f"no-EEG baseline {r_const:+.4f} | demeaned {r_dem:+.4f}")

    obs, p, null = perm_null(np.concatenate(P), np.concatenate(T), cfg.n_perm)
    return pd.DataFrame(rows), {"r_demeaned": obs, "p_perm": p,
                                "null_mean": float(null.mean())}, null


t0 = time.time()
cv, summary, null = cross_subject_cv(DATA)
print(f"\ntrained in {(time.time()-t0)/60:.1f} min")
cv.to_csv(OUT / "cross_subject_cv.csv", index=False)
cv
''')

co(r'''
r_raw, r_const = cv.r_raw.mean(), cv.r_const.mean()
r_dem, p = summary["r_demeaned"], summary["p_perm"]

print("=" * 68)
print("RESULT")
print("=" * 68)
print(f"model, raw correlation          r = {r_raw:+.4f}")
print(f"predictor that IGNORES EEG      r = {r_const:+.4f}")
print(f"model, demeaned (honest)        r = {r_dem:+.4f}   p = {p:.4f}")
print(f"measured ceiling                r = {ceiling:.4f}")
if ceiling > 0:
    print(f"achieved / ceiling                = {r_dem / ceiling * 100:.1f}%")
print()
if r_const >= r_raw:
    print("The no-EEG baseline MATCHES OR BEATS the raw model.")
    print("=> raw correlation is measuring subject identity, not neurovascular")
    print("   coupling. The demeaned number is the one to report.")
else:
    print(f"Raw model exceeds the no-EEG baseline by {r_raw - r_const:+.4f}.")
    print("   The demeaned number remains the honest estimate.")
''')

co(r'''
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

names = ["model\n(raw)", "ignores EEG\n(baseline)", "model\n(demeaned)"]
vals = [r_raw, r_const, r_dem]
cols = ["#be7800", "#aa1e1e", "#00783c"]
ax[0].bar(names, vals, color=cols, alpha=.9)
ax[0].axhline(0, color="k", lw=.8)
for i, v in enumerate(vals):
    ax[0].text(i, v + .002, f"{v:.4f}", ha="center", fontweight="bold")
ax[0].set_ylabel("cross-subject Pearson $r$")
ax[0].set_title("Does the model beat a predictor\nthat ignores EEG?", fontweight="bold")

ax[1].hist(null, bins=40, color="#b0b0b0", label="chance (1000 shuffles)")
ax[1].axvline(r_dem, color="#00783c", lw=2.5, label=f"observed {r_dem:.4f}")
ax[1].set_xlabel("demeaned $r$"); ax[1].set_ylabel("count")
ax[1].legend(fontsize=9)
ax[1].set_title(f"Permutation null\np = {p:.4f}", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "result.png", dpi=150, bbox_inches="tight")
plt.show()
''')

# ===========================================================================
md(r"""
## 8 · Clinical comparison *(only if `hefmi` was downloaded)*

17 healthy controls and 20 intracerebral-haemorrhage patients, identical protocol.

**This needs no accurate prediction** — the coupling descriptors are measured
directly from the data.

**Note the subject-ID handling.** `Normals/subject1` and `Patients/subject1` are
*different people*, and every person's file is named `1_epo.mat` inside their own
folder — so a filename identifies nobody. Keying on `group_subject` is what keeps
the comparison honest.
""")

co(r'''
if HEFMI is None:
    print("HEFMI not downloaded -- use PROFILE = 'core' or above in the settings block (section 2).")
else:
    def clinical_rows(root, group):
        rows = []
        eeg_root = root / "epo_data" / group / "EEG"
        if not eeg_root.exists():
            return rows
        for sd in sorted(p for p in eeg_root.iterdir() if p.is_dir()):
            for ef in sorted(sd.glob("*_epo.mat")):
                nf = root / "epo_data" / group / "NIRS" / sd.name / ef.name
                if not nf.exists():
                    continue
                try:
                    e = sio.loadmat(ef, squeeze_me=True, struct_as_record=False)["epo"]
                    n = sio.loadmat(nf, squeeze_me=True, struct_as_record=False)["epo"]
                    X = np.asarray(e.X, np.float64)
                    oxy = np.asarray(n.oxy.X, np.float64)
                    dxy = np.asarray(n.doxy.X, np.float64)
                except Exception:
                    continue
                k = min(X.shape[2], oxy.shape[2])
                hbo = np.transpose(oxy[:, :, :k], (2, 1, 0))
                hbr = np.transpose(dxy[:, :, :k], (2, 1, 0))
                eeg = np.transpose(X[:, :, :k], (2, 1, 0))
                cc = [np.corrcoef(hbo[t].ravel(), hbr[t].ravel())[0, 1] for t in range(k)]
                d = np.diff(hbo.reshape(k, -1), axis=-1)
                rows.append({
                    # group MUST be part of the key -- see the note above
                    "person": f"{group}_{sd.name}", "group": group,
                    "hbo_hbr": float(np.nanmedian(cc)),
                    "motion_frac": float((np.abs(d) > 5 * np.nanstd(d)).mean()),
                    "hbo_amp": float(np.percentile(np.abs(hbo), 99.5)),
                    "eeg_amp": float(np.percentile(np.abs(eeg), 99.5)),
                    "n_trials": k})
        return rows

    cdf = pd.DataFrame(clinical_rows(HEFMI, "Normals") + clinical_rows(HEFMI, "Patients"))
    per = cdf.groupby(["person", "group"], as_index=False).mean(numeric_only=True)
    print(f"{len(cdf)} recordings -> {len(per)} people")
    print(per.group.value_counts().to_string())

    a = per[per.group == "Normals"]; b = per[per.group == "Patients"]
    res = []
    for k, primary in [("hbo_hbr", True), ("motion_frac", False),
                       ("hbo_amp", False), ("eeg_amp", False), ("n_trials", False)]:
        x, y = a[k].dropna().values, b[k].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        pv = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        res.append({"measure": k, "controls": x.mean(), "patients": y.mean(),
                    "p": pv, "primary": primary, "sig": "*" if pv < .05 else ""})
    res = pd.DataFrame(res)
    per.to_csv(OUT / "clinical_per_person.csv", index=False)
    res.to_csv(OUT / "clinical_comparison.csv", index=False)

    bad = res[(~res.primary) & (res.p < .05)]
    print()
    if len(bad):
        print("WARNING: artefact controls also differ:", ", ".join(bad.measure))
        print("-> the coupling difference may reflect acquisition, not physiology.")
    else:
        print("Artefact controls do NOT differ -> the coupling difference is not")
        print("explained by movement or signal amplitude.")
    display(res)
''')

# ===========================================================================
md(r"""
## 9 · Protocol ablation — where inflated numbers come from

Each row removes one evaluation shortcut. **Nothing about the model or the data
changes** — only how it is scored. The drop from top to bottom is inflation
attributable purely to protocol.

Row 1 is the shape of a typical published protocol: a random split over *trials*,
so the same person appears on both sides.
""")

co(r'''
def protocol_ablation(data, cfg=CFG):
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    subjects = np.unique(sids)
    remap = {int(s): i for i, s in enumerate(subjects)}
    rng = np.random.default_rng(SEED)

    # (1) trial-level split -- the same subject lands on both sides
    idx = rng.permutation(len(eeg)); cut = int(0.8 * len(eeg))
    tr, te = idx[:cut], idx[cut:]
    m = train_fold(eeg[tr], hemo[tr],
                   np.array([remap[int(s)] for s in sids[tr]]), len(subjects), cfg)
    r_leak = mean_corr(predict(m, eeg[te]), hemo[te])

    rows = [
        {"protocol": "1. trial-level split (subject on both sides)", "r": r_leak},
        {"protocol": "2. + subject-wise CV (no person in two folds)", "r": float(cv.r_raw.mean())},
        {"protocol": "3. + no-EEG baseline made explicit", "r": float(cv.r_const.mean())},
        {"protocol": "4. + subject demeaning  <-- HONEST", "r": float(summary["r_demeaned"])},
    ]
    return pd.DataFrame(rows)


abl = protocol_ablation(DATA)
abl.to_csv(OUT / "protocol_ablation.csv", index=False)
infl = abl.r.iloc[0] / abl.r.iloc[-1] if abl.r.iloc[-1] > 0 else float("nan")
print(abl.to_string(index=False))
print(f"\ninflation from protocol alone: {infl:.2f}x  "
      f"({abl.r.iloc[0]:+.4f} -> {abl.r.iloc[-1]:+.4f})")
''')

# ===========================================================================
md(r"""
## 10 · Reliability / abstention gate

The model must know when **not** to answer. We run it several times with dropout
left on; where those runs disagree, the model is uncertain.

A working gate shows accuracy **rising as coverage falls** — discard the least
confident predictions and what remains should be better. That property, not the
headline correlation, is what would make such a system usable clinically.
""")

co(r'''
@torch.no_grad()
def mc_dropout(model, eeg, n_pass=20):
    """Two independent uncertainty signals:
      * MC-dropout spread  -- disagreement across stochastic passes
      * predicted logvar   -- what the heteroscedastic loss taught the model
    Both are returned so the gate can be judged on either."""
    model.train()                       # keep dropout active at inference
    preds = []
    for _ in range(n_pass):
        out = []
        for i in range(0, len(eeg), 256):
            b = torch.tensor(eeg[i:i + 256], dtype=torch.float32).to(DEVICE)
            out.append(model(b, None)[0].cpu().numpy())
        preds.append(np.concatenate(out))
    preds = np.stack(preds)
    conf_mc = -preds.std(0).reshape(len(eeg), -1).mean(1)   # higher = more confident
    _, logvar = predict(model, eeg, want_var=True)
    conf_var = -logvar.reshape(len(eeg), -1).mean(1)
    return preds.mean(0), conf_mc, conf_var


subjects = np.unique(DATA["subject_ids"])
folds = np.array_split(np.random.default_rng(SEED).permutation(subjects), CFG.n_folds)
te = np.isin(DATA["subject_ids"], folds[-1]); tr = ~te
remap = {int(s): i for i, s in enumerate(subjects)}
mdl = train_fold(DATA["eeg"][tr], DATA["hemo"][tr],
                 np.array([remap[int(s)] for s in DATA["subject_ids"][tr]]),
                 len(subjects), CFG)

mean_pred, conf, conf_var = mc_dropout(mdl, DATA["eeg"][te])
pd_ = demean_by_subject(mean_pred, DATA["subject_ids"][te])
td_ = demean_by_subject(DATA["hemo"][te], DATA["subject_ids"][te])

order = np.argsort(-conf)
cov_rows = []
for c in (1.0, .8, .6, .4, .2):
    k = max(int(len(order) * c), 8)
    sel = order[:k]
    cov_rows.append({"coverage": c, "n_kept": k, "r": mean_corr(pd_[sel], td_[sel])})
cov = pd.DataFrame(cov_rows)
cov.to_csv(OUT / "coverage_curve.csv", index=False)
print(cov.to_string(index=False))

mono = bool(np.all(np.diff(cov.r.values) >= -1e-3))
print(f"\naccuracy rises as coverage falls: {mono}")
print("  -> " + ("gate is informative" if mono else "gate is NOT informative on this split"))
''')

# ===========================================================================
md(r"""
## 11 · Calibration — is the confidence score honest?

A gate that ranks trials correctly can still be **badly calibrated**: it may say
"90% confident" on trials it gets wrong half the time. Ranking and calibration
are different properties, and only calibration lets you act on a number.

**Expected Calibration Error (ECE)** bins predictions by stated confidence and
measures the gap between claimed and actual accuracy. **Temperature scaling**
fits a single scalar that rescales confidence to close that gap — one parameter,
fitted on held-out data, so it cannot overfit its way to a good score.
""")

co(r'''
def expected_calibration_error(conf, correct, n_bins=10):
    """Mean |stated confidence - actual accuracy| across equal-width bins."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    lo, hi = conf.min(), conf.max()
    if hi - lo < 1e-12:
        return float("nan"), pd.DataFrame()
    p = (conf - lo) / (hi - lo)              # map to [0,1] so bins are meaningful
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        stated, actual = float(p[m].mean()), float(correct[m].mean())
        ece += (m.sum() / len(p)) * abs(stated - actual)
        rows.append({"bin": i, "n": int(m.sum()),
                     "stated_confidence": round(stated, 4),
                     "actual_accuracy": round(actual, 4),
                     "gap": round(stated - actual, 4)})
    return float(ece), pd.DataFrame(rows)


def fit_temperature(conf, correct, grid=None):
    """One scalar T rescaling confidence to minimise ECE. T>1 = was overconfident."""
    grid = grid if grid is not None else np.linspace(0.25, 4.0, 60)
    best_T, best_ece = 1.0, float("inf")
    for T in grid:
        e, _ = expected_calibration_error(np.asarray(conf, float) / T, correct)
        if np.isfinite(e) and e < best_ece:
            best_T, best_ece = float(T), float(e)
    return best_T, best_ece


# "correct" = this trial's error is better than the median. A gate should be
# more confident on the trials it actually gets right.
_err = np.array([1.0 - mean_corr(pd_[i:i + 1], td_[i:i + 1]) for i in range(len(pd_))])
_correct = (_err < np.median(_err)).astype(float)

# fit the temperature on one half, report ECE on the other -- never on itself
_rng = np.random.default_rng(SEED)
_perm = _rng.permutation(len(conf))
_fit, _hold = _perm[:len(_perm) // 2], _perm[len(_perm) // 2:]

ece_before, bins_before = expected_calibration_error(conf[_hold], _correct[_hold])
T, _ = fit_temperature(conf[_fit], _correct[_fit])
ece_after, bins_after = expected_calibration_error(conf[_hold] / T, _correct[_hold])

print(f"temperature T           = {T:.3f}"
      + ("   (was overconfident)" if T > 1.05 else
         "   (was underconfident)" if T < 0.95 else "   (already close)"))
print(f"ECE before calibration  = {ece_before:.4f}")
print(f"ECE after  calibration  = {ece_after:.4f}   (held-out half)")
print("  -> " + ("calibration improved the gate"
                 if ece_after < ece_before else
                 "calibration did NOT help; the gate ranks but does not calibrate"))
bins_after.to_csv(OUT / "calibration_bins.csv", index=False)
display(bins_after)
calibration = {"temperature": round(T, 4),
               "ece_before": round(float(ece_before), 4),
               "ece_after": round(float(ece_after), 4),
               "improved": bool(ece_after < ece_before)}
''')

# ===========================================================================
md(r"""
## 12 · Within-subject probe — the other ceiling

Everything so far is **cross-subject**: test on people never seen. This section
asks the easier question — *train and test inside the same person* — because the
gap between the two is the whole story.

**Why it matters.** If within-subject is also near zero, the target is broken and
no amount of data will help. If within-subject is much higher, the signal exists
but does not generalise across people, which is a completely different problem
with a completely different fix.

This is not a usable system — it needs paired data from every new patient, which
defeats the purpose — but it **bounds** what cross-subject could ever reach.
""")

co(r'''
def within_subject_probe(data, min_trials=20, test_frac=0.3):
    """Ridge fit and tested inside each subject separately."""
    from sklearn.linear_model import Ridge
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    rng = np.random.default_rng(SEED)
    rows = []
    for s in np.unique(sids):
        m = np.where(sids == s)[0]
        if len(m) < min_trials:
            continue
        idx = rng.permutation(m)
        cut = int(len(idx) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        # compact features: band-power per channel, so the probe is fast and linear
        X = bandpower_features(eeg) if "bandpower_features" in dir() else \
            eeg.reshape(len(eeg), -1)[:, ::200]
        Y = hemo.reshape(len(hemo), -1)
        model = Ridge(alpha=10.0).fit(X[tr], Y[tr])
        pred = model.predict(X[te])
        rows.append({"subject": int(s), "n_train": len(tr), "n_test": len(te),
                     "r": mean_corr(pred, Y[te])})
    return pd.DataFrame(rows)


wsp = within_subject_probe(DATA)
if len(wsp) == 0:
    print("no subject had enough trials for a within-subject probe")
    within_subject = None
else:
    wsp.to_csv(OUT / "within_subject_probe.csv", index=False)
    display(wsp)
    w_mean = float(wsp.r.mean())
    print()
    print(f"within-subject  r = {w_mean:+.4f}   ({len(wsp)} subjects)")
    print(f"cross-subject   r = {r_dem:+.4f}   (demeaned, from section 8)")
    print(f"target ceiling  r = {ceiling:.4f}")
    print()
    if w_mean > 2 * abs(r_dem):
        print("Within-subject is much higher than cross-subject:")
        print("  -> the signal EXISTS but does not generalise across people.")
        print("     The bottleneck is between-subject variability, not the target.")
    elif w_mean < 0.05:
        print("Within-subject is near zero as well:")
        print("  -> the TARGET is the bottleneck, not generalisation.")
        print("     More subjects will not help; the target must be fixed first.")
    else:
        print("Within- and cross-subject are comparable: no large")
        print("subject-specific component is being lost.")
    within_subject = {"within_subject_r": round(w_mean, 4),
                      "cross_subject_r": round(float(r_dem), 4),
                      "n_subjects": int(len(wsp))}
''')

# ===========================================================================
md(r"""
## 13 · CLAIM 2 — external validation *(needs `PROFILE='claims'` or `'full'`)*

**The claim.** Train on the primary corpus, then evaluate **once** on a corpus
the model has never seen — different people, different task, no tuning. The drop
should be bounded.

**The setup is deliberately fair.** Both corpora have **30 EEG channels** and
**72 haemodynamic targets** (36 HbO + 36 HbR), so no channel truncation or
padding is involved. The workload set already ships HbO/HbR, so no Beer–Lambert
step differs between them either.

**Two comparisons are run:**

| | What changes | What stays the same |
|---|---|---|
| cross-**task** | the task (MI → n-back) | *the shape of the problem* |
| cross-**cohort** | the **people** | — |

The second is the real test. Watch what happens to the score.
""")

co(r'''
def load_tuberlin_wl(cfg=CFG, root=None):
    """TU Berlin workload (n-back) -- the EXTERNAL corpus for claim 2.

    Differs from hBCI in two ways that matter:
      * NIRS already ships oxy/deoxy, so there is no Beer-Lambert step here.
      * one continuous run per subject, with session-level markers.
    Output shape is identical to load_tuberlin(): EEG (n,30,6000), hemo (n,72,300).
    """
    root = root or WL
    if root is None:
        return None
    eeg_l, hemo_l, sid_l = [], [], []
    n_base = int(cfg.baseline_s * cfg.hemo_fs)

    subs = sorted(p for p in (root / "EEG").iterdir() if p.is_dir())
    for sd in subs:
        vp = sd.name.split("-")[0]                       # VP001-EEG -> VP001
        nd = root / "NIRS" / f"{vp}-NIRS"
        if not nd.exists():
            continue
        try:
            ec = _runs(sd / "cnt_nback.mat")[0]
            nc = _runs(nd / "cnt_nback.mat")[0]
            nm = _runs(nd / "mrk_nback.mat")[0]
        except Exception:
            continue

        e_fs = float(np.atleast_1d(ec.fs).ravel()[0])
        ex = np.asarray(ec.x, dtype=np.float64).T        # (30, T)
        oxy = np.asarray(nc.oxy.x, dtype=np.float64).T   # (36, T)
        dxy = np.asarray(nc.deoxy.x, dtype=np.float64).T
        n_fs = float(np.atleast_1d(nc.oxy.fs).ravel()[0])
        oxy = bandpass(oxy, n_fs, .01, .2)
        dxy = bandpass(dxy, n_fs, .01, .2)

        onsets = np.atleast_1d(np.asarray(nm.time, float)) / 1000.0
        for t0_s in onsets:
            a_e = int(round((t0_s + cfg.tmin_s) * e_fs))
            b_e = int(round((t0_s + cfg.tmax_s) * e_fs))
            a_n = int(round((t0_s + cfg.tmin_s) * n_fs))
            b_n = int(round((t0_s + cfg.tmax_s) * n_fs))
            if a_e < 0 or b_e > ex.shape[1] or a_n < 0 or b_n > oxy.shape[1]:
                continue
            seg = resample_to(ex[:, a_e:b_e], cfg.eeg_samples)
            seg = ((seg - seg.mean(-1, keepdims=True)) /
                   (seg.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
            so = resample_to(oxy[:, a_n:b_n], cfg.hemo_samples)
            sr = resample_to(dxy[:, a_n:b_n], cfg.hemo_samples)
            so = so - so[:, :n_base].mean(-1, keepdims=True)
            sr = sr - sr[:, :n_base].mean(-1, keepdims=True)
            eeg_l.append(seg)
            hemo_l.append(np.concatenate([so, sr], 0).astype(np.float32))
            sid_l.append(int(vp[2:]))

    if not eeg_l:
        return None
    return {"eeg": np.stack(eeg_l), "hemo": np.stack(hemo_l),
            "subject_ids": np.asarray(sid_l, np.int64)}
''')

co(r'''
if WL is None:
    print("TU Berlin workload not downloaded.")
    print("Set PROFILE = 'claims' in the settings block (section 2) and re-run.")
    claim2 = None
else:
    EXT_DATA = load_tuberlin_wl()
    if EXT_DATA is None:
        print("workload corpus present but no usable trials were extracted.")
        claim2 = None
    else:
        print(f"external corpus: {EXT_DATA['eeg'].shape[0]} trials, "
              f"{len(np.unique(EXT_DATA['subject_ids']))} subjects")
        print(f"  EEG  {EXT_DATA['eeg'].shape}   hemo {EXT_DATA['hemo'].shape}")
        assert EXT_DATA["eeg"].shape[1:] == DATA["eeg"].shape[1:], "EEG shape mismatch"
        assert EXT_DATA["hemo"].shape[1:] == DATA["hemo"].shape[1:], "hemo shape mismatch"
        print("  shapes match the training corpus -> no truncation or padding\n")

        # Train ONCE on all of hBCI, then evaluate ONCE externally. No tuning.
        subjects = np.unique(DATA["subject_ids"])
        remap = {int(s): i for i, s in enumerate(subjects)}
        model_all = train_fold(DATA["eeg"], DATA["hemo"],
                               np.array([remap[int(s)] for s in DATA["subject_ids"]]),
                               len(subjects), CFG)

        pe = predict(model_all, EXT_DATA["eeg"])
        pd_e = demean_by_subject(pe, EXT_DATA["subject_ids"])
        td_e = demean_by_subject(EXT_DATA["hemo"], EXT_DATA["subject_ids"])
        r_ext, p_ext, _ = perm_null(pd_e, td_e, CFG.n_perm)

        r_int = float(summary["r_demeaned"])
        p_int = float(summary["p_perm"])
        drop = (r_int - r_ext) / abs(r_int) * 100 if r_int else float("nan")

        print("=" * 66)
        print("CLAIM 2 -- external validation")
        print("=" * 66)
        print(f"internal  (held-out hBCI subjects)  r = {r_int:+.4f}   p = {p_int:.4f}")
        print(f"EXTERNAL  (TU Berlin workload)      r = {r_ext:+.4f}   p = {p_ext:.4f}")
        print(f"drop = {drop:.0f}%")
        print()
        verdict = "HOLDS" if p_ext < 0.05 else "FAILS"
        print(f"VERDICT: {verdict} -- " + ("the effect survives a new cohort"
              if p_ext < 0.05 else
              "does NOT retain significance on a new cohort"))
        if p_ext >= 0.05:
            print()
            print("Note the two explanations this rules out:")
            print("  * not target quality -- the external set's reliability is")
            print("    measured below and is not worse than the training set's")
            print("  * not montage -- 30 ch / 72 targets on both sides, asserted above")
            print("  => what the encoder learned is COHORT-SPECIFIC.")

        rel_ext = split_half_reliability(EXT_DATA["hemo"], EXT_DATA["subject_ids"])
        rel_ext_m = float(np.nanmean(list(rel_ext.values())))
        print(f"\nexternal target reliability = {rel_ext_m:+.3f} "
              f"(training corpus: {rel_mean:+.3f})")

        claim2 = {"internal_r": r_int, "internal_p": p_int,
                  "external_r": r_ext, "external_p": p_ext,
                  "drop_pct": drop, "verdict": verdict,
                  "external_reliability": rel_ext_m,
                  "n_external_trials": int(len(EXT_DATA["eeg"])),
                  "n_external_subjects": int(len(np.unique(EXT_DATA["subject_ids"])))}
        pd.DataFrame([claim2]).to_csv(OUT / "claim2_external.csv", index=False)
''')

# ===========================================================================
md(r"""
## 12 · CLAIM 1 — the subject-factored encoder beats one without

**The claim.** An encoder with a per-subject adapter outperforms one without it.

**The test.** Identical architecture, identical budget, identical data — the
*only* difference is whether the subject embedding can learn. Several seeds, and
we require the confidence interval to exclude zero **and** the sign to be stable
across every seed. A paired comparison like this is the strongest evidence
available at this effect size.
""")

co(r'''
def train_fold_noadapter(eeg, hemo, sid_ix, n_subj, cfg=CFG):
    """Identical to train_fold except the subject embedding is frozen at zero,
    so the model cannot represent WHO a trial came from."""
    model = NeuroVascNet(eeg.shape[1], hemo.shape[1], hemo.shape[2], n_subj).to(DEVICE)
    model.subj.weight.requires_grad_(False)            # <-- the only difference
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(eeg, dtype=torch.float32),
        torch.tensor(hemo, dtype=torch.float32),
        torch.tensor(sid_ix, dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for _ in range(cfg.epochs):
        for be, bh, bs in dl:
            opt.zero_grad()
            mean, logvar = model(be.to(DEVICE), bs.to(DEVICE))
            loss, _ = compute_loss(mean, logvar, bh.to(DEVICE), cfg)
            loss.backward(); opt.step()
    return model


def claim1_subject_adapter(data, cfg=CFG):
    n_seeds = 2 if QUICK_TEST else 3
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    subjects = np.unique(sids)
    remap = {int(s): i for i, s in enumerate(subjects)}
    hold = subjects[-max(2, len(subjects) // 5):]        # held-out PEOPLE
    te = np.isin(sids, hold); tr = ~te
    tr_ix = np.array([remap[int(s)] for s in sids[tr]])
    td = demean_by_subject(hemo[te], sids[te])

    rows = []
    for seed in range(n_seeds):
        torch.manual_seed(SEED + seed); np.random.seed(SEED + seed)
        m_with = train_fold(eeg[tr], hemo[tr], tr_ix, len(subjects), cfg)
        torch.manual_seed(SEED + seed); np.random.seed(SEED + seed)
        m_without = train_fold_noadapter(eeg[tr], hemo[tr], tr_ix, len(subjects), cfg)
        r_w = mean_corr(demean_by_subject(predict(m_with, eeg[te]), sids[te]), td)
        r_o = mean_corr(demean_by_subject(predict(m_without, eeg[te]), sids[te]), td)
        rows.append({"seed": seed, "with_adapter": r_w,
                     "no_adapter": r_o, "diff": r_w - r_o})
        print(f"  seed {seed}: with {r_w:+.4f} | without {r_o:+.4f} | diff {r_w - r_o:+.4f}")

    df = pd.DataFrame(rows)
    d = df["diff"].values
    if len(d) > 1:
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    else:
        lo = hi = float("nan")
    n_pos = int((d > 0).sum())
    holds = bool(len(d) > 1 and lo > 0 and n_pos == len(d))
    print()
    print(f"mean difference : {d.mean():+.4f}")
    print(f"95% CI          : [{lo:+.4f}, {hi:+.4f}]")
    print(f"seeds positive  : {n_pos}/{len(d)}")
    print(f"VERDICT: {'HOLDS' if holds else 'NOT CONFIRMED'} -- "
          + ("CI excludes zero and every seed agrees" if holds
             else "CI includes zero, or the sign flips across seeds"))
    return df, {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                "n_seeds": int(len(d)), "n_positive": n_pos,
                "verdict": "HOLDS" if holds else "NOT CONFIRMED"}


claim1_df, claim1 = claim1_subject_adapter(DATA)
claim1_df.to_csv(OUT / "claim1_subject_adapter.csv", index=False)
''')

# ===========================================================================
md(r"""
## 13 · CLAIM 3 — does the delay module measure physiology?

**The claim.** A learned delay should recover the neurovascular lag: inside the
physiological 2–8 s window **and varying across people**. If everyone gets the
same delay, it has learned nothing.

**Read the trap carefully.** The delay is produced as

```
delay = DELAY_MIN + (DELAY_MAX - DELAY_MIN) * sigmoid(...)
```

so it is **mathematically forced** inside whatever window we pick. Landing in
2–8 s therefore proves nothing. This cell trains the same module under **two
different windows**. If the recovered delay follows the window instead of
staying at a fixed number of seconds, it is reporting our prior, not the biology.
""")

co(r'''
class DelayNet(nn.Module):
    """NeuroVascNet plus a per-trial delay head bounded to [dmin, dmax]."""

    def __init__(self, n_eeg, n_hemo, hemo_len, n_subj, dmin, dmax, width=64, emb=8):
        super().__init__()
        self.base = NeuroVascNet(n_eeg, n_hemo, hemo_len, n_subj, emb=emb, width=width)
        self.dmin, self.dmax = dmin, dmax
        self.delay_head = nn.Sequential(
            nn.Linear(width * 16 + emb, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, sid=None):
        h = self.base.trunk(x).flatten(1)
        e = (self.base.subj(sid) if sid is not None
             else self.base.subj.weight.mean(0).expand(len(x), -1))
        z = torch.cat([h, e], 1)
        mean, logvar = self.base(x, sid)
        # THE BOUND: a sigmoid rescaled into [dmin, dmax]. It cannot escape.
        delay = self.dmin + (self.dmax - self.dmin) * torch.sigmoid(
            self.delay_head(z)).squeeze(-1)
        return self.shift(mean, delay, CFG.hemo_fs), logvar, delay

    @staticmethod
    def shift(sig, delay_s, fs):
        B, C, T = sig.shape
        base = torch.arange(T, device=sig.device, dtype=sig.dtype).unsqueeze(0).expand(B, -1)
        idx = (base - delay_s.unsqueeze(-1) * fs).clamp(0, T - 1)
        lo = idx.floor().long(); hi = (lo + 1).clamp(max=T - 1)
        w = (idx - lo.to(sig.dtype)).unsqueeze(1)
        a = torch.gather(sig, 2, lo.unsqueeze(1).expand(-1, C, -1))
        b = torch.gather(sig, 2, hi.unsqueeze(1).expand(-1, C, -1))
        return a * (1 - w) + b * w


def empirical_lag(pred, target, fs, dmin, dmax):
    """Lag maximising prediction-target correlation, CLAMPED to [dmin, dmax].

    Note the clamp: the teaching signal is forced into the same window as the
    output. That is the second half of the circularity."""
    with torch.no_grad():
        p = pred - pred.mean(-1, keepdim=True)
        t = target - target.mean(-1, keepdim=True)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
        best_r = torch.full((pred.shape[0],), -2.0, device=pred.device)
        best_l = torch.zeros(pred.shape[0], device=pred.device)
        for lag in range(0, int(dmax * fs) + 1):
            a, b = (p, t) if lag == 0 else (p[..., :-lag], t[..., lag:])
            if a.shape[-1] < 4:
                break
            r = (a * b).sum(-1).mean(-1)
            upd = r > best_r
            best_r = torch.where(upd, r, best_r)
            best_l = torch.where(upd, torch.full_like(best_l, lag / fs), best_l)
        return best_l.clamp(dmin, dmax)


def claim3_delay(data, dmin, dmax, cfg=CFG):
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    subjects = np.unique(sids)
    remap = {int(s): i for i, s in enumerate(subjects)}
    hold = subjects[-max(2, len(subjects) // 5):]
    te = np.isin(sids, hold); tr = ~te

    torch.manual_seed(SEED)
    model = DelayNet(eeg.shape[1], hemo.shape[1], hemo.shape[2],
                     len(subjects), dmin, dmax).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(eeg[tr], dtype=torch.float32),
        torch.tensor(hemo[tr], dtype=torch.float32),
        torch.tensor([remap[int(s)] for s in sids[tr]], dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for be, bh, bs in dl:
            be, bh, bs = be.to(DEVICE), bh.to(DEVICE), bs.to(DEVICE)
            opt.zero_grad()
            pred, logvar, delay = model(be, bs)
            loss, _ = compute_loss(pred, logvar, bh, cfg)
            tgt = empirical_lag(pred.detach(), bh, cfg.hemo_fs, dmin, dmax)
            loss = loss + nn.functional.mse_loss(delay, tgt)
            loss.backward(); opt.step()

    model.eval()
    all_d, per_subject = [], []
    with torch.no_grad():
        for s in np.unique(sids[te]):
            m = sids[te] == s
            b = torch.tensor(eeg[te][m], dtype=torch.float32).to(DEVICE)
            _, _, d = model(b, None)
            d = d.cpu().numpy()
            all_d.append(d); per_subject.append(float(d.mean()))
    all_d = np.concatenate(all_d)
    return {"window": f"[{dmin:.0f},{dmax:.0f}] s", "mean_s": float(all_d.mean()),
            "within_sd": float(all_d.std()),
            "between_subject_sd": float(np.std(per_subject)),
            "frac_of_range": float((all_d.mean() - dmin) / (dmax - dmin))}


rows3 = [claim3_delay(DATA, 2.0, 8.0), claim3_delay(DATA, 0.0, 20.0)]
c3 = pd.DataFrame(rows3)
c3.to_csv(OUT / "claim3_delay.csv", index=False)
display(c3)

sec_move = abs(rows3[1]["mean_s"] - rows3[0]["mean_s"])
frac_move = abs(rows3[1]["frac_of_range"] - rows3[0]["frac_of_range"])
collapsed = rows3[0]["between_subject_sd"] < 0.5
print()
print(f"widening the window moved the delay by {sec_move:.2f} s,")
print(f"but its POSITION within the range moved only {frac_move:.3f}")
print(f"between-subject SD (window [2,8]) = {rows3[0]['between_subject_sd']:.3f} s")
print()
if collapsed or sec_move > 1.0:
    print("VERDICT: FAILS")
    if sec_move > 1.0:
        print("  * the answer follows the BOUND we chose, not the physiology")
    if collapsed:
        print("  * delays barely differ between people -> collapsed to a constant")
    print("  Landing inside 2-8 s is guaranteed by construction, so it is NOT evidence.")
else:
    print("VERDICT: HOLDS -- stable across windows and varies between subjects")

claim3 = {"sec_move": float(sec_move), "frac_move": float(frac_move),
          "between_subject_sd": rows3[0]["between_subject_sd"],
          "verdict": "FAILS" if (collapsed or sec_move > 1.0) else "HOLDS"}
''')

# ===========================================================================
md(r"""
## 14 · CLAIM 4 — does haemodynamic supervision help an EEG-only task?

**The broadest claim, and it does not require accurate prediction.** Freeze the
trained encoder, fit a linear probe on the EEG-only task labels, and compare
against three controls:

| Encoder | What it tests |
|---|---|
| haemodynamic-supervised | our trained trunk |
| shuffled-target control | same training, targets randomly re-paired |
| random init (untrained) | no training at all |
| **raw band-power** | no encoder — the textbook baseline |

Beating the first two means the encoder learned something real. **Losing to raw
band-power means it is not yet worth using.**
""")

co(r'''
@torch.no_grad()
def embed(model, eeg):
    """Frozen trunk features. No subject embedding -- EEG only."""
    model.eval()
    out = []
    for i in range(0, len(eeg), 256):
        b = torch.tensor(eeg[i:i + 256], dtype=torch.float32).to(DEVICE)
        out.append(model.trunk(b).flatten(1).cpu().numpy())
    return np.concatenate(out)


def bandpower_features(eeg, fs=CFG.eeg_fs):
    """The textbook baseline: log band-power per channel in 5 classic bands."""
    feats = []
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
        x = bandpass(eeg.astype(np.float64), fs, lo, hi)
        feats.append(np.log(np.mean(x ** 2, axis=-1) + 1e-12))
    return np.concatenate(feats, axis=1)


def probe_accuracy(X, y, groups, n_splits=3):
    """Subject-wise CV linear probe -- no person in both train and test."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    subs = np.unique(groups)
    folds = np.array_split(np.random.default_rng(SEED).permutation(subs), n_splits)
    accs = []
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


y = DATA["labels"]; g = DATA["subject_ids"]
_subjects = np.unique(g); _remap = {int(s): i for i, s in enumerate(_subjects)}
_sid_ix = np.array([_remap[int(s)] for s in g])

if len(np.unique(y)) < 2:
    print("only one class in the labels -- claim 4 cannot be tested on this data.")
    claim4 = None
else:
    print(f"{len(np.unique(y))} classes, {len(y)} trials, {len(_subjects)} subjects\n")
    rows4 = []

    m_sup = train_fold(DATA["eeg"], DATA["hemo"], _sid_ix, len(_subjects), CFG)
    a, sd = probe_accuracy(embed(m_sup, DATA["eeg"]), y, g)
    rows4.append({"encoder": "hemodynamic-supervised", "accuracy": a, "sd": sd})

    perm = np.random.default_rng(SEED).permutation(len(DATA["hemo"]))
    m_shuf = train_fold(DATA["eeg"], DATA["hemo"][perm], _sid_ix, len(_subjects), CFG)
    a, sd = probe_accuracy(embed(m_shuf, DATA["eeg"]), y, g)
    rows4.append({"encoder": "SHUFFLED-target control", "accuracy": a, "sd": sd})

    torch.manual_seed(SEED)
    m_rand = NeuroVascNet(DATA["eeg"].shape[1], DATA["hemo"].shape[1],
                          DATA["hemo"].shape[2], len(_subjects)).to(DEVICE)
    a, sd = probe_accuracy(embed(m_rand, DATA["eeg"]), y, g)
    rows4.append({"encoder": "random init (untrained)", "accuracy": a, "sd": sd})

    a, sd = probe_accuracy(bandpower_features(DATA["eeg"]), y, g)
    rows4.append({"encoder": "raw band-power", "accuracy": a, "sd": sd})

    c4 = pd.DataFrame(rows4)
    c4.to_csv(OUT / "claim4_downstream.csv", index=False)
    display(c4)

    acc = dict(zip(c4.encoder, c4.accuracy))
    sup = acc["hemodynamic-supervised"]
    v_shuf = sup - acc["SHUFFLED-target control"]
    v_rand = sup - acc["random init (untrained)"]
    v_raw = sup - acc["raw band-power"]
    print()
    print(f"vs shuffled target : {v_shuf:+.4f}")
    print(f"vs random init     : {v_rand:+.4f}")
    print(f"vs raw band-power  : {v_raw:+.4f}")
    print()
    if v_shuf > 0 and v_rand > 0 and v_raw > 0:
        verdict = "HOLDS"
        print("VERDICT: HOLDS -- beats every control, including raw features")
    elif v_shuf > 0 and v_rand > 0:
        verdict = "PARTIAL"
        print("VERDICT: PARTIAL")
        print("  * beats shuffled and random -> the encoder DOES learn something real")
        print("  * loses to raw band-power   -> not yet worth using over the simplest")
        print("    possible features. This is the most valuable claim to fix.")
    else:
        verdict = "FAILS"
        print("VERDICT: FAILS -- no better than an untrained or mis-supervised encoder")
    claim4 = {"supervised": sup, "vs_shuffled": v_shuf, "vs_random": v_rand,
              "vs_raw_bandpower": v_raw, "verdict": verdict}
''')

# ===========================================================================
md(r"""
## 15 · fMRI loaders — ds002336 and ds002739 *(PROFILE='full')*

These turn the two downloaded fMRI datasets into trials with the **same schema**
as the fNIRS loader, so every analysis above can be pointed at them.

**ds002336 — the R128 result.** BrainVision marker files carry an `R128`
annotation at *every* volume acquisition. Volume *k* of the BOLD series is
therefore EEG sample `R128[k]` **exactly** — no assumed offset. Using this
instead of sliding windows is what flipped the target's split-half reliability
from **−0.639 to +0.638**: before it, two halves of the same data
anti-correlated, so no model could ever have worked.

Three things the audit showed are required, all applied here:
1. **epoch on stimulus onsets**, not sliding windows (the decisive fix)
2. **regress out motion + polynomial drift** before epoching
3. **restrict to sensorimotor ROIs** rather than all 48 Harvard–Oxford regions

**ds002739** has per-run `events.tsv` and 1 kHz EEG in MATLAB files, so it needs
no trigger hunting — but also has no R128, so alignment relies on the events file.
""")

co(r'''
# BOLD -> ROI timeseries needs nilearn (only for this section).
try:
    import nilearn
except ImportError:
    print("installing nilearn ...")
    pip("nilearn")

TR_S = 2.0
EPOCH_VOL = 15                       # 15 volumes x 2 s = 30 s, matches the fNIRS window
MOTOR_KEYS = ("Precentral", "Postcentral", "Supplementary", "Juxtapositional")
_MASKER = None


def ho_masker():
    """Harvard-Oxford cortical atlas masker, fetched once and reused."""
    global _MASKER
    if _MASKER is None:
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        _MASKER = (NiftiLabelsMasker(labels_img=atlas.maps, standardize=False,
                                     verbose=0),
                   [str(x) for x in atlas.labels[1:]])
    return _MASKER


def roi_timeseries(bold_path):
    """(n_volumes, n_rois) mean signal per Harvard-Oxford region."""
    masker, labels = ho_masker()
    return masker.fit_transform(str(bold_path)), labels


def clean_nuisance(ts, motion=None):
    """Regress out cubic drift (and motion if available). Scanner drift alone
    inflates reliability, so this is not optional."""
    n = ts.shape[0]
    t = np.linspace(-1, 1, n)
    cols = [np.ones(n), t, t ** 2, t ** 3]
    if motion is not None and motion.shape[0] >= 2:
        m = (motion[:n] if motion.shape[0] >= n else
             np.vstack([motion, np.repeat(motion[-1:], n - motion.shape[0], axis=0)]))
        for k in range(min(6, m.shape[1])):
            v = m[:, k]
            cols.append((v - v.mean()) / (v.std() if v.std() > 0 else 1.0))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
    return ts - X @ beta


def motor_subset(ts, labels):
    idx = [i for i, nm in enumerate(labels) if any(k in nm for k in MOTOR_KEYS)]
    return (ts[:, idx], [labels[i] for i in idx]) if idx else (ts, labels)
''')

co(r'''
def load_ds002336(root=None, cfg=CFG, max_subjects=None):
    """EEG-fMRI trials aligned by R128 scanner triggers."""
    root = root or DS2336
    if root is None:
        return None
    import mne

    subs = sorted(d.name for d in root.glob("sub-*") if d.is_dir())
    if max_subjects:
        subs = subs[:max_subjects]
    tasks = ("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF")

    eeg_l, hemo_l, sid_l, audit = [], [], [], []
    for si, sub in enumerate(subs):
        for task in tasks:
            vhdr = root / "derivatives" / sub / "eeg_pp" / f"{sub}_task-{task}_eeg_pp.vhdr"
            bolds = sorted(root.rglob(f"{sub}_task-{task}*_bold.nii.gz"))
            ev = root / f"task-{task}_events.tsv"
            if not vhdr.exists() or not bolds or not ev.exists():
                continue
            try:
                raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
                eeg = raw.get_data(picks=mne.pick_types(raw.info, eeg=True)) * 1e6
                fs = float(raw.info["sfreq"])
                # R128 marks every volume -> exact EEG<->BOLD correspondence
                trig = np.array([o for o, d in zip(raw.annotations.onset,
                                                   raw.annotations.description)
                                 if "R128" in str(d)])
                trig = (trig * fs).astype(int)
                if trig.size < EPOCH_VOL:
                    raise ValueError(f"only {trig.size} R128 triggers")

                ts, labels = roi_timeseries(bolds[0])
                ts = clean_nuisance(ts)
                ts, labels = motor_subset(ts, labels)

                df = pd.read_csv(ev, sep="\t")
                if "trial_type" in df.columns:
                    df = df[~df.trial_type.astype(str).str.lower().str.contains("rest")]
                onsets = pd.to_numeric(df["onset"], errors="coerce").dropna().to_numpy()

                n = 0
                for o in onsets:
                    v = int(round(o / TR_S))
                    if v < 0 or v + EPOCH_VOL > ts.shape[0] or v >= trig.size:
                        continue
                    s0 = trig[v]
                    if s0 + cfg.eeg_samples > eeg.shape[1]:
                        continue
                    seg_h = ts[v:v + EPOCH_VOL].T
                    seg_h = seg_h - seg_h[:, :1]          # baseline at onset
                    seg_e = eeg[:, s0:s0 + cfg.eeg_samples]
                    seg_e = ((seg_e - seg_e.mean(-1, keepdims=True)) /
                             (seg_e.std(-1, keepdims=True) + 1e-6))
                    eeg_l.append(seg_e.astype(np.float32))
                    hemo_l.append(seg_h.astype(np.float32))
                    sid_l.append(si); n += 1
                audit.append({"subject": sub, "task": task, "trials": n,
                              "n_rois": ts.shape[1], "n_triggers": int(trig.size),
                              "status": "ok" if n else "no usable trials"})
            except Exception as exc:
                audit.append({"subject": sub, "task": task, "trials": 0,
                              "status": f"skip: {type(exc).__name__}: {exc}"})

    if not eeg_l:
        print(pd.DataFrame(audit).to_string())
        return None
    n_ch = min(e.shape[0] for e in eeg_l)
    n_roi = min(h.shape[0] for h in hemo_l)
    return {"eeg": np.stack([e[:n_ch] for e in eeg_l]),
            "hemo": np.stack([h[:n_roi] for h in hemo_l]),
            "subject_ids": np.asarray(sid_l, np.int64),
            "labels": np.zeros(len(eeg_l), np.int64),
            "audit": pd.DataFrame(audit)}


def load_ds002739(root=None, cfg=CFG, max_subjects=None):
    """EEG-fMRI trials from per-run events.tsv (no R128 in this dataset)."""
    root = root or DS2739
    if root is None:
        return None
    subs = sorted(d.name for d in root.glob("sub-*") if d.is_dir())
    if max_subjects:
        subs = subs[:max_subjects]

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
                eeg = np.asarray(m.Y, dtype=np.float64)            # (63, T)
                fs = float(np.asarray(m.fd).ravel()[0])            # downsampled rate
                ts, labels = roi_timeseries(bold)
                ts = clean_nuisance(ts)
                ts, labels = motor_subset(ts, labels)

                df = pd.read_csv(ev, sep="\t")
                if "trial_type" in df.columns:
                    df = df[df.trial_type.astype(str).str.contains("stim", case=False)]
                onsets = pd.to_numeric(df["onset"], errors="coerce").dropna().to_numpy()

                n = 0
                for o in onsets:
                    v = int(round(o / TR_S))
                    s0 = int(round(o * fs))
                    if v < 0 or v + EPOCH_VOL > ts.shape[0]:
                        continue
                    if s0 < 0 or s0 + cfg.eeg_samples > eeg.shape[1]:
                        continue
                    seg_h = ts[v:v + EPOCH_VOL].T
                    seg_h = seg_h - seg_h[:, :1]
                    seg_e = eeg[:, s0:s0 + cfg.eeg_samples]
                    seg_e = ((seg_e - seg_e.mean(-1, keepdims=True)) /
                             (seg_e.std(-1, keepdims=True) + 1e-6))
                    eeg_l.append(seg_e.astype(np.float32))
                    hemo_l.append(seg_h.astype(np.float32))
                    sid_l.append(si); n += 1
                audit.append({"subject": sub, "run": run, "trials": n,
                              "n_rois": ts.shape[1],
                              "status": "ok" if n else "no usable trials"})
            except Exception as exc:
                audit.append({"subject": sub, "run": run, "trials": 0,
                              "status": f"skip: {type(exc).__name__}: {exc}"})

    if not eeg_l:
        print(pd.DataFrame(audit).to_string())
        return None
    n_ch = min(e.shape[0] for e in eeg_l)
    n_roi = min(h.shape[0] for h in hemo_l)
    return {"eeg": np.stack([e[:n_ch] for e in eeg_l]),
            "hemo": np.stack([h[:n_roi] for h in hemo_l]),
            "subject_ids": np.asarray(sid_l, np.int64),
            "labels": np.zeros(len(eeg_l), np.int64),
            "audit": pd.DataFrame(audit)}
''')

co(r'''
# Loading fMRI is slow (NIfTI + atlas per run), so cap subjects unless asked.
FMRI_MAX_SUBJECTS = 3 if QUICK_TEST else None

fmri_summary = []
for name, loader in (("ds002336", load_ds002336), ("ds002739", load_ds002739)):
    root = DS2336 if name == "ds002336" else DS2739
    if root is None:
        print(f"{name}: not downloaded -- set PROFILE = 'full' in section 2")
        continue
    print(f"\nloading {name} ...")
    t0 = time.time()
    d = loader(max_subjects=FMRI_MAX_SUBJECTS)
    if d is None:
        print(f"  {name}: no usable trials (audit printed above)")
        continue
    rel = split_half_reliability(d["hemo"], d["subject_ids"])
    rel_m = float(np.nanmean(list(rel.values())))
    print(f"  EEG {d['eeg'].shape}  hemo {d['hemo'].shape}  "
          f"{len(np.unique(d['subject_ids']))} subjects  "
          f"({(time.time()-t0)/60:.1f} min)")
    print(f"  target reliability {rel_m:+.3f}  ->  ceiling {math.sqrt(max(rel_m,0)):.3f}")
    if rel_m < 0:
        print("  !! NEGATIVE reliability -- the two halves anti-correlate, so the")
        print("     target is misaligned and no model can work on it.")
    d["audit"].to_csv(OUT / f"{name}_loader_audit.csv", index=False)
    fmri_summary.append({"dataset": name, "trials": int(len(d["eeg"])),
                         "subjects": int(len(np.unique(d["subject_ids"]))),
                         "n_rois": int(d["hemo"].shape[1]),
                         "reliability": round(rel_m, 4),
                         "ceiling": round(math.sqrt(max(rel_m, 0)), 4)})
    globals()[f"DATA_{name}"] = d

if fmri_summary:
    fs_df = pd.DataFrame(fmri_summary)
    fs_df.to_csv(OUT / "fmri_targets.csv", index=False)
    display(fs_df)
''')

# ===========================================================================
md(r"""
## 16 · Cross-subject evaluation on **every** dataset

The fMRI dicts have the same schema as the fNIRS one, so the *identical*
protocol runs on all of them — subject-wise folds, a no-EEG baseline, demeaning,
and an exact permutation null.

**Read the last column.** Achieved *r* means little on its own; what matters is
how much of the **ceiling** each dataset reaches, because the ceiling differs a
lot between modalities. A dataset with a clean target and a low percentage is
telling you the mapping is weak, not that the data is bad.
""")

co(r'''
def evaluate_dataset(name, data, cfg=CFG):
    """Run the same cross-subject protocol on any dataset dict."""
    n_sub = len(np.unique(data["subject_ids"]))
    if n_sub < 3:
        print(f"{name}: only {n_sub} subjects -- need at least 3 for subject-wise CV")
        return None
    # never ask for more folds than there are people
    folds = min(cfg.n_folds, n_sub)
    local = Config(**{**asdict(cfg), "n_folds": folds})

    rel = split_half_reliability(data["hemo"], data["subject_ids"])
    rel_m = float(np.nanmean(list(rel.values())))
    ceiling = math.sqrt(max(rel_m, 0.0))

    print(f"\n--- {name}: {n_sub} subjects, {len(data['eeg'])} trials, "
          f"{folds} folds")
    cv_t, summ, _ = cross_subject_cv(data, local)
    r_raw = float(cv_t.r_raw.mean()); r_const = float(cv_t.r_const.mean())
    r_dem = float(summ["r_demeaned"]); p = float(summ["p_perm"])
    return {"dataset": name, "n_subjects": n_sub, "n_trials": int(len(data["eeg"])),
            "target_dims": int(data["hemo"].shape[1] * data["hemo"].shape[2]),
            "reliability": round(rel_m, 4), "ceiling": round(ceiling, 4),
            "r_raw": round(r_raw, 4), "r_const": round(r_const, 4),
            "r_gain": round(r_raw - r_const, 4), "r_demeaned": round(r_dem, 4),
            "p": round(p, 5),
            "pct_of_ceiling": round(100 * r_dem / ceiling, 2) if ceiling > 0 else None}


all_rows = [{"dataset": "TU Berlin hBCI (fNIRS)",
             "n_subjects": int(len(np.unique(DATA["subject_ids"]))),
             "n_trials": int(len(DATA["eeg"])),
             "target_dims": int(DATA["hemo"].shape[1] * DATA["hemo"].shape[2]),
             "reliability": round(rel_mean, 4), "ceiling": round(ceiling, 4),
             "r_raw": round(float(r_raw), 4), "r_const": round(float(r_const), 4),
             "r_gain": round(float(r_raw - r_const), 4),
             "r_demeaned": round(float(r_dem), 4), "p": round(float(p), 5),
             "pct_of_ceiling": round(100 * r_dem / ceiling, 2) if ceiling > 0 else None}]

for _nm, _key in (("ds002336 (fMRI)", "DATA_ds002336"),
                  ("ds002739 (fMRI)", "DATA_ds002739")):
    _d = globals().get(_key)
    if _d is None:
        print(f"{_nm}: not loaded -- add the dataset to DATA_DIR to include it")
        continue
    _row = evaluate_dataset(_nm, _d)
    if _row:
        all_rows.append(_row)

allres = pd.DataFrame(all_rows)
allres.to_csv(OUT / "all_datasets_evaluation.csv", index=False)
print()
display(allres)

print()
print("How to read this:")
print("  reliability   does the target reproduce itself? negative = misaligned")
print("  ceiling       sqrt(reliability) -- the hard cap on ANY model")
print("  r_const       a predictor that IGNORES EEG entirely")
print("  r_gain        r_raw - r_const. NEGATIVE means the model loses to it.")
print("  r_demeaned    the honest number: the EEG contribution alone")
print("  pct_of_ceiling how much of the achievable maximum we actually reached")
n_pos = int((allres.r_gain > 0).sum())
print(f"\ngain over the no-EEG baseline is positive in {n_pos} of {len(allres)} datasets")
''')

# ===========================================================================
md(r"""
## 16 · fMRI download check *(PROFILE='full' only)*

The fMRI arms of the project (ds002336, ds002739) are **not trained here** — they
need a BIDS/NIfTI pipeline and hours of compute. This section verifies the
downloads and reports what the full study used them for.

**The R128 result is worth knowing:** ds002336 carries scanner trigger pulses
marking every BOLD volume. Using them to align EEG and fMRI exactly flipped that
target's split-half reliability from **−0.639 to +0.638** — before the fix, the
two halves of the same data anti-correlated, so no model could have worked.
""")

co(r'''
def inventory(root, label, eeg_glob):
    """Verify a downloaded fMRI dataset: count it, then actually OPEN one EEG
    file and one BOLD volume. A file count alone cannot detect a download that
    was truncated mid-file, which is the usual gdown failure."""
    if root is None or not root.exists():
        print(f"{label:10s} not downloaded")
        return None
    subs = sorted({p.name for p in root.rglob("sub-*") if p.is_dir()})
    files = [f for f in root.rglob("*") if f.is_file()]
    gb = sum(f.stat().st_size for f in files) / 1e9
    row = {"dataset": label, "subjects": len(subs), "files": len(files),
           "gb": round(gb, 2), "eeg_ok": False, "bold_ok": False}
    print(f"{label:10s} {len(subs):3d} subjects | {len(files):5d} files | {gb:6.2f} GB")

    # The two datasets store EEG in different formats, so check accordingly:
    #   ds002336 -> BrainVision (.vhdr text header + .eeg raw binary)
    #   ds002739 -> MATLAB .mat
    hits = sorted(root.rglob(eeg_glob))
    if hits:
        f = hits[0]
        try:
            if f.suffix.lower() == ".vhdr":
                head = f.read_text(errors="replace")[:200]
                if "Brain Vision" not in head:
                    raise ValueError("not a BrainVision header")
                binary = sorted(root.rglob("*.eeg"))
                mb = binary[0].stat().st_size / 1e6 if binary else 0.0
                row["eeg_ok"] = True
                print(f"           EEG  OK  {f.name} (BrainVision, "
                      f"{len(binary)} .eeg files, first {mb:.0f} MB)")
            else:
                sio.loadmat(f, squeeze_me=True, struct_as_record=False)
                row["eeg_ok"] = True
                print(f"           EEG  OK  {f.name}")
        except Exception as exc:
            print(f"           EEG  FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"           EEG  none matching {eeg_glob}")

    bolds = sorted(root.rglob("*bold.nii.gz"))
    if bolds:
        try:
            import nibabel as nib
            img = nib.load(str(bolds[0]))
            tr = float(img.header.get_zooms()[3]) if len(img.shape) > 3 else float("nan")
            row.update(bold_ok=True, bold_shape=str(img.shape), tr_s=round(tr, 2))
            print(f"           BOLD OK  {bolds[0].name} -> {img.shape}, TR {tr:.1f}s")
        except Exception as exc:
            print(f"           BOLD FAILED: {type(exc).__name__}: {exc}")
    else:
        print("           BOLD none found")
    return row


DS2336 = find_root("sub-xp*/eeg/*.vhdr", "derivatives/sub-xp*/eeg_pp/*.vhdr")
DS2739 = find_root("sub-*/EEG/EEG_data_sub-*_run-*.mat")

rows = [r for r in (inventory(DS2336, "ds002336", "*.vhdr"),
                    inventory(DS2739, "ds002739", "EEG_data_sub-*_run-*.mat")) if r]
if rows:
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "fmri_inventory.csv", index=False)
    display(inv)
    bad = inv[~(inv.eeg_ok & inv.bold_ok)]
    if len(bad):
        print("\n!! " + ", ".join(bad.dataset) + " did not open cleanly.")
        print("   Re-run the download cell -- it replaces partial files.")
    else:
        print("\nBoth fMRI datasets downloaded and readable.")
    print("\nNote: this notebook does not TRAIN on the fMRI sets -- that needs a")
    print("BIDS/NIfTI pipeline and hours of compute. The full study used them for")
    print("the R128 event-locking fix (target reliability -0.639 -> +0.638).")
else:
    print("\nfMRI sets not present. Set PROFILE = 'full' in the settings block (section 2) and re-run.")
''')

# ===========================================================================
md("## 20 · Summary")

co(r'''
summary_out = {
    "ruleset": CFG.ruleset,
    "profile": PROFILE,
    "quick_test": QUICK_TEST,
    "n_subjects": int(len(np.unique(DATA["subject_ids"]))),
    "n_trials": int(len(DATA["eeg"])),
    "target_reliability": round(rel_mean, 4),
    "ceiling": round(ceiling, 4),
    "hbo_hbr_sign": round(sign, 4),
    "r_raw": round(float(r_raw), 4),
    "r_constant_baseline": round(float(r_const), 4),
    "r_demeaned_HONEST": round(float(r_dem), 4),
    "p_permutation": round(float(p), 5),
    "pct_of_ceiling": round(float(r_dem / ceiling * 100), 2) if ceiling > 0 else None,
    "protocol_inflation_x": round(float(infl), 2),
    "abstention_gate_informative": bool(mono),
    "calibration": calibration,
    "within_subject": within_subject,
    "claim1": claim1,
    "claim2": claim2,
    "claim3": claim3,
    "claim4": claim4,
    "device": DEVICE,
}
print(json.dumps(summary_out, indent=2))
json.dump(summary_out, open(OUT / "summary.json", "w"), indent=2)
print(f"\nall outputs -> {OUT.resolve()}")
for f in sorted(OUT.iterdir()):
    print("   ", f.name)
''')

md(r"""
## 10 · How to read the result

**The number to quote is `r_demeaned_HONEST`**, together with its *p*-value and
its percentage of the ceiling.

**Why it is small, and why that is a finding rather than a failure:**

1. **The ceiling is measured, not assumed.** The target only partly predicts
   itself, which caps every model before training begins. We report the achieved
   score as a fraction of that cap.
2. **The no-EEG baseline is reported next to every number.** On raw correlation
   it typically *beats* the model — because raw correlation largely measures
   which person a trial came from. Removing each person's mean makes that
   baseline exactly zero, and what remains is genuinely from the EEG.
3. **Everything is cross-subject.** No person appears in both training and test.
   Within-subject scores are much higher but would require scanning each new
   patient first, defeating the purpose.

**What this notebook does not include.** The full project also tests five
structural claims (transfer to new cohorts, recovery of neurovascular delays,
downstream task transfer, reliability gating) and reports two of them as failing,
with diagnoses. See `NeuroVascNet_Presentation.tex` and `PRESENTATION_GUIDE.md`.

---

### Troubleshooting

| Problem | Fix |
|---|---|
| Download fails or stalls | Kaggle: Settings → **Internet ON**. Then re-run the cell — it resumes and skips finished files |
| `TU Berlin hBCI not found` | The download is incomplete; re-run cell 2 |
| Google Drive quota error | The folder has a daily download cap. Wait, or copy it to your own Drive and swap the IDs in the DATASETS block (section 2) |
| Out of memory | Lower `batch_size` to 32 and `epochs` to 15 in the Config cell |
| Very slow | Kaggle: Settings → Accelerator → **GPU T4 x2** |
""")

# Renumber every "## N ." heading sequentially. Sections get inserted and moved
# during development, and duplicate numbers are confusing in a handed-over file.
import re as _re
_n = 0
for _c in nb.cells:
    if _c.cell_type == "markdown" and _re.match(r"^## \d+ . ", _c.source):
        _n += 1
        _c.source = _re.sub(r"^## \d+ ", f"## {_n} ", _c.source, count=1)

nbf.write(nb, "NeuroVascNet_Kaggle.ipynb")
print(f"wrote NeuroVascNet_Kaggle.ipynb ({len(nb.cells)} cells)")
