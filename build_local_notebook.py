"""Emit NeuroVascNet_Local.ipynb -- a LOCAL-ONLY version, no Kaggle, no downloads.

Takes the full notebook and replaces its setup with a single settings cell:
you point DATA_DIR at your downloaded datasets and run. Everything after the
setup is identical -- same loaders, same model, same five claims.

Removed: gdown, Google Drive ids, PROFILE, the download gate, Kaggle paths,
         the disk-space checks, and the RUN_SMOKE_TEST switch.
Kept:    all analysis, and the smoke test (now always on -- it costs 60 s and
         catches plumbing errors before an hour of training).

Usage:  python build_kaggle_notebook.py && python build_local_notebook.py
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

SRC = Path("NeuroVascNet_Kaggle.ipynb")
DST = Path("NeuroVascNet_Local.ipynb")

if not SRC.exists():
    raise SystemExit(f"{SRC} not found -- run build_kaggle_notebook.py first")

nb = nbf.read(str(SRC), as_version=4)

# --------------------------------------------------------------- new header
HEADER = r"""
# NeuroVascNet — local run

**Set one path in the next cell, then Run All.** Nothing is downloaded.

---

### What this does

Trains an EEG-only model to predict the haemodynamic (blood-flow) response and
evaluates it **honestly** — cross-subject, against a baseline that ignores EEG,
with an exact permutation null — then tests the project's five claims.

### What you need

Python 3.9+ and the datasets already on disk. A GPU is optional; it runs on CPU.

| Section | Needs |
|---|---|
| main result, claims 1 / 3 / 4 / 5, ablation | **TU Berlin hBCI** (required) |
| clinical comparison | HEFMI-ICH |
| claim 2, external validation | TU Berlin EEG + NIRS (workload) |
| fMRI loaders | ds002336, ds002739 |

Missing datasets are skipped and the notebook says which sections that disables.

### Runtime

| Step | Time |
|---|---|
| Load + preprocess | 5–10 min |
| Smoke test | < 1 min |
| Train + evaluate (5 folds) | 10–25 min CPU, 3–8 min GPU |
| Claims 1–5 | 15–40 min |
| fMRI loaders *(optional)* | 10–30 min |

Set **`QUICK_TEST = True`** for a ~5 minute trial run over the whole notebook
before committing to the real one.
"""

# --------------------------------------------------------------- settings cell
SETTINGS = r"""
# =============================================================================
#                        >>>  EDIT THESE FOUR PATHS  <<<
#
#   The design: ONE shared EEG encoder, supervised by BOTH modalities.
#   Two datasets train it, two completely separate datasets validate it.
#
#     TRAIN    fNIRS  +  fMRI      ->   shared encoder   ->   EEG-only at test
#     VALIDATE fNIRS  +  fMRI      ->   never seen during training
# =============================================================================

# ---- 1. TRAINING -- fNIRS supervision ----------------------------------------
# HEFMI-ICH. 37 people (17 control, 20 haemorrhage), motor imagery L/R.
# Target: 90 HbO + 90 HbR = 180 channels.
# Folder contains:  "epo_data/Normals/"  and  "epo_data/Patients/"
PATH_TRAIN_FNIRS = r"G:\research_bio_medical\dataset\HEFMI-ICH EEG+FNRI"

# ---- 2. TRAINING -- fMRI supervision -----------------------------------------
# OpenNeuro ds002336. 10 subjects. Carries R128 scanner triggers, which give
# exact EEG<->BOLD alignment. Target: Harvard-Oxford ROI BOLD.
# Folder contains:  "derivatives/"  and  "task-*_events.tsv"
PATH_TRAIN_FMRI = r"G:\research_bio_medical\dataset\OpenNeuro ds002336"

# ---- 3. EXTERNAL VALIDATION -- fNIRS -----------------------------------------
# TU Berlin hBCI. 29 DIFFERENT people, motor imagery. Never trained on.
# Folder contains:  "EEG data/"  and  "NRI data/"
PATH_EXTERNAL_FNIRS = r"G:\research_bio_medical\dataset\TU Berlin hBCI EEG+FNRI"

# ---- 4. EXTERNAL VALIDATION -- fMRI ------------------------------------------
# OpenNeuro ds002739. 24 DIFFERENT subjects. Never trained on.
# Folder contains:  "sub-01/EEG/"  and  "sub-01/func/"
PATH_EXTERNAL_FMRI = r"G:\research_bio_medical\dataset\External fMRI validation OpenNeuro ds002739"

# ---- TRIAL RUN ---------------------------------------------------------------
# True -> few subjects, 2 epochs, ~5 min. False -> the real run.
QUICK_TEST = False

# Set a path to None to skip it. At least one TRAINING path is required.
# =============================================================================

from pathlib import Path

DATASET_PATHS = {
    "TRAIN    fNIRS (HEFMI)":     (PATH_TRAIN_FNIRS,    "epo_data",
                                   "fNIRS supervision + clinical comparison"),
    "TRAIN    fMRI  (ds002336)":  (PATH_TRAIN_FMRI,     "derivatives",
                                   "fMRI supervision"),
    "EXTERNAL fNIRS (hBCI)":      (PATH_EXTERNAL_FNIRS, "EEG data",
                                   "held-out fNIRS validation"),
    "EXTERNAL fMRI  (ds002739)":  (PATH_EXTERNAL_FMRI,  "sub-01",
                                   "held-out fMRI validation"),
}

# A training set and its matching external set must never be the same folder,
# or the external validation is just testing on the training data.
for _a, _b, _mod in (("PATH_TRAIN_FNIRS", "PATH_EXTERNAL_FNIRS", "fNIRS"),
                     ("PATH_TRAIN_FMRI", "PATH_EXTERNAL_FMRI", "fMRI")):
    _pa, _pb = eval(_a), eval(_b)
    if _pa and _pb and Path(_pa).resolve() == Path(_pb).resolve():
        raise ValueError(
            f"{_a} and {_b} point at the same folder.\n"
            f"The {_mod} external validation would be testing on training data.")

RESOLVED = {}
_have_train = False
for _name, (_raw, _marker, _what) in DATASET_PATHS.items():
    if not _raw:
        print(f"  [off ] {_name:26s} not set -- {_what} skipped")
        RESOLVED[_name] = None
        continue
    _p = Path(_raw)
    if not _p.exists():
        print(f"  [BAD ] {_name:26s} path does not exist: {_p}")
        RESOLVED[_name] = None
    elif not (_p / _marker).exists():
        print(f"  [BAD ] {_name:26s} no '{_marker}/' inside: {_p}")
        RESOLVED[_name] = None
    else:
        print(f"  [ok  ] {_name:26s} {_what}")
        RESOLVED[_name] = _p
        if _name.startswith("TRAIN"):
            _have_train = True

if not _have_train:
    raise FileNotFoundError(
        "\nNo TRAINING dataset resolved. Set PATH_TRAIN_FNIRS and/or "
        "PATH_TRAIN_FMRI to a folder that exists.")

print()
if QUICK_TEST:
    print("+==============================================================+")
    print("|                    RUNNING: SMOKE TEST                       |")
    print("|                                                              |")
    print("|  A few subjects, 2 epochs, ~5 minutes.                       |")
    print("|  Purpose: prove every cell runs end to end.                  |")
    print("|  THE NUMBERS ARE NOISY -- do not report them.                |")
    print("|                                                              |")
    print("|  Set QUICK_TEST = False above for the real run.              |")
    print("+==============================================================+")
else:
    print("+==============================================================+")
    print("|                      RUNNING: FULL RUN                       |")
    print("|                                                              |")
    print("|  All subjects, 30 epochs, 1000 permutations.                 |")
    print("|  Expect 1-3 hours depending on your GPU.                     |")
    print("|  These are the numbers you report.                           |")
    print("|                                                              |")
    print("|  Set QUICK_TEST = True above for a 5-minute check first.     |")
    print("+==============================================================+")
"""

# --------------------------------------------------------------- root finding
LOCATE = r"""
# --- the four datasets, by ROLE ----------------------------------------------
HEFMI    = RESOLVED["TRAIN    fNIRS (HEFMI)"]      # train, fNIRS supervision
DS2336   = RESOLVED["TRAIN    fMRI  (ds002336)"]   # train, fMRI supervision
TUBERLIN = RESOLVED["EXTERNAL fNIRS (hBCI)"]       # held out, fNIRS
DS2739   = RESOLVED["EXTERNAL fMRI  (ds002739)"]   # held out, fMRI

WL = None                      # workload set is not part of this design
DATA_DIR = (HEFMI or DS2336 or TUBERLIN).parent
SUBJECTS_AVAILABLE = None
OUT = Path("neurovascnet_outputs_local")
OUT.mkdir(exist_ok=True)

print("roles:")
print(f"  TRAIN    fNIRS   {HEFMI    if HEFMI    else '-- skipped'}")
print(f"  TRAIN    fMRI    {DS2336   if DS2336   else '-- skipped'}")
print(f"  EXTERNAL fNIRS   {TUBERLIN if TUBERLIN else '-- skipped'}")
print(f"  EXTERNAL fMRI    {DS2739   if DS2739   else '-- skipped'}")
print(f"  outputs          {OUT.resolve()}")
"""

cells = nb.cells

# 1. header
cells[0] = nbf.v4.new_markdown_cell(HEADER.strip())

# 2. install cell: drop gdown, keep the rest
inst = cells[2].source
inst = inst.replace('("gdown", "gdown>=5.1.0"),\n                 ', "")
inst = inst.replace('("gdown", "gdown>=5.1.0"), ', "")
inst = re.sub(r"\nON_KAGGLE = .*?\n(?:            .*\n)?", "\n", inst)
inst = inst.replace(
    'print(f"device: {DEVICE} | environment: '
    "{'Kaggle' if ON_KAGGLE else 'local'}\")",
    'print(f"device: {DEVICE}")')
cells[2].source = inst

# 3. replace the three download cells (3,4,5,6) with settings + locate
new_setup = [
    nbf.v4.new_markdown_cell("## 1 · Settings — point this at your data"),
    nbf.v4.new_code_cell(SETTINGS.strip()),
    nbf.v4.new_markdown_cell("## 2 · Locate the datasets"),
    nbf.v4.new_code_cell(LOCATE.strip()),
]
cells[3:7] = new_setup

nb.cells = cells

# 4. strip leftovers anywhere in the notebook
for c in nb.cells:
    if c.cell_type != "code":
        continue
    s = c.source
    s = s.replace("if RUN_SMOKE_TEST:\n    assert smoke_test(), "
                  '"smoke test FAILED -- see the list above"\n'
                  'else:\n    print("smoke test skipped (RUN_SMOKE_TEST = False)")',
                  'assert smoke_test(), "smoke test FAILED -- see the list above"')
    s = s.replace('"/kaggle/working/outputs" if ON_KAGGLE else "outputs"',
                  '"neurovascnet_outputs_local"')
    s = s.replace("Path(\"/kaggle/temp/data\")", "DATA_DIR")
    c.source = s

# 4b. drop the download-verification section: there is no download locally, and
#     the fMRI loader section already reports what it managed to read.
_drop = []
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and "fMRI download check" in c.source:
        _drop = [i, i + 1]
        break
for i in reversed(_drop):
    if i < len(nb.cells):
        del nb.cells[i]

# 4c. rewrite language that only made sense with the downloader
SWAPS = [
    ("*(only if `hefmi` was downloaded)*", "*(needs the HEFMI dataset)*"),
    ("*(needs `PROFILE='claims'` or `'full'`)*", "*(needs the TU Berlin workload dataset)*"),
    ("*(PROFILE='full')*", "*(needs ds002336 / ds002739)*"),
    ("*(PROFILE='full' only)*", "*(needs ds002336 / ds002739)*"),
    ("## 3 · Settings", "## 3 · Analysis configuration"),
    ("Set PROFILE = 'full' in the settings block (section 2) and re-run.",
     "Add the dataset to DATA_DIR and re-run."),
    ("Set PROFILE = 'claims' in the settings block (section 2) and re-run.",
     "Add 'TU Berlin EEG + NIRS' to DATA_DIR and re-run."),
    ("use PROFILE = 'core' or above in the settings block (section 2).",
     "add 'HEFMI-ICH EEG+FNRI' to DATA_DIR."),
    ("set PROFILE = 'full' in section 2", "add it to DATA_DIR"),
    ("Set PROFILE = 'full' in cell 3 and re-run.", "Add the dataset to DATA_DIR."),
    ("not downloaded -- set PROFILE = 'full' in section 2",
     "not present in DATA_DIR"),
    ("TU Berlin workload not downloaded.", "TU Berlin workload dataset not present."),
    ("HEFMI not downloaded -- set GET['hefmi'] = True in cell 3 and re-run.",
     "HEFMI not present in DATA_DIR."),
    ("fMRI sets not present. Set PROFILE = 'full' in the settings block "
     "(section 2) and re-run.", "fMRI datasets not present in DATA_DIR."),
    ('"profile": PROFILE,', '"data_dir": str(DATA_DIR),'),
    ('ruleset: str = "kaggle-frozen-2026-08-07"',
     'ruleset: str = "frozen-2026-08-07"'),
    # leftover Kaggle-detection comment: meaningless in the local build
    ('# Detect Kaggle from its env var first: on Windows, Path("/kaggle") resolves\n'
     '# against the current drive (e.g. G:\\kaggle), so a stray folder of that name\n'
     '# would otherwise make a local machine look like Kaggle.\n', ""),
    ("| Download fails or stalls | Kaggle: Settings → **Internet ON**. "
     "Then re-run the cell — it resumes and skips finished files |",
     "| `DATA_DIR does not exist` | Fix the path in section 2 |"),
    ("| `TU Berlin hBCI not found` | The download is incomplete; re-run cell 2 |",
     "| `TU Berlin hBCI not found` | Folder name must match exactly — see the layout in section 2 |"),
    ("| Google Drive quota error | The folder has a daily download cap. Wait, "
     "or copy it to your own Drive and swap the IDs in the DATASETS block (section 2) |",
     "| A section is skipped | That dataset is not in `DATA_DIR` — the log says which |"),
    ("| Very slow | Kaggle: Settings → Accelerator → **GPU T4 x2** |",
     "| Very slow | Set `QUICK_TEST = True` first, or use a machine with a GPU |"),
    ("is the download complete?", "is the dataset complete?"),
    ("The download is INCOMPLETE. Re-run the download cell; it",
     "The dataset is incomplete. Check DATA_DIR against the"),
    ("resumes and skips files already on disk. Wait for it to",
     "layout in section 2 -- subject folders are missing."),
    ("print('-> N/346 files' before running this cell.", "print(''))"),
]
for c in nb.cells:
    for a, b in SWAPS:
        if a in c.source:
            c.source = c.source.replace(a, b)

# 5. renumber every "## N · " heading sequentially, so inserting/removing
#    sections above cannot leave duplicate or skipped numbers
_n = 0
for c in nb.cells:
    if c.cell_type == "markdown" and re.match(r"^## \d+ · ", c.source):
        _n += 1
        c.source = re.sub(r"^## \d+ · ", f"## {_n} · ", c.source, count=1)

# 6. a reference cell naming every variable, so nothing is a mystery
VARS_MD = r"""
## Every variable in this notebook

Only the first two are meant to be edited. The rest are listed so nothing in the
code is unexplained.

### You set these

| Variable | Default | What it does |
|---|---|---|
| `DATA_DIR` | — | Folder containing the dataset folders |
| `QUICK_TEST` | `False` | `True` = few subjects, 2 epochs, ~5 min trial run |

### Resolved for you (section 2)

| Variable | What it is |
|---|---|
| `TUBERLIN` | TU Berlin hBCI root — **required** |
| `HEFMI` | HEFMI-ICH root, or `None` → clinical section skipped |
| `WL` | TU Berlin workload root, or `None` → claim 2 skipped |
| `DS2336`, `DS2739` | fMRI roots, or `None` → fMRI loader skipped |
| `OUT` | Where every CSV and figure is written |
| `DEVICE` | `cuda` if a GPU is present, else `cpu` |
| `SUBJECTS` | Which subject numbers get loaded |

### Analysis settings (`CFG`, section 3)

| Variable | Default | Meaning |
|---|---|---|
| `tmin_s`, `tmax_s` | −2.0, 28.0 | Trial window, seconds relative to stimulus |
| `eeg_fs`, `hemo_fs` | 200, 10 | Sampling rates (Hz) |
| `eeg_samples` | 6000 | 30 s × 200 Hz |
| `hemo_samples` | 300 | 30 s × 10 Hz |
| `baseline_s` | 2.0 | Pre-stimulus baseline used for correction |
| `n_folds` | 5 | Subject-wise CV folds — no person in two folds |
| `n_perm` | 1000 | Permutation-null shuffles; sets the *p* floor at 0.001 |
| `apply_gsr` | `False` | Global signal regression. **Off** — it removes 83% of the evoked response |
| `task` | `"MI"` | Motor imagery runs |
| `epochs`, `batch_size`, `lr` | 30, 64, 1e-3 | Training budget |

### Loss weights (`CFG`)

| Variable | Default | Term |
|---|---|---|
| `w_main` | 1.0 | Pearson — shape, not amplitude |
| `w_hetero` | 0.1 | Gaussian NLL — trains the variance head |
| `w_delay` | 0.5 | Lag-tolerant cross-correlation |
| `w_smooth` | 0.05 | Penalises jitter |
| `max_lag` | 3 | Samples of timing slack allowed |

### Constants you should not change

| Variable | Value | Why |
|---|---|---|
| `SEED` | 1337 | Reproducibility |
| `EXT`, `DPF`, `SD_CM` | Prahl / 6.4, 5.8 / 3.0 cm | Beer–Lambert physics |
| `MIN_SUBJECTS` | 8 | Fewest subjects worth training on |
| `TR_S` | 2.0 | fMRI repetition time (seconds) |
| `EPOCH_VOL` | 15 | 15 volumes × 2 s = the same 30 s window |
| `MOTOR_KEYS` | Precentral, … | Sensorimotor ROIs; the audit showed this raises reliability |
"""
nb.cells.insert(1, nbf.v4.new_markdown_cell(VARS_MD.strip()))

nbf.write(nb, str(DST))
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"wrote {DST} ({len(nb.cells)} cells, {n_code} code)")
