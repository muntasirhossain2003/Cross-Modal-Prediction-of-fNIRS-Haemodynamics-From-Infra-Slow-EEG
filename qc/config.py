"""Paths, dataset registry, and shared constants for the NeuroVascNet data-quality audit.

Everything the QC pipeline touches is declared here so the manifest, the metric
code and the report all agree on what "the data" is.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO / "dataset"
QC_ROOT = REPO / "qc_outputs"

MANIFEST_CSV = QC_ROOT / "manifest.csv"
FIG_DIR = QC_ROOT / "figures"
ARRAY_DIR = QC_ROOT / "arrays"       # PSDs and other saved vectors
TABLE_DIR = QC_ROOT / "tables"

for _d in (QC_ROOT, FIG_DIR, ARRAY_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset registry.
#
# `dir_name` is what is actually on disk. `documented_*` fields come from
# study_reference_facts.csv and are deliberately kept separate so the audit can
# report disagreements between the documentation and the files.
# --------------------------------------------------------------------------
DATASETS = {
    "HEFMI": {
        "dir_name": "HEFMI-ICH EEG+FNRI",
        "role": "primary EEG-fNIRS supervision (train)",
        "modalities": ("eeg", "fnirs"),
        "documented_subjects": 37,
        "documented_task": "left/right hand motor imagery",
        "documented_eeg_fs": 256.0,
        "documented_eeg_channels": 32,
        "documented_fnirs_fs": 11.0,
        "documented_fnirs_channels": 90,
        "simultaneous_fmri": False,
    },
    "ds002336": {
        "dir_name": "OpenNeuro ds002336",
        "role": "primary EEG-fMRI supervision (train)",
        "modalities": ("eeg", "fmri"),
        "documented_subjects": 10,
        "documented_task": "motor imagery + neurofeedback",
        "documented_eeg_fs": 5000.0,
        "documented_eeg_channels": 64,
        "documented_tr": 2.0,
        "simultaneous_fmri": True,
    },
    "TUBerlin": {
        "dir_name": "TU Berlin EEG + NIRS",
        "role": "external EEG-fNIRS validation",
        "modalities": ("eeg", "fnirs"),
        "documented_subjects": 29,
        "documented_task": "motor imagery + mental arithmetic/rest",
        "documented_eeg_fs": 200.0,
        "documented_eeg_channels": 30,
        "documented_fnirs_fs": 10.0,
        "documented_fnirs_channels": 36,
        "simultaneous_fmri": False,
    },
    "ds002739": {
        "dir_name": "External fMRI validation OpenNeuro ds002739",
        "role": "external EEG-fMRI validation",
        "modalities": ("eeg", "fmri"),
        "documented_subjects": 24,
        "documented_task": "random-dot motion decision + confidence",
        "documented_eeg_fs": 1000.0,
        "documented_eeg_channels": 64,
        "documented_tr": 2.0,
        "simultaneous_fmri": True,
    },
}


def dataset_path(key: str) -> Path:
    return DATASET_ROOT / DATASETS[key]["dir_name"]


def clean_path(value) -> Path | None:
    """Normalise a manifest path cell to a Path or None.

    Empty CSV cells come back from pandas as NaN, which is a float *and* is
    truthy -- so a plain `if not value` guard lets it through to Path() and
    raises TypeError. Anything that is not a usable path string becomes None.
    """
    if value is None:
        return None
    if isinstance(value, float):      # NaN (or any stray numeric cell)
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    return Path(s)


# --------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------
LINE_FREQ_CANDIDATES = (50.0, 60.0)   # decided per-recording by which has more power
EEG_BAND_FILTER = (0.5, 40.0)         # the "post-filter" band used for the amplitude distribution
MU_BAND = (8.0, 13.0)                 # sensorimotor rhythm, used for the ERD sanity check
BETA_BAND = (13.0, 30.0)
CARDIAC_BAND = (0.8, 2.0)             # BCG fundamental search range
FNIRS_CARDIAC_BAND = (0.5, 2.5)       # used for scalp-coupling-style checks
HRF_PEAK_S = 6.0

# Harvard-Oxford cortical, 48 ROIs -- matches the modelling pipeline in
# NeuroVascNet_v3.ipynb so ROI coverage QC describes the actual target.
ATLAS_NAME = "cort-maxprob-thr25-2mm"
N_ROIS = 48

# Motor / sensorimotor ROI subset of Harvard-Oxford cortical (1-indexed labels),
# used for the event-locked ROI sanity check on motor tasks.
MOTOR_ROI_LABELS = (
    "Precentral Gyrus",
    "Postcentral Gyrus",
    "Supplementary Motor Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)",
)
