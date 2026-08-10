# ═══ CELL 3 ═══  Import all libraries (torch, scipy, sklearn, nilearn, matplotlib).
# why: Everything below depends on these; loaded once at the top.
import os
import math
import json
import random
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.signal import butter, filtfilt, resample, hilbert
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
# ═══ CELL 4 ═══  Set random seed and pick GPU/CPU device.
# why: Makes runs reproducible and uses the GPU when available.
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Torch: {torch.__version__}")
# ═══ CELL 6 ═══  CONFIG dataclass — every setting/hyperparameter in one place.
# why: Single source of truth: sampling rates, window length, channel counts, training settings.
@dataclass
class Config:
    # ============================================================
    # SIGNAL PARAMETERS
    # Sampling rates each modality is RESAMPLED to inside the loaders.
    # One shared EEG encoder needs identical input timing for every
    # dataset, so all four are forced onto these common rates.
    # ============================================================
    eeg_fs: int = 200          # EEG samples/sec. ds002336 & TU Berlin are already 200 Hz.
                               #   -> Resample Stroop EEG 1000->200 and ds002739 EEG ->200.
    fnirs_fs: int = 10         # fNIRS samples/sec. TU Berlin NIRS is 10 Hz.
                               #   -> Resample Stroop fNIRS (raw 100 Hz) ->10 Hz.
    fmri_tr: float = 2.0       # fMRI repetition time (sec/volume). ds002336 & ds002739 = 2.0 s.

    # ------------------------------------------------------------
    # ANALYSIS WINDOW LENGTH (seconds)  <-- KEY FIX (was 10.0)
    # ------------------------------------------------------------
    # Each training example = a window_seconds slice of EEG paired
    # with the same-length slice of hemodynamic signal.
    # The BOLD response is SLOW: peaks ~5-6 s after neural activity,
    # takes ~20-30 s to rise and fall; the delay module also shifts
    # it 2-8 s. With a 10 s window the fMRI head saw only 10/2 = 5
    # BOLD samples -> too short -> it learned almost nothing
    # (Pearson ~0.02). 30 s gives 15 BOLD samples: room for the full
    # response PLUS the delay shift.
    window_seconds: float = 30.0

    # ============================================================
    # EEG CHANNEL LAYOUT
    # Must be the COMMON channel set shared by all four datasets
    # (one shared encoder = one fixed channel count).
    # Native: Stroop 29 scalp (drop M1/M2/HEO/VEO), TU Berlin 30,
    # ds002336 64. Stroop's 29 is the cap.
    # TODO: set this to the intersection count the loader prints.
    # ============================================================
    n_eeg_channels: int = 29   # COMMON_EEG_CHANNELS (Stroop ∩ ds002336). Shrink when TU Berlin/ds002739 wired.

    # ============================================================
    # fNIRS CHANNEL LAYOUT
    # Common cortical subset of Stroop (20 ch) and TU Berlin (36 ch),
    # so this must be <= 20. Each channel has HbO and HbR.
    # ============================================================
    n_fnirs_channels: int = 20   # Stroop has 20 fNIRS channels (HbO+HbR -> 40). Reduce to common subset later.
    fnirs_hb_types: int = 2      # HbO, HbR

    # ============================================================
    # fMRI ROI LAYOUT
    # n_rois is a MODELING CHOICE, not a data property. We do not
    # predict ~100k voxels; we average the BOLD volume into n_rois
    # atlas regions (Schaefer / Harvard-Oxford) and predict those.
    # Keep 20 unless you want finer granularity.
    # ============================================================
    n_rois: int = 48

    # ============================================================
    # ENCODER ARCHITECTURE (leave as-is for now)
    # ============================================================
    encoder_temporal_filters: int = 16
    encoder_spatial_filters: int = 2
    encoder_kernel_length: int = 64
    encoder_pool: int = 4
    tcn_channels: int = 32   # Fix#1: smaller model (was 64)
    tcn_layers: int = 4
    tcn_kernel: int = 5
    latent_dim: int = 64    # Fix#1: smaller model (was 128)
    dropout: float = 0.5    # Fix#1: heavy dropout against overfitting

    # ============================================================
    # DELAY MODULE (physiologically plausible lag window, seconds)
    # ============================================================
    delay_min_s: float = 2.0
    delay_max_s: float = 8.0

    # ============================================================
    # TRAINING (leave as-is; lower epochs only for quick test runs)
    # ============================================================
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4   # Fix#1: stronger L2 (was 1e-5)
    epochs_stage1: int = 25   # Fix2: many more gradient updates (was 8)
    epochs_stage2: int = 8    # (was 6)
    epochs_stage3: int = 4

    # ============================================================
    # LOSS WEIGHTS (later: coarse grid-search on internal validation)
    # ============================================================
    w_hb: float = 1.0
    w_bold: float = 1.0
    w_delay_fnirs: float = 0.3
    w_delay_fmri: float = 0.3
    w_align: float = 0.2
    w_hetero: float = 0.1
    w_smooth: float = 0.05

    # ============================================================
    # RELIABILITY
    # ============================================================
    mc_dropout_passes: int = 10
    ensemble_size: int = 3

    # ============================================================
    # DATA SIZING
    # With REAL data these are set by the datasets, not by these
    # knobs. They only matter for the synthetic generators.
    # ============================================================
    n_subjects_train: int = 12
    n_subjects_val: int = 4
    n_trials_per_subject: int = 30

    # Output
    output_dir: str = "./neurovascnet_outputs"

CFG = Config()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# DERIVED WINDOW LENGTHS (computed from the values above).
# These set each head's OUTPUT length, so they must match the real
# data after resampling. With window_seconds=30:
#   EEG_WINDOW   = 200 * 30  = 6000 samples
#   FNIRS_WINDOW = 10  * 30  = 300  samples
#   FMRI_WINDOW  = 30  / 2.0 = 15   samples
# ------------------------------------------------------------
EEG_WINDOW = int(CFG.eeg_fs * CFG.window_seconds)
FNIRS_WINDOW = int(CFG.fnirs_fs * CFG.window_seconds)
FMRI_WINDOW = int(CFG.window_seconds / CFG.fmri_tr)

print(f"EEG window samples: {EEG_WINDOW}")
print(f"fNIRS window samples: {FNIRS_WINDOW}")
print(f"fMRI window samples: {FMRI_WINDOW}")
# ═══ CELL 8 ═══  canonical_hrf — the textbook blood-flow response shape.
# why: Reference/sanity tool; synthetic data also uses it to fake hemodynamics.
def canonical_hrf(length: int, fs: float, peak: float = 5.0, undershoot: float = 12.0) -> np.ndarray:
    """Double-gamma hemodynamic response function."""
    t = np.arange(length) / fs
    from math import gamma as Gamma
    a1, b1 = peak, 1.0
    a2, b2 = undershoot, 1.0
    h = (t**(a1-1) * np.exp(-t/b1) / (b1**a1 * Gamma(a1)) -
         0.35 * t**(a2-1) * np.exp(-t/b2) / (b2**a2 * Gamma(a2)))
    h = h / (np.max(np.abs(h)) + 1e-9)
    return h
# ═══ CELL 9 ═══  generate_paired_eeg_hemo — synthetic data template.
# why: Defines the exact dict schema (eeg/hemo/labels/subject_ids) every real loader must copy.
def generate_paired_eeg_hemo(
    n_subjects: int,
    n_trials: int,
    n_eeg_ch: int,
    n_hemo_ch: int,
    eeg_fs: int,
    hemo_fs: float,
    win_s: float,
    task_amplitude_range: Tuple[float, float],
    delay_range_s: Tuple[float, float],
    coupling_strength: float = 0.6,
    rng_seed: int = 0,
):
    """Generate paired EEG and hemodynamic windows with controlled coupling and delay."""
    rng = np.random.default_rng(rng_seed)
    eeg_len = int(eeg_fs * win_s)
    hemo_len = int(hemo_fs * win_s)
    hrf = canonical_hrf(int(hemo_fs * 20), hemo_fs)

    eeg_all = np.zeros((n_subjects * n_trials, n_eeg_ch, eeg_len), dtype=np.float32)
    hemo_all = np.zeros((n_subjects * n_trials, n_hemo_ch, hemo_len), dtype=np.float32)
    labels = np.zeros(n_subjects * n_trials, dtype=np.int64)
    subj_ids = np.zeros(n_subjects * n_trials, dtype=np.int64)
    delays = np.zeros(n_subjects * n_trials, dtype=np.float32)

    idx = 0
    for s in range(n_subjects):
        subj_delay = rng.uniform(*delay_range_s)
        subj_gain = rng.uniform(0.7, 1.3)
        for t in range(n_trials):
            label = int(rng.integers(0, 2))
            amp = rng.uniform(*task_amplitude_range) * (1.0 if label == 1 else 0.4)

            # neural-like envelope: low-frequency modulation across trial
            env_t = np.linspace(0, win_s, eeg_len)
            envelope = amp * np.exp(-((env_t - win_s/2)**2) / (2 * (win_s/4)**2))

            # multi-channel EEG: bandlimited noise modulated by envelope
            eeg = rng.standard_normal((n_eeg_ch, eeg_len)).astype(np.float32) * 0.3
            for f, w in [(2.5, 0.4), (10.0, 0.6), (20.0, 0.5), (40.0, 0.3)]:
                phase = rng.uniform(0, 2*np.pi, size=(n_eeg_ch, 1))
                eeg += w * np.sin(2*np.pi*f*env_t + phase).astype(np.float32)
            spatial_mask = rng.uniform(0.3, 1.0, size=(n_eeg_ch, 1)).astype(np.float32)
            eeg = eeg + spatial_mask * envelope[None, :].astype(np.float32) * subj_gain
            eeg_all[idx] = eeg

            # hemodynamic ground truth: HRF convolution of downsampled envelope + delay
            env_lo = np.interp(np.linspace(0, win_s, hemo_len),
                               np.linspace(0, win_s, eeg_len), envelope)
            bold_like = np.convolve(env_lo, hrf, mode="full")[:hemo_len]
            shift_samples = int(round(subj_delay * hemo_fs))
            shifted = np.concatenate([np.zeros(shift_samples), bold_like])[:hemo_len]

            hemo = np.zeros((n_hemo_ch, hemo_len), dtype=np.float32)
            for c in range(n_hemo_ch):
                region_gain = rng.uniform(0.4, 1.2)
                noise = rng.standard_normal(hemo_len).astype(np.float32) * 0.15
                slow_drift = 0.1 * np.sin(2*np.pi*0.05*np.arange(hemo_len)/hemo_fs + rng.uniform(0, 2*np.pi))
                hemo[c] = coupling_strength * region_gain * shifted + noise + slow_drift.astype(np.float32)
            hemo_all[idx] = hemo

            labels[idx] = label
            subj_ids[idx] = s
            delays[idx] = subj_delay
            idx += 1

    return {
        "eeg": eeg_all,
        "hemo": hemo_all,
        "labels": labels,
        "subject_ids": subj_ids,
        "true_delays": delays,
    }
# ═══ CELL 10 ═══  REAL loaders (Stroop, ds002336) + synthetic (TU Berlin, ds002739) + common-channel helper.
# why: Reads your actual brain recordings and returns them in the shared dict format.
# =====================================================================
# REAL DATA LOADERS  (Section 3 substitution points)
# =====================================================================
# Stroop EEG-fNIRS  and  ds002336 EEG-fMRI are REAL. TU Berlin and
# ds002739 are still synthetic (replace next). Every loader returns the
# same 5-key dict so cells 11+ are unchanged.
import io as _io
import zipfile as _zipfile
import scipy.io as _sio

# ---------------------------------------------------------------------
# COMMON EEG CHANNELS
# One shared encoder => channel i must be the SAME electrode in every
# dataset. So all loaders select+reorder EEG to this canonical list.
# This is the intersection of Stroop (30 scalp) and ds002336 (64);
# ds002336 lacks FCz, so the shared set is 29. Shrink further when
# TU Berlin / ds002739 are wired (set cfg.n_eeg_channels to len()).
# ---------------------------------------------------------------------
COMMON_EEG_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FC4",
    "FT8", "T7", "C3", "CZ", "C4", "T8", "TP7", "CP3", "CPZ", "CP4",
    "TP8", "P7", "P3", "PZ", "P4", "P8", "O1", "OZ", "O2",
]   # 29 channels


def _select_common_channels(x, labels):
    """x: (n_native_ch, T); labels: native channel names in the same order.
    Returns (29, T) reordered to COMMON_EEG_CHANNELS (case-insensitive)."""
    up = {lab.upper(): i for i, lab in enumerate(labels)}
    idx = [up[c] for c in COMMON_EEG_CHANNELS]   # raises if a channel is missing
    return x[idx]


def _read_brainvision_dat(vhdr_path):
    """Minimal BrainVision reader (no MNE needed) for the BINARY/MULTIPLEXED/
    IEEE_FLOAT_32 files in ds002336 derivatives. Returns (data, fs, labels)."""
    info, labels, section = {}, [], None
    for line in open(vhdr_path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("["):
            section = line
            continue
        if line.startswith(";") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if section == "[Channel Infos]" and k.startswith("Ch"):
            labels.append(v.split(",")[0])
        else:
            info[k] = v
    n_ch = int(info["NumberOfChannels"])
    fs = 1e6 / float(info["SamplingInterval"])
    dat = np.fromfile(os.path.join(os.path.dirname(vhdr_path), info["DataFile"]),
                      dtype="<f4")
    data = dat.reshape(-1, n_ch).T          # (n_ch, T)
    return data, fs, labels


# =====================================================================
# LOADER #1 -- Stroop EEG-fNIRS  (primary fNIRS supervision)  [REAL]
# =====================================================================
STROOP_ROOT = Path(r"g:/research_bio_medical/dataset/Stroop EEG + fNIRS/fNIRS-EEG_Stroop")
STROOP_EEG_DIR = STROOP_ROOT / "Raw Data" / "Raw EEG Data" / "Raw EEG Data In mat"
# Unzipped preprocessed fNIRS dir. IMPORTANT: 2 of the 8 filenames are
# MISLABELED (e.g. HbO_Incongruent_1.mat actually contains variable HbO_N_2).
# We therefore key blocks by the INTERNAL variable name, not the filename.
STROOP_FNIRS_DIR = (STROOP_ROOT / "Pre-processed Data" / "Pre-processed fNIRS Data"
                    / "Pre-Preocessed fNIRS Data In mat")
STROOP_N_SUBJECTS = 21
STROOP_FNIRS_FS_RAW = 100.0
STROOP_BLOCK_PRE_S = 0.11
STROOP_N_VAL_SUBJECTS = 4
STROOP_WIN_STRIDE_S = 2.0   # Fix1: slide overlapping 30s windows inside each ~37s block -> more data
# OFFICIAL block order (Raw EEG Dataset Description): the four blocks are
# Neutral, Neutral, Incongruent, Incongruent -- in that fixed order. The EEG
# trigger codes (1,2,1,2) are NOT condition labels; condition is POSITIONAL.
STROOP_BLOCK_SEQUENCE = [("Neutral", 1), ("Neutral", 2),
                         ("Incongruent", 1), ("Incongruent", 2)]
STROOP_COND_LABEL = {"Neutral": 0, "Incongruent": 1}


def _load_stroop_fnirs_blocks():
    """dict[(hb, cond, occ)] -> (21, 3711, 20). hb in {HbO,Hb}.
    Keyed by the INTERNAL variable name (e.g. 'HbO_N_2'), because 2 of the 8
    filenames are mislabeled. This fixes the EEG<->fNIRS block pairing."""
    cmap = {"N": "Neutral", "I": "Incongruent"}
    out = {}
    for f in os.listdir(STROOP_FNIRS_DIR):
        if not f.endswith(".mat"):
            continue
        m = _sio.loadmat(str(STROOP_FNIRS_DIR / f), squeeze_me=True)
        for k, v in m.items():
            if k.startswith("__") or getattr(v, "ndim", 0) != 3:
                continue
            hb, c, occ = k.split("_")                # 'HbO','N','2'
            out[(hb, cmap[c], int(occ))] = np.asarray(v, dtype=np.float32)
    return out


def _stroop_subject_split(split):
    all_subj = list(range(STROOP_N_SUBJECTS))
    val_subj = set(all_subj[-STROOP_N_VAL_SUBJECTS:])
    keep_val = (split == "val")
    return [s for s in all_subj if (s in val_subj) == keep_val]


def generate_eeg_fnirs_stroop(cfg: Config, split: str = "train"):
    eeg_len = int(cfg.eeg_fs * cfg.window_seconds)
    fnirs_len = int(cfg.fnirs_fs * cfg.window_seconds)
    fnirs = _load_stroop_fnirs_blocks()
    eeg_list, hemo_list, lab_list, sid_list = [], [], [], []

    for s in _stroop_subject_split(split):
        E = _sio.loadmat(STROOP_EEG_DIR / f"S{s+1}.mat",
                         squeeze_me=True, struct_as_record=False)["EEG"]
        srate = float(E.srate)
        data = np.asarray(E.data, dtype=np.float32)
        ch_labels = [c.labels for c in E.chanlocs]
        data = _select_common_channels(data, ch_labels)      # (29, T) common order

        evs = E.event if hasattr(E.event, "__len__") else [E.event]
        stim = []
        for e in evs:
            try:
                ti = int(getattr(e, "type"))
            except (TypeError, ValueError):
                continue
            if ti in (1, 2):
                stim.append((ti, float(getattr(e, "latency"))))
        if len(stim) != 64:
            print(f"  [skip S{s+1}] expected 64 stimulus events, got {len(stim)}")
            continue
        ev_type = [t for t, _ in stim]
        ev_lat = [lat for _, lat in stim]

        win_eeg = int(round(cfg.window_seconds * srate))
        win_fnirs = int(round(cfg.window_seconds * STROOP_FNIRS_FS_RAW))
        # Fix1: each block is ~37s but the window is 30s -> slide several
        # overlapping windows per block (stride STROOP_WIN_STRIDE_S) for more data.
        block_dur = fnirs[("HbO", "Incongruent", 1)].shape[1] / STROOP_FNIRS_FS_RAW
        offsets = np.arange(0.0, max(0.0, block_dur - cfg.window_seconds) + 1e-6,
                            STROOP_WIN_STRIDE_S)
        if len(offsets) == 0:
            offsets = np.array([0.0])
        for b, (cond, occ) in enumerate(STROOP_BLOCK_SEQUENCE):
            block_start = max(0, int(round(ev_lat[b * 16] - STROOP_BLOCK_PRE_S * srate)))
            hbo_block = fnirs[("HbO", cond, occ)][s]
            hbr_block = fnirs[("Hb", cond, occ)][s]
            for off in offsets:
                e0 = block_start + int(round(off * srate))
                seg = data[:, e0:e0 + win_eeg]
                if seg.shape[1] < win_eeg:
                    seg = np.pad(seg, ((0, 0), (0, win_eeg - seg.shape[1])))
                eeg_win = resample(seg, eeg_len, axis=1).astype(np.float32)
                f0 = int(round(off * STROOP_FNIRS_FS_RAW))
                hbo = resample(hbo_block[f0:f0 + win_fnirs], fnirs_len, axis=0)
                hbr = resample(hbr_block[f0:f0 + win_fnirs], fnirs_len, axis=0)
                hemo = np.concatenate([hbo, hbr], axis=1).T.astype(np.float32)
                eeg_list.append(eeg_win); hemo_list.append(hemo)
                lab_list.append(STROOP_COND_LABEL[cond]); sid_list.append(s)

    return {
        "eeg": np.stack(eeg_list), "hemo": np.stack(hemo_list),
        "labels": np.asarray(lab_list, dtype=np.int64),
        "subject_ids": np.asarray(sid_list, dtype=np.int64),
        "true_delays": np.zeros(len(lab_list), dtype=np.float32),
    }


# =====================================================================
# LOADER #3 -- ds002336 EEG-fMRI  (primary fMRI supervision)  [REAL]
# =====================================================================
# Task: eegfmriNF = simultaneous EEG+fMRI neurofeedback (the only task with
# both a preprocessed, MR/BCG-corrected EEG and a BOLD volume). EEG comes
# from derivatives/eeg_pp (.dat float32, 64 ch @ 200 Hz). BOLD -> ROI time
# series via a Harvard-Oxford cortical atlas (48 ROIs). One 30 s window is
# slid across the run (15 s stride). NF non-stationarity is acknowledged in
# the paper's limitations.
DS2336_ROOT = Path(r"g:/research_bio_medical/dataset/OpenNeuro ds002336")
DS2336_TASK = "eegfmriNF"
DS2336_SUBJECTS = [f"sub-xp10{i}" if i < 10 else f"sub-xp1{i}" for i in range(1, 11)]
DS2336_N_VAL = 2          # last 2 subjects -> internal validation
DS2336_STRIDE_S = 8.0     # Fix1: smaller stride -> more overlapping windows -> more data

_HO_MASKER = None
def _get_ho_masker(cfg):
    """Build the Harvard-Oxford ROI masker once (atlas auto-downloaded + cached)."""
    global _HO_MASKER
    if _HO_MASKER is None:
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker
        ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        _HO_MASKER = NiftiLabelsMasker(
            labels_img=ho["maps"], standardize="zscore_sample", detrend=True,
            low_pass=0.1, high_pass=0.01, t_r=cfg.fmri_tr, verbose=0)
    return _HO_MASKER


def generate_eeg_fmri_ds002336(cfg: Config, split: str = "train"):
    eeg_len = int(cfg.eeg_fs * cfg.window_seconds)        # 6000
    fmri_len = int(cfg.window_seconds / cfg.fmri_tr)      # 15
    masker = _get_ho_masker(cfg)

    val_set = set(DS2336_SUBJECTS[-DS2336_N_VAL:])
    keep_val = (split == "val")
    use = [(i, s) for i, s in enumerate(DS2336_SUBJECTS) if (s in val_set) == keep_val]

    eeg_list, hemo_list, lab_list, sid_list = [], [], [], []
    for gi, sub in use:
        vhdr = DS2336_ROOT / "derivatives" / sub / "eeg_pp" / f"{sub}_task-{DS2336_TASK}_eeg_pp.vhdr"
        bold = DS2336_ROOT / sub / "func" / f"{sub}_task-{DS2336_TASK}_bold.nii.gz"
        if not vhdr.exists() or not bold.exists():
            print(f"  [skip {sub}] missing eeg_pp or bold for task {DS2336_TASK}")
            continue
        x, fs, labels = _read_brainvision_dat(vhdr)          # (64, T) @ ~200 Hz
        x = _select_common_channels(x, labels).astype(np.float32)   # (29, T)
        ts = masker.fit_transform(str(bold)).astype(np.float32)     # (n_TR, k<=48)
        # Reindex to a FIXED n_rois layout: some subjects drop empty ROIs,
        # so map each returned column back to its atlas label position.
        roi = np.zeros((ts.shape[0], cfg.n_rois), dtype=np.float32)
        for col in range(ts.shape[1]):
            lab = masker.region_ids_.get(col)
            if isinstance(lab, (int, np.integer)) and 1 <= lab <= cfg.n_rois:
                roi[:, lab - 1] = ts[:, col]

        dur = min(x.shape[1] / fs, roi.shape[0] * cfg.fmri_tr)
        n_win = max(0, int((dur - cfg.window_seconds) // DS2336_STRIDE_S) + 1)
        win_eeg = int(round(cfg.window_seconds * fs))
        for w in range(n_win):
            t0 = w * DS2336_STRIDE_S
            e0 = int(round(t0 * fs))
            seg = x[:, e0:e0 + win_eeg]
            if seg.shape[1] < win_eeg:
                seg = np.pad(seg, ((0, 0), (0, win_eeg - seg.shape[1])))
            eeg_win = resample(seg, eeg_len, axis=1).astype(np.float32)
            b0 = int(round(t0 / cfg.fmri_tr))
            bseg = roi[b0:b0 + fmri_len]
            if bseg.shape[0] < fmri_len:
                bseg = np.pad(bseg, ((0, fmri_len - bseg.shape[0]), (0, 0)))
            eeg_list.append(eeg_win)
            hemo_list.append(bseg.T.astype(np.float32))      # (48, 15)
            lab_list.append(0)                               # NF run: no trial labels
            sid_list.append(gi)

    return {
        "eeg": np.stack(eeg_list), "hemo": np.stack(hemo_list),
        "labels": np.asarray(lab_list, dtype=np.int64),
        "subject_ids": np.asarray(sid_list, dtype=np.int64),
        "true_delays": np.zeros(len(lab_list), dtype=np.float32),
    }


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# EXTERNAL VALIDATION LOADERS  (blueprint §7.4, §8.2)  [REAL]
# Both held-out external sets. NOTE the §12 harmonization caveats:
#  - EEG mapped to the common-29 montage, zero-filling channels a dataset
#    lacks (TU Berlin shares 14/29; ds002739 shares all 29).
#  - TU Berlin EEG & fNIRS were recorded in SEPARATE sessions -> paired
#    block-by-block via condition markers, not sample-by-sample.
#  - TU Berlin fNIRS uses a different probe layout -> restricted to the
#    first 20 channels as the common cortical subset (approximate).
# ---------------------------------------------------------------------
def _map_eeg_to_common(x, labels):
    """x:(n_native_ch, T) -> (29, T) in COMMON_EEG_CHANNELS order, zero-fill missing."""
    up = {str(l).upper(): i for i, l in enumerate(labels)}
    out = np.zeros((len(COMMON_EEG_CHANNELS), x.shape[1]), dtype=np.float32)
    for j, c in enumerate(COMMON_EEG_CHANNELS):
        if c in up:
            out[j] = x[up[c]]
    return out

TUBERLIN_ROOT = Path(r"g:/research_bio_medical/dataset/TU Berlin EEG + NIRS")
TUBERLIN_EEG_FS = 200.0
TUBERLIN_NIRS_FS = 10.0

def generate_eeg_fnirs_tuberlin(cfg: Config):
    """External fNIRS: TU Berlin n-back. EEG & NIRS are separate sessions,
    paired block-by-block via the 0/2/3-back session markers."""
    import glob
    eeg_len = int(cfg.eeg_fs * cfg.window_seconds)
    fnirs_len = int(cfg.fnirs_fs * cfg.window_seconds)
    n_hb = cfg.n_fnirs_channels                     # 20 channels per Hb type
    subs = sorted(os.path.basename(d).split("-")[0]
                  for d in glob.glob(str(TUBERLIN_ROOT / "NIRS" / "VP*-NIRS")))
    eeg_list, hemo_list, lab_list, sid_list = [], [], [], []
    for si, vp in enumerate(subs):
        try:
            E = _sio.loadmat(TUBERLIN_ROOT / "EEG" / f"{vp}-EEG" / "cnt_nback.mat",
                             squeeze_me=True, struct_as_record=False)["cnt_nback"]
            Em = _sio.loadmat(TUBERLIN_ROOT / "EEG" / f"{vp}-EEG" / "mrk_nback.mat",
                              squeeze_me=True, struct_as_record=False)["mrk_nback"]
            N = _sio.loadmat(TUBERLIN_ROOT / "NIRS" / f"{vp}-NIRS" / "cnt_nback.mat",
                             squeeze_me=True, struct_as_record=False)["cnt_nback"]
            Nm = _sio.loadmat(TUBERLIN_ROOT / "NIRS" / f"{vp}-NIRS" / "mrk_nback.mat",
                              squeeze_me=True, struct_as_record=False)["mrk_nback"]
        except Exception as e:
            print(f"  [TUBerlin skip {vp}] {e}"); continue
        eeg = _map_eeg_to_common(np.asarray(E.x, float).T, [str(c) for c in np.atleast_1d(E.clab)])
        oxy = np.asarray(N.oxy.x, float); deoxy = np.asarray(N.deoxy.x, float)   # (T,36)@10
        # EEG session-block onsets (className rows 5..7 = 0/2/3-back session)
        ey = np.atleast_2d(Em.y); et = np.atleast_1d(Em.time).astype(float)
        sess_cols = np.where(ey[5:8].sum(0) > 0)[0] if ey.shape[0] >= 8 else np.arange(0)
        e_on = et[sess_cols]; e_cond = np.argmax(ey[5:8][:, sess_cols], 0) if len(sess_cols) else []
        n_on = np.atleast_1d(Nm.time).astype(float); n_cond = np.argmax(np.atleast_2d(Nm.y), 0)
        for b in range(min(len(e_on), len(n_on))):
            e0 = int(round(e_on[b] / 1000.0 * TUBERLIN_EEG_FS))
            seg = eeg[:, e0:e0 + int(cfg.window_seconds * TUBERLIN_EEG_FS)]
            if seg.shape[1] < int(cfg.window_seconds * TUBERLIN_EEG_FS):
                seg = np.pad(seg, ((0, 0), (0, int(cfg.window_seconds * TUBERLIN_EEG_FS) - seg.shape[1])))
            eeg_win = resample(seg, eeg_len, axis=1).astype(np.float32)
            n0 = int(round(n_on[b] / 1000.0 * TUBERLIN_NIRS_FS))
            wl = int(cfg.window_seconds * TUBERLIN_NIRS_FS)
            ho = oxy[n0:n0 + wl, :n_hb]; hr = deoxy[n0:n0 + wl, :n_hb]
            if ho.shape[0] < wl:
                ho = np.pad(ho, ((0, wl - ho.shape[0]), (0, 0))); hr = np.pad(hr, ((0, wl - hr.shape[0]), (0, 0)))
            hemo = np.concatenate([resample(ho, fnirs_len, axis=0),
                                   resample(hr, fnirs_len, axis=0)], axis=1).T.astype(np.float32)
            eeg_list.append(eeg_win); hemo_list.append(hemo)
            lab_list.append(int(n_cond[b] > 0)); sid_list.append(si)
    return {"eeg": np.stack(eeg_list), "hemo": np.stack(hemo_list),
            "labels": np.asarray(lab_list, np.int64),
            "subject_ids": np.asarray(sid_list, np.int64),
            "true_delays": np.zeros(len(lab_list), np.float32)}

DS2739_ROOT = Path(r"g:/research_bio_medical/dataset/External fMRI validation OpenNeuro ds002739")
def _ds2739_eeg_labels():
    labs = []
    for line in open(DS2739_ROOT / "additional_files" / "electrode_info.elp", encoding="utf-8", errors="replace"):
        p = line.split()
        if p: labs.append(p[0])
    return labs

def generate_eeg_fmri_ds002739(cfg: Config):
    """External fMRI: ds002739. EEG .mat (map to common-29) + BOLD->ROI (same atlas)."""
    import glob
    eeg_len = int(cfg.eeg_fs * cfg.window_seconds); fmri_len = int(cfg.window_seconds / cfg.fmri_tr)
    masker = _get_ho_masker(cfg); elp = _ds2739_eeg_labels()
    subs = sorted(d for d in os.listdir(DS2739_ROOT) if d.startswith("sub-"))
    eeg_list, hemo_list, lab_list, sid_list = [], [], [], []
    for si, sub in enumerate(subs):
        for run in ["run-01", "run-02"]:
            ef = DS2739_ROOT / sub / "EEG" / f"EEG_data_{sub}_{run}.mat"
            bf = DS2739_ROOT / sub / "func" / f"{sub}_task-main_{run}_bold.nii.gz"
            if not ef.exists() or not bf.exists(): continue
            try:
                Ed = _sio.loadmat(ef, squeeze_me=True, struct_as_record=False)["EEGdata"]
                Y = np.asarray(Ed.Y, float)                      # (nch, T)
                fs = float(getattr(Ed, "fd", 1000))              # use downsampled rate
                labs = elp[:Y.shape[0]]
                eeg = _map_eeg_to_common(Y, labs)
                eeg = resample(eeg, int(eeg.shape[1] * cfg.eeg_fs / fs), axis=1)  # ->200Hz
                ts = masker.fit_transform(str(bf)).astype(np.float32)
                roi = np.zeros((ts.shape[0], cfg.n_rois), np.float32)
                for c in range(ts.shape[1]):
                    lab = masker.region_ids_.get(c)
                    if isinstance(lab, (int, np.integer)) and 1 <= lab <= cfg.n_rois: roi[:, lab-1] = ts[:, c]
            except Exception as e:
                print(f"  [ds2739 skip {sub} {run}] {e}"); continue
            dur = min(eeg.shape[1] / cfg.eeg_fs, roi.shape[0] * cfg.fmri_tr)
            n_win = max(0, int((dur - cfg.window_seconds) // 30.0) + 1)
            we = int(cfg.window_seconds * cfg.eeg_fs)
            for w in range(n_win):
                t0 = w * 30.0; e0 = int(t0 * cfg.eeg_fs)
                seg = eeg[:, e0:e0 + we]
                if seg.shape[1] < we: seg = np.pad(seg, ((0, 0), (0, we - seg.shape[1])))
                b0 = int(t0 / cfg.fmri_tr); bseg = roi[b0:b0 + fmri_len]
                if bseg.shape[0] < fmri_len: bseg = np.pad(bseg, ((0, fmri_len - bseg.shape[0]), (0, 0)))
                eeg_list.append(resample(seg, eeg_len, axis=1).astype(np.float32))
                hemo_list.append(bseg.T.astype(np.float32)); lab_list.append(0); sid_list.append(si)
    return {"eeg": np.stack(eeg_list), "hemo": np.stack(hemo_list),
            "labels": np.asarray(lab_list, np.int64),
            "subject_ids": np.asarray(sid_list, np.int64),
            "true_delays": np.zeros(len(lab_list), np.float32)}

# ═══ CELL 11 ═══  Build dataset splits (disk-cached so re-runs are fast).
# why: the real loaders (esp. ds002739) are slow; cache each split to .npz once.
def _cached_build(key, builder):
    cdir = Path(CFG.output_dir) / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"{key}_w{int(CFG.window_seconds)}_e{CFG.n_eeg_channels}_r{CFG.n_rois}.npz"
    if cpath.exists():
        d = np.load(cpath)
        return {k: d[k] for k in d.files}
    out = builder(); np.savez(cpath, **out); return out

stroop_train = _cached_build("stroop_train", lambda: generate_eeg_fnirs_stroop(CFG, "train"))
stroop_val   = _cached_build("stroop_val",   lambda: generate_eeg_fnirs_stroop(CFG, "val"))
tuberlin_ext = _cached_build("tuberlin_ext", lambda: generate_eeg_fnirs_tuberlin(CFG))
ds002336_train = _cached_build("ds002336_train", lambda: generate_eeg_fmri_ds002336(CFG, "train"))
ds002336_val   = _cached_build("ds002336_val",   lambda: generate_eeg_fmri_ds002336(CFG, "val"))
ds002739_ext   = _cached_build("ds002739_ext",   lambda: generate_eeg_fmri_ds002739(CFG))

for name, d in [("stroop_train", stroop_train), ("stroop_val", stroop_val),
                ("tuberlin_ext", tuberlin_ext), ("ds002336_train", ds002336_train),
                ("ds002336_val", ds002336_val), ("ds002739_ext", ds002739_ext)]:
    print(f"{name:18s} EEG {d['eeg'].shape}  HEMO {d['hemo'].shape}  labels {d['labels'].shape}")

# ═══ CELL 13 ═══  Preprocessing functions: bandpass / notch / z-score per modality.
# why: Cleans EEG and normalizes signals so the model sees comparable inputs.
def bandpass_filter(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, x, axis=-1).astype(np.float32)

def notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    from scipy.signal import iirnotch
    b, a = iirnotch(freq/(0.5*fs), q)
    return filtfilt(b, a, x, axis=-1).astype(np.float32)

def zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + eps
    return ((x - mu) / sd).astype(np.float32)

def preprocess_eeg(x: np.ndarray, fs: int) -> np.ndarray:
    x = bandpass_filter(x, fs, 0.5, 45.0)
    x = notch_filter(x, fs, 50.0)
    x = zscore_per_channel(x)
    return x

def preprocess_fnirs(x: np.ndarray, fs: float) -> np.ndarray:
    # bandpass for hemodynamic frequencies, exclude cardiac
    x = bandpass_filter(x, fs, 0.01, 0.2)
    x = zscore_per_channel(x)
    return x

def preprocess_fmri_roi(x: np.ndarray, fs: float) -> np.ndarray:
    # ROI time series preprocessing: bandpass when window length permits, otherwise z-score only
    min_required = 30  # filtfilt padlen for order-4 bandpass
    if x.shape[-1] > min_required:
        x = bandpass_filter(x, fs, 0.008, 0.15)
    x = zscore_per_channel(x)
    return x
# ═══ CELL 14 ═══  Apply the preprocessing to all 6 splits.
# why: Runs the cleaning once and stores the cleaned arrays.
def apply_preprocessing(dataset: dict, modality: str, cfg: Config):
    eeg_proc = np.stack([preprocess_eeg(e, cfg.eeg_fs) for e in dataset["eeg"]])
    if modality == "fnirs":
        hemo_proc = np.stack([preprocess_fnirs(h, cfg.fnirs_fs) for h in dataset["hemo"]])
    elif modality == "fmri":
        hemo_proc = np.stack([preprocess_fmri_roi(h, 1.0/cfg.fmri_tr) for h in dataset["hemo"]])
    else:
        raise ValueError(modality)
    return {**dataset, "eeg": eeg_proc, "hemo": hemo_proc}

stroop_train = apply_preprocessing(stroop_train, "fnirs", CFG)
stroop_val = apply_preprocessing(stroop_val, "fnirs", CFG)
tuberlin_ext = apply_preprocessing(tuberlin_ext, "fnirs", CFG)
ds002336_train = apply_preprocessing(ds002336_train, "fmri", CFG)
ds002336_val = apply_preprocessing(ds002336_val, "fmri", CFG)
ds002739_ext = apply_preprocessing(ds002739_ext, "fmri", CFG)
print("Preprocessing complete for all six dataset splits.")
# ═══ CELL 15 ═══  Plot one sample (EEG / fNIRS / fMRI).
# why: Visual sanity check that the signals look reasonable.
# Inspect a sample
fig, axes = plt.subplots(3, 1, figsize=(12, 7))
axes[0].plot(stroop_train["eeg"][0, :5].T, alpha=0.7)
axes[0].set_title("EEG (5 channels) — preprocessed Stroop sample")
axes[0].set_xlabel("samples")
axes[1].plot(stroop_train["hemo"][0, :4].T, alpha=0.7)
axes[1].set_title("fNIRS HbO/HbR (4 channels) — preprocessed Stroop sample")
axes[1].set_xlabel("samples")
axes[2].plot(ds002336_train["hemo"][0, :4].T, alpha=0.7)
axes[2].set_title("fMRI ROI BOLD (4 ROIs) — preprocessed ds002336 sample")
axes[2].set_xlabel("samples")
plt.tight_layout()
plt.show()
# ═══ CELL 17 ═══  PairedHemoDataset — wraps arrays as a PyTorch Dataset.
# why: Lets the DataLoader fetch one (eeg, hemo, label) example at a time.
class PairedHemoDataset(Dataset):
    """Holds paired EEG and hemodynamic windows for a single dataset split."""
    def __init__(self, data: dict, modality: str):
        assert modality in ("fnirs", "fmri")
        self.eeg = torch.from_numpy(data["eeg"]).float()
        self.hemo = torch.from_numpy(data["hemo"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()
        self.subject_ids = torch.from_numpy(data["subject_ids"]).long()
        self.true_delays = torch.from_numpy(data["true_delays"]).float()
        self.modality = modality

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],
            "hemo": self.hemo[idx],
            "label": self.labels[idx],
            "subject_id": self.subject_ids[idx],
            "true_delay": self.true_delays[idx],
            "modality": self.modality,
        }
# ═══ CELL 18 ═══  collate function + DataLoaders (batching).
# why: Groups examples into batches the model trains on.
def collate_modality(batch: List[dict]) -> dict:
    eeg = torch.stack([b["eeg"] for b in batch])
    hemo = torch.stack([b["hemo"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    subj = torch.stack([b["subject_id"] for b in batch])
    delays = torch.stack([b["true_delay"] for b in batch])
    modality = batch[0]["modality"]
    return {"eeg": eeg, "hemo": hemo, "label": labels,
            "subject_id": subj, "true_delay": delays, "modality": modality}

ds_stroop_tr = PairedHemoDataset(stroop_train, "fnirs")
ds_stroop_va = PairedHemoDataset(stroop_val, "fnirs")
ds_tuberlin = PairedHemoDataset(tuberlin_ext, "fnirs")
ds_fmri_tr = PairedHemoDataset(ds002336_train, "fmri")
ds_fmri_va = PairedHemoDataset(ds002336_val, "fmri")
ds_fmri_ext = PairedHemoDataset(ds002739_ext, "fmri")

dl_stroop_tr = DataLoader(ds_stroop_tr, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_modality)
dl_stroop_va = DataLoader(ds_stroop_va, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_modality)
dl_tuberlin = DataLoader(ds_tuberlin, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_modality)
dl_fmri_tr = DataLoader(ds_fmri_tr, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_modality)
dl_fmri_va = DataLoader(ds_fmri_va, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_modality)
dl_fmri_ext = DataLoader(ds_fmri_ext, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_modality)

print(f"fNIRS train batches: {len(dl_stroop_tr)}  fMRI train batches: {len(dl_fmri_tr)}")
# ═══ CELL 20 ═══  Shared EEG encoder (EEGNet front-end + TCN).
# why: The backbone that turns EEG into a latent vector; shared by both modalities.
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class EEGNetFrontend(nn.Module):
    """EEGNet-style spatial-temporal feature extractor."""
    def __init__(self, n_channels, n_samples, F1=16, D=2, F2=None, kern_len=64, pool=4, dropout=0.25):
        super().__init__()
        F2 = F2 or F1 * D
        self.temporal = nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, pool))
        self.drop1 = nn.Dropout(dropout)
        self.sep_dw = nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False)
        self.sep_pw = nn.Conv2d(F1*D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool*2))
        self.drop2 = nn.Dropout(dropout)
        self.out_channels = F2

    def forward(self, x):
        # x: [B, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.bn1(self.temporal(x))
        x = self.bn2(self.spatial(x))
        x = F.elu(x)
        x = self.drop1(self.pool1(x))
        x = self.sep_dw(x)
        x = self.bn3(self.sep_pw(x))
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        # [B, F2, 1, T'] -> [B, F2, T']
        return x.squeeze(2)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        h = F.elu(self.bn1(self.conv1(x)[..., :-self.pad]))
        h = self.drop(h)
        h = F.elu(self.bn2(self.conv2(h)[..., :-self.pad]))
        return F.elu(h + self.proj(x))


class SharedEncoder(nn.Module):
    def __init__(self, cfg: Config, eeg_window: int):
        super().__init__()
        self.frontend = EEGNetFrontend(
            n_channels=cfg.n_eeg_channels,
            n_samples=eeg_window,
            F1=cfg.encoder_temporal_filters,
            D=cfg.encoder_spatial_filters,
            kern_len=cfg.encoder_kernel_length,
            pool=cfg.encoder_pool,
            dropout=getattr(cfg, "dropout", 0.25),
        )
        layers = []
        in_ch = self.frontend.out_channels
        for i in range(cfg.tcn_layers):
            layers.append(TCNBlock(in_ch, cfg.tcn_channels, cfg.tcn_kernel, dilation=2**i,
                                   dropout=getattr(cfg, "dropout", 0.2)))
            in_ch = cfg.tcn_channels
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Conv1d(cfg.tcn_channels, cfg.latent_dim, 1)

    def forward(self, eeg):
        h = self.frontend(eeg)         # [B, F2, T']
        h = self.tcn(h)                # [B, tcn_channels, T']
        z = self.proj(h)               # [B, latent_dim, T']
        return z
# ═══ CELL 22 ═══  Hemodynamic heads (predict mean + uncertainty).
# why: Turn the latent into HbO/HbR (fNIRS) or ROI-BOLD (fMRI) predictions.
class HemodynamicHead(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, target_length: int, hidden: int = 64):
        super().__init__()
        self.target_length = target_length
        self.body = nn.Sequential(
            nn.Conv1d(latent_dim, hidden, 1),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ELU(),
        )
        self.mean_head = nn.Conv1d(hidden, out_channels, 1)
        self.logvar_head = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, z):
        # resample latent to target temporal length
        z_resampled = F.interpolate(z, size=self.target_length, mode="linear", align_corners=False)
        h = self.body(z_resampled)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(min=-7.0, max=4.0)
        return mean, logvar
# ═══ CELL 24 ═══  DelayModule — learns a physiological 2-8 s lag.
# why: Blood-flow lags electrical activity; this shifts predictions to match.
class DelayModule(nn.Module):
    def __init__(self, latent_dim: int, fs: float, delay_min_s: float, delay_max_s: float):
        super().__init__()
        self.fs = fs
        self.delay_min = delay_min_s
        self.delay_max = delay_max_s
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        # estimate delay in seconds, bounded to physiological window
        s = torch.sigmoid(self.estimator(z))
        delay_s = self.delay_min + (self.delay_max - self.delay_min) * s
        return delay_s.squeeze(-1)  # [B]

    @staticmethod
    def apply_shift(signal: torch.Tensor, delay_s: torch.Tensor, fs: float) -> torch.Tensor:
        """Apply per-sample temporal shift to a [B, C, T] signal using linear interpolation."""
        B, C, T = signal.shape
        device = signal.device
        # build a grid sampled in time, shifted by delay_s * fs samples
        base = torch.arange(T, device=device, dtype=signal.dtype).unsqueeze(0).expand(B, -1)
        shift_samples = delay_s * fs
        shifted_idx = base - shift_samples.unsqueeze(-1)
        # normalize to [-1, 1] for grid_sample-like indexing using manual interp
        shifted_idx = shifted_idx.clamp(0, T - 1)
        left = shifted_idx.floor().long()
        right = (left + 1).clamp(max=T - 1)
        frac = (shifted_idx - left.float()).unsqueeze(1)  # [B, 1, T]

        left_idx = left.unsqueeze(1).expand(-1, C, -1)
        right_idx = right.unsqueeze(1).expand(-1, C, -1)
        out = (1 - frac) * torch.gather(signal, 2, left_idx) + frac * torch.gather(signal, 2, right_idx)
        return out
# ═══ CELL 26 ═══  DomainConfidence — input-distance reliability signal.
# why: Flags inputs far from training data (part of the confidence score).
class DomainConfidence(nn.Module):
    """Tracks training-set latent statistics and produces a confidence score per input."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(latent_dim))
        self.register_buffer("running_var", torch.ones(latent_dim))
        self.register_buffer("count", torch.zeros(1))

    @torch.no_grad()
    def update(self, z_pooled: torch.Tensor):
        # z_pooled: [B, latent_dim]
        batch_mean = z_pooled.mean(dim=0)
        batch_var = z_pooled.var(dim=0, unbiased=False)
        n = self.count.item()
        m = z_pooled.shape[0]
        new_count = n + m
        delta = batch_mean - self.running_mean
        self.running_mean += delta * (m / max(new_count, 1))
        self.running_var = (n * self.running_var + m * batch_var + (n * m / max(new_count, 1)) * delta**2) / max(new_count, 1)
        self.count += m

    def forward(self, z_pooled: torch.Tensor) -> torch.Tensor:
        # negative log Mahalanobis-style distance, higher = more in-distribution
        diff = z_pooled - self.running_mean
        d2 = (diff**2 / (self.running_var + 1e-6)).sum(dim=-1)
        # convert to bounded confidence
        return torch.exp(-d2 / z_pooled.shape[-1])
# ═══ CELL 28 ═══  NeuroVascNet — assembles encoder + heads + delay + reliability.
# why: The full model object used for training and evaluation.
class NeuroVascNet(nn.Module):
    def __init__(self, cfg: Config, eeg_window: int, fnirs_len: int, fmri_len: int,
                 fnirs_channels: int, fmri_channels: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedEncoder(cfg, eeg_window)
        self.fnirs_head = HemodynamicHead(cfg.latent_dim, fnirs_channels, fnirs_len)
        self.fmri_head = HemodynamicHead(cfg.latent_dim, fmri_channels, fmri_len)
        self.delay_fnirs = DelayModule(cfg.latent_dim, cfg.fnirs_fs, cfg.delay_min_s, cfg.delay_max_s)
        self.delay_fmri = DelayModule(cfg.latent_dim, 1.0/cfg.fmri_tr, cfg.delay_min_s, cfg.delay_max_s)
        self.domain_conf = DomainConfidence(cfg.latent_dim)

    def encode(self, eeg):
        return self.encoder(eeg)

    def latent_pool(self, z):
        return z.mean(dim=-1)  # [B, latent_dim]

    def forward(self, eeg, modality: str):
        z = self.encoder(eeg)
        z_pool = self.latent_pool(z)
        if modality == "fnirs":
            mean, logvar = self.fnirs_head(z)
            delay = self.delay_fnirs(z)
            mean_shifted = DelayModule.apply_shift(mean, delay, self.cfg.fnirs_fs)
        elif modality == "fmri":
            mean, logvar = self.fmri_head(z)
            delay = self.delay_fmri(z)
            mean_shifted = DelayModule.apply_shift(mean, delay, 1.0/self.cfg.fmri_tr)
        else:
            raise ValueError(modality)
        return {
            "z": z, "z_pool": z_pool,
            "pred_mean": mean_shifted, "pred_raw": mean,
            "pred_logvar": logvar, "delay": delay,
        }
# ═══ CELL 30 ═══  Loss functions: heteroscedastic (value+uncertainty) + smoothness.
# why: Define how the model is graded; the uncertainty term is the one that needs care.
def heteroscedastic_loss(pred_mean, target, pred_logvar):
    # Gaussian NLL with predicted log variance
    precision = torch.exp(-pred_logvar)
    return (precision * (pred_mean - target)**2 + pred_logvar).mean()

def cross_correlation_loss(pred, target, max_lag: int = 5):
    """1 - peak normalized cross-correlation within a small lag window."""
    B, C, T = pred.shape
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    targ_c = target - target.mean(dim=-1, keepdim=True)
    pred_n = pred_c / (pred_c.norm(dim=-1, keepdim=True) + 1e-6)
    targ_n = targ_c / (targ_c.norm(dim=-1, keepdim=True) + 1e-6)
    best = torch.full((B, C), -1.0, device=pred.device)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            p, t = pred_n[..., :lag], targ_n[..., -lag:]
        elif lag > 0:
            p, t = pred_n[..., lag:], targ_n[..., :-lag]
        else:
            p, t = pred_n, targ_n
        # avoid degenerate empty slice
        if p.shape[-1] < 4:
            continue
        # renormalize after slicing
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
        corr = (p * t).sum(dim=-1)
        best = torch.maximum(best, corr)
    return (1.0 - best).mean()

def smoothness_loss(pred):
    diff = pred[..., 1:] - pred[..., :-1]
    return (diff**2).mean()

def moment_matching_loss(z_a: torch.Tensor, z_b: torch.Tensor):
    """Match first and second moments of two latent distributions."""
    mean_diff = (z_a.mean(dim=0) - z_b.mean(dim=0)).pow(2).mean()
    cov_a = torch.cov(z_a.T)
    cov_b = torch.cov(z_b.T)
    cov_diff = (cov_a - cov_b).pow(2).mean()
    return mean_diff + cov_diff
# ═══ CELL 31 ═══  compute_modality_loss — combine the loss terms for one modality.
# why: Used each training step to produce a single number to minimize.
def compute_modality_loss(out, batch, cfg: Config, modality: str):
    pred_mean = out["pred_mean"]
    pred_logvar = out["pred_logvar"]
    target = batch["hemo"]

    l_mse = F.mse_loss(pred_mean, target)
    l_hetero = heteroscedastic_loss(pred_mean, target, pred_logvar)
    l_xcorr = cross_correlation_loss(pred_mean, target, max_lag=3)
    l_smooth = smoothness_loss(pred_mean)

    if modality == "fnirs":
        w_main, w_delay = cfg.w_hb, cfg.w_delay_fnirs
    else:
        w_main, w_delay = cfg.w_bold, cfg.w_delay_fmri

    loss = (w_main * l_mse +
            cfg.w_hetero * l_hetero +
            w_delay * l_xcorr +
            cfg.w_smooth * l_smooth)

    return loss, {
        "mse": l_mse.item(),
        "hetero": l_hetero.item(),
        "xcorr": l_xcorr.item(),
        "smooth": l_smooth.item(),
        "delay_pred_mean_s": out["delay"].mean().item(),
    }
# ═══ CELL 33 ═══  build_model + iter_balanced (alternate two loaders).
# why: Feeds one fNIRS and one fMRI batch in turn so the encoder learns from both.
def build_model(cfg: Config) -> NeuroVascNet:
    return NeuroVascNet(
        cfg=cfg,
        eeg_window=EEG_WINDOW,
        fnirs_len=FNIRS_WINDOW,
        fmri_len=FMRI_WINDOW,
        fnirs_channels=cfg.n_fnirs_channels * cfg.fnirs_hb_types,
        fmri_channels=cfg.n_rois,
    ).to(DEVICE)

def iter_balanced(loader_a, loader_b):
    """Yield alternating batches from two loaders, restarting the shorter one."""
    it_a, it_b = iter(loader_a), iter(loader_b)
    while True:
        try:
            batch_a = next(it_a)
        except StopIteration:
            return
        try:
            batch_b = next(it_b)
        except StopIteration:
            it_b = iter(loader_b)
            batch_b = next(it_b)
        yield batch_a, batch_b
# ═══ CELL 34 ═══  move_batch + stage1_train — Stage 1 joint supervised training.
# why: Main training: teaches encoder + both heads from real data.
def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def stage1_train(model, dl_fnirs, dl_fmri, cfg, epochs, log_every=5):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    for ep in range(epochs):
        model.train()
        running = {"fnirs_mse": 0.0, "fmri_mse": 0.0, "fnirs_xcorr": 0.0, "fmri_xcorr": 0.0,
                   "fnirs_n": 0, "fmri_n": 0, "n": 0}
        for step, (b_a, b_b) in enumerate(iter_balanced(dl_fnirs, dl_fmri)):
            b_a = move_batch(b_a, DEVICE)
            b_b = move_batch(b_b, DEVICE)
            mod_a, mod_b = b_a["modality"], b_b["modality"]

            out_a = model(b_a["eeg"], mod_a)
            out_b = model(b_b["eeg"], mod_b)
            loss_a, log_a = compute_modality_loss(out_a, b_a, cfg, mod_a)
            loss_b, log_b = compute_modality_loss(out_b, b_b, cfg, mod_b)
            loss = loss_a + loss_b

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            model.domain_conf.update(out_a["z_pool"].detach())
            model.domain_conf.update(out_b["z_pool"].detach())

            for mod, log in [(mod_a, log_a), (mod_b, log_b)]:
                running[f"{mod}_mse"] += log["mse"]
                running[f"{mod}_xcorr"] += log["xcorr"]
                running[f"{mod}_n"] += 1
            running["n"] += 1

        def _safe(k, n):
            return running[k] / n if n > 0 else 0.0
        msg = (f"[Stage1 ep {ep+1}/{epochs}] "
               f"fNIRS mse={_safe('fnirs_mse', running['fnirs_n']):.4f} "
               f"xcorr={_safe('fnirs_xcorr', running['fnirs_n']):.4f} | "
               f"fMRI mse={_safe('fmri_mse', running['fmri_n']):.4f} "
               f"xcorr={_safe('fmri_xcorr', running['fmri_n']):.4f}")
        print(msg)
        history.append(dict(running))
    return history
# ═══ CELL 35 ═══  stage2_align — Stage 2 cross-modal latent alignment.
# why: Nudges the fNIRS and fMRI latents to live in a shared space.
def stage2_align(model, dl_fnirs, dl_fmri, cfg, epochs):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)
    history = []
    for ep in range(epochs):
        model.train()
        running = {"task": 0.0, "align": 0.0, "n": 0}
        for b_a, b_b in iter_balanced(dl_fnirs, dl_fmri):
            b_a = move_batch(b_a, DEVICE)
            b_b = move_batch(b_b, DEVICE)
            mod_a, mod_b = b_a["modality"], b_b["modality"]
            out_a = model(b_a["eeg"], mod_a)
            out_b = model(b_b["eeg"], mod_b)
            loss_a, _ = compute_modality_loss(out_a, b_a, cfg, mod_a)
            loss_b, _ = compute_modality_loss(out_b, b_b, cfg, mod_b)
            l_align = moment_matching_loss(out_a["z_pool"], out_b["z_pool"])
            loss = loss_a + loss_b + cfg.w_align * l_align

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running["task"] += (loss_a.item() + loss_b.item())
            running["align"] += l_align.item()
            running["n"] += 1
        n = max(running["n"], 1)
        print(f"[Stage2 ep {ep+1}/{epochs}] task={running['task']/n:.4f}  align={running['align']/n:.4f}")
        history.append({k: v/n if k != "n" else v for k, v in running.items()})
    return history
# ═══ CELL 36 ═══  RUN training (Stage 1 then Stage 2).  «slow»
# why: This is where the model actually learns; prints loss per epoch.
model = build_model(CFG)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
hist1 = stage1_train(model, dl_stroop_tr, dl_fmri_tr, CFG, CFG.epochs_stage1)
hist2 = stage2_align(model, dl_stroop_tr, dl_fmri_tr, CFG, CFG.epochs_stage2)
# ═══ CELL 38 ═══  evaluate_modality — compute metrics on a loader.
# why: Measures MAE / Pearson / delay on held-out data.
@torch.no_grad()
def evaluate_modality(model, loader, modality: str, max_lag: int = 5):
    model.eval()
    all_pred, all_true, all_logvar, all_delay = [], [], [], []
    all_z = []
    for batch in loader:
        batch = move_batch(batch, DEVICE)
        out = model(batch["eeg"], modality)
        all_pred.append(out["pred_mean"].cpu().numpy())
        all_true.append(batch["hemo"].cpu().numpy())
        all_logvar.append(out["pred_logvar"].cpu().numpy())
        all_delay.append(out["delay"].cpu().numpy())
        all_z.append(out["z_pool"].cpu().numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    logvar = np.concatenate(all_logvar)
    delay = np.concatenate(all_delay)
    z = np.concatenate(all_z)

    mae = mean_absolute_error(true.flatten(), pred.flatten())
    rmse = np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))
    # per-sample per-channel Pearson correlation
    corrs = []
    for b in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            try:
                r, _ = pearsonr(pred[b, c], true[b, c])
                if not np.isnan(r):
                    corrs.append(r)
            except Exception:
                continue
    pearson = float(np.mean(corrs)) if corrs else 0.0

    # peak xcorr metric
    peak_xcorr_vals = []
    for b in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            p = pred[b, c] - pred[b, c].mean()
            t = true[b, c] - true[b, c].mean()
            denom = (np.linalg.norm(p) * np.linalg.norm(t)) + 1e-9
            xc = np.correlate(p, t, mode="full") / denom
            mid = len(xc) // 2
            window = xc[mid-max_lag:mid+max_lag+1]
            peak_xcorr_vals.append(window.max())
    peak_xcorr = float(np.mean(peak_xcorr_vals))

    return {
        "mae": mae, "rmse": rmse, "pearson": pearson, "peak_xcorr": peak_xcorr,
        "pred": pred, "true": true, "logvar": logvar, "delay": delay, "z": z,
    }
# ═══ CELL 39 ═══  Run evaluation on internal + external sets.
# why: Produces the key validation numbers (Pearson is the one that matters).
res_stroop_val = evaluate_modality(model, dl_stroop_va, "fnirs")
res_tuberlin = evaluate_modality(model, dl_tuberlin, "fnirs")
res_fmri_val = evaluate_modality(model, dl_fmri_va, "fmri")
res_fmri_ext = evaluate_modality(model, dl_fmri_ext, "fmri")

print("\n=== Internal validation ===")
print(f"Stroop fNIRS:  MAE {res_stroop_val['mae']:.4f}  Pearson {res_stroop_val['pearson']:.4f}  peak xcorr {res_stroop_val['peak_xcorr']:.4f}")
print(f"ds002336 fMRI: MAE {res_fmri_val['mae']:.4f}  Pearson {res_fmri_val['pearson']:.4f}  peak xcorr {res_fmri_val['peak_xcorr']:.4f}")
print("\n=== External validation ===")
print(f"TU Berlin fNIRS:  MAE {res_tuberlin['mae']:.4f}  Pearson {res_tuberlin['pearson']:.4f}  peak xcorr {res_tuberlin['peak_xcorr']:.4f}")
print(f"ds002739 fMRI:    MAE {res_fmri_ext['mae']:.4f}  Pearson {res_fmri_ext['pearson']:.4f}  peak xcorr {res_fmri_ext['peak_xcorr']:.4f}")
# ═══ CELL 40 ═══  Plot predicted vs measured blood-flow.
# why: Visual check of prediction quality.
# Predicted vs measured hemodynamic traces
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, res, title in [
    (axes[0, 0], res_stroop_val, "Stroop fNIRS (internal val)"),
    (axes[0, 1], res_tuberlin, "TU Berlin fNIRS (external)"),
    (axes[1, 0], res_fmri_val, "ds002336 fMRI (internal val)"),
    (axes[1, 1], res_fmri_ext, "ds002739 fMRI (external)"),
]:
    ax.plot(res["true"][0, 0], label="measured", alpha=0.8)
    ax.plot(res["pred"][0, 0], label="predicted", alpha=0.8)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
# ═══ CELL 41 ═══  Plot recovered delay distributions.
# why: Shows the per-sample neurovascular lag the model learned.
# Delay distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].hist(res_stroop_val["delay"], bins=20, alpha=0.7, label="Stroop val")
axes[0].hist(res_tuberlin["delay"], bins=20, alpha=0.7, label="TU Berlin ext")
axes[0].set_title("Recovered fNIRS delay (s)")
axes[0].set_xlabel("seconds"); axes[0].legend()
axes[1].hist(res_fmri_val["delay"], bins=20, alpha=0.7, label="ds002336 val")
axes[1].hist(res_fmri_ext["delay"], bins=20, alpha=0.7, label="ds002739 ext")
axes[1].set_title("Recovered fMRI delay (s)")
axes[1].set_xlabel("seconds"); axes[1].legend()
plt.tight_layout()
plt.show()
# ═══ HbO/HbR PREDICTION — RELATIVE EVALUATION (blueprint §4.3, §8.1) ═══
# why: the headline fNIRS claim is RELATIVE — cross-modal supervision should
#      improve HbO/HbR prediction OVER single-modality & independent-encoder
#      baselines, judged by MAE/RMSE/Pearson/peak-xcorr AND task-condition
#      decoding from the PREDICTED hemodynamics (not absolute Pearson alone).
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def _train_fnirs_only(cfg, epochs=8):
    """Single-modality baseline: encoder + fNIRS head trained ONLY on fNIRS."""
    m = build_model(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(epochs):
        m.train()
        for b in dl_stroop_tr:
            b = move_batch(b, DEVICE)
            out = m(b["eeg"], "fnirs")
            loss, _ = compute_modality_loss(out, b, cfg, "fnirs")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

@torch.no_grad()
def _collect_fnirs(m, loader):
    m.eval(); P, T, L = [], [], []
    for b in loader:
        bb = move_batch(b, DEVICE)
        P.append(m(bb["eeg"], "fnirs")["pred_mean"].cpu().numpy())
        T.append(bb["hemo"].cpu().numpy()); L.append(b["label"].numpy())
    return np.concatenate(P), np.concatenate(T), np.concatenate(L)

def _peak_xcorr(pred, true, max_lag=5):
    best = []
    for i in range(len(pred)):
        a, b = pred[i], true[i]; bb = -1.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:   x, y = a[..., :lag], b[..., -lag:]
            elif lag > 0: x, y = a[..., lag:], b[..., :-lag]
            else:         x, y = a, b
            if x.shape[-1] < 4: continue
            xc = x - x.mean(-1, keepdims=True); yc = y - y.mean(-1, keepdims=True)
            c = (xc * yc).sum(-1) / ((np.linalg.norm(xc, axis=-1) * np.linalg.norm(yc, axis=-1)) + 1e-9)
            bb = max(bb, float(c.mean()))
        best.append(bb)
    return np.mean(best)

def _hbo_metrics(pred, true, labels):
    mae = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))
    pr = float(np.mean([np.corrcoef(pred[i].ravel(), true[i].ravel())[0, 1] for i in range(len(pred))]))
    px = _peak_xcorr(pred, true)
    feat = pred.reshape(len(pred), -1)
    dec = (cross_val_score(LogisticRegression(max_iter=1000), feat, labels, cv=3).mean()
           if len(np.unique(labels)) > 1 else np.nan)
    return dict(MAE=mae, RMSE=rmse, Pearson=pr, peak_xcorr=px, decode_from_pred=float(dec))

print("Training single-modality (fNIRS-only) baseline for the relative comparison...")
m_fnirs_only = _train_fnirs_only(CFG, epochs=CFG.epochs_stage1)

print("Evaluating HbO/HbR prediction on internal fNIRS validation (Stroop)...")
fnirs_eval = {}
for name, m in [("Full (cross-modal)", model), ("fNIRS-only baseline", m_fnirs_only)]:
    P, T, L = _collect_fnirs(m, dl_stroop_va)
    fnirs_eval[name] = _hbo_metrics(P, T, L)

print("\n=== HbO/HbR prediction — relative comparison (internal validation) ===")
print(f"{'model':22s} {'MAE':>7} {'RMSE':>7} {'Pearson':>8} {'peakXC':>7} {'decode':>7}")
for name, r in fnirs_eval.items():
    print(f"{name:22s} {r['MAE']:7.3f} {r['RMSE']:7.3f} {r['Pearson']:8.3f} "
          f"{r['peak_xcorr']:7.3f} {r['decode_from_pred']:7.3f}")
print(f"(chance decode ≈ {1.0/len(np.unique(L)):.2f}; claim holds if Full > baseline on "
      f"peak-xcorr / decode-from-pred)")
# ═══ CELL 43 ═══  mc_dropout_predictions — uncertainty via dropout passes.
# why: Estimates how much the model disagrees with itself (epistemic uncertainty).
@torch.no_grad()
def mc_dropout_predictions(model, loader, modality, n_passes=10):
    """Run forward passes with dropout active and return per-sample disagreement."""
    # enable dropout while keeping batchnorm in eval mode
    def enable_dropout(m):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()
    model.eval()
    enable_dropout(model)

    preds = []
    for _ in range(n_passes):
        batch_preds = []
        for batch in loader:
            batch = move_batch(batch, DEVICE)
            out = model(batch["eeg"], modality)
            batch_preds.append(out["pred_mean"].cpu().numpy())
        preds.append(np.concatenate(batch_preds))
    preds = np.stack(preds)  # [P, N, C, T]
    disagreement = preds.std(axis=0).mean(axis=(1, 2))  # per-sample
    model.eval()
    return disagreement
# ═══ CELL 44 ═══  compute_reliability + LEARNED gating (blueprint §6.5, §10.2)
# why: fuse aleatoric + epistemic + domain uncertainties via a LEARNED gate
#      (logistic regression fit on internal validation) — not hardcoded weights.
from sklearn.preprocessing import StandardScaler

@torch.no_grad()
def compute_reliability(model, loader, modality, n_passes=10):
    model.eval()
    pred_logvars, z_pools, pred_means, targets = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, DEVICE)
        out = model(batch["eeg"], modality)
        pred_logvars.append(out["pred_logvar"].cpu().numpy())
        z_pools.append(out["z_pool"]); pred_means.append(out["pred_mean"].cpu().numpy())
        targets.append(batch["hemo"].cpu().numpy())
    pred_logvar = np.concatenate(pred_logvars); z_pool_t = torch.cat(z_pools, dim=0)
    pred_mean = np.concatenate(pred_means); target = np.concatenate(targets)
    aleatoric = np.exp(pred_logvar).mean(axis=(1, 2))
    epistemic = mc_dropout_predictions(model, loader, modality, n_passes=n_passes)
    domain_conf = model.domain_conf(z_pool_t).cpu().numpy()
    err = np.abs(pred_mean - target).mean(axis=(1, 2))
    return {"aleatoric": aleatoric, "epistemic": epistemic, "domain_conf": domain_conf,
            "error": err, "pred": pred_mean, "true": target}

def fit_reliability_gate(rel):
    """LEARNED gating (blueprint §6.5/§10.2): logistic regression over the three
    uncertainty signals, trained on the internal-validation split."""
    X = np.stack([rel["aleatoric"], rel["epistemic"], rel["domain_conf"]], axis=1)
    y = (rel["error"] <= np.median(rel["error"])).astype(int)
    sc = StandardScaler().fit(X)
    gate = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)
    return (sc, gate)

def reliability_score(gate, rel):
    """Apply the learned gate -> a single scalar reliability per prediction in [0,1]."""
    sc, clf = gate
    X = np.stack([rel["aleatoric"], rel["epistemic"], rel["domain_conf"]], axis=1)
    return clf.predict_proba(sc.transform(X))[:, 1]
# ═══ CELL 45 ═══  Learned gate -> TEMPERATURE scaling -> isotonic (blueprint §10.3)
# why: calibrate the learned reliability score — temperature scaling FIRST, then
#      isotonic refinement; fit once on internal validation and frozen.
from scipy.optimize import minimize_scalar
rel_internal = compute_reliability(model, dl_stroop_va, "fnirs", n_passes=CFG.mc_dropout_passes)
rel_external = compute_reliability(model, dl_tuberlin, "fnirs", n_passes=CFG.mc_dropout_passes)

rel_gate = fit_reliability_gate(rel_internal)                 # LEARNED gating (internal only)
score_int = reliability_score(rel_gate, rel_internal)
score_ext = reliability_score(rel_gate, rel_external)
err_threshold = np.median(rel_internal["error"])
y_internal = (rel_internal["error"] <= err_threshold).astype(float)
y_external = (rel_external["error"] <= err_threshold).astype(float)

def fit_temperature(scores, y):
    eps = 1e-6; s = np.clip(scores, eps, 1 - eps); logit = np.log(s / (1 - s))
    def nll(T):
        p = np.clip(1.0 / (1.0 + np.exp(-logit / T)), eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)

def apply_temperature(scores, T):
    eps = 1e-6; s = np.clip(scores, eps, 1 - eps); logit = np.log(s / (1 - s))
    return 1.0 / (1.0 + np.exp(-logit / T))

RELIABILITY_TEMP = fit_temperature(score_int, y_internal)            # step 1: temperature scaling
ts_int = apply_temperature(score_int, RELIABILITY_TEMP)
ts_ext = apply_temperature(score_ext, RELIABILITY_TEMP)
reliability_iso = IsotonicRegression(out_of_bounds="clip").fit(ts_int, y_internal)  # step 2: isotonic
calibrated_internal = reliability_iso.predict(ts_int)
calibrated_external = reliability_iso.predict(ts_ext)
print(f"Learned-gate reliability-error corr (internal): {np.corrcoef(score_int, -rel_internal['error'])[0,1]:.4f}")
print(f"Learned-gate reliability-error corr (external): {np.corrcoef(score_ext, -rel_external['error'])[0,1]:.4f}")
print(f"Fitted temperature T = {RELIABILITY_TEMP:.3f}")
# ═══ CELL 46 ═══  Expected Calibration Error (raw vs calibrated).
# why: Measures whether the confidence numbers are trustworthy.
def expected_calibration_error(confidence, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() == 0:
            continue
        acc = correctness[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece

y_external = (rel_external["error"] <= err_threshold).astype(float)
ece_raw = expected_calibration_error(score_ext, y_external)
ece_cal = expected_calibration_error(calibrated_external, y_external)
print(f"ECE raw (external): {ece_raw:.4f}")
print(f"ECE calibrated (external): {ece_cal:.4f}")
# ═══ CELL 47 ═══  Selective-prediction curve.
# why: Shows error when the model is allowed to abstain on low-confidence inputs.
def selective_prediction_curve(reliability, error, n_steps=20):
    # for each rejection rate, compute mean error on remaining samples
    sorted_idx = np.argsort(-reliability)  # most confident first
    rates = np.linspace(0.0, 0.9, n_steps)
    curve = []
    for r in rates:
        keep_n = max(int((1 - r) * len(reliability)), 1)
        kept = sorted_idx[:keep_n]
        curve.append(error[kept].mean())
    return rates, np.array(curve)

rates_int, curve_int = selective_prediction_curve(calibrated_internal, rel_internal["error"])
rates_ext, curve_ext = selective_prediction_curve(calibrated_external, rel_external["error"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(rates_int, curve_int, marker="o", label="internal val")
axes[0].plot(rates_ext, curve_ext, marker="s", label="external")
axes[0].set_xlabel("Rejection rate")
axes[0].set_ylabel("Mean prediction error on kept samples")
axes[0].set_title("Selective prediction curves")
axes[0].legend()

axes[1].scatter(calibrated_external, rel_external["error"], alpha=0.5, s=15)
axes[1].set_xlabel("Calibrated reliability")
axes[1].set_ylabel("Per-sample error")
axes[1].set_title("Reliability vs error (external)")
plt.tight_layout()
plt.show()
# ═══ INFERENCE WITH ABSTENTION (blueprint §3.1) ═══
# why: abstention is BUILT INTO the deployment inference path — below a configurable
#      reliability threshold the model returns "insufficient confidence" (NaN)
#      instead of a numerical prediction, rather than abstaining post-hoc in eval.
@torch.no_grad()
def deploy_infer(model, eeg, modality, gate, temp, iso, threshold,
                 n_passes=CFG.mc_dropout_passes):
    """Deployment inference: predict + calibrated reliability + abstain in one call."""
    model.eval()
    out = model(eeg.to(DEVICE), modality)
    aleatoric = np.exp(out["pred_logvar"].cpu().numpy()).mean(axis=(1, 2))
    domain_conf = model.domain_conf(out["z_pool"]).cpu().numpy()
    def _enable_dropout(m):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout): mod.train()
    _enable_dropout(model)
    mc = np.stack([model(eeg.to(DEVICE), modality)["pred_mean"].cpu().numpy()
                   for _ in range(n_passes)])
    model.eval()
    epistemic = mc.std(0).mean(axis=(1, 2))
    rel = {"aleatoric": aleatoric, "epistemic": epistemic, "domain_conf": domain_conf}
    score = iso.predict(apply_temperature(reliability_score(gate, rel), temp))  # calibrated
    pred = out["pred_mean"].cpu().numpy()
    abstain = score < threshold
    pred_gated = pred.copy(); pred_gated[abstain] = np.nan      # "insufficient confidence"
    return {"prediction": pred_gated, "reliability": score, "abstained": abstain,
            "n_abstained": int(abstain.sum())}

# Configurable deployment threshold (derived from internal selective-prediction).
DEPLOY_THRESHOLD = float(np.quantile(calibrated_internal, 0.25))
_demo = deploy_infer(model, next(iter(dl_tuberlin))["eeg"], "fnirs",
                     rel_gate, RELIABILITY_TEMP, reliability_iso, DEPLOY_THRESHOLD)
print(f"Deployment inference @ threshold={DEPLOY_THRESHOLD:.3f}: "
      f"{_demo['n_abstained']}/{len(_demo['abstained'])} windows returned "
      f"'insufficient confidence'; the rest returned numerical HbO/HbR predictions.")
# ═══ CELL 49 ═══  LinearProbe + extract_features (downstream EEG-only).
# why: Tests whether the learned EEG features help a simple task.
class LinearProbe(nn.Module):
    def __init__(self, latent_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)
    def forward(self, z_pool):
        return self.fc(z_pool)

def extract_features(model, loader):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, DEVICE)
            z = model.encode(batch["eeg"])
            feats.append(model.latent_pool(z).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)

def train_probe(feats_tr, labels_tr, feats_va, labels_va, n_classes=2, epochs=30):
    probe = LinearProbe(feats_tr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Xtr = torch.from_numpy(feats_tr).float().to(DEVICE)
    ytr = torch.from_numpy(labels_tr).long().to(DEVICE)
    Xva = torch.from_numpy(feats_va).float().to(DEVICE)
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(Xtr)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(Xva).argmax(-1).cpu().numpy()
    return (pred == labels_va).mean()
# ═══ CELL 50 ═══  Run downstream transfer (hemo-aware vs scratch).
# why: Key 'deploy with EEG only' experiment: does hemodynamic training help?
# Hemodynamic-aware encoder transfer
feats_tr, labs_tr = extract_features(model, dl_stroop_tr)
feats_va, labs_va = extract_features(model, dl_stroop_va)
acc_hemo = train_probe(feats_tr, labs_tr, feats_va, labs_va)

# Scratch encoder baseline: same architecture, no hemodynamic supervision
scratch = build_model(CFG)
feats_tr_s, _ = extract_features(scratch, dl_stroop_tr)
feats_va_s, _ = extract_features(scratch, dl_stroop_va)
acc_scratch = train_probe(feats_tr_s, labs_tr, feats_va_s, labs_va)

print(f"Linear probe accuracy (Stroop): hemodynamic-aware={acc_hemo:.4f}  scratch={acc_scratch:.4f}")
# ═══ CELL 52 ═══  run_ablation — train a model variant with a part removed.
# why: Lets us prove each component (alignment/delay/etc.) contributes.
def run_ablation(name: str, cfg_overrides: dict, no_fnirs: bool = False,
                 no_fmri: bool = False, no_align: bool = False, no_delay: bool = False):
    cfg = Config(**{**asdict(CFG), **cfg_overrides})
    m = build_model(cfg)

    if no_delay:
        # freeze delay to physiological midpoint
        for p in m.delay_fnirs.parameters():
            p.requires_grad = False
        for p in m.delay_fmri.parameters():
            p.requires_grad = False

    if no_fnirs and no_fmri:
        raise ValueError("at least one modality required")

    dl_a = dl_fmri_tr if no_fnirs else dl_stroop_tr
    dl_b = dl_stroop_tr if no_fmri else dl_fmri_tr
    # if one modality disabled, both loaders point at the same dataset
    if no_fnirs:
        dl_a = dl_fmri_tr; dl_b = dl_fmri_tr
    if no_fmri:
        dl_a = dl_stroop_tr; dl_b = dl_stroop_tr

    _ = stage1_train(m, dl_a, dl_b, cfg, max(cfg.epochs_stage1 // 2, 2))
    if not no_align:
        _ = stage2_align(m, dl_stroop_tr, dl_fmri_tr, cfg, max(cfg.epochs_stage2 // 2, 1))

    r_int_f = evaluate_modality(m, dl_stroop_va, "fnirs")
    r_ext_f = evaluate_modality(m, dl_tuberlin, "fnirs")
    r_int_m = evaluate_modality(m, dl_fmri_va, "fmri")
    r_ext_m = evaluate_modality(m, dl_fmri_ext, "fmri")
    return {
        "name": name,
        "fnirs_int_pearson": r_int_f["pearson"],
        "fnirs_ext_pearson": r_ext_f["pearson"],
        "fmri_int_pearson": r_int_m["pearson"],
        "fmri_ext_pearson": r_ext_m["pearson"],
    }
# ═══ CELL 53 ═══  Run the ablation suite.  «slow»
# why: Re-trains several variants and tabulates their scores.
# Reduced epoch budget for ablations to keep runtime tractable
ablation_cfg = {"epochs_stage1": 3, "epochs_stage2": 2}

ablations = []
print("Running ablation: full model...")
ablations.append(run_ablation("full",        ablation_cfg))
print("Running ablation: no cross-modal alignment...")
ablations.append(run_ablation("no_align",    ablation_cfg, no_align=True))
print("Running ablation: no delay module...")
ablations.append(run_ablation("no_delay",    ablation_cfg, no_delay=True))
print("Running ablation: fNIRS-only encoder...")
ablations.append(run_ablation("fnirs_only",  ablation_cfg, no_fmri=True))
print("Running ablation: fMRI-only encoder...")
ablations.append(run_ablation("fmri_only",   ablation_cfg, no_fnirs=True))

import pandas as pd
abl_df = pd.DataFrame(ablations)
abl_df
# ═══ CELL 55 ═══  Build summary.json of all results.
# why: Collects every metric into one file for the paper.
summary = {
    "internal_validation": {
        "stroop_fnirs": {
            "mae": float(res_stroop_val["mae"]),
            "rmse": float(res_stroop_val["rmse"]),
            "pearson": float(res_stroop_val["pearson"]),
            "peak_xcorr": float(res_stroop_val["peak_xcorr"]),
        },
        "ds002336_fmri": {
            "mae": float(res_fmri_val["mae"]),
            "rmse": float(res_fmri_val["rmse"]),
            "pearson": float(res_fmri_val["pearson"]),
            "peak_xcorr": float(res_fmri_val["peak_xcorr"]),
        },
    },
    "external_validation": {
        "tuberlin_fnirs": {
            "mae": float(res_tuberlin["mae"]),
            "rmse": float(res_tuberlin["rmse"]),
            "pearson": float(res_tuberlin["pearson"]),
            "peak_xcorr": float(res_tuberlin["peak_xcorr"]),
        },
        "ds002739_fmri": {
            "mae": float(res_fmri_ext["mae"]),
            "rmse": float(res_fmri_ext["rmse"]),
            "pearson": float(res_fmri_ext["pearson"]),
            "peak_xcorr": float(res_fmri_ext["peak_xcorr"]),
        },
    },
    "reliability": {
        "internal_corr": float(np.corrcoef(score_int, -rel_internal["error"])[0, 1]),
        "external_corr": float(np.corrcoef(score_ext, -rel_external["error"])[0, 1]),
        "ece_raw_external": float(ece_raw),
        "ece_calibrated_external": float(ece_cal),
    },
    "downstream_transfer": {
        "hemodynamic_aware_acc": float(acc_hemo),
        "scratch_acc": float(acc_scratch),
    },
    "ablations": ablations,
}

out_path = Path(CFG.output_dir) / "summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {out_path}")
print(json.dumps(summary, indent=2))
# ═══ CELL 56 ═══  Save the trained model checkpoint (.pt).
# why: Persists the model so it can be reloaded/deployed later.
# Persist trained model
ckpt_path = Path(CFG.output_dir) / "neurovascnet.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "config": asdict(CFG),
    "isotonic_calibrator_X": iso.X_thresholds_.tolist() if hasattr(iso, "X_thresholds_") else None,
    "isotonic_calibrator_y": iso.y_thresholds_.tolist() if hasattr(iso, "y_thresholds_") else None,
    "error_threshold_for_calibration": float(err_threshold),
}, ckpt_path)
print(f"Model checkpoint saved to {ckpt_path}")