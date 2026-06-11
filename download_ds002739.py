"""
OpenNeuro ds002739 — EEG + fMRI Dataset Downloader
====================================================
Dataset: Simultaneous EEG-fMRI — Confidence in Perceptual Decisions
Authors: Sabina Gherman & Marios G. Philiastides
Source : https://openneuro.org/datasets/ds002739
DOI    : 10.18112/openneuro.ds002739.v1.0.0
License: CC0 (public domain)

Dataset details:
  - 24 subjects (sub-01 … sub-24)
  - 2 sessions per subject (ses-01, ses-02)
  - Task: visual perceptual decision making + confidence rating
  - EEG: preprocessed, 1000 Hz, inside and outside MR environment
  - fMRI: BOLD NIfTI + anatomical T1w
  - Behavioural: events TSV files

Note: EEG data is stored under a custom folder structure
  (EEG_data / EEG_events) — NOT the standard BIDS eeg/ folder.
  This script handles both BIDS and legacy folder structures.

Usage:
  pip install requests tqdm
  python download_ds002739.py

Edit OPTIONS below to customise what gets downloaded.
"""

import os
import sys
import time
import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: pip install tqdm  for a progress bar\n")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS  ← edit these
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_ID  = "ds002739"
VERSION     = "1.0.0"
OUTPUT_DIR  = "OpenNeuro_ds002739"

# Subjects to download (1–24). Use ["all"] or list specific e.g. ["sub-01"]
SUBJECTS = ["all"]

# Sessions per subject: this dataset has ses-01 and ses-02
SESSIONS = ["ses-01", "ses-02"]

# Which data to download per subject
# Options: "eeg", "fmri", "anat", "beh", "all"
DATA_TYPES = ["all"]

# Resume interrupted downloads
RESUME = True

MAX_RETRIES = 5
RETRY_WAIT  = 5   # seconds between retries

# ═══════════════════════════════════════════════════════════════════════════════

S3_BASE     = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/{VERSION}"
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 25)]   # sub-01 … sub-24

DATASET_LEVEL_FILES = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
    "task-perceptualDecision_bold.json",
    "task-perceptualDecision_eeg.json",
]


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_file(url: str, dest_path: str, label: str = "") -> bool:
    """Download url → dest_path with resume + retry support."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    existing = os.path.getsize(dest_path) if (RESUME and os.path.exists(dest_path)) else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 416:
                return True   # already complete

            if resp.status_code == 404:
                return True   # file not present in this dataset — skip

            if resp.status_code not in (200, 206):
                print(f"\n  ✗ HTTP {resp.status_code}: {url}")
                return False

            total = int(resp.headers.get("content-length", 0)) + existing
            mode  = "ab" if existing else "wb"
            name  = label or os.path.basename(dest_path)

            if HAS_TQDM:
                bar = tqdm(
                    total=total, initial=existing,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"  {name[:48]:<48}", ncols=80, leave=True
                )
            else:
                size_str = sizeof_fmt(total) if total else "?"
                print(f"  ↓ {name}  ({size_str})")

            downloaded = existing
            start      = time.time()

            with open(dest_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if HAS_TQDM:
                            bar.update(len(chunk))
                        else:
                            elapsed = time.time() - start or 0.001
                            speed   = (downloaded - existing) / elapsed
                            pct     = (downloaded / total * 100) if total else 0
                            print(
                                f"\r    {pct:5.1f}%  "
                                f"{sizeof_fmt(downloaded):>10}  "
                                f"{sizeof_fmt(speed):>10}/s   ",
                                end="", flush=True
                            )

            if HAS_TQDM:
                bar.close()
            else:
                print(f"\r    ✓ {sizeof_fmt(downloaded)}" + " " * 30)

            return True

        except (requests.ConnectionError, requests.Timeout) as exc:
            try:
                if HAS_TQDM: bar.close()
            except Exception:
                pass
            print(f"\n  ! Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {RETRY_WAIT}s …")
                time.sleep(RETRY_WAIT)
            else:
                print(f"  ✗ Giving up: {url}")
                return False

    return False


def get_subject_files(subject: str) -> list:
    """
    Return list of (url, local_path) tuples for a subject.
    Handles both BIDS standard folders and the legacy EEG_data/EEG_events
    folder structure used by this dataset.
    """
    files = []
    want_all  = "all"  in DATA_TYPES
    want_eeg  = "eeg"  in DATA_TYPES or want_all
    want_fmri = "fmri" in DATA_TYPES or want_all
    want_anat = "anat" in DATA_TYPES or want_all
    want_beh  = "beh"  in DATA_TYPES or want_all

    for ses in SESSIONS:

        # ── EEG (legacy structure: EEG_data + EEG_events) ───────────────────
        if want_eeg:
            for folder in ["EEG_data", "EEG_events"]:
                # MATLAB files (.mat) — this dataset stores preprocessed EEG as .mat
                fname = f"{subject}_{ses}_{folder}.mat"
                url   = f"{S3_BASE}/{subject}/{ses}/{folder}/{fname}"
                dest  = os.path.join(OUTPUT_DIR, subject, ses, folder, fname)
                files.append((url, dest))

            # Also try BIDS-style eeg/ folder (some versions have both)
            for ext in [".vhdr", ".vmrk", ".eeg", ".json",
                        "_channels.tsv", "_electrodes.tsv",
                        "_coordsystem.json", "_events.tsv"]:
                if ext.startswith("_"):
                    fname = f"{subject}_{ses}_task-perceptualDecision_eeg{ext}"
                else:
                    fname = f"{subject}_{ses}_task-perceptualDecision_eeg{ext}"
                url  = f"{S3_BASE}/{subject}/{ses}/eeg/{fname}"
                dest = os.path.join(OUTPUT_DIR, subject, ses, "eeg", fname)
                files.append((url, dest))

        # ── fMRI BOLD ────────────────────────────────────────────────────────
        if want_fmri:
            for run in ["run-01", "run-02", "run-03", "run-04"]:
                for ext in [".nii.gz", ".json"]:
                    fname = f"{subject}_{ses}_task-perceptualDecision_{run}_bold{ext}"
                    url   = f"{S3_BASE}/{subject}/{ses}/func/{fname}"
                    dest  = os.path.join(OUTPUT_DIR, subject, ses, "func", fname)
                    files.append((url, dest))
                # events
                fname = f"{subject}_{ses}_task-perceptualDecision_{run}_events.tsv"
                url   = f"{S3_BASE}/{subject}/{ses}/func/{fname}"
                dest  = os.path.join(OUTPUT_DIR, subject, ses, "func", fname)
                files.append((url, dest))

        # ── Anatomical T1w ───────────────────────────────────────────────────
        if want_anat:
            for ext in [".nii.gz", ".json"]:
                fname = f"{subject}_{ses}_T1w{ext}"
                url   = f"{S3_BASE}/{subject}/{ses}/anat/{fname}"
                dest  = os.path.join(OUTPUT_DIR, subject, ses, "anat", fname)
                files.append((url, dest))

        # ── Behavioural ──────────────────────────────────────────────────────
        if want_beh:
            for ext in [".tsv", ".json"]:
                fname = f"{subject}_{ses}_task-perceptualDecision_beh{ext}"
                url   = f"{S3_BASE}/{subject}/{ses}/beh/{fname}"
                dest  = os.path.join(OUTPUT_DIR, subject, ses, "beh", fname)
                files.append((url, dest))

    return files


def download_dataset_level():
    print("\n[1] Dataset-level files …")
    for fname in DATASET_LEVEL_FILES:
        url  = f"{S3_BASE}/{fname}"
        dest = os.path.join(OUTPUT_DIR, fname)
        download_file(url, dest, fname)


def download_subjects():
    subjects = ALL_SUBJECTS if "all" in SUBJECTS else SUBJECTS
    total    = len(subjects)
    failed   = []

    print(f"\n[2] Subject data ({total} subjects × {len(SESSIONS)} sessions) …")

    for i, subj in enumerate(subjects, 1):
        print(f"\n── {subj}  [{i}/{total}] ──────────────────────────────────")
        for url, dest in get_subject_files(subj):
            ok = download_file(url, dest)
            if not ok:
                failed.append(url)

    return failed


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"OpenNeuro {DATASET_ID} v{VERSION} Downloader")
    print("EEG + fMRI — Confidence in Perceptual Decisions")
    print("Gherman & Philiastides | 24 subjects | 2 sessions")
    print("=" * 60)
    print(f"Subjects  : {SUBJECTS}")
    print(f"Sessions  : {SESSIONS}")
    print(f"Data types: {DATA_TYPES}")
    print(f"Save to   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"S3 base   : {S3_BASE}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    download_dataset_level()
    failed = download_subjects()

    print("\n" + "=" * 60)
    if failed:
        print(f"⚠  {len(failed)} file(s) could not be downloaded:")
        for f in failed:
            print(f"   {f}")
    else:
        print("✓  All files downloaded successfully.")
    print(f"Saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)