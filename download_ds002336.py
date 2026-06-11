"""
OpenNeuro ds002336 — EEG + fMRI Dataset Downloader
====================================================
Dataset: Simultaneous EEG-fMRI during Motor Imagery and Neurofeedback
Source : https://openneuro.org/datasets/ds002336
Size   : ~18 GB total (10 subjects)
License: CC0 (public domain)

Files per subject:
  - EEG (.vhdr/.vmrk/.eeg)
  - fMRI BOLD NIfTI (.nii.gz)
  - Anatomical T1w (.nii.gz)
  - Events (.tsv)
  - Metadata (.json / .tsv)

How it works:
  OpenNeuro stores all data on AWS S3 (public bucket).
  No account needed. Downloads are fast worldwide.

Usage:
  pip install requests tqdm boto3
  python download_ds002336.py

  OR without boto3 (direct HTTP):
  pip install requests tqdm
  python download_ds002336.py

Configuration: edit the OPTIONS section below.
"""

import os
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: pip install tqdm  for a progress bar\n")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS  ← edit these
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_ID   = "ds002336"
VERSION      = "2.0.0"
OUTPUT_DIR   = r"G:\research_bio_medical\dataset\OpenNeuro ds002336"

# Subjects to download. Dataset has sub-01 … sub-10.
# Use ["all"] to get everything, or list specific ones e.g. ["sub-01","sub-02"]
SUBJECTS = ["all"]

# Which data types to download per subject.
# Options: "eeg", "fmri", "anat", "all"
DATA_TYPES = ["all"]

# Resume downloads if interrupted
RESUME = True

# Retry settings
MAX_RETRIES = 5
RETRY_WAIT  = 5   # seconds

# ═══════════════════════════════════════════════════════════════════════════════

# AWS S3 public base URL for OpenNeuro
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/{VERSION}"

# All subjects in this dataset
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]

# Dataset-level files (always downloaded)
DATASET_LEVEL_FILES = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
    "task-motorImagery_eeg.json",
    "task-motorImagery_bold.json",
    "task-neurofeedback_bold.json",
]


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_file(url: str, dest_path: str, label: str = "") -> bool:
    """Download a single file with resume + retry support."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    existing = os.path.getsize(dest_path) if (RESUME and os.path.exists(dest_path)) else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 416:
                # Already fully downloaded
                return True

            if resp.status_code == 404:
                # File doesn't exist in this dataset version — skip silently
                return True

            if resp.status_code not in (200, 206):
                print(f"\n  ✗ HTTP {resp.status_code}: {url}")
                return False

            total    = int(resp.headers.get("content-length", 0)) + existing
            mode     = "ab" if existing else "wb"
            name     = label or os.path.basename(dest_path)

            if HAS_TQDM:
                bar = tqdm(
                    total=total, initial=existing,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"  {name[:45]:<45}", ncols=80, leave=True
                )
            else:
                print(f"  ↓ {name}  ({sizeof_fmt(total)})")

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
            if HAS_TQDM:
                try: bar.close()
                except: pass
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
    Return list of (url, local_path) tuples for a given subject.
    Covers EEG, fMRI BOLD, anatomical, events, and sidecar files.
    """
    files = []
    want_all  = "all"  in DATA_TYPES
    want_eeg  = "eeg"  in DATA_TYPES or want_all
    want_fmri = "fmri" in DATA_TYPES or want_all
    want_anat = "anat" in DATA_TYPES or want_all

    tasks_eeg  = ["motorImagery"]
    tasks_bold = ["motorImagery", "neurofeedback"]
    runs_bold  = ["run-01", "run-02", "run-03"]   # adjust if needed
    sessions   = ["ses-01"]                         # this dataset has 1 session

    for ses in sessions:

        # ── EEG ──────────────────────────────────────────────────────────────
        if want_eeg:
            for task in tasks_eeg:
                for ext in [".vhdr", ".vmrk", ".eeg", ".json", "_channels.tsv",
                            "_electrodes.tsv", "_coordsystem.json", "_events.tsv"]:
                    if ext.startswith("_"):
                        fname = f"{subject}_{ses}_task-{task}_eeg{ext}"
                    else:
                        fname = f"{subject}_{ses}_task-{task}_eeg{ext}"
                    url  = f"{S3_BASE}/{subject}/{ses}/eeg/{fname}"
                    dest = os.path.join(OUTPUT_DIR, subject, ses, "eeg", fname)
                    files.append((url, dest))

        # ── fMRI BOLD ────────────────────────────────────────────────────────
        if want_fmri:
            for task in tasks_bold:
                for run in runs_bold:
                    for ext in [".nii.gz", ".json"]:
                        fname = f"{subject}_{ses}_task-{task}_{run}_bold{ext}"
                        url   = f"{S3_BASE}/{subject}/{ses}/func/{fname}"
                        dest  = os.path.join(OUTPUT_DIR, subject, ses, "func", fname)
                        files.append((url, dest))
                    # events
                    fname = f"{subject}_{ses}_task-{task}_{run}_events.tsv"
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

    return files


def download_dataset_level():
    """Download top-level dataset files."""
    print("\n[1] Dataset-level files …")
    for fname in DATASET_LEVEL_FILES:
        url  = f"{S3_BASE}/{fname}"
        dest = os.path.join(OUTPUT_DIR, fname)
        download_file(url, dest, fname)


def download_subjects():
    """Download all selected subjects."""
    subjects = ALL_SUBJECTS if "all" in SUBJECTS else SUBJECTS
    total    = len(subjects)

    print(f"\n[2] Subject data ({total} subjects) …")
    failed = []

    for i, subj in enumerate(subjects, 1):
        print(f"\n── Subject {subj}  [{i}/{total}] ──────────────────────────")
        files = get_subject_files(subj)
        for url, dest in files:
            ok = download_file(url, dest)
            if not ok:
                failed.append(url)

    return failed


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"OpenNeuro {DATASET_ID} v{VERSION} Downloader")
    print("EEG + fMRI Motor Imagery & Neurofeedback")
    print("=" * 60)
    print(f"Subjects  : {SUBJECTS}")
    print(f"Data types: {DATA_TYPES}")
    print(f"Save to   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"S3 base   : {S3_BASE}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    download_dataset_level()
    failed = download_subjects()

    print("\n" + "=" * 60)
    if failed:
        print(f"⚠ {len(failed)} file(s) could not be downloaded:")
        for f in failed:
            print(f"  {f}")
    else:
        print("✓ All files downloaded successfully.")
    print(f"Saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)