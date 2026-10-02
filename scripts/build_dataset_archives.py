"""Pack each dataset into ONE zip containing only the files the loaders read.

Why: Google Drive charges its download quota per FILE. Pulling TU Berlin hBCI
means 116 separate requests, and the shared folder's daily quota trips partway
through -- which is exactly what happened. One archive per dataset is one
request, so the quota is a non-issue, and it is far faster (no per-file redirect
and confirm handshake).

It is also much smaller, because most of each dataset is never opened:

    TU Berlin hBCI   5.75 GB of 7.06   (drops montage + *_artifact copies)
    HEFMI            6.91 GB of 13.17  (drops source_data/, keeps epo_data/)
    TU Berlin WL     2.61 GB of 24.67  (drops EEG_brainAmp/ and non-nback tasks)

Upload the resulting zips to Drive, share each "anyone with the link", and put
the ids in DATASETS_ZIP in the notebook.

Usage:  python scripts/build_dataset_archives.py
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

SRC = Path("dataset")
OUT = Path("dataset_archives")
OUT.mkdir(exist_ok=True)

MANIFEST = Path("tuberlin_manifest.json")      # written by build_kaggle_notebook


def tuberlin_files():
    """Exactly the 116 files load_tuberlin() opens."""
    root = SRC / "TU Berlin hBCI EEG+FNRI"
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        return root, [(root / rel, rel) for _fid, rel in man if (root / rel).exists()]
    out = []
    for grp in sorted((root / "EEG data").glob("EEG_*")):
        for sub in sorted(grp.glob("subject *")):
            d = sub / "with occular artifact"
            for nm in ("cnt.mat", "mrk.mat"):
                if (d / nm).exists():
                    out.append((d / nm, str((d / nm).relative_to(root))))
    for sub in sorted((root / "NRI data" / "NIRS_01-29").glob("subject *")):
        for nm in ("cnt.mat", "mrk.mat"):
            if (sub / nm).exists():
                out.append((sub / nm, str((sub / nm).relative_to(root))))
    return root, out


def hefmi_files():
    """Only epo_data/ -- source_data/ and code/ are never read."""
    root = SRC / "HEFMI-ICH EEG+FNRI"
    files = [f for f in (root / "epo_data").rglob("*_epo.mat")]
    return root, [(f, str(f.relative_to(root))) for f in files]


def wl_files():
    """Only the n-back task, and only EEG/ + NIRS/ (not EEG_brainAmp/)."""
    root = SRC / "TU Berlin EEG + NIRS"
    out = []
    for sub in ("EEG", "NIRS"):
        d = root / sub
        if not d.exists():
            continue
        for f in d.rglob("*nback.mat"):
            out.append((f, str(f.relative_to(root))))
    return root, out


JOBS = [
    ("tuberlin",    tuberlin_files),
    ("hefmi",       hefmi_files),
    ("tuberlin_wl", wl_files),
]


def main():
    for key, fn in JOBS:
        root, files = fn()
        if not files:
            print(f"{key}: no files found under {root} -- skipped")
            continue
        total = sum(p.stat().st_size for p, _ in files)
        zp = OUT / f"{key}.zip"
        if zp.exists() and zp.stat().st_size > total * 0.5:
            print(f"{key}: {zp} already exists ({zp.stat().st_size/1e9:.2f} GB) -- skipped")
            continue
        print(f"{key}: packing {len(files)} files, {total/1e9:.2f} GB -> {zp}")
        t0 = time.time()
        # compresslevel=1: .mat files barely compress, so trade ratio for speed.
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for i, (src, rel) in enumerate(files, 1):
                z.write(src, rel)
                if i % 25 == 0 or i == len(files):
                    print(f"    {i}/{len(files)}  ({(time.time()-t0)/60:.1f} min)",
                          flush=True)
        print(f"    done: {zp.stat().st_size/1e9:.2f} GB "
              f"in {(time.time()-t0)/60:.1f} min\n")

    print("=" * 62)
    for key, _ in JOBS:
        zp = OUT / f"{key}.zip"
        if zp.exists():
            print(f"  {zp.stat().st_size/1e9:6.2f} GB  {zp}")
    print("=" * 62)
    print("Next: upload these to Google Drive, set each to")
    print("'Anyone with the link', and put the file ids in DATASETS_ZIP.")


if __name__ == "__main__":
    main()
