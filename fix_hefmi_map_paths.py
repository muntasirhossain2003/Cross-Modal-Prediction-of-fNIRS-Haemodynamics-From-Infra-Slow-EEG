"""Rewrite hefmi_recording_map.csv with RELATIVE paths.

The map was generated with absolute paths pointing into dataset_v31/, a set of
directory junctions that no longer exist. Absolute paths also mean the map only
works on one machine, in one folder layout -- useless for a teammate or a
supervisor.

Relative paths (epo_data/Normals/EEG/subject1/1_epo.mat) resolve against
whatever HEFMI_ROOT the notebook finds, so the same file works everywhere.
The notebook explicitly supports this: "Paths may be absolute or relative to
HEFMI_ROOT."

Usage:  python fix_hefmi_map_paths.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

SRC = Path("hefmi_recording_map.csv")
BAK = Path("hefmi_recording_map.absolute.bak.csv")
HEFMI = Path("dataset/HEFMI-ICH EEG+FNRI")
COLS = ["recording_key", "biological_subject_id", "eeg_path", "fnirs_path"]


def to_relative(p: str) -> str:
    """Keep the path from the 'epo_data' component onward."""
    parts = str(p).replace("/", "\\").split("\\")
    for i, part in enumerate(parts):
        if part.lower() == "epo_data":
            return "/".join(parts[i:])          # forward slashes: portable
    return str(p)


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found")
    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"backed up original -> {BAK.name}")

    # always convert from the ABSOLUTE backup, so re-running is safe
    rows = list(csv.DictReader(open(BAK, encoding="utf-8")))
    missing = []
    for r in rows:
        for col in ("eeg_path", "fnirs_path"):
            r[col] = to_relative(r[col])
            if not (HEFMI / r[col]).exists():
                missing.append(r[col])

    with open(SRC, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"rewrote {len(rows)} rows with relative paths")
    print(f"unresolved under {HEFMI}: {len(missing)}")
    for m in missing[:5]:
        print("   MISSING:", m)
    print("\nsample:")
    for r in rows[:3]:
        print(f"   {r['recording_key']:26s} {r['eeg_path']}")


if __name__ == "__main__":
    main()
