"""Consolidate every QC table into single CSVs for sharing.

Produces:
  qc_all_recordings.csv  -- one row per recording, all modalities, all metrics
  qc_all_subjects.csv    -- one row per subject, decision + reason
  qc_summary_counts.csv  -- decision counts by dataset/modality

De-duplicates on the recording identity key (the repair-and-resume cycle left
two duplicate HEFMI fNIRS rows), keeping the last write for each recording.

Usage:  python qc/export_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

SOURCES = {"eeg": "qc_eeg.csv", "fnirs": "qc_fnirs.csv", "fmri": "qc_fmri.csv"}
ID = ["dataset", "subject", "session", "task", "modality", "data_role"]

LEAD = ID + ["qc_decision", "qc_reason", "n_rule_hits", "path"]


def main():
    frames = []
    for mod, name in SOURCES.items():
        p = TABLE_DIR / name
        if not p.exists():
            print(f"  {name:16s} missing, skipped")
            continue
        df = pd.read_csv(p, low_memory=False)
        df["modality"] = mod
        print(f"  {name:16s} {len(df):4d} rows, {df.shape[1]:3d} columns")
        frames.append(df)

    if not frames:
        raise SystemExit("No QC tables found. Run qc/run_qc.py first.")

    allrec = pd.concat(frames, ignore_index=True, sort=False)

    before = len(allrec)
    allrec = allrec.drop_duplicates(subset=ID, keep="last").reset_index(drop=True)
    if before != len(allrec):
        print(f"\n  de-duplicated {before - len(allrec)} row(s) on {ID}")

    # put the identity + decision columns first, everything else after
    rest = [c for c in allrec.columns if c not in LEAD]
    allrec = allrec[[c for c in LEAD if c in allrec.columns] + rest]
    allrec = allrec.sort_values(ID).reset_index(drop=True)

    out = TABLE_DIR / "qc_all_recordings.csv"
    allrec.to_csv(out, index=False)
    print(f"\nwrote {out}\n      {len(allrec)} recordings x {allrec.shape[1]} columns")

    # subjects
    subj_src = TABLE_DIR / "subject_decisions.csv"
    if subj_src.exists():
        subj = pd.read_csv(subj_src, low_memory=False)
        out_s = TABLE_DIR / "qc_all_subjects.csv"
        subj.to_csv(out_s, index=False)
        print(f"wrote {out_s}\n      {len(subj)} subjects x {subj.shape[1]} columns")

    # summary counts
    counts = (allrec.groupby(["dataset", "modality", "data_role", "qc_decision"])
              .size().rename("n_recordings").reset_index())
    out_c = TABLE_DIR / "qc_summary_counts.csv"
    counts.to_csv(out_c, index=False)
    print(f"wrote {out_c}\n      {len(counts)} rows")

    print("\ndecision totals:")
    print(allrec.groupby(["dataset", "modality", "qc_decision"]).size().to_string())


if __name__ == "__main__":
    main()
