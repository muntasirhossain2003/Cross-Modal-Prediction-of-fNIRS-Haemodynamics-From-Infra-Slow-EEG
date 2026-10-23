"""Salvage QC tables written by the pre-fix, schema-unstable appender.

The old `_append` wrote each row with its own columns, so lines whose key set
differed from the header ended up with a different field count and pandas
refuses to parse the file. Those lines cannot be safely renamed after the fact
-- we do not know which column each surplus/missing field belongs to.

So: keep every line that matches the header, drop the ones that do not, and
report exactly which recordings were dropped. Re-running `run_qc.py` (without
--force) then recomputes precisely those, because the resume logic keys off
what is present in the table.

The original files are backed up to *.corrupt.bak first.

Usage:  python qc/repair_tables.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import TABLE_DIR

TABLES = ["qc_eeg.csv", "qc_fnirs.csv", "qc_fmri.csv", "qc_raw_vs_corrected.csv"]
# These leading identity columns are written by every code path, so a dropped
# row can still be named even when the rest of its schema is ambiguous.
ID_COLS = 6


def repair(path: Path) -> dict:
    if not path.exists():
        return {"file": path.name, "status": "absent"}

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {"file": path.name, "status": "empty"}

    header, body = rows[0], rows[1:]
    n = len(header)
    good = [r for r in body if len(r) == n]
    bad = [r for r in body if len(r) != n]

    if not bad:
        return {"file": path.name, "status": "clean", "rows": len(good)}

    shutil.copy2(path, path.with_suffix(".corrupt.bak"))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(good)

    dropped = [" / ".join(r[:ID_COLS]) for r in bad]
    return {"file": path.name, "status": "repaired", "rows": len(good),
            "dropped": len(bad), "dropped_ids": dropped,
            "field_counts": sorted({len(r) for r in bad})}


if __name__ == "__main__":
    print(f"header-matching repair over {TABLE_DIR}\n")
    total_dropped = 0
    for name in TABLES:
        res = repair(TABLE_DIR / name)
        st = res["status"]
        if st == "absent":
            print(f"{name:28s} -- not present")
        elif st == "clean":
            print(f"{name:28s} OK        {res['rows']} rows, no schema drift")
        elif st == "empty":
            print(f"{name:28s} -- empty")
        else:
            total_dropped += res["dropped"]
            print(f"{name:28s} REPAIRED  kept {res['rows']}, dropped "
                  f"{res['dropped']} (field counts {res['field_counts']} "
                  f"vs header)")
            for d in res["dropped_ids"][:40]:
                print(f"      dropped: {d}")
            if len(res["dropped_ids"]) > 40:
                print(f"      ... and {len(res['dropped_ids']) - 40} more")
            print(f"      backup: {name}.corrupt.bak")

    print(f"\n{total_dropped} rows dropped in total.")
    if total_dropped:
        print("Re-run to recompute exactly those (no --force, so nothing else "
              "is redone):\n\n    python qc/run_qc.py\n")
