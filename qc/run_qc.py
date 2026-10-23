"""Run the full QC pass over every recording in the manifest.

Resumable: each modality writes rows incrementally, and completed rows are
skipped on re-run unless --force is given. Failures are recorded as rows with
qc_decision=ERROR rather than aborting the run, so one unreadable file cannot
hide the status of the other 609.

Usage
-----
  python qc/run_qc.py                       # everything
  python qc/run_qc.py --dataset HEFMI       # one dataset
  python qc/run_qc.py --modality eeg        # one modality
  python qc/run_qc.py --limit 5             # smoke test
  python qc/run_qc.py --no-figs             # metrics only (faster)
  python qc/run_qc.py --force               # ignore existing rows
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc import readers
from qc.build_manifest import build as build_manifest
from qc.config import MANIFEST_CSV, TABLE_DIR, clean_path
from qc.qc_eeg import compare_raw_vs_corrected, qc_eeg_recording
from qc.qc_fmri import qc_fmri_recording
from qc.qc_fnirs import qc_fnirs_recording
from qc.rules import RULESET_VERSION

OUT = {
    "eeg": TABLE_DIR / "qc_eeg.csv",
    "fnirs": TABLE_DIR / "qc_fnirs.csv",
    "fmri": TABLE_DIR / "qc_fmri.csv",
    "raw_vs_corrected": TABLE_DIR / "qc_raw_vs_corrected.csv",
}


def _key(row) -> str:
    return "|".join(str(row[k]) for k in
                    ("dataset", "subject", "session", "task", "modality", "data_role"))


def _done_keys(path: Path, modality: str) -> set[str]:
    """Keys already processed. ERROR rows are deliberately NOT counted as done,
    so a re-run retries them after a bug fix instead of silently skipping."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return set()
    need = {"dataset", "subject", "session", "task", "data_role"}
    if not need.issubset(df.columns):
        return set()
    if "qc_decision" in df.columns:
        df = df[df.qc_decision != "ERROR"]
    return {"|".join([str(r.dataset), str(r.subject), str(r.session),
                      str(r.task), modality, str(r.data_role)])
            for r in df.itertuples()}


def _append(path: Path, row: dict):
    """Append one row, keeping the file's schema stable.

    Rows do not all carry identical keys (ERROR rows are short; some branches
    add a column). Appending a bare DataFrame would write the row's own columns
    in its own order regardless of the existing header, producing a CSV whose
    lines have different field counts. So: align every row to the header on
    disk, and when a row introduces genuinely new columns, rewrite the file
    under the union schema.
    """
    if not path.exists():
        pd.DataFrame([row]).to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    new_cols = [c for c in row if c not in header]
    if new_cols:
        old = pd.read_csv(path, low_memory=False)
        combined = pd.concat([old, pd.DataFrame([row])], ignore_index=True, sort=False)
        combined.to_csv(path, index=False)
        return
    aligned = {c: row.get(c, "") for c in header}
    pd.DataFrame([aligned])[header].to_csv(path, mode="a", header=False, index=False)


# =========================================================================
def _events_for(row: dict):
    """Load whatever event description exists for a recording."""
    ds, p = row["dataset"], clean_path(row.get("events_path"))
    if p is None or not p.exists():
        return None
    try:
        if ds == "TUBerlin":
            return readers.read_tuberlin_markers(p)
        if ds == "ds002739":
            return readers.read_ds002739_events(p)
        df = readers.read_bids_events(p)
        if df is not None and "onset" in df.columns:
            return {"onsets_s": df["onset"].to_numpy(dtype=float),
                    "labels": (df["trial_type"].to_numpy()
                               if "trial_type" in df.columns else np.array([]))}
    except Exception:
        return None
    return None


def _attach_events(row, data):
    """Put event onsets on the data dict so the EEG module can report coverage."""
    ev = _events_for(row)
    if ev and "onsets_s" in ev:
        data["event_onsets_s"] = np.asarray(ev["onsets_s"], dtype=float)
    elif data.get("events"):
        data["event_onsets_s"] = np.array([o for o, _ in data["events"]], dtype=float)
    else:
        data["event_onsets_s"] = np.array([])
    return ev


# =========================================================================
def run(dataset=None, modality=None, limit=None, save_figs=True, force=False):
    if not MANIFEST_CSV.exists():
        build_manifest()
    man = pd.read_csv(MANIFEST_CSV)
    if dataset:
        man = man[man.dataset == dataset]
    if modality:
        man = man[man.modality == modality]
    man = man[man.exists.astype(str).str.lower() == "true"]
    if limit:
        man = man.groupby(["dataset", "modality", "data_role"]).head(limit)

    if force:
        for m, p in OUT.items():
            if (modality in (None, m)) and p.exists():
                p.unlink()

    done = {m: (set() if force else _done_keys(p, m)) for m, p in OUT.items()
            if m != "raw_vs_corrected"}

    print(f"ruleset {RULESET_VERSION}")
    print(f"{len(man)} recordings selected\n")

    n_ok = n_err = n_skip = 0
    for i, row in enumerate(man.to_dict("records"), 1):
        mod = row["modality"]
        k = _key(row)
        if k in done.get(mod, set()):
            n_skip += 1
            continue
        tag = f"[{i}/{len(man)}] {row['dataset']:9s} {row['subject']:22s} " \
              f"{row['task']:14s} {mod:5s} {row['data_role']:12s}"
        try:
            if mod == "fmri":
                r = qc_fmri_recording(row, save_figs=save_figs)
            else:
                data = readers.read_recording(row)
                ev = _attach_events(row, data)
                if mod == "eeg":
                    r = qc_eeg_recording(row, data, save_figs=save_figs)
                else:
                    r = qc_fnirs_recording(row, data, events=ev, save_figs=save_figs)
                del data
            _append(OUT[mod], r)
            n_ok += 1
            print(f"{tag} -> {r.get('qc_decision','?')}")
        except Exception as exc:
            n_err += 1
            _append(OUT[mod], {
                "dataset": row["dataset"], "subject": row["subject"],
                "session": row["session"], "task": row["task"],
                "data_role": row["data_role"], "path": row["primary_path"],
                "qc_decision": "ERROR",
                "qc_reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
            print(f"{tag} -> ERROR {type(exc).__name__}: {str(exc)[:120]}")
            traceback.print_exc(limit=1, file=sys.stderr)

    print(f"\ndone: {n_ok} processed, {n_skip} skipped (already done), {n_err} errors")

    if modality in (None, "eeg"):
        run_raw_vs_corrected(dataset=dataset, save_figs=save_figs, force=force)


# =========================================================================
def run_raw_vs_corrected(dataset=None, save_figs=True, force=False):
    """Pair raw and provider-corrected in-scanner EEG and compare them."""
    man = pd.read_csv(MANIFEST_CSV)
    man = man[(man.modality == "eeg") & (man.dataset == "ds002336")]
    if dataset and dataset != "ds002336":
        return
    if force and OUT["raw_vs_corrected"].exists():
        OUT["raw_vs_corrected"].unlink()

    raw = man[man.data_role == "raw"]
    corr = man[man.data_role == "corrected"].set_index(["subject", "task"])
    print(f"\nraw-vs-corrected comparison: {len(raw)} candidate pairs")

    for row in raw.to_dict("records"):
        key = (row["subject"], row["task"])
        if key not in corr.index:
            continue
        c = corr.loc[key]
        c = c.iloc[0] if isinstance(c, pd.DataFrame) else c
        crow = {**c.to_dict(), "subject": row["subject"], "task": row["task"],
                "dataset": "ds002336"}
        try:
            rd = readers.read_brainvision(Path(row["primary_path"]))
            cd = readers.read_brainvision(Path(crow["primary_path"]))
            res = compare_raw_vs_corrected(row, rd, crow, cd, save_figs=save_figs)
            _append(OUT["raw_vs_corrected"], res)
            print(f"  {row['subject']} {row['task']:12s} "
                  f"raw {res['raw_robust_amplitude_uv']:9.1f} uV -> "
                  f"corr {res['corrected_robust_amplitude_uv']:7.1f} uV "
                  f"({res['power_reduction_db']:.1f} dB)")
            del rd, cd
        except Exception as exc:
            print(f"  {row['subject']} {row['task']}: ERROR {type(exc).__name__}: {exc}")


# =========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["HEFMI", "ds002336", "TUBerlin", "ds002739"])
    ap.add_argument("--modality", choices=["eeg", "fnirs", "fmri"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(dataset=a.dataset, modality=a.modality, limit=a.limit,
        save_figs=not a.no_figs, force=a.force)
