"""Enumerate every recording on disk across the four datasets.

The manifest is the audit's unit of work: one row per (dataset, subject,
session/task, modality, data_role) with the concrete file paths attached.
Everything downstream iterates over this file, so "QC ran over every file in
the manifest" is checkable rather than asserted.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc.config import DATASETS, MANIFEST_CSV, dataset_path

FIELDS = [
    "dataset", "subject", "session", "task", "modality", "data_role",
    "primary_path", "aux_path", "events_path", "size_mb", "exists",
]


def _row(dataset, subject, session, task, modality, data_role,
         primary, aux=None, events=None):
    p = Path(primary)
    return {
        "dataset": dataset,
        "subject": subject,
        "session": session,
        "task": task,
        "modality": modality,
        "data_role": data_role,
        "primary_path": str(p),
        "aux_path": str(aux) if aux else "",
        "events_path": str(events) if events else "",
        "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else 0.0,
        "exists": p.exists(),
    }


# --------------------------------------------------------------------------
def manifest_hefmi():
    root = dataset_path("HEFMI")
    rows = []
    for group in ("Normals", "Patients"):
        eeg_root = root / "epo_data" / group / "EEG"
        nirs_root = root / "epo_data" / group / "NIRS"
        if not eeg_root.exists():
            continue
        for subj_dir in sorted(eeg_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            subject = f"{group}/{subj_dir.name}"
            for f in sorted(subj_dir.glob("*_epo.mat")):
                session = f.stem.replace("_epo", "")
                rows.append(_row("HEFMI", subject, session, "motor_imagery",
                                 "eeg", "epoched", f))
                nirs_f = nirs_root / subj_dir.name / f.name
                if nirs_f.exists():
                    rows.append(_row("HEFMI", subject, session, "motor_imagery",
                                     "fnirs", "epoched_conc", nirs_f))
    return rows


def manifest_ds002336():
    root = dataset_path("ds002336")
    rows = []
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        # raw in-scanner EEG (BrainVision)
        for vhdr in sorted((subj_dir / "eeg").glob("*.vhdr")):
            task = vhdr.stem.split("task-")[1].replace("_eeg", "")
            ev = root / f"task-{task}_events.tsv"
            rows.append(_row("ds002336", subject, "ses-01", task, "eeg", "raw",
                             vhdr, events=ev if ev.exists() else None))
        # provider-preprocessed (gradient/BCG corrected) EEG
        pp_dir = root / "derivatives" / subject / "eeg_pp"
        for vhdr in sorted(pp_dir.glob("*.vhdr")):
            task = vhdr.stem.split("task-")[1].replace("_eeg_pp", "")
            rows.append(_row("ds002336", subject, "ses-01", task, "eeg", "corrected", vhdr))
        # BOLD
        for nii in sorted((subj_dir / "func").glob("*_bold.nii.gz")):
            task = nii.name.split("task-")[1].split("_")[0]
            ev = root / f"task-{task}_events.tsv"
            hm = (root / "derivatives" / subject / "head_motion" /
                  f"d_{subject}_task-{task}_headmotion.tsv")
            rows.append(_row("ds002336", subject, "ses-01", task, "fmri", "raw",
                             nii, aux=hm if hm.exists() else None,
                             events=ev if ev.exists() else None))
    return rows


def manifest_tuberlin():
    root = dataset_path("TUBerlin")
    rows = []
    for subj_dir in sorted((root / "EEG").glob("VP*-EEG")):
        subject = subj_dir.name.replace("-EEG", "")
        for cnt in sorted(subj_dir.glob("cnt_*.mat")):
            task = cnt.stem.replace("cnt_", "")
            mrk = subj_dir / f"mrk_{task}.mat"
            mnt = subj_dir / f"mnt_{task}.mat"
            rows.append(_row("TUBerlin", subject, "ses-01", task, "eeg", "raw",
                             cnt, aux=mnt if mnt.exists() else None,
                             events=mrk if mrk.exists() else None))
    for subj_dir in sorted((root / "NIRS").glob("VP*-NIRS")):
        subject = subj_dir.name.replace("-NIRS", "")
        for cnt in sorted(subj_dir.glob("cnt_*.mat")):
            task = cnt.stem.replace("cnt_", "")
            mrk = subj_dir / f"mrk_{task}.mat"
            mnt = subj_dir / f"mnt_{task}.mat"
            rows.append(_row("TUBerlin", subject, "ses-01", task, "fnirs", "conc",
                             cnt, aux=mnt if mnt.exists() else None,
                             events=mrk if mrk.exists() else None))
    return rows


def manifest_ds002739():
    root = dataset_path("ds002739")
    rows = []
    for subj_dir in sorted(root.glob("sub-*")):
        subject = subj_dir.name
        for mat in sorted((subj_dir / "EEG").glob("EEG_data_*.mat")):
            run = mat.stem.split("run-")[1]
            ev = subj_dir / "EEG" / mat.name.replace("EEG_data_", "EEG_events_")
            rows.append(_row("ds002739", subject, f"run-{run}", "rdm_decision",
                             "eeg", "corrected", mat,
                             events=ev if ev.exists() else None))
        for nii in sorted((subj_dir / "func").glob("*_bold.nii.gz")):
            run = nii.name.split("run-")[1].split("_")[0]
            ev = nii.parent / nii.name.replace("_bold.nii.gz", "_events.tsv")
            rows.append(_row("ds002739", subject, f"run-{run}", "rdm_decision",
                             "fmri", "raw", nii,
                             events=ev if ev.exists() else None))
    return rows


BUILDERS = {
    "HEFMI": manifest_hefmi,
    "ds002336": manifest_ds002336,
    "TUBerlin": manifest_tuberlin,
    "ds002739": manifest_ds002739,
}


def build(out_path: Path = MANIFEST_CSV):
    rows = []
    for key, fn in BUILDERS.items():
        got = fn()
        print(f"{key:12s} {len(got):5d} recordings")
        rows.extend(got)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")
    return rows


if __name__ == "__main__":
    rows = build()
    # quick shape of the audit workload
    from collections import Counter
    c = Counter((r["dataset"], r["modality"], r["data_role"]) for r in rows)
    print("\nworkload breakdown:")
    for k in sorted(c):
        print(f"  {k[0]:10s} {k[1]:6s} {k[2]:14s} {c[k]:4d}")
    tot_gb = sum(r["size_mb"] for r in rows) / 1000
    print(f"\ntotal manifest volume: {tot_gb:.1f} GB")
