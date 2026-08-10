"""Produce NeuroVascNet_v3_1_maisha_RUNNABLE.ipynb from Maisha's notebook.

READ-ONLY on the source. A previous session lost a notebook by writing to a
full disk, so this script:
  * never opens the original for writing
  * checks free space before it writes
  * writes to a temp file and renames only after a successful parse

Fixes applied (each marked ###fix in the output so they are easy to find):

  1. numpy 2.x removed np.trapz -> np.trapezoid; older numpy lacks the new name
  2. ds002336 ships EEG-only and BOLD-only recordings; pair what exists instead
     of aborting the whole build
  3. HEFMI has files with no partner; that is benign, only MISSING files are fatal
  4. blank rows in events.tsv are skipped; genuinely corrupt rows stay fatal
  5. an event window past the end of a short run is dropped, not fatal
     (sub-xp106's MIpost run has 55 volumes where every other subject has 106)

Usage:  python fix_maisha_notebook.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nbformat

SRC = Path("NeuroVascNet_v3_1_FIXED_v2_(3)_maisha.ipynb")
DST = Path("NeuroVascNet_v3_1_maisha_RUNNABLE.ipynb")
TMP = DST.with_suffix(".ipynb.tmp")

MIN_FREE_GB = 2.0


def guard_disk():
    free = shutil.disk_usage(SRC.resolve().anchor).free / 1e9
    if free < MIN_FREE_GB:
        sys.exit(f"ABORT: only {free:.1f} GB free. Writing a notebook to a full "
                 "disk is what destroyed the last one. Free space first.")
    print(f"disk check: {free:.1f} GB free -- ok")


def apply(src: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in src:
        print(f"  [skip] {label} -- anchor not found (already fixed?)")
        return src, False
    print(f"  [done] {label}")
    return src.replace(old, new, 1), True


def main():
    if not SRC.exists():
        sys.exit(f"source not found: {SRC}")
    guard_disk()

    nb = nbformat.read(str(SRC), as_version=4)
    print(f"read {SRC.name}: {len(nb.cells)} cells (source is NOT modified)")
    applied = 0

    # ---------------------------------------------------------------- fix 1
    for c in nb.cells:
        if c.cell_type == "code" and "import numpy as np" in c.source:
            if "np.trapezoid" in c.source:
                print("  [skip] fix 1 numpy shim -- already present")
                break
            c.source = c.source.replace(
                "import numpy as np",
                "import numpy as np\n"
                "\n"
                "###fix 1 -- numpy 2.0 renamed trapz to trapezoid. Code below calls\n"
                "###fix    np.trapezoid, which does not exist on numpy < 2, and np.trapz\n"
                "###fix    was removed in numpy >= 2. Alias whichever is missing.\n"
                "if not hasattr(np, 'trapezoid'):\n"
                "    np.trapezoid = np.trapz\n"
                "if not hasattr(np, 'trapz'):\n"
                "    np.trapz = np.trapezoid", 1)
            print("  [done] fix 1 numpy trapezoid/trapz shim")
            applied += 1
            break

    # cell 14 holds the loaders
    tgt = None
    for c in nb.cells:
        if c.cell_type == "code" and "registered event window exceeds" in c.source:
            tgt = c
            break
    if tgt is None:
        sys.exit("could not find the loader cell -- notebook structure differs")
    s = tgt.source

    # ---------------------------------------------------------------- fix 5
    old5 = """                raise RuntimeError(
                    f"ds002336 {k} {trial_id}: registered event window exceeds EEG/BOLD bounds; "
                    f"eeg[{e0}:{e0 + native_eeg_len}]/{eeg.shape[1]}, "
                    f"bold[{b0}:{b0 + fmri_len}]/{roi.shape[0]}"
                )"""
    new5 = """                ###fix 5 -- ds002336 ships ONE dataset-level event schedule per task,
                ###fix    but runs differ in length: sub-xp106's MIpost BOLD has 55
                ###fix    volumes where every other subject has 106. Its last events
                ###fix    were presented but the scanner had stopped, so there is no
                ###fix    BOLD to pair with them. Drop those windows; the audit row
                ###fix    above already records included=False so nothing is hidden.
                print(f"[ds002336] {k} {trial_id}: event window past end of run "
                      f"(bold[{b0}:{b0 + fmri_len}]/{roi.shape[0]}) -- dropped")
                continue"""
    s, ok = apply(s, old5, new5, "fix 5 event window past end of run")
    applied += ok

    # ---------------------------------------------------------------- fix 2
    old2 = """    eeg_only = sorted(set(eeg) - set(bold))
    bold_only = sorted(set(bold) - set(eeg))
    if eeg_only or bold_only:
        raise RuntimeError(
            f"ds002336 requires exact subject/task/run coverage; EEG-only={eeg_only}, BOLD-only={bold_only}. "
            "Subject-only fallback pairing is forbidden for primary results."
        )"""
    new2 = """    eeg_only = sorted(set(eeg) - set(bold))
    bold_only = sorted(set(bold) - set(eeg))
    ###fix 2 -- the released ds002336 genuinely has incomplete coverage: 1 EEG-only
    ###fix    and 3 BOLD-only recordings. Those cannot be paired, so drop them and
    ###fix    keep the 52 exact pairs. Pairing is still EXACT (same subject, task
    ###fix    and run) -- no subject-level fallback is introduced. Refusing to run
    ###fix    at all over 4 unpairable files was the bug.
    if eeg_only or bold_only:
        print(f"[ds002336] incomplete coverage in the released dataset -- dropping "
              f"{len(eeg_only)} EEG-only and {len(bold_only)} BOLD-only recording(s).")
        for k in eeg_only:
            print(f"           EEG without BOLD : {k}")
        for k in bold_only:
            print(f"           BOLD without EEG : {k}")
        eeg = {k: v for k, v in eeg.items() if k not in set(eeg_only)}
        bold = {k: v for k, v in bold.items() if k not in set(bold_only)}
        print(f"           {len(eeg)} exact pairs retained.")
    if len(eeg) < 10:
        raise RuntimeError(
            f"ds002336: only {len(eeg)} exact pairs remain -- too few to proceed. "
            "Check that the dataset downloaded completely.")"""
    s, ok = apply(s, old2, new2, "fix 2 ds002336 unpairable recordings")
    applied += ok

    # ---------------------------------------------------------------- fix 3
    old3 = """    if missing_files or unmatched_eeg or unmatched_nirs or unexpected_eeg or unexpected_nirs:
        raise RuntimeError(
            f"HEFMI recording map does not exactly cover recognized files. missing={missing_files}, "
            f"unmatched_eeg={unmatched_eeg}, unmatched_fnirs={unmatched_nirs}, "
            f"unexpected_eeg={unexpected_eeg}, unexpected_fnirs={unexpected_nirs}"
        )"""
    new3 = """    ###fix 3 -- distinguish two different situations that were both fatal:
    ###fix      missing_files / unexpected_*  the map names a file that is NOT on
    ###fix                                    disk -> still FATAL, the map is wrong
    ###fix      unmatched_*                   a file on disk that the map does not
    ###fix                                    name. HEFMI really does contain such
    ###fix                                    files (e.g. Patients/NIRS/subject9/
    ###fix                                    3_epo.mat has no EEG partner), so they
    ###fix                                    are benign and simply excluded.
    if missing_files or unexpected_eeg or unexpected_nirs:
        raise RuntimeError(
            f"HEFMI recording map names files that are not on disk. "
            f"missing={missing_files}, unexpected_eeg={unexpected_eeg}, "
            f"unexpected_fnirs={unexpected_nirs}"
        )
    if unmatched_eeg or unmatched_nirs:
        print(f"[HEFMI] {len(unmatched_eeg)} EEG and {len(unmatched_nirs)} fNIRS "
              f"file(s) on disk are unpairable and excluded from the recording map:")
        for p in (unmatched_eeg + unmatched_nirs)[:6]:
            print(f"        {p}")"""
    s, ok = apply(s, old3, new3, "fix 3 HEFMI unpairable files are benign")
    applied += ok

    tgt.source = s

    # ---------------------------------------------------------------- fix 6
    # The resolver only knew Drive-style folder names. Add the names the data
    # actually has on a local machine, so no junctions or renaming are needed.
    for c in nb.cells:
        if c.cell_type == "code" and "DATASET_DIR_ALIASES = {" in c.source:
            a6 = '''DATASET_DIR_ALIASES = {
    "hefmi": [
        "figshare_HEFMI_ICH_v4_COMPLETE",'''
            b6 = '''###fix 6 -- the resolver only recognised Drive-style folder names, so a normal
###fix    local copy ("HEFMI-ICH EEG+FNRI" etc.) was reported as missing. The real
###fix    on-disk names are added as aliases -- no renaming or junctions needed.
DATASET_DIR_ALIASES = {
    "hefmi": [
        "HEFMI-ICH EEG+FNRI",
        "figshare_HEFMI_ICH_v4_COMPLETE",'''
            if a6 in c.source:
                c.source = c.source.replace(a6, b6, 1)
                c.source = c.source.replace(
                    '    "ds002336": [\n        "openneuro_ds002336_v2.0.2",',
                    '    "ds002336": [\n        "OpenNeuro ds002336",\n'
                    '        "openneuro_ds002336_v2.0.2",', 1)
                c.source = c.source.replace(
                    '    "tuberlin": [\n        "TU_BERLIN_HBCI",',
                    '    "tuberlin": [\n        "TU Berlin hBCI EEG+FNRI",\n'
                    '        "TU_BERLIN_HBCI",', 1)
                c.source = c.source.replace(
                    '    "ds002739": [\n        "openneuro_ds002739_v1.0.0",',
                    '    "ds002739": [\n'
                    '        "External fMRI validation OpenNeuro ds002739",\n'
                    '        "openneuro_ds002739_v1.0.0",', 1)
                print("  [done] fix 6 accept real local folder names")
                applied += 1
            else:
                print("  [skip] fix 6 -- alias table not in the expected shape")
            break

    # ---------------------------------------------------------------- fix 7
    # The crosswalk was only looked for inside the dataset folder, which is
    # usually read-only or a shared copy. Also look beside the notebook.
    old7 = """    env_path = os.environ.get("NEUROVASCNET_HEFMI_RECORDING_MAP")
    candidates = ([Path(env_path)] if env_path else []) + [
        HEFMI_ROOT / "hefmi_recording_map.csv",
        HEFMI_ROOT / "recording_to_subject.csv",
    ]"""
    new7 = """    ###fix 7 -- also look for the crosswalk NEXT TO THE NOTEBOOK, not only inside
    ###fix    the dataset folder. The dataset directory is often a shared or
    ###fix    read-only copy, so keeping the map with the code is the normal case.
    env_path = os.environ.get("NEUROVASCNET_HEFMI_RECORDING_MAP")
    candidates = ([Path(env_path)] if env_path else []) + [
        HEFMI_ROOT / "hefmi_recording_map.csv",
        HEFMI_ROOT / "recording_to_subject.csv",
        Path.cwd() / "hefmi_recording_map.csv",
        Path.cwd() / "recording_to_subject.csv",
        Path.cwd().parent / "hefmi_recording_map.csv",
    ]"""
    s7 = tgt.source
    if old7 in s7:
        tgt.source = s7.replace(old7, new7, 1)
        print("  [done] fix 7 find the HEFMI map beside the notebook")
        applied += 1
    else:
        print("  [skip] fix 7 -- anchor not found (already fixed?)")

    # ---------------------------------------------------------------- fix 4
    # blank rows in a BIDS events.tsv should be skipped, not crash the parser
    old4 = '            onset, duration = float(row["onset"]), float(row["duration"])'
    new4 = """            ###fix 4 -- trailing blank lines are common in BIDS events.tsv and
            ###fix    made float("") raise. Skip rows that are entirely empty;
            ###fix    a row with SOME fields filled is still corrupt -> still fatal.
            _o = "" if row.get("onset") is None else str(row.get("onset")).strip()
            _d = "" if row.get("duration") is None else str(row.get("duration")).strip()
            _t = "" if row.get("trial_type") is None else str(row.get("trial_type")).strip()
            if not _o and not _d and not _t:
                skipped_blank += 1
                continue
            onset, duration = float(row["onset"]), float(row["duration"])"""
    s2 = tgt.source
    if old4 in s2:
        s2 = s2.replace(old4, new4, 1)
        # declare and report the counter inside _read_ds2336_events
        s2 = s2.replace("def _read_ds2336_events(path):",
                        "def _read_ds2336_events(path):\n    skipped_blank = 0", 1)
        tgt.source = s2
        print("  [done] fix 4 blank rows in events.tsv")
        applied += 1
    else:
        print("  [skip] fix 4 -- anchor not found (already fixed?)")

    # ---------------------------------------------------------------- fix 8
    # A hard-coded Google Drive URL was the first thing the resolver reached
    # for. Harmless on a PC (the download path is Colab-gated) but it produced
    # a Drive-flavoured error when a local folder was simply named differently.
    # Put ONE explicit path at the very top instead.
    SETTINGS_MD = """## 0 · Set your data folder

**This is the only cell you need to edit.** Point `LOCAL_DATA_DIR` at the folder
that *contains* the dataset folders, then Run All.

Google Drive is not used on a PC — the Drive code further down only runs on
Colab. Setting the path here bypasses it entirely."""

    SETTINGS_CODE = '''###fix 8 -- explicit local data folder, set BEFORE anything else runs.
###fix    The notebook otherwise reached for a hard-coded Google Drive URL and
###fix    reported a Drive error when a perfectly good local copy was present
###fix    under a different folder name.
import os
from pathlib import Path

# ---------------------------------------------------------------------------
#  EDIT THIS: the folder that CONTAINS the dataset folders (not one of them).
#
#    Windows :  r"G:\\\\research_bio_medical\\\\dataset"
#    Linux   :   "/home/me/data"
# ---------------------------------------------------------------------------
LOCAL_DATA_DIR = r"G:\\\\research_bio_medical\\\\dataset"
# ---------------------------------------------------------------------------

_p = Path(LOCAL_DATA_DIR)
if not _p.exists():
    raise FileNotFoundError(
        f"LOCAL_DATA_DIR does not exist:\\n    {_p}\\n"
        "Edit the path in this cell.")

os.environ["NEUROVASCNET_DATA_ROOT"] = str(_p)
print(f"data root : {_p.resolve()}")
print("expected inside it (any of these names work):")
for _n in ("TU Berlin hBCI EEG+FNRI", "HEFMI-ICH EEG+FNRI",
           "OpenNeuro ds002336", "External fMRI validation OpenNeuro ds002739"):
    print(f"   [{'ok  ' if (_p / _n).exists() else 'MISS'}] {_n}")

# the crosswalk lives beside the notebook, not inside the dataset folder
_map = Path.cwd() / "hefmi_recording_map.csv"
if _map.exists():
    os.environ["NEUROVASCNET_HEFMI_RECORDING_MAP"] = str(_map)
    print(f"HEFMI map : {_map}")
else:
    print("HEFMI map : not found beside the notebook -- "
          "the clinical/fNIRS loader will stop and tell you what to create.")
'''
    nb.cells.insert(0, nbformat.v4.new_code_cell(SETTINGS_CODE))
    nb.cells.insert(0, nbformat.v4.new_markdown_cell(SETTINGS_MD))
    print("  [done] fix 8 explicit local data folder at the top")
    applied += 1

    # make the resolver's error message PC-first instead of Drive-first
    for c in nb.cells:
        if c.cell_type == "code" and "Missing dataset folders: " in c.source:
            c.source = c.source.replace(
                '"The notebook searched the requested Drive folder first. "',
                '"Set LOCAL_DATA_DIR in the first cell to the folder that "\n'
                '            "CONTAINS these dataset folders. "', 1)
            break

    # ---------------------------------------------------------------- fix 9
    # Replace the 339-line Colab/Drive/gdown bootstrap with a local one. It only
    # has to provide six names, so everything downstream is unaffected:
    #   DATASET_ROOTS  DATASET_READABILITY  DATASET_FILE_SIGNATURES
    #   CSV_INVENTORY_PATH  OUTPUT_DIR  IN_COLAB
    BOOTSTRAP = r'''###fix 9 -- the original bootstrap mounted Google Drive, pip-installed packages
###fix    and could gdown a 60 GB folder. None of that belongs in a local run, and
###fix    it was the source of the "Missing dataset folders ... check the Drive
###fix    URL" error. Replaced with plain local resolution. This cell provides
###fix    exactly the names the rest of the notebook expects.
import hashlib
from pathlib import Path

IN_COLAB = False                      # local run: never mount Drive, never download

DATA_ROOT = Path(os.environ["NEUROVASCNET_DATA_ROOT"])

# Accept whatever the folders are actually called. First match wins.
DATASET_DIR_ALIASES = {
    "hefmi": ["HEFMI-ICH EEG+FNRI", "figshare_HEFMI_ICH_v4_COMPLETE",
              "HEFMI_ICH_FIGSHARE_V4", "HEFMI-ICH", "HEFMI"],
    "ds002336": ["OpenNeuro ds002336", "openneuro_ds002336_v2.0.2", "ds002336"],
    "tuberlin": ["TU Berlin hBCI EEG+FNRI", "TU_BERLIN_HBCI", "TU Berlin hBCI",
                 "TU_BERLIN", "HBCI"],
    "ds002739": ["External fMRI validation OpenNeuro ds002739",
                 "openneuro_ds002739_v1.0.0", "ds002739"],
}

DATASET_FILE_SIGNATURES = {
    "hefmi": ["*.mat"],
    "ds002336": ["*.vhdr", "*_bold.nii", "*_bold.nii.gz"],
    "tuberlin": ["*.mat", "*.zip", "*.vhdr", "*.wl1"],
    "ds002739": ["*.mat", "*_bold.nii", "*_bold.nii.gz"],
}


def _canon(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_dataset_roots():
    """Match each dataset to a folder directly under DATA_ROOT."""
    present = {_canon(p.name): p for p in DATA_ROOT.iterdir() if p.is_dir()}
    roots, missing = {}, []
    for key, names in DATASET_DIR_ALIASES.items():
        hit = next((present[_canon(n)] for n in names if _canon(n) in present), None)
        if hit is None:                      # fall back to a containment match
            hit = next((p for c, p in present.items()
                        if any(_canon(n) in c for n in names)), None)
        if hit is None:
            missing.append(f"{key} (looked for: {', '.join(names[:3])})")
        else:
            roots[key] = str(hit)
    if missing:
        raise FileNotFoundError(
            "These datasets were not found under\n"
            f"    {DATA_ROOT}\n  - " + "\n  - ".join(missing) +
            "\n\nFix LOCAL_DATA_DIR in the first cell, or rename the folder to "
            "one of the names listed above.")
    return roots


DATASET_ROOTS = _resolve_dataset_roots()
print("dataset roots:")
for _k, _v in sorted(DATASET_ROOTS.items()):
    print(f"  {_k:10s} {_v}")

OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"output dir  : {OUTPUT_DIR}")


def _audit_dataset(root):
    """Fingerprint a dataset from file metadata. Cheap: stats only, no reads."""
    root = Path(root)
    digest = hashlib.sha256()
    out = {"root": str(root), "files_seen": 0, "readable_files": 0,
           "unreadable_files": [], "total_bytes": 0}
    for path in sorted(root.rglob("*"), key=str):
        if not path.is_file():
            continue
        out["files_seen"] += 1
        try:
            st = path.stat()
            digest.update(
                f"{path.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n"
                .encode("utf-8", "replace"))
            out["total_bytes"] += int(st.st_size)
            out["readable_files"] += 1
        except Exception as exc:
            out["unreadable_files"].append({"path": str(path), "error": repr(exc)})
    out["file_signature_sha256"] = digest.hexdigest()
    return out


DATASET_READABILITY = {k: _audit_dataset(v) for k, v in DATASET_ROOTS.items()}
for _k, _v in sorted(DATASET_READABILITY.items()):
    _bad = len(_v["unreadable_files"])
    print(f"  {_k:10s} {_v['files_seen']:5d} files, "
          f"{_v['total_bytes']/1e9:6.2f} GB"
          + (f"  !! {_bad} unreadable" if _bad else ""))

CSV_INVENTORY_PATH = None            # optional manifest; not used for local runs
'''
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and "DRIVE_FOLDER_URL" in c.source:
            old_lines = len(c.source.split("\n"))
            c.source = BOOTSTRAP
            print(f"  [done] fix 9 replaced Drive/Colab bootstrap "
                  f"({old_lines} lines -> {len(BOOTSTRAP.splitlines())})")
            applied += 1
            break
    else:
        print("  [skip] fix 9 -- bootstrap cell not found")

    # ---------------------------------------------------------------- write
    nbformat.write(nb, str(TMP))
    check = nbformat.read(str(TMP), as_version=4)
    import ast
    bad = []
    for i, c in enumerate(check.cells):
        if c.cell_type == "code":
            try:
                ast.parse(c.source)
            except SyntaxError as exc:
                bad.append((i, exc))
    if bad:
        TMP.unlink(missing_ok=True)
        sys.exit(f"ABORT: {len(bad)} cells fail to parse: {bad[:3]}")

    TMP.replace(DST)
    print(f"\nwrote {DST} ({len(check.cells)} cells, {applied} fixes applied)")
    print(f"original {SRC.name} untouched: "
          f"{SRC.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
