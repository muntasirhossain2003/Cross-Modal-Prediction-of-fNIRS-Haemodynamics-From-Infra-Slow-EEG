"""Run Maisha's v2.3 notebook once per remaining seed, each in a fresh kernel.

The notebook trains one seed per kernel (low-resource protocol). This script
starts a new kernel for every seed that has no provenance-valid summary in
output/, passes the seed through NEUROVASCNET_RUN_SEED, saves the executed copy
and stops at the first failure. The run that completes the fifth seed also
writes the multi-seed aggregate (last notebook cell).

Usage (from the repo root):
    python scripts/run_maisha_all_seeds.py              # all missing seeds
    python scripts/run_maisha_all_seeds.py --seeds 7 42 # only these
    python scripts/run_maisha_all_seeds.py --rerun      # all five, even if done
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED.ipynb"
OUTPUT = ROOT / "output"
EXECUTED = OUTPUT / "executed_seed_notebooks"
SEEDS = [1337, 20260726, 7, 42, 2718]
MODEL_VERSION = "v3.1-v2.3-personalized-physio-fixed-delay"


def summary_valid(seed):
    path = OUTPUT / f"summary_{MODEL_VERSION}_seed{seed}.json"
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        prov = doc.get("provenance", {})
        return (prov.get("model_version") == MODEL_VERSION and int(prov.get("seed")) == seed
                and doc.get("claim_guard", {}).get("passed") is True)
    except Exception:
        return False


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_seed(seed):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    os.environ["NEUROVASCNET_RUN_SEED"] = str(seed)     # inherited by the new kernel
    n = len(nb.cells)
    t0 = time.time()

    def on_start(cell, cell_index, **_):
        if cell.cell_type == "code":
            first = (cell.source.splitlines() or [""])[0][:70]
            log(f"seed {seed}  cell {cell_index + 1}/{n}  {first}")

    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}},
                            on_cell_start=on_start)
    out = EXECUTED / f"{NOTEBOOK.stem}_seed{seed}.ipynb"
    try:
        client.execute()
    finally:
        EXECUTED.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out)
        os.environ.pop("NEUROVASCNET_RUN_SEED", None)
    log(f"seed {seed} finished in {(time.time() - t0) / 60:.1f} min; executed copy: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", help="run only these seeds")
    ap.add_argument("--rerun", action="store_true", help="re-run seeds that are already complete")
    args = ap.parse_args()

    wanted = args.seeds or SEEDS
    bad = [s for s in wanted if s not in SEEDS]
    if bad:
        sys.exit(f"seeds {bad} are not in the frozen protocol {SEEDS}")
    todo = [s for s in wanted if args.rerun or not summary_valid(s)]
    log(f"complete already: {[s for s in SEEDS if summary_valid(s)]}")
    log(f"to run now      : {todo or 'nothing'}")

    for seed in todo:
        try:
            run_seed(seed)
        except CellExecutionError as exc:
            log(f"seed {seed} FAILED. See the executed copy in {EXECUTED}. Stopping.")
            print(str(exc)[-3000:], flush=True)
            sys.exit(1)
        if not summary_valid(seed):
            log(f"seed {seed} ran but left no valid summary in {OUTPUT}. Stopping.")
            sys.exit(1)

    agg = OUTPUT / f"multi_seed_summary_{MODEL_VERSION}.json"
    if all(summary_valid(s) for s in SEEDS):
        log(f"all five seeds complete. Aggregate: {agg if agg.is_file() else 'run the last cell once more to write it'}")
    else:
        log(f"still missing: {[s for s in SEEDS if not summary_valid(s)]}")


if __name__ == "__main__":
    main()
