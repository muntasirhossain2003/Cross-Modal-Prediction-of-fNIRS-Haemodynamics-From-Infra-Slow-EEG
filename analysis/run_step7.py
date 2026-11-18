"""README Step 7: merged notebook with EEG_FILTER=False, run to the end of section 8 (cell 25).

Runs from merge/ so the cache and outputs land next to the real ones (a new run tag,
because eeg_filter is part of the fingerprint). The executed copy is saved here.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]   # repository root

SRC = Path(str(REPO / "pipeline/main_experiment.ipynb"))
OUT = Path(__file__).with_name("step7_nofilter_executed.ipynb")
LAST_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 25

nb = nbformat.read(SRC, as_version=4)
cfg = nb.cells[4]
assert "EEG_FILTER = True" in cfg.source and "QUICK_TEST = False" in cfg.source
cfg.source = cfg.source.replace("EEG_FILTER = True", "EEG_FILTER = False")
nb.cells = nb.cells[:LAST_CELL + 1]

t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3",
                        resources={"metadata": {"path": str(SRC.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, OUT)
    print(f"saved {OUT} after {(time.time() - t0) / 60:.1f} min")
