"""FINAL_PLAN P2.1: the full merged notebook with EEG_FILTER = False (seed picked by the notebook)."""
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]   # repository root

SRC = Path(str(REPO / "pipeline/CrossModal_InfraSlowEEG_fNIRS_fMRI.ipynb"))
OUT = Path(__file__).with_name("p2_full_executed.ipynb")

nb = nbformat.read(SRC, as_version=4)
cfg = nb.cells[4]
assert "EEG_FILTER = True" in cfg.source and "QUICK_TEST = False" in cfg.source
cfg.source = cfg.source.replace("EEG_FILTER = True", "EEG_FILTER = False")

t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3",
                        resources={"metadata": {"path": str(SRC.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, OUT)
    print(f"saved {OUT} after {(time.time() - t0) / 60:.1f} min")
