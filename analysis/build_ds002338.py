"""FINAL_PLAN P4.1: ds002338 through the notebook's own load_ds002336 (root + task list changed only)."""
import json
import os
from pathlib import Path

import nbformat
import numpy as np

REPO = Path(__file__).resolve().parents[1]   # repository root

NB = Path(str(REPO / "pipeline/main_experiment.ipynb"))
ROOT = Path(str(REPO / "dataset/OpenNeuro ds002338"))
CACHE = Path(str(REPO / "pipeline/neurovascnet_merge_v1_cache/replication_ds002338"))

nb = nbformat.read(NB, as_version=4)
nb.cells[4].source = nb.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")
ns = {"__name__": "__main__", "display": print}
cwd = os.getcwd(); os.chdir(NB.parent)
for c in nb.cells[:12]:
    if c.cell_type == "code":
        exec("\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!"))), ns)
os.chdir(cwd)

tasks = sorted({p.name.split("_task-")[1].split("_eeg_pp")[0]
                for p in (ROOT / "derivatives").rglob("*_eeg_pp.vhdr") if "_task-" in p.name})
print("tasks with preprocessed EEG:", tasks)
print("task-level event files:", sorted(p.name for p in ROOT.glob("task-*_events.tsv")))
import ast
src = "\n".join(l for l in nb.cells[11].source.splitlines() if not l.lstrip().startswith(("%", "!")))
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_ds002336")
code = ast.get_source_segment(src, fn)
a1 = 'vhdr = root / "derivatives" / sub / "eeg_pp" / f"{sub}_task-{task}_eeg_pp.vhdr"'
a2 = 'ev = root / f"task-{task}_events.tsv"'
assert a1 in code and a2 in code
code = code.replace(a1, a1.replace('f"{sub}_task', 'f"d_{sub}_task')).replace(a2, "ev = root / f\"task-{task.split('_run-')[0]}_events.tsv\"")
exec(code, ns)
ns["DS2336_TASKS"] = tuple(tasks)
d = ns["load_ds002336"](ROOT)
assert d is not None, "no trials"
print(d["audit"].to_string())
CACHE.mkdir(parents=True, exist_ok=True)
for k in ("eeg", "hemo", "labels", "subject_ids", "task_ids", "rec_ids"):
    np.save(CACHE / f"{k}.npy", np.asarray(d[k]))
(CACHE / "meta.json").write_text(json.dumps({"info": {"name": "ds002338", "modality": "fmri", "ch_names": d["ch_names"],
                                                      "tasks": tasks}}, indent=1))
print(f"DONE {len(d['subject_ids'])} trials, {len(np.unique(d['subject_ids']))} people, EEG {d['eeg'].shape}, "
      f"hemo {d['hemo'].shape}")
