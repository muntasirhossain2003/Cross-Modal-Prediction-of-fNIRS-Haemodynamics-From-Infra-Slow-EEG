"""Execute NeuroVascNet_v6_hBCI_trains.ipynb top to bottom, logging progress.

Run standalone so it does not depend on any editor or agent session staying
alive. Writes v6_hbci_fullrun.log next to the notebook.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "NeuroVascNet_v6_hBCI_trains.ipynb"


def main():
    import nbformat

    nb = nbformat.read(str(NB), as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    env = {"__name__": "__main__", "display": lambda *a, **k: None}

    print(f"START {time.strftime('%Y-%m-%d %H:%M:%S')}  {NB.name}", flush=True)
    print(f"{len(code)} code cells\n", flush=True)

    t0 = time.time()
    for i, c in enumerate(code):
        ts = time.time()
        try:
            exec(compile(c.source, f"<cell {i}>", "exec"), env)
        except Exception as exc:
            import traceback
            print(f"!!! cell {i} FAILED after {time.time()-ts:.1f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            print(f"\nSTOPPED at cell {i} after "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            return 1
        print(f"=== cell {i}/{len(code)-1} OK  {time.time()-ts:.1f}s  "
              f"(total {(time.time()-t0)/60:.1f} min)", flush=True)

    print(f"\nALL CELLS COMPLETED in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"END {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
