import warnings, nbformat
warnings.filterwarnings("ignore")
nb = nbformat.read("_feas_smoke.ipynb", as_version=4)
G = {"__name__": "__main__", "display": lambda *a, **k: None}
for i, c in enumerate([x for x in nb.cells if x.cell_type == "code"]):
    try:
        exec(compile(c.source, f"<cell {i}>", "exec"), G)
    except Exception as e:
        import traceback
        print(f"!!! cell {i} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        break
else:
    print("FEAS SMOKE OK", flush=True)
