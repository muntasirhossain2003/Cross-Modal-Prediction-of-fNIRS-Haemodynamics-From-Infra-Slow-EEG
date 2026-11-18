"""FINAL_PLAN T1 + T2: band-split training and global/local scoring on hBCI.

Runs the merged notebook's own code (cells 0-23 + the cross_subject_cv definition) with
EEG_FILTER=False, so the unfiltered cache is reused, then trains one CV per band.
"""
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]   # repository root

SRC = Path(str(REPO / "pipeline/CrossModal_InfraSlowEEG_fNIRS_fMRI.ipynb"))
HERE = Path(__file__).parent
OUT = HERE / "bands_executed.ipynb"

nb = nbformat.read(SRC, as_version=4)
assert "EEG_FILTER = True" in nb.cells[4].source
nb.cells[4].source = nb.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")
cv = nb.cells[25].source
cut = cv.index('print(f"training on:')
nb.cells = nb.cells[:25] + [nbformat.v4.new_code_cell(cv[:cut])]

ANALYSIS = r'''
import json, gc, time
from pathlib import Path
import scipy.signal as sps

RESULTS = Path(r"__HERE__") / "bands_results.json"
BANDS = [("unfiltered", None, None, False), ("lt0.5", None, 0.5, False), ("lt0.1", None, 0.1, False),
         ("0.1-0.5", 0.1, 0.5, False), ("gt45", 45.0, None, True)]


def band_copy(x, fs, lo, hi, notch, chunk=128):
    """Same steps as filter_eeg_inplace (4th-order Butterworth, zero phase, z-score), any band."""
    if lo is None and hi is None:
        return x
    if lo is None:
        sos = sps.butter(4, hi, btype="low", fs=fs, output="sos")
    elif hi is None:
        sos = sps.butter(4, lo, btype="high", fs=fs, output="sos")
    else:
        sos = sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    bn, an = sps.iirnotch(50.0, 30.0, fs=fs) if notch else (None, None)
    out = np.empty_like(x)
    for i in range(0, len(x), chunk):
        seg = sps.sosfiltfilt(sos, np.asarray(x[i:i + chunk], np.float64), axis=-1)
        if bn is not None:
            seg = sps.filtfilt(bn, an, seg, axis=-1)
        seg = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
        out[i:i + chunk] = seg.astype(np.float32)
    return out


def global_local(pred, d):
    k = d["hemo"].shape[1] // 2
    g, sids = strata(d), d["subject_ids"]
    res = {}
    for chrom, sl in (("HbO", slice(0, k)), ("HbR", slice(k, None))):
        T, P = d["hemo"][:, sl].astype(np.float64), pred[:, sl].astype(np.float64)
        Tg, Pg = T.mean(1, keepdims=True), P.mean(1, keepdims=True)
        for part, (pp, tt) in (("global", (Pg, Tg)), ("local", (P - Pg, T - Tg))):
            st = subject_level_test(demean_by_subject(pp, g), demean_by_subject(tt, g), sids)
            res[f"{chrom}_{part}"] = {k_: st[k_] for k_ in ("r_subject", "ci_low", "ci_high", "p_signflip")
                                      if k_ in st}
    return res


# T1/T2 only use the two training sets: free the external cohorts (~9 GB of EEG) and envelopes
import psutil
_ext = {id(DATA[k]) for k in ("external_fnirs", "external_fmri") if k in DATA}
for _n, _v in list(globals().items()):
    if isinstance(_v, dict) and id(_v) in _ext:
        globals()[_n] = None
    elif isinstance(_v, list) and any(id(x) in _ext for x in _v):
        globals()[_n] = [x for x in _v if id(x) not in _ext]
for k in ("external_fnirs", "external_fmri"):
    DATA.pop(k, None)
ENV_CACHE.clear()
gc.collect()
print(f"kernel RSS after freeing externals: {psutil.Process().memory_info().rss / 1e9:.1f} GB", flush=True)

hbci = TRAIN_SETS[0]
others = TRAIN_SETS[1:]
raw = {id(d): d["eeg"] for d in TRAIN_SETS}
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
for name, lo, hi, notch in BANDS:
    if name in results:
        print("skip", name, results[name]["r_subject"]); continue
    t0 = time.time()
    sets = [dict(d, eeg=band_copy(raw[id(d)], CFG.eeg_fs, lo, hi, notch)) for d in TRAIN_SETS]
    print(f"\n===== band {name}", flush=True)
    tbl, st = cross_subject_cv(sets)
    gl = global_local(OOF[hbci["name"]]["pred"], hbci)
    results[name] = {"r_subject": st["r_subject"], "ci_low": st.get("ci_low"), "ci_high": st.get("ci_high"),
                     "p_signflip": st.get("p_signflip"), "r_trial_pooled": st["r_trial_pooled"],
                     "global_local": gl, "minutes": (time.time() - t0) / 60}
    print(f"  {name}: {fmt(st)}")
    for k_, v in gl.items():
        print(f"     {k_:11s} r {v['r_subject']:+.4f} [{v.get('ci_low', float('nan')):+.4f}, {v.get('ci_high', float('nan')):+.4f}]")
    RESULTS.write_text(json.dumps(results, indent=1, default=float))
    del sets; gc.collect(); torch.cuda.empty_cache()
print("DONE")
'''.replace("__HERE__", str(HERE))
nb.cells.append(nbformat.v4.new_code_cell(ANALYSIS))

t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3",
                        resources={"metadata": {"path": str(SRC.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, OUT)
    print(f"saved {OUT} after {(time.time() - t0) / 60:.1f} min")
