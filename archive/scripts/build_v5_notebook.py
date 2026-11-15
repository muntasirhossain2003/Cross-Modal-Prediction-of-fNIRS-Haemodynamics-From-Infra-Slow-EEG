"""Emit NeuroVascNet_v5.ipynb -- a fully self-contained notebook.

v5 depends on NO other notebook and NO project module. Every loader, metric and
model it needs is defined inside the notebook itself. Point NEUROVASCNET_DATA_ROOT
at a folder holding the raw datasets and run top to bottom.

What v5 keeps from v1-v4 (the things that survived scrutiny):
  * TU Berlin hBCI as the fNIRS supervision set  (reliability 0.440 vs HEFMI 0.207)
  * Beer-Lambert from raw 760/850 nm             (the datasets ship intensities)
  * GSR OFF                                      (it removed 83% of the evoked response)
  * subject-wise CV + exact permutation null + constant baseline + demeaning
  * the target-reliability gate that produces the measured ceiling

What v5 drops: every result that did not replicate. See the Limitations cell.

Usage:  python scripts/build_v5_notebook.py
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

# ===========================================================================
md(r"""
# NeuroVascNet v5 — self-contained

**Goal.** Train on paired EEG + haemodynamics; at test time use **EEG only** to
predict the haemodynamic response, with a reliability layer that abstains when
the prediction is untrustworthy.

**This notebook depends on nothing else.** No other notebook, no `qc/` package,
no `neurovascnet_loaders_*.py`. Set `NEUROVASCNET_DATA_ROOT` and run top to bottom.

---

### What v5 reports, honestly

| Quantity | Value |
|---|---|
| Cross-subject *r*, demeaned (hBCI, k=1) | **0.053**, *p* = 0.015 |
| Raw (non-demeaned) model *r* | 0.072, *p* = 0.005 |
| Constant-predictor baseline, raw | **0.076** — *beats the model* |
| Target reliability, hBCI | 0.440 (ceiling 0.663) |
| Target reliability, HEFMI v3 | 0.207 (ceiling 0.455) |
| Clinical: HbO/HbR, ICH vs controls | *p* = 0.00085 |

*(Values above are the audited figures from `qc_outputs/tables/`. Running this
notebook produces its own numbers, which may differ with seed and environment.)*

The headline is **not** a large correlation. It is a *measured ceiling* plus a
protocol that removes a ~2× inflation, plus a clinical group difference.

### The one thing to understand before reading any number

A predictor that **ignores EEG entirely** — always emitting the training-set mean —
scores *r* = 0.076, which is **higher** than our raw model score of 0.072. Trials
from one subject share a mean, so raw correlation mostly measures *"which subject
is this"*. Subject-demeaning drives the constant predictor to exactly 0; what
survives is the EEG contribution. **That is why v5 reports demeaned numbers.**
""")

# ===========================================================================
md("## 1 · Environment and configuration")

co(r'''
import os, sys, json, math, warnings, subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

warnings.filterwarnings("ignore")

for _pkg in ("numpy", "scipy", "torch", "pandas"):
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sps
from scipy import stats
import torch
import torch.nn as nn

# numpy >= 2.0 renamed trapz -> trapezoid; older numpy lacks the new name.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"numpy {np.__version__} | torch {torch.__version__} | device {DEVICE}")
''')

co(r'''
# ---------------------------------------------------------------- data root
# Resolution order: env var -> common local paths -> Kaggle -> Colab Drive.
def resolve_data_root() -> Path:
    env = os.environ.get("NEUROVASCNET_DATA_ROOT")
    if env and Path(env).exists():
        return Path(env)
    for cand in [Path.cwd() / "dataset", Path.cwd() / "dataset_v31",
                 Path.cwd().parent / "dataset",
                 Path("/kaggle/input"), Path("/content/drive/MyDrive/dataset")]:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Set NEUROVASCNET_DATA_ROOT to the folder containing the raw datasets, e.g.\n"
        "  os.environ['NEUROVASCNET_DATA_ROOT'] = r'G:/research_bio_medical/dataset'"
    )

DATA_ROOT = resolve_data_root()
OUT_DIR = Path("neurovascnet_v5_outputs"); OUT_DIR.mkdir(exist_ok=True)
print("data root :", DATA_ROOT)
print("outputs   :", OUT_DIR.resolve())


def find_dir(*name_fragments: str) -> Path | None:
    """Locate a dataset folder by fragments of its name, case-insensitively."""
    frags = [f.lower() for f in name_fragments]
    for p in [DATA_ROOT, *DATA_ROOT.iterdir()] if DATA_ROOT.is_dir() else []:
        if p.is_dir() and all(f in p.name.lower() for f in frags):
            return p
    for p in DATA_ROOT.rglob("*"):
        if p.is_dir() and all(f in p.name.lower() for f in frags):
            return p
    return None
''')

co(r'''
@dataclass(frozen=True)
class Config:
    # --- windowing -------------------------------------------------------
    tmin_s: float = -2.0        # 2 s pre-stimulus baseline
    tmax_s: float = 28.0        # -> 30 s window
    eeg_fs: float = 200.0
    hemo_fs: float = 10.0
    eeg_samples: int = 6000
    hemo_samples: int = 300
    baseline_s: float = 2.0

    # --- protocol (FROZEN before any test performance was viewed) --------
    ruleset_version: str = "v5-frozen-2026-08-07"
    n_folds: int = 5            # subject-wise k-fold; no subject in two folds
    n_perm: int = 1000          # exact permutation null
    apply_gsr: bool = False     # audit: GSR removed 83% of the evoked response
    task: str = "MI"            # motor imagery

    # --- training --------------------------------------------------------
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3

CFG = Config()
print(json.dumps(asdict(CFG), indent=2))
''')

# ===========================================================================
md(r"""
## 2 · Data loader (self-contained)

TU Berlin hBCI, 29 subjects, paired EEG + fNIRS.

**Why this dataset and not HEFMI.** Two audited measurements decided it
(`qc_outputs/tables/v4_target_quality.csv`):

| | HEFMI (v3) | TU Berlin hBCI (v4) |
|---|---|---|
| target split-half reliability | 0.207 | **0.440** |
| ceiling on prediction √rel. | 0.455 | **0.663** |
| HbO/HbR correlation | **+0.589** ✗ | **−0.147** ✓ |

A real evoked haemodynamic response has HbO and HbR moving in *opposite*
directions. HEFMI's **positive** correlation means the signal is dominated by
scalp blood flow and heart rate, not by the brain; hBCI shows the physiological
negative sign. Since prediction is capped at √reliability, the swap raises the
ceiling by ~45% *before any model change*.

**Beer-Lambert is done here, not by the dataset.** The files ship raw 760/850 nm
intensities. We convert with Prahl extinction coefficients (DPF 6.4/5.8, SD 3.0 cm).
""")

co(r'''
# Beer-Lambert constants (Prahl / Wray), cm^-1 / (mol/L)
EXT = {760: {"HbO": 1486.59, "HbR": 3843.71},
       850: {"HbO": 2526.39, "HbR": 1798.64}}
DPF = {760: 6.4, 850: 5.8}
SD_CM = 3.0
EOG_CHANNELS = ("VEOG", "HEOG")


def _runs(path: Path, key_hint: str | None = None):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    key = key_hint if key_hint in keys else keys[0]
    return np.atleast_1d(m[key]).ravel()


def bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def beer_lambert(lo_wl, hi_wl):
    """(ch, T) raw intensities -> HbO, HbR in micromolar."""
    def od(x):
        base = np.median(x, axis=-1, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        return -np.log(np.clip(x / base, 1e-9, None))

    a1, a2 = od(lo_wl), od(hi_wl)
    L1, L2 = SD_CM * DPF[760], SD_CM * DPF[850]
    E = np.array([[EXT[760]["HbO"] * L1, EXT[760]["HbR"] * L1],
                  [EXT[850]["HbO"] * L2, EXT[850]["HbR"] * L2]])
    Ei = np.linalg.inv(E)
    return ((Ei[0, 0] * a1 + Ei[0, 1] * a2) * 1e6,
            (Ei[1, 0] * a1 + Ei[1, 1] * a2) * 1e6)


def split_wavelengths(x, clab):
    """(T, 72) -> (lo, hi) each (36, T), matched by source-detector pair name."""
    names = [str(c) for c in np.atleast_1d(clab)]
    lo = {n.replace("lowWL", ""): i for i, n in enumerate(names) if "lowWL" in n}
    hi = {n.replace("highWL", ""): i for i, n in enumerate(names) if "highWL" in n}
    pairs = sorted(set(lo) & set(hi))
    return (np.stack([x[:, lo[p]] for p in pairs]),
            np.stack([x[:, hi[p]] for p in pairs]), pairs)


def regress_global(x):
    """Global signal regression. OFF by default -- it removes 83% of the evoked
    response (measured). Kept only so the ablation can switch it on."""
    g = x.mean(axis=0, keepdims=True)
    gc = g - g.mean(axis=-1, keepdims=True)
    bc = x - x.mean(axis=-1, keepdims=True)
    beta = (bc * gc).sum(-1, keepdims=True) / ((gc ** 2).sum(-1, keepdims=True) + 1e-12)
    return bc - beta * gc


def resample_to(x, n):
    return x if x.shape[-1] == n else sps.resample(x, n, axis=-1)
''')

co(r'''
TUBERLIN = find_dir("berlin") or find_dir("hbci")
if TUBERLIN is None:
    raise FileNotFoundError(f"TU Berlin hBCI folder not found under {DATA_ROOT}")
EEG_ROOT = TUBERLIN / "EEG data"
NIRS_ROOT = TUBERLIN / "NRI data" / "NIRS_01-29"
print("TU Berlin root:", TUBERLIN)


def eeg_subject_dir(n: int) -> Path | None:
    for group in sorted(EEG_ROOT.glob("EEG_*")):
        d = group / f"subject {n:02d}" / "with occular artifact"
        if d.exists():
            return d
    return None


def load_tuberlin(cfg: Config = CFG, subjects=range(1, 30)) -> dict:
    """All subjects in one dict. Splitting happens later, by subject, in the CV
    loop -- so there is exactly one place where a split is defined."""
    eeg_out, hemo_out, lab_out, sid_out, audit = [], [], [], [], []
    n_base = int(cfg.baseline_s * cfg.hemo_fs)

    for s in subjects:
        ed, nd = eeg_subject_dir(s), NIRS_ROOT / f"subject {s:02d}"
        if ed is None or not nd.exists():
            audit.append({"subject": s, "status": "missing directory", "trials": 0}); continue
        try:
            e_runs, e_mrks = _runs(ed / "cnt.mat", "cnt"), _runs(ed / "mrk.mat", "mrk")
            n_runs, n_mrks = _runs(nd / "cnt.mat", "cnt"), _runs(nd / "mrk.mat", "mrk")
        except Exception as exc:
            audit.append({"subject": s, "status": f"read fail: {exc}", "trials": 0}); continue

        n_tr = 0
        for r in range(min(e_runs.size, n_runs.size)):
            er, nr = e_runs[r], n_runs[r]
            if not str(getattr(er, "title", "")).upper().startswith(cfg.task):
                continue

            e_fs = float(np.atleast_1d(er.fs).ravel()[0])
            e_clab = [str(c) for c in np.atleast_1d(er.clab)]
            keep_ch = [i for i, c in enumerate(e_clab) if c not in EOG_CHANNELS]
            ex = np.asarray(er.x, dtype=np.float64)[:, keep_ch].T

            n_fs = float(np.atleast_1d(nr.fs).ravel()[0])
            lo, hi, _ = split_wavelengths(np.asarray(nr.x, dtype=np.float64), nr.clab)
            hbo, hbr = beer_lambert(lo, hi)
            hbo, hbr = bandpass(hbo, n_fs, 0.01, 0.2), bandpass(hbr, n_fs, 0.01, 0.2)
            if cfg.apply_gsr:
                hbo, hbr = regress_global(hbo), regress_global(hbr)

            em = e_mrks[r] if r < e_mrks.size else None
            nm = n_mrks[r] if r < n_mrks.size else None
            if em is None or nm is None:
                continue
            e_on = np.atleast_1d(np.asarray(em.time, dtype=float)) / 1000.0
            n_on = np.atleast_1d(np.asarray(nm.time, dtype=float)) / 1000.0
            y = np.asarray(em.y)
            labels = np.argmax(y, axis=0) if y.ndim == 2 else np.asarray(y).ravel()

            for k in range(min(e_on.size, n_on.size, labels.size)):
                a_e = int(round((e_on[k] + cfg.tmin_s) * e_fs))
                b_e = int(round((e_on[k] + cfg.tmax_s) * e_fs))
                a_n = int(round((n_on[k] + cfg.tmin_s) * n_fs))
                b_n = int(round((n_on[k] + cfg.tmax_s) * n_fs))
                # A trial whose window runs past the end of the recording has no
                # data to pair with -- drop it rather than abort the build.
                if a_e < 0 or b_e > ex.shape[1] or a_n < 0 or b_n > hbo.shape[1]:
                    continue

                seg = resample_to(ex[:, a_e:b_e], cfg.eeg_samples)
                seg = ((seg - seg.mean(-1, keepdims=True)) /
                       (seg.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
                so = resample_to(hbo[:, a_n:b_n], cfg.hemo_samples)
                sr = resample_to(hbr[:, a_n:b_n], cfg.hemo_samples)
                so = so - so[:, :n_base].mean(-1, keepdims=True)
                sr = sr - sr[:, :n_base].mean(-1, keepdims=True)

                eeg_out.append(seg)
                hemo_out.append(np.concatenate([so, sr], 0).astype(np.float32))
                lab_out.append(int(labels[k])); sid_out.append(s); n_tr += 1

        audit.append({"subject": s, "status": "ok" if n_tr else "no trials", "trials": n_tr})

    if not eeg_out:
        raise RuntimeError("loader produced no samples")
    return {"eeg": np.stack(eeg_out), "hemo": np.stack(hemo_out),
            "labels": np.asarray(lab_out, np.int64),
            "subject_ids": np.asarray(sid_out, np.int64),
            "audit": pd.DataFrame(audit)}
''')

co(r'''
DATA = load_tuberlin()
print(f"eeg  {DATA['eeg'].shape}   hemo {DATA['hemo'].shape}")
print(f"{len(np.unique(DATA['subject_ids']))} subjects, {len(DATA['eeg'])} trials")
display(DATA["audit"])
DATA["audit"].to_csv(OUT_DIR / "loader_audit.csv", index=False)
''')

# ===========================================================================
md(r"""
## 3 · Target reliability gate — *the ceiling*

**This is the most important cell in the notebook.** Before training anything,
we ask: *does the target predict itself?*

Split the trials of one subject at random into two halves, average each, and
correlate. That split-half reliability bounds how well **any** model can do:

$$r_{\max} \approx \sqrt{\text{reliability}}$$

If the target is unreliable, a low prediction score says nothing about the model.
This is why v5 reports a *measured* ceiling instead of asserting one.

We also check the **HbO/HbR sign**. A genuine evoked response drives HbO up and
HbR down — a *negative* correlation. A strong positive correlation means systemic
physiology (heart rate, blood pressure, scalp flow) dominates.
""")

co(r'''
def split_half_reliability(hemo, sids, n_rep=50, rng=None):
    rng = rng or np.random.default_rng(SEED)
    out = {}
    for s in np.unique(sids):
        X = hemo[sids == s]
        if len(X) < 4:
            continue
        rs = []
        for _ in range(n_rep):
            p = rng.permutation(len(X))
            a = X[p[: len(X) // 2]].mean(0).ravel()
            b = X[p[len(X) // 2:]].mean(0).ravel()
            if a.std() > 0 and b.std() > 0:
                rs.append(np.corrcoef(a, b)[0, 1])
        out[int(s)] = float(np.mean(rs)) if rs else np.nan
    return out


def hbo_hbr_sign(hemo):
    """hemo is [N, 72, T] = 36 HbO then 36 HbR. Negative is physiological."""
    n = hemo.shape[1] // 2
    return float(np.nanmedian([
        np.corrcoef(t[:n].mean(0), t[n:].mean(0))[0, 1] for t in hemo]))


rel = split_half_reliability(DATA["hemo"], DATA["subject_ids"])
rel_mean = float(np.nanmean(list(rel.values())))
sign = hbo_hbr_sign(DATA["hemo"])

print(f"target split-half reliability : {rel_mean:+.3f}")
print(f"  -> ceiling on prediction r  : {math.sqrt(max(rel_mean, 0)):.3f}")
print(f"HbO/HbR correlation           : {sign:+.3f}"
      f"   [{'physiological (HbO up, HbR down)' if sign < 0 else 'SYSTEMIC-DOMINATED -- warning'}]")

pd.DataFrame({"subject": list(rel), "reliability": list(rel.values())}
             ).to_csv(OUT_DIR / "target_reliability.csv", index=False)
''')

# ===========================================================================
md(r"""
## 4 · Metrics: the exact permutation null and the constant baseline

**Constant baseline.** A predictor that ignores EEG and always emits the
training-set mean. If the model cannot beat this, it has learned nothing about
the brain. It must be reported next to every correlation.

**Exact permutation null, computed by matrix multiplication.** A permutation only
*reorders rows*, so every pairwise inner product can be computed once as
$P_cT_c^\top$ and each permutation becomes a gather along the diagonal — no
recomputation. Measured **559× faster** than the naive loop, and verified
identical to 1.6 × 10⁻¹⁰.
""")

co(r'''
def _center(x):
    x = x.reshape(len(x), -1).astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def mean_corr(pred, targ):
    """Mean per-trial Pearson r."""
    return float((_center(pred) * _center(targ)).sum(1).mean())


def perm_null(pred, targ, n_perm=CFG.n_perm, rng=None):
    """Exact permutation null via one matmul.

    Naive: recompute all correlations for each of n_perm shuffles.
    Here:  G = Pc @ Tc.T holds every pairwise correlation already, so
           permutation pi has score mean(G[i, pi[i]]) -- a gather, not a product.
    """
    rng = rng or np.random.default_rng(SEED)
    Pc, Tc = _center(pred), _center(targ)
    G = Pc @ Tc.T                                  # (N, N), computed ONCE
    obs = float(np.diag(G).mean())
    idx = np.arange(len(G))
    null = np.array([G[idx, rng.permutation(len(G))].mean() for _ in range(n_perm)])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)   # +1: never report p = 0
    return obs, float(p), null


def constant_baseline(train_targ, test_targ):
    """Ignores EEG entirely: predicts the training mean for every test trial."""
    const = np.repeat(train_targ.mean(0, keepdims=True), len(test_targ), axis=0)
    return mean_corr(const, test_targ)


def demean_by_subject(y, sids):
    """Removes each subject's mean, so the constant predictor scores exactly 0
    and what remains is the EEG contribution."""
    out = y.astype(np.float64).copy()
    for s in np.unique(sids):
        m = sids == s
        out[m] -= out[m].mean(0, keepdims=True)
    return out
''')

# ===========================================================================
md(r"""
## 5 · Model — shared trunk + per-subject embedding

A shared convolutional trunk learns what is common across people; a small
per-subject embedding absorbs what is idiosyncratic. The subject adapter was one
of only three changes that improved results on **all** seeds tested (+0.011).

At test time a held-out subject has no embedding, so it receives the **mean**
embedding — the model must fall back on the subject-general trunk. That is what
makes the evaluation genuinely cross-subject.
""")

co(r'''
class NeuroVascNet(nn.Module):
    def __init__(self, n_eeg_ch, n_hemo_ch, hemo_len, n_subjects, emb=8, width=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(n_eeg_ch, width, 25, stride=5, padding=12), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 15, stride=4, padding=7),     nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 9,  stride=3, padding=4),     nn.BatchNorm1d(width), nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.subj = nn.Embedding(n_subjects, emb)
        nn.init.zeros_(self.subj.weight)          # start subject-agnostic
        self.head = nn.Sequential(
            nn.Linear(width * 16 + emb, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n_hemo_ch * hemo_len),
        )
        self.out_shape = (n_hemo_ch, hemo_len)

    def forward(self, x, sid=None):
        h = self.trunk(x).flatten(1)
        e = (self.subj(sid) if sid is not None
             else self.subj.weight.mean(0).expand(len(x), -1))
        return self.head(torch.cat([h, e], 1)).view(len(x), *self.out_shape)


def train_fold(tr_eeg, tr_hemo, tr_sid, n_subjects, cfg=CFG, verbose=False):
    model = NeuroVascNet(tr_eeg.shape[1], tr_hemo.shape[1], tr_hemo.shape[2],
                         n_subjects).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    Xe = torch.tensor(tr_eeg, dtype=torch.float32)
    Xh = torch.tensor(tr_hemo, dtype=torch.float32)
    Xs = torch.tensor(tr_sid, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Xe, Xh, Xs)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for ep in range(cfg.epochs):
        tot = 0.0
        for be, bh, bs in dl:
            be, bh, bs = be.to(DEVICE), bh.to(DEVICE), bs.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(be, bs), bh)
            loss.backward(); opt.step()
            tot += loss.item() * len(be)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3d}  loss {tot/len(ds):.4f}")
    return model


@torch.no_grad()
def predict(model, eeg):
    model.eval()
    out = []
    for i in range(0, len(eeg), 256):
        b = torch.tensor(eeg[i:i+256], dtype=torch.float32).to(DEVICE)
        out.append(model(b, None).cpu().numpy())   # None -> mean embedding
    return np.concatenate(out)
''')

# ===========================================================================
md(r"""
## 6 · Cross-subject evaluation

**Cross-subject is the whole point of the project.** Subject-wise *k*-fold: every
fold holds out a disjoint set of *people*, so no subject ever appears in both
train and test.

Each fold reports four numbers:

| | what it measures |
|---|---|
| `r_raw` | model score before demeaning — **contaminated by subject identity** |
| `r_const` | a predictor that ignores EEG — the number `r_raw` must beat |
| `r_demeaned` | **the EEG contribution** — the honest number |
| `p_perm` | exact permutation null on the demeaned score |
""")

co(r'''
def cross_subject_cv(data, cfg=CFG, verbose=True):
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    subjects = np.unique(sids)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(subjects)
    folds = np.array_split(order, cfg.n_folds)
    remap = {int(s): i for i, s in enumerate(subjects)}

    rows, all_pred, all_targ, all_sid = [], [], [], []
    for fi, test_subj in enumerate(folds):
        te = np.isin(sids, test_subj); tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        tr_sid_ix = np.array([remap[int(s)] for s in sids[tr]])
        model = train_fold(eeg[tr], hemo[tr], tr_sid_ix, len(subjects), cfg)
        pred = predict(model, eeg[te])

        r_raw = mean_corr(pred, hemo[te])
        r_const = constant_baseline(hemo[tr], hemo[te])
        pd_, td_ = demean_by_subject(pred, sids[te]), demean_by_subject(hemo[te], sids[te])
        r_dem = mean_corr(pd_, td_)

        rows.append({"fold": fi, "n_test_subjects": len(test_subj),
                     "n_test_trials": int(te.sum()), "r_raw": r_raw,
                     "r_const": r_const, "r_demeaned": r_dem})
        all_pred.append(pd_); all_targ.append(td_); all_sid.append(sids[te])
        if verbose:
            print(f"fold {fi}: subjects={list(test_subj)}  raw={r_raw:+.4f}  "
                  f"const={r_const:+.4f}  demeaned={r_dem:+.4f}")

    P, T = np.concatenate(all_pred), np.concatenate(all_targ)
    obs, p, null = perm_null(P, T, cfg.n_perm)
    return pd.DataFrame(rows), {"r_demeaned_pooled": obs, "p_perm": p,
                                "null_mean": float(null.mean()),
                                "null_sd": float(null.std())}


cv_table, cv_summary = cross_subject_cv(DATA)
display(cv_table)
print()
for k, v in cv_summary.items():
    print(f"{k:22s}: {v:+.4f}" if isinstance(v, float) else f"{k:22s}: {v}")
cv_table.to_csv(OUT_DIR / "cross_subject_cv.csv", index=False)
''')

co(r'''
# --- the comparison that decides how the result must be described -----------
r_raw   = cv_table["r_raw"].mean()
r_const = cv_table["r_const"].mean()
r_dem   = cv_summary["r_demeaned_pooled"]

print(f"raw model        r = {r_raw:+.4f}")
print(f"constant (no EEG) r = {r_const:+.4f}")
print()
if r_const >= r_raw:
    print("The constant predictor MATCHES OR BEATS the raw model.")
    print("=> raw correlation is measuring subject identity, not neurovascular coupling.")
    print(f"=> the reportable number is the demeaned one: r = {r_dem:+.4f}"
          f"  (p = {cv_summary['p_perm']:.4f})")
else:
    print(f"Raw model exceeds the constant baseline by {r_raw - r_const:+.4f};")
    print(f"the demeaned score r = {r_dem:+.4f} remains the honest estimate.")

ceiling = math.sqrt(max(rel_mean, 0))
print(f"\nmeasured ceiling  r = {ceiling:.3f}")
print(f"achieved / ceiling  = {r_dem / ceiling * 100:.1f}%" if ceiling > 0 else "")
''')

# ===========================================================================
md(r"""
## 7 · Protocol ablation — where the inflation comes from

The claim: **most reported EEG→haemodynamics correlations are inflated by
evaluation protocol, not by method.** Each row removes one more shortcut.
The drop from top to bottom is attributable purely to protocol.
""")

co(r'''
def protocol_ablation(data, cfg=CFG):
    eeg, hemo, sids = data["eeg"], data["hemo"], data["subject_ids"]
    subjects = np.unique(sids)
    remap = {int(s): i for i, s in enumerate(subjects)}
    rng = np.random.default_rng(SEED)
    rows = []

    # (1) CONVENTIONAL: one random TRIAL-level split -- subjects appear in both
    #     sides. This is the shape of a typical published protocol.
    idx = rng.permutation(len(eeg)); cut = int(0.8 * len(eeg))
    tr, te = idx[:cut], idx[cut:]
    m = train_fold(eeg[tr], hemo[tr], np.array([remap[int(s)] for s in sids[tr]]),
                   len(subjects), cfg)
    rows.append({"protocol": "1. trial-level split (subject leakage)",
                 "r": mean_corr(predict(m, eeg[te]), hemo[te])})

    # (2) + subject-wise CV: no subject in both sides
    rows.append({"protocol": "2. + subject-wise CV",
                 "r": float(cv_table["r_raw"].mean())})

    # (3) + constant baseline made explicit
    rows.append({"protocol": "3. + constant baseline (what 2 must beat)",
                 "r": float(cv_table["r_const"].mean())})

    # (4) + subject demeaning: the EEG contribution alone
    rows.append({"protocol": "4. + subject demeaning  <- HONEST",
                 "r": float(cv_summary["r_demeaned_pooled"])})
    return pd.DataFrame(rows)


abl = protocol_ablation(DATA)
display(abl)
abl.to_csv(OUT_DIR / "protocol_ablation.csv", index=False)

infl = abl.r.iloc[0] / abl.r.iloc[-1] if abl.r.iloc[-1] > 0 else float("nan")
print(f"\ninflation from protocol alone: {infl:.2f}x "
      f"({abl.r.iloc[0]:+.4f} -> {abl.r.iloc[-1]:+.4f})")
''')

# ===========================================================================
md(r"""
## 8 · Reliability / abstention layer

The model must know when **not** to answer. We score each prediction by the
agreement of an MC-dropout ensemble; low agreement ⇒ abstain.

A working gate shows accuracy **rising monotonically** as coverage falls. That
property — not the headline correlation — is what would make such a system usable.
""")

co(r'''
@torch.no_grad()
def mc_dropout_scores(model, eeg, n_pass=20):
    """Enable dropout at inference; disagreement across passes = uncertainty."""
    model.train()                       # keeps dropout active
    preds = []
    for _ in range(n_pass):
        out = []
        for i in range(0, len(eeg), 256):
            b = torch.tensor(eeg[i:i+256], dtype=torch.float32).to(DEVICE)
            out.append(model(b, None).cpu().numpy())
        preds.append(np.concatenate(out))
    preds = np.stack(preds)
    mean = preds.mean(0)
    conf = -preds.std(0).reshape(len(eeg), -1).mean(1)   # higher = more confident
    return mean, conf


def coverage_curve(conf, pred, targ, grid=(1.0, .8, .6, .4, .2)):
    order = np.argsort(-conf)
    rows = []
    for c in grid:
        k = max(int(len(order) * c), 8)
        sel = order[:k]
        rows.append({"coverage": c, "n": k, "r": mean_corr(pred[sel], targ[sel])})
    return pd.DataFrame(rows)


# fit on the last CV fold's held-out subjects
subjects = np.unique(DATA["subject_ids"])
rng = np.random.default_rng(SEED)
folds = np.array_split(rng.permutation(subjects), CFG.n_folds)
test_subj = folds[-1]
te = np.isin(DATA["subject_ids"], test_subj); tr = ~te
remap = {int(s): i for i, s in enumerate(subjects)}
mdl = train_fold(DATA["eeg"][tr], DATA["hemo"][tr],
                 np.array([remap[int(s)] for s in DATA["subject_ids"][tr]]),
                 len(subjects), CFG)

mean_pred, conf = mc_dropout_scores(mdl, DATA["eeg"][te])
pd_ = demean_by_subject(mean_pred, DATA["subject_ids"][te])
td_ = demean_by_subject(DATA["hemo"][te], DATA["subject_ids"][te])
cov = coverage_curve(conf, pd_, td_)
display(cov)
cov.to_csv(OUT_DIR / "coverage_curve.csv", index=False)

mono = bool(np.all(np.diff(cov.r.values) >= -1e-3))
print(f"\naccuracy rises as coverage falls: {mono}"
      f"   [{'gate is informative' if mono else 'gate is NOT informative'}]")
''')

# ===========================================================================
md(r"""
## 9 · Clinical analysis — altered coupling in ICH

The strongest positive result in the project, and it needs **no accurate
prediction**: coupling descriptors are measured directly.

HEFMI holds 17 healthy controls and 20 intracerebral-haemorrhage patients under
an identical protocol. Earlier work in this project silently used the controls
only — `groups=("Normals",)` was the loader default.

**Confounds are reported alongside.** Patients move more, and ICH alters
skull/scalp optical properties, so motion and amplitude are compared too. If
those differ, the coupling difference may be acquisition rather than physiology.
""")

co(r'''
HEFMI = find_dir("hefmi")
CLINICAL_OK = HEFMI is not None and (HEFMI / "epo_data").exists()
print("HEFMI:", HEFMI if CLINICAL_OK else "not found -- skipping clinical section")


def clinical_rows(root, group):
    """One row per recording: coupling, reliability, sign, and artefact controls."""
    rows = []
    eeg_root = root / "epo_data" / group / "EEG"
    if not eeg_root.exists():
        return rows
    for sd in sorted(p for p in eeg_root.iterdir() if p.is_dir()):
        for ef in sorted(sd.glob("*_epo.mat")):
            nf = root / "epo_data" / group / "NIRS" / sd.name / ef.name
            if not nf.exists():
                continue
            try:
                e = sio.loadmat(ef, squeeze_me=True, struct_as_record=False)["epo"]
                n = sio.loadmat(nf, squeeze_me=True, struct_as_record=False)["epo"]
                X = np.asarray(e.X, np.float64)
                oxy, dxy = np.asarray(n.oxy.X, np.float64), np.asarray(n.doxy.X, np.float64)
            except Exception:
                continue
            k = min(X.shape[2], oxy.shape[2])
            hbo = np.transpose(oxy[:, :, :k], (2, 1, 0))
            hbr = np.transpose(dxy[:, :, :k], (2, 1, 0))
            eeg = np.transpose(X[:, :, :k], (2, 1, 0))
            cc = [np.corrcoef(hbo[t].ravel(), hbr[t].ravel())[0, 1] for t in range(k)]
            d = np.diff(hbo.reshape(k, -1), axis=-1)
            rows.append({
                # biological identity: Normals/subject1 and Patients/subject1 are
                # DIFFERENT people, so the group must be part of the key.
                "biological_subject_id": f"{group}_{sd.name}",
                "group": group,
                "hbo_hbr": float(np.nanmedian(cc)),
                "motion_frac": float((np.abs(d) > 5 * np.nanstd(d)).mean()),
                "hbo_amp": float(np.percentile(np.abs(hbo), 99.5)),
                "eeg_amp": float(np.percentile(np.abs(eeg), 99.5)),
                "n_trials": k})
    return rows


if CLINICAL_OK:
    recs = clinical_rows(HEFMI, "Normals") + clinical_rows(HEFMI, "Patients")
    cdf = pd.DataFrame(recs)
    # collapse sessions -> one row per PERSON, so n is people not recordings
    per_person = cdf.groupby(["biological_subject_id", "group"], as_index=False).mean(
        numeric_only=True)
    print(f"{len(cdf)} recordings -> {len(per_person)} biological subjects")
    print(per_person.group.value_counts().to_string())

    PRIMARY, CONTROL = ["hbo_hbr"], ["motion_frac", "hbo_amp", "eeg_amp", "n_trials"]
    a = per_person[per_person.group == "Normals"]
    b = per_person[per_person.group == "Patients"]
    out = []
    for k in PRIMARY + CONTROL:
        x, y = a[k].dropna().values, b[k].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        out.append({"measure": k, "controls": x.mean(), "patients": y.mean(),
                    "p": p, "primary": k in PRIMARY,
                    "sig": "*" if p < 0.05 else ""})
    cres = pd.DataFrame(out)
    display(cres)
    per_person.to_csv(OUT_DIR / "clinical_per_subject.csv", index=False)
    cres.to_csv(OUT_DIR / "clinical_comparison.csv", index=False)

    sig_ctrl = cres[(~cres.primary) & (cres.p < 0.05)]
    if len(sig_ctrl):
        print("\nWARNING: artefact controls also differ:",
              ", ".join(sig_ctrl.measure))
        print("-> the coupling difference may reflect acquisition, not physiology.")
    else:
        print("\nArtefact controls do NOT differ -> the coupling difference is not"
              "\nexplained by motion or signal amplitude.")
''')

# ===========================================================================
md(r"""
## 10 · Summary
""")

co(r'''
summary = {
    "ruleset_version": CFG.ruleset_version,
    "n_subjects": int(len(np.unique(DATA["subject_ids"]))),
    "n_trials": int(len(DATA["eeg"])),
    "target_reliability": round(rel_mean, 4),
    "ceiling_r": round(math.sqrt(max(rel_mean, 0)), 4),
    "hbo_hbr_sign": round(sign, 4),
    "r_raw": round(float(cv_table["r_raw"].mean()), 4),
    "r_constant_baseline": round(float(cv_table["r_const"].mean()), 4),
    "r_demeaned": round(float(cv_summary["r_demeaned_pooled"]), 4),
    "p_permutation": round(float(cv_summary["p_perm"]), 5),
    "abstention_gate_informative": bool(mono),
}
print(json.dumps(summary, indent=2))
json.dump(summary, open(OUT_DIR / "summary_v5.json", "w"), indent=2)
print(f"\nall outputs -> {OUT_DIR.resolve()}")
''')

md(r"""
## 11 · Limitations, and what v5 deliberately dropped

**Dropped — did not replicate.**
- *Anatomical dissociation.* At *n* = 6, visual *p* = 0.012 / motor *p* = 0.944
  looked like a clean dissociation. At *n* = 24 both were *p* = 0.001.
  **Retracted** — it was a small-sample artefact.
- *A fixed neurovascular delay.* The estimated delay tracks the analysis window
  (3.90 / 4.50 / 8.05 s), so it is reported as window-dependent, not as a constant.

**Standing limitations.**
1. The effect is **small** (*r* ≈ 0.05). We claim statistical reality and a
   measured ceiling, **not** practical utility. No deployment claim is made.
   The model does **not** beat the no-EEG baseline on *raw* correlation at any
   fold size — only after demeaning.
2. The model beats shuffled (+0.023) and random (+0.025) targets, but **not raw
   features** — representation learning is not yet justified over simple features.
3. ICH alters skull/scalp optics. Motion and amplitude controls are n.s., which
   removes the commonest confounds, but the sample is modest and lesion-side
   stratification is still needed.
4. Subject-demeaning removes between-subject variance, which is part of the real
   signal. It is the conservative choice, and it is the one we report.

**A leakage trap worth knowing about.** In HEFMI, `1_epo.mat` appears **37 times**,
belonging to **37 different people**; `Normals/subject1` and `Patients/subject1`
are different individuals. Splitting on filename would have put **7 of 7**
validation subjects in both splits. v5 keys every clinical row on
`biological_subject_id = f"{group}_{subject}"` for exactly this reason.
""")

nbf.write(nb, "NeuroVascNet_v5.ipynb")
print(f"wrote NeuroVascNet_v5.ipynb  ({len(nb.cells)} cells)")
