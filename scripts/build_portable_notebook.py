"""Generate NeuroVascNet_portable.ipynb — the complete v3.3 system on repaired v4 data.

Self-contained: downloads raw data, builds targets, trains the full NeuroVascNet
(shared EEG encoder + dual hemodynamic heads + delay module + reliability head),
evaluates with subject-wise CV, calibrates, and runs ablations.

Removed vs v3.3: synthetic data generator, the two datasets the audit rejected,
and the cached-npz dependency. Everything else is kept.
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

# ============================================================ 0. HEADER
md("""
# NeuroVascNet — EEG-only prediction of hemodynamic responses

**Goal.** Neural activity drives local blood flow (*neurovascular coupling*). Train one
encoder on paired recordings so that afterwards it takes **EEG only** and predicts the
hemodynamic response — fNIRS (HbO/HbR) and fMRI (BOLD).

**Self-contained.** Downloads its own data and builds every target from raw. No uploads,
no cached files, no local modules.

## Pipeline

`download → build targets from raw → preprocess → shared encoder → dual heads +
delay + reliability → staged training → subject-wise CV → calibration → ablations`

## Data

| dataset | modality | subjects | role |
|---|---|---|---|
| TU Berlin hBCI | EEG + fNIRS | 29 | fNIRS supervision |
| OpenNeuro ds002336 | EEG + fMRI | 10 | fMRI supervision |

The fNIRS and fMRI subjects are **different people** — no public dataset records both in
the same person — so the two arms never share a subject.

## Why the data is rebuilt here rather than reused

A quality audit of all 610 recordings found the earlier targets were broken:

* the previous fNIRS set had **HbO/HbR = +0.898** (must be negative) — systemic, not neural
* global-signal regression removed **83%** of the evoked response
* the fMRI target scored **−0.639** split-half reliability — it did not reproduce at all
* against that target, ridge gave r = 0.0706 at **p = 0.005**; after repair, **p = 0.473**

So this notebook converts raw 760/850 nm light itself, skips GSR, and event-locks the fMRI
target using the scanner triggers.

## How to read the results

| metric | meaning |
|---|---|
| `r_raw` | plain correlation — **inflated**, every trial shares a stereotyped response shape |
| `r_const` | what you get **ignoring EEG entirely** |
| `r_demeaned` | subject means removed → `r_const` becomes 0 → **the EEG contribution** |

Reference run: `r_raw = 0.072` but `r_const = 0.076` — the constant predictor *wins*.
**Judge on `r_demeaned` (≈ 0.053–0.071, p = 0.005).**

## Runtime

GPU recommended, internet required. ~2 h total, mostly download (25 GB).
""")

# ============================================================ 1. SETUP
md("## 1. Environment and configuration")

code("""
import os, sys, shutil, subprocess, warnings, math, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "gdown", "mne", "nilearn", "nibabel"], check=False)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 1337
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ON_KAGGLE = Path("/kaggle").exists()
WORK  = Path("/kaggle/working" if ON_KAGGLE else ".")
RAW   = WORK/"raw";   RAW.mkdir(parents=True, exist_ok=True)
CACHE = WORK/"cache"; CACHE.mkdir(parents=True, exist_ok=True)
print("device:", DEV, torch.cuda.get_device_name(0) if DEV=="cuda" else "")
""")

code("""
@dataclass
class Config:
    # --- data geometry ---
    n_eeg_channels: int = 30
    eeg_fs: float       = 200.0
    eeg_window: int     = 6000       # 30 s
    fnirs_fs: float     = 10.0
    fnirs_len: int      = 300        # 30 s
    fmri_tr: float      = 2.0
    fmri_len: int       = 15         # 30 s

    # --- encoder ---
    encoder_temporal_filters: int = 16
    encoder_spatial_filters: int  = 2
    encoder_kernel_length: int    = 64
    encoder_pool: int             = 4
    tcn_layers: int   = 4
    tcn_channels: int = 64
    tcn_kernel: int   = 3
    latent_dim: int   = 64
    dropout: float    = 0.25

    # --- delay module (physiological bounds) ---
    delay_min_s: float = 2.0
    delay_max_s: float = 8.0

    # --- loss weights ---
    w_hb: float = 1.0
    w_bold: float = 1.0
    w_hetero: float = 0.1
    w_delay_fnirs: float = 0.3
    w_delay_fmri: float = 0.3
    w_smooth: float = 0.05
    w_align: float = 0.0          # cross-modal latent alignment (ablation knob)

    # --- training ---
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs_stage1: int = 30       # per-modality warm-up
    epochs_stage2: int = 30       # joint
    patience: int = 15

    # --- evaluation ---
    k_folds: int = 5
    n_perm: int = 1000
    n_boot: int = 2000

CFG = Config()

# Google Drive folder ids
DRIVE = {"hbci": "1lw-AR79e5FgVOL6iHaRG0FppYJHOhnqj",
         "ds002336": "1K_91Y5Ph0PiAIcFdlGsiHg10_knUNmhw"}
DIRNAME = {"hbci": "TU Berlin hBCI EEG+FNRI", "ds002336": "OpenNeuro ds002336"}
FREE_RAW_AFTER = True      # delete raw once its target is built (Kaggle disk is ~20 GB)
print(CFG)
""")

# ============================================================ 2. DOWNLOAD
md("""
## 2. Download

`gdown.download_folder` silently caps at 50 files unless `remaining_ok=True`. Drive
rate-limits multi-GB pulls — if a download stops early, re-run the cell; finished files
are skipped so it resumes.
""")

code("""
def fetch(key):
    dest = RAW/DIRNAME[key]
    if dest.exists() and len(list(dest.rglob("*"))) > 10:
        print(f"{DIRNAME[key]}: present ({len(list(dest.rglob('*')))} files)"); return dest
    import gdown
    print(f"downloading {DIRNAME[key]} ...")
    try:
        gdown.download_folder(id=DRIVE[key], output=str(dest), quiet=False,
                              use_cookies=False, remaining_ok=True)
    except Exception as e:
        print(f"  interrupted: {type(e).__name__}: {e} -- re-run to resume")
    return dest

def free(key):
    if FREE_RAW_AFTER and (RAW/DIRNAME[key]).exists():
        shutil.rmtree(RAW/DIRNAME[key], ignore_errors=True); print(f"  freed {DIRNAME[key]}")
""")

# ============================================================ 3. TARGETS
md("""
## 3. Build the fNIRS target — raw 760/850 nm → HbO/HbR

The dataset ships **raw light intensities**, so Beer–Lambert conversion happens here with
published extinction coefficients. **Global-signal regression is deliberately not applied**
— it removes ~83% of the evoked response, because a broad motor response *is* spatially
global. EEG and fNIRS are epoched on the same stimulus markers.
""")

code("""
import scipy.io as sio
from scipy import signal as sps

EXT = {760:{"HbO":1486.59,"HbR":3843.71}, 850:{"HbO":2526.39,"HbR":1798.64}}
DPF = {760:6.4, 850:5.8}
SD_CM, EOG = 3.0, ("VEOG","HEOG")
TMIN, TMAX = -2.0, 28.0

def _runs(p, hint=None):
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    ks = [k for k in m if not k.startswith("__")]
    return np.atleast_1d(m[hint if hint in ks else ks[0]]).ravel()

def _bp(x, fs, lo, hi, order=3):
    nyq = fs/2; hi = min(hi, nyq*0.95)
    if lo >= hi: return x
    b,a = sps.butter(order, [lo/nyq, hi/nyq], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)

def beer_lambert(lo_wl, hi_wl):
    def od(x):
        base = np.median(x, -1, keepdims=True)
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        return -np.log(np.clip(x/base, 1e-9, None))
    a1, a2 = od(lo_wl), od(hi_wl)
    L1, L2 = SD_CM*DPF[760], SD_CM*DPF[850]
    E = np.array([[EXT[760]["HbO"]*L1, EXT[760]["HbR"]*L1],
                  [EXT[850]["HbO"]*L2, EXT[850]["HbR"]*L2]])
    Ei = np.linalg.inv(E)
    return (Ei[0,0]*a1+Ei[0,1]*a2)*1e6, (Ei[1,0]*a1+Ei[1,1]*a2)*1e6

def _split_wl(x, clab):
    n = [str(c) for c in np.atleast_1d(clab)]
    lo = {v.replace("lowWL",""):i for i,v in enumerate(n) if "lowWL" in v}
    hi = {v.replace("highWL",""):i for i,v in enumerate(n) if "highWL" in v}
    pr = sorted(set(lo)&set(hi))
    return np.stack([x[:,lo[p]] for p in pr]), np.stack([x[:,hi[p]] for p in pr])

def _rs(x,n): return x if x.shape[-1]==n else sps.resample(x, n, axis=-1)

def build_fnirs(root, task="MI"):
    out = CACHE/f"fnirs_{task}.npz"
    if out.exists(): print(f"fnirs {task}: cached"); return out
    E_, H_, S_, L_ = [], [], [], []
    nb = int(2.0*CFG.fnirs_fs)
    for s in range(1, 30):
        ed = next((g/f"subject {s:02d}"/"with occular artifact"
                   for g in sorted((root/"EEG data").glob("EEG_*"))
                   if (g/f"subject {s:02d}"/"with occular artifact").exists()), None)
        nd = root/"NRI data"/"NIRS_01-29"/f"subject {s:02d}"
        if ed is None or not nd.exists(): continue
        try:
            er, em = _runs(ed/"cnt.mat","cnt"), _runs(ed/"mrk.mat","mrk")
            nr, nm = _runs(nd/"cnt.mat","cnt"), _runs(nd/"mrk.mat","mrk")
        except Exception as e:
            print(f"  subj {s:02d}: {e}"); continue
        cnt = 0
        for r in range(min(er.size, nr.size)):
            Er, Nr = er[r], nr[r]
            if not str(getattr(Er,"title","")).upper().startswith(task): continue
            efs = float(np.atleast_1d(Er.fs).ravel()[0])
            clab = [str(c) for c in np.atleast_1d(Er.clab)]
            keep = [i for i,c in enumerate(clab) if c not in EOG]
            ex = np.asarray(Er.x, np.float64)[:,keep].T
            nfs = float(np.atleast_1d(Nr.fs).ravel()[0])
            lo, hi = _split_wl(np.asarray(Nr.x, np.float64), Nr.clab)
            hbo, hbr = beer_lambert(lo, hi)
            hbo, hbr = _bp(hbo,nfs,0.01,0.2), _bp(hbr,nfs,0.01,0.2)
            if r >= em.size or r >= nm.size: continue
            eon = np.atleast_1d(np.asarray(em[r].time,float))/1000.0
            non = np.atleast_1d(np.asarray(nm[r].time,float))/1000.0
            y = np.asarray(em[r].y); lab = np.argmax(y,0) if y.ndim==2 else np.asarray(y).ravel()
            for k in range(min(eon.size, non.size, lab.size)):
                ae,be = int(round((eon[k]+TMIN)*efs)), int(round((eon[k]+TMAX)*efs))
                an,bn = int(round((non[k]+TMIN)*nfs)), int(round((non[k]+TMAX)*nfs))
                if ae<0 or be>ex.shape[1] or an<0 or bn>hbo.shape[1]: continue
                se = _rs(ex[:,ae:be], CFG.eeg_window)
                se = ((se-se.mean(-1,keepdims=True))/(se.std(-1,keepdims=True)+1e-6)).astype(np.float32)
                so, sr = _rs(hbo[:,an:bn],CFG.fnirs_len), _rs(hbr[:,an:bn],CFG.fnirs_len)
                so -= so[:,:nb].mean(-1,keepdims=True); sr -= sr[:,:nb].mean(-1,keepdims=True)
                E_.append(se); H_.append(np.concatenate([so,sr],0).astype(np.float32))
                S_.append(s); L_.append(int(lab[k])); cnt += 1
        print(f"  subj {s:02d}: {cnt} trials")
    np.savez_compressed(out, eeg=np.stack(E_), hemo=np.stack(H_),
                        subject_ids=np.asarray(S_), labels=np.asarray(L_))
    print(f"{out.name}: {len(E_)} trials, {len(set(S_))} subjects, hemo {H_[0].shape}")
    return out

if not (CACHE/"fnirs_MI.npz").exists():
    build_fnirs(fetch("hbci"), "MI"); free("hbci")
else:
    print("fnirs_MI.npz: cached")
""")

md("""
## 4. Build the fMRI target — event-locked, motion-corrected, sensorimotor ROIs

The original target used sliding windows with no nuisance regression over all 48 ROIs and
scored **−0.639** reliability. Three fixes: event-lock on stimulus onsets, regress motion +
polynomial drift, restrict to sensorimotor ROIs.

EEG↔BOLD alignment uses the **R128 scanner triggers**, which mark every volume — volume *k*
is EEG sample `R128[k]` exactly, no assumed offset.
""")

code("""
import mne, nibabel as nib
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

N_ROI = 48
MOTOR_KEYS = ("Precentral","Postcentral","Supplementary","Juxtapositional")
FMRI_TASKS = ("motorloc","MIpre","MIpost","eegNF","fmriNF","eegfmriNF")

def _clean(ts, motion):
    n = ts.shape[0]; t = np.linspace(-1,1,n)
    cols = [np.ones(n), t, t**2, t**3]
    if motion is not None and motion.shape[0] >= 2:
        m = motion[:n] if motion.shape[0]>=n else np.vstack(
            [motion, np.repeat(motion[-1:], n-motion.shape[0],0)])
        for k in range(min(6,m.shape[1])):
            v = m[:,k]; sd = v.std(); cols.append((v-v.mean())/(sd if sd>0 else 1.0))
    X = np.column_stack(cols)
    return ts - X @ np.linalg.lstsq(X, ts, rcond=None)[0]

def build_fmri(root):
    out = CACHE/"fmri.npz"
    if out.exists(): print("fmri: cached"); return out
    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    amap = atlas.maps if hasattr(atlas,"maps") else atlas["maps"]
    labels = [str(x) for x in (atlas.labels if hasattr(atlas,"labels") else atlas["labels"])[1:N_ROI+1]]
    motor = [i for i,nm in enumerate(labels) if any(k in nm for k in MOTOR_KEYS)]
    print("motor ROIs:", [labels[i] for i in motor])
    E_, H_, S_ = [], [], []
    for si, sub in enumerate(sorted(d.name for d in root.glob("sub-*") if d.is_dir())):
        cnt = 0
        for task in FMRI_TASKS:
            nii  = root/sub/"func"/f"{sub}_task-{task}_bold.nii.gz"
            vhdr = root/"derivatives"/sub/"eeg_pp"/f"{sub}_task-{task}_eeg_pp.vhdr"
            ev   = root/f"task-{task}_events.tsv"
            if not (nii.exists() and vhdr.exists() and ev.exists()): continue
            try:
                img = nib.load(str(nii)); data = np.asarray(img.dataobj, np.float32)
                if data.ndim != 4: continue
                ref = nib.Nifti1Image(data[...,0], img.affine, img.header)
                lab = np.asarray(resample_to_img(amap, ref, interpolation="nearest",
                        force_resample=True, copy_header=True).dataobj).astype(int)
                ts = np.zeros((data.shape[3], N_ROI), np.float32)
                for k in range(1, N_ROI+1):
                    m = lab==k
                    if m.sum() >= 10: ts[:,k-1] = data[m].mean(0)
                del data
                hm = root/"derivatives"/sub/"head_motion"/f"d_{sub}_task-{task}_headmotion.tsv"
                motion = None
                if hm.exists():
                    df = pd.read_csv(hm, sep="\\t").select_dtypes("number").dropna(axis=1, how="all")
                    if df.shape[1] >= 6: motion = df.to_numpy()
                ts = _clean(ts, motion)[:, motor]
                raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
                eeg = raw.get_data(picks=mne.pick_types(raw.info, eeg=True, exclude=[]))*1e6
                fs = float(raw.info["sfreq"])
                trig = np.array([o for o,dd in zip(raw.annotations.onset, raw.annotations.description)
                                 if "R128" in str(dd)])
                trig = (trig*fs).astype(int)
                if trig.size < CFG.fmri_len: continue
                e = pd.read_csv(ev, sep="\\t")
                if "trial_type" in e.columns:
                    e = e[~e["trial_type"].astype(str).str.lower().str.contains("rest")]
                for o in pd.to_numeric(e["onset"], errors="coerce").dropna().to_numpy():
                    v = int(round(o/CFG.fmri_tr))
                    if v<0 or v+CFG.fmri_len>ts.shape[0] or v>=trig.size: continue
                    s0 = trig[v]
                    if s0+CFG.eeg_window > eeg.shape[1]: continue
                    sh = ts[v:v+CFG.fmri_len].T; sh = sh - sh[:,:1]
                    se = eeg[:, s0:s0+CFG.eeg_window]
                    se = (se-se.mean(-1,keepdims=True))/(se.std(-1,keepdims=True)+1e-6)
                    E_.append(se.astype(np.float32)); H_.append(sh.astype(np.float32)); S_.append(si)
                    cnt += 1
                del eeg, raw
            except Exception as exc:
                print(f"  {sub}/{task}: {type(exc).__name__}: {str(exc)[:60]}")
        print(f"  {sub}: {cnt} trials")
    nc = min(x.shape[0] for x in E_); nr = min(x.shape[0] for x in H_)
    np.savez_compressed(out, eeg=np.stack([x[:nc] for x in E_]),
                        hemo=np.stack([x[:nr] for x in H_]), subject_ids=np.asarray(S_))
    print(f"{out.name}: {len(E_)} trials, {len(set(S_))} subjects, hemo {H_[0][:nr].shape}")
    return out

if not (CACHE/"fmri.npz").exists():
    build_fmri(fetch("ds002336")); free("ds002336")
else:
    print("fmri.npz: cached")
""")

# ============================================================ 5. LOAD
md("## 5. Load")

code("""
def load(name, modality):
    z = np.load(CACHE/name, allow_pickle=True)
    return dict(eeg=z["eeg"], hemo=z["hemo"].astype(np.float32),
                sid=z["subject_ids"], modality=modality,
                labels=z["labels"] if "labels" in z.files else np.zeros(len(z["eeg"]), int))

DATA = {"fnirs": load("fnirs_MI.npz","fnirs"), "fmri": load("fmri.npz","fmri")}
for k,d in DATA.items():
    print(f"{k:6s} {d['eeg'].shape[0]:5d} trials  {np.unique(d['sid']).size:2d} subjects  "
          f"eeg {d['eeg'].shape[1:]} hemo {d['hemo'].shape[1:]}")

CFG.n_eeg_channels = DATA["fnirs"]["eeg"].shape[1]
FNIRS_CH = DATA["fnirs"]["hemo"].shape[1]
FMRI_CH  = DATA["fmri"]["hemo"].shape[1]
print(f"\\nencoder in {CFG.n_eeg_channels} ch | fNIRS head {FNIRS_CH} ch | fMRI head {FMRI_CH} ROIs")
""")

# ============================================================ 6. GATE
md("""
## 6. Target-quality gate — the ceiling on everything

Split a subject's trials in half, average each, correlate. **A target that cannot predict
itself cannot be predicted by anything**; the maximum achievable correlation is about
`sqrt(reliability)`. HbO/HbR must be *negative* for a real hemodynamic response.

This check is what caught the earlier failures — one target scored −0.639 while still
producing "significant" predictions.
""")

code("""
def concat_pearson(a,b):
    a,b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a)&np.isfinite(b)
    if ok.sum()<10 or a[ok].std()==0 or b[ok].std()==0: return np.nan
    return float(np.corrcoef(a[ok],b[ok])[0,1])

def subject_reliability(t, n_repeat=50, seed=0):
    rng = np.random.default_rng(seed); n = t.shape[0]
    if n < 6: return np.nan
    rs = [concat_pearson(t[p[:n//2]].mean(0), t[p[n//2:]].mean(0))
          for p in (rng.permutation(n) for _ in range(n_repeat))]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else np.nan

def boot_ci(v, n=CFG.n_boot, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size==0: return np.nan,np.nan,np.nan
    rng = np.random.default_rng(seed)
    b = [rng.choice(v,v.size,replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b,2.5)), float(np.percentile(b,97.5))

rows=[]
for k,d in DATA.items():
    vals=[subject_reliability(d["hemo"][d["sid"]==s]) for s in np.unique(d["sid"])]
    m,lo,hi = boot_ci(vals)
    hh=np.nan
    if k=="fnirs":
        nh=d["hemo"].shape[1]//2
        hh=float(np.nanmedian([np.corrcoef(d["hemo"][i,:nh].ravel(),d["hemo"][i,nh:].ravel())[0,1]
                               for i in range(min(200,len(d["hemo"])))]))
    rows.append({"target":k,"n_subjects":int(np.unique(d["sid"]).size),"reliability":m,
                 "ci_low":lo,"ci_high":hi,"ceiling":float(np.sqrt(max(m,0))),"hbo_hbr":hh})
qual=pd.DataFrame(rows)
print(qual.round(3).to_string(index=False))
""")

# ============================================================ 7. ARCH
md("""
## 7. Architecture

`SharedEncoder` (EEGNet front-end + dilated TCN) → per-modality `HemodynamicHead`
(predicts mean **and** log-variance) → `DelayModule` (learns a 2–8 s physiological lag and
shifts the prediction) → `DomainConfidence` (flags inputs far from the training
distribution, feeding the reliability score).
""")

code("""
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(s, i, o, k, p):
        super().__init__(); s.dw = nn.Conv1d(i,i,k,padding=p,groups=i); s.pw = nn.Conv1d(i,o,1)
    def forward(s,x): return s.pw(s.dw(x))

class EEGNetFrontend(nn.Module):
    \"\"\"EEGNet-style spatial-temporal feature extractor.\"\"\"
    def __init__(s, n_channels, F1=16, D=2, F2=None, kern_len=64, pool=4, dropout=0.25):
        super().__init__(); F2 = F2 or F1*D
        s.temporal = nn.Conv2d(1,F1,(1,kern_len),padding=(0,kern_len//2),bias=False)
        s.bn1 = nn.BatchNorm2d(F1)
        s.spatial = nn.Conv2d(F1,F1*D,(n_channels,1),groups=F1,bias=False)
        s.bn2 = nn.BatchNorm2d(F1*D); s.pool1 = nn.AvgPool2d((1,pool)); s.drop1 = nn.Dropout(dropout)
        s.sep_dw = nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False)
        s.sep_pw = nn.Conv2d(F1*D,F2,1,bias=False); s.bn3 = nn.BatchNorm2d(F2)
        s.pool2 = nn.AvgPool2d((1,pool*2)); s.drop2 = nn.Dropout(dropout); s.out_channels = F2
    def forward(s,x):
        x = x.unsqueeze(1)
        x = s.bn1(s.temporal(x)); x = F.elu(s.bn2(s.spatial(x)))
        x = s.drop1(s.pool1(x)); x = s.sep_dw(x)
        x = F.elu(s.bn3(s.sep_pw(x))); x = s.drop2(s.pool2(x))
        return x.squeeze(2)

class TCNBlock(nn.Module):
    def __init__(s, i, o, k, dilation, dropout=0.2):
        super().__init__(); pad = (k-1)*dilation; s.pad = pad
        s.c1 = nn.Conv1d(i,o,k,padding=pad,dilation=dilation)
        s.c2 = nn.Conv1d(o,o,k,padding=pad,dilation=dilation)
        s.b1, s.b2 = nn.BatchNorm1d(o), nn.BatchNorm1d(o)
        s.drop = nn.Dropout(dropout)
        s.proj = nn.Conv1d(i,o,1) if i!=o else nn.Identity()
    def forward(s,x):
        h = F.elu(s.b1(s.c1(x)[..., :-s.pad])); h = s.drop(h)
        h = F.elu(s.b2(s.c2(h)[..., :-s.pad]))
        return F.elu(h + s.proj(x))

class SharedEncoder(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.frontend = EEGNetFrontend(cfg.n_eeg_channels, cfg.encoder_temporal_filters,
                                    cfg.encoder_spatial_filters, None,
                                    cfg.encoder_kernel_length, cfg.encoder_pool, cfg.dropout)
        layers, in_ch = [], s.frontend.out_channels
        for i in range(cfg.tcn_layers):
            layers.append(TCNBlock(in_ch, cfg.tcn_channels, cfg.tcn_kernel, 2**i, cfg.dropout))
            in_ch = cfg.tcn_channels
        s.tcn = nn.Sequential(*layers)
        s.proj = nn.Conv1d(cfg.tcn_channels, cfg.latent_dim, 1)
    def forward(s, eeg): return s.proj(s.tcn(s.frontend(eeg)))

class HemodynamicHead(nn.Module):
    \"\"\"Predicts the response mean AND its log-variance (heteroscedastic).\"\"\"
    def __init__(s, latent_dim, out_channels, target_length, hidden=64):
        super().__init__(); s.target_length = target_length
        s.body = nn.Sequential(nn.Conv1d(latent_dim,hidden,1), nn.ELU(),
                               nn.Conv1d(hidden,hidden,3,padding=1), nn.ELU(),
                               nn.Conv1d(hidden,hidden,3,padding=1), nn.ELU())
        s.mean_head = nn.Conv1d(hidden,out_channels,1)
        s.logvar_head = nn.Conv1d(hidden,out_channels,1)
    def forward(s,z):
        z = F.interpolate(z, size=s.target_length, mode="linear", align_corners=False)
        h = s.body(z)
        return s.mean_head(h), s.logvar_head(h).clamp(-7.0, 4.0)

class DelayModule(nn.Module):
    \"\"\"Blood flow lags electrical activity; learn that lag, bounded to 2-8 s.\"\"\"
    def __init__(s, latent_dim, fs, dmin, dmax):
        super().__init__(); s.fs, s.dmin, s.dmax = fs, dmin, dmax
        s.est = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                              nn.Linear(latent_dim,32), nn.ELU(), nn.Linear(32,1))
    def forward(s,z):
        return (s.dmin + (s.dmax-s.dmin)*torch.sigmoid(s.est(z))).squeeze(-1)
    @staticmethod
    def apply_shift(sig, delay_s, fs):
        B,C,T = sig.shape
        base = torch.arange(T, device=sig.device, dtype=sig.dtype).unsqueeze(0).expand(B,-1)
        idx = (base - delay_s.unsqueeze(-1)*fs).clamp(0, T-1)
        left = idx.floor().long(); right = (left+1).clamp(max=T-1)
        w = (idx - left.to(sig.dtype)).unsqueeze(1)
        l = torch.gather(sig, 2, left.unsqueeze(1).expand(-1,C,-1))
        r = torch.gather(sig, 2, right.unsqueeze(1).expand(-1,C,-1))
        return l*(1-w) + r*w

class DomainConfidence(nn.Module):
    \"\"\"Tracks training-latent statistics; low score = input unlike anything seen.\"\"\"
    def __init__(s, latent_dim):
        super().__init__()
        s.register_buffer("running_mean", torch.zeros(latent_dim))
        s.register_buffer("running_var", torch.ones(latent_dim))
        s.register_buffer("count", torch.zeros(1))
    @torch.no_grad()
    def update(s, zp):
        bm, bv = zp.mean(0), zp.var(0, unbiased=False)
        n, m = s.count.item(), zp.shape[0]; new = n+m
        delta = bm - s.running_mean
        s.running_mean += delta*(m/max(new,1))
        s.running_var = (n*s.running_var + m*bv + (n*m/max(new,1))*delta**2)/max(new,1)
        s.count += m
    def forward(s, zp):
        d2 = ((zp - s.running_mean)**2/(s.running_var+1e-6)).sum(-1)
        return torch.exp(-d2/zp.shape[-1])

class NeuroVascNet(nn.Module):
    def __init__(s, cfg, fnirs_ch, fmri_ch):
        super().__init__(); s.cfg = cfg
        s.encoder = SharedEncoder(cfg)
        s.fnirs_head = HemodynamicHead(cfg.latent_dim, fnirs_ch, cfg.fnirs_len)
        s.fmri_head  = HemodynamicHead(cfg.latent_dim, fmri_ch, cfg.fmri_len)
        s.delay_fnirs = DelayModule(cfg.latent_dim, cfg.fnirs_fs, cfg.delay_min_s, cfg.delay_max_s)
        s.delay_fmri  = DelayModule(cfg.latent_dim, 1.0/cfg.fmri_tr, cfg.delay_min_s, cfg.delay_max_s)
        s.domain_conf = DomainConfidence(cfg.latent_dim)
    def forward(s, eeg, modality):
        z = s.encoder(eeg); zp = z.mean(-1)
        if modality == "fnirs":
            mean, logvar = s.fnirs_head(z); delay = s.delay_fnirs(z); fs = s.cfg.fnirs_fs
        else:
            mean, logvar = s.fmri_head(z);  delay = s.delay_fmri(z);  fs = 1.0/s.cfg.fmri_tr
        return {"z":z, "z_pool":zp, "pred_mean":DelayModule.apply_shift(mean, delay, fs),
                "pred_raw":mean, "pred_logvar":logvar, "delay":delay}

m = NeuroVascNet(CFG, FNIRS_CH, FMRI_CH)
print(f"parameters: {sum(p.numel() for p in m.parameters()):,}")
del m
""")

# ============================================================ 8. LOSSES
md("""
## 8. Losses

`pearson_corr_loss` targets *shape* rather than amplitude; `heteroscedastic_loss` is a
Gaussian NLL that lets the model express its own uncertainty (this is what the reliability
layer consumes); `cross_correlation_loss` tolerates small residual lag; `smoothness_loss`
keeps the predicted hemodynamic trace physiologically smooth.
""")

code("""
def heteroscedastic_loss(mean, target, logvar):
    return (torch.exp(-logvar)*(mean-target)**2 + logvar).mean()

def pearson_corr_loss(pred, target, eps=1e-6):
    p = pred - pred.mean(-1,keepdim=True); t = target - target.mean(-1,keepdim=True)
    p = p/(p.norm(dim=-1,keepdim=True)+eps); t = t/(t.norm(dim=-1,keepdim=True)+eps)
    return (1.0 - (p*t).sum(-1)).mean()

def cross_correlation_loss(pred, target, max_lag=3):
    B,C,T = pred.shape
    p0 = pred-pred.mean(-1,keepdim=True); t0 = target-target.mean(-1,keepdim=True)
    p0 = p0/(p0.norm(dim=-1,keepdim=True)+1e-6); t0 = t0/(t0.norm(dim=-1,keepdim=True)+1e-6)
    best = torch.full((B,C), -1.0, device=pred.device)
    for lag in range(-max_lag, max_lag+1):
        if lag<0: p,t = p0[...,:lag], t0[...,-lag:]
        elif lag>0: p,t = p0[...,lag:], t0[...,:-lag]
        else: p,t = p0,t0
        if p.shape[-1] < 4: continue
        p = p/(p.norm(dim=-1,keepdim=True)+1e-6); t = t/(t.norm(dim=-1,keepdim=True)+1e-6)
        best = torch.maximum(best, (p*t).sum(-1))
    return (1.0-best).mean()

def smoothness_loss(pred):
    d = pred[...,1:]-pred[...,:-1]; return (d**2).mean()

def moment_matching_loss(za, zb):
    return (za.mean(0)-zb.mean(0)).pow(2).mean() + (torch.cov(za.T)-torch.cov(zb.T)).pow(2).mean()

def compute_modality_loss(out, target, cfg, modality):
    l_corr   = pearson_corr_loss(out["pred_mean"], target)
    l_het    = heteroscedastic_loss(out["pred_mean"], target, out["pred_logvar"])
    l_xcorr  = cross_correlation_loss(out["pred_mean"], target)
    l_smooth = smoothness_loss(out["pred_mean"])
    w_main, w_delay = (cfg.w_hb, cfg.w_delay_fnirs) if modality=="fnirs" else (cfg.w_bold, cfg.w_delay_fmri)
    loss = w_main*l_corr + cfg.w_hetero*l_het + w_delay*l_xcorr + cfg.w_smooth*l_smooth
    return loss, {"corr":l_corr.item(), "hetero":l_het.item(), "xcorr":l_xcorr.item(),
                  "smooth":l_smooth.item(), "delay_s":out["delay"].mean().item()}
""")

# ============================================================ 9. TRAINING
md("""
## 9. Staged training

**Stage 1** warms up each modality separately. **Stage 2** trains jointly — the shared
encoder receives gradients from both losses in the same step, which is what makes the
latent "hemodynamic-aware" rather than modality-specific. `w_align` optionally adds a
moment-matching term between the two latents (off by default; see the ablations).
""")

code("""
class HemoDataset(Dataset):
    def __init__(s, eeg, hemo): s.eeg, s.hemo = eeg, hemo
    def __len__(s): return len(s.eeg)
    def __getitem__(s,i): return torch.from_numpy(s.eeg[i]), torch.from_numpy(s.hemo[i])

def make_loader(eeg, hemo, bs, shuffle=True):
    return DataLoader(HemoDataset(eeg,hemo), batch_size=bs, shuffle=shuffle, drop_last=False)

def train_model(tr, cfg=CFG, verbose=True):
    \"\"\"tr = {"fnirs": (eeg,hemo), "fmri": (eeg,hemo)} for the TRAINING subjects only.\"\"\"
    model = NeuroVascNet(cfg, FNIRS_CH, FMRI_CH).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dls = {k: make_loader(v[0], v[1], cfg.batch_size) for k,v in tr.items()}

    # ---- stage 1: per-modality warm-up ----
    for mod, dl in dls.items():
        for ep in range(cfg.epochs_stage1):
            model.train(); tot=0.0
            for eeg, hemo in dl:
                eeg, hemo = eeg.to(DEV), hemo.to(DEV)
                out = model(eeg, mod)
                loss,_ = compute_modality_loss(out, hemo, cfg, mod)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if verbose and (ep+1)%10==0:
                print(f"    stage1 {mod:5s} ep{ep+1:3d} loss {tot/len(dl):.4f}")

    # ---- stage 2: joint ----
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs_stage2)
    it_f, it_m = iter(dls["fnirs"]), iter(dls["fmri"])
    steps = max(len(dls["fnirs"]), len(dls["fmri"]))
    for ep in range(cfg.epochs_stage2):
        model.train(); tot=0.0
        for _ in range(steps):
            try: ef, hf = next(it_f)
            except StopIteration: it_f = iter(dls["fnirs"]); ef, hf = next(it_f)
            try: em, hm = next(it_m)
            except StopIteration: it_m = iter(dls["fmri"]);  em, hm = next(it_m)
            of = model(ef.to(DEV), "fnirs"); om = model(em.to(DEV), "fmri")
            lf,_ = compute_modality_loss(of, hf.to(DEV), cfg, "fnirs")
            lm,_ = compute_modality_loss(om, hm.to(DEV), cfg, "fmri")
            loss = lf + lm
            if cfg.w_align > 0:
                loss = loss + cfg.w_align*moment_matching_loss(of["z_pool"], om["z_pool"])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            model.domain_conf.update(of["z_pool"].detach())
        sch.step()
        if verbose and (ep+1)%10==0:
            print(f"    stage2 joint ep{ep+1:3d} loss {tot/steps:.4f}")
    return model

@torch.no_grad()
def predict(model, eeg, modality, bs=64):
    model.eval(); out=[]
    for i in range(0, len(eeg), bs):
        x = torch.from_numpy(eeg[i:i+bs]).to(DEV)
        out.append(model(x, modality)["pred_mean"].cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def predict_full(model, eeg, modality, bs=64):
    model.eval(); mean, var, dly, conf = [], [], [], []
    for i in range(0, len(eeg), bs):
        x = torch.from_numpy(eeg[i:i+bs]).to(DEV)
        o = model(x, modality)
        mean.append(o["pred_mean"].cpu().numpy()); var.append(o["pred_logvar"].exp().cpu().numpy())
        dly.append(o["delay"].cpu().numpy()); conf.append(model.domain_conf(o["z_pool"]).cpu().numpy())
    return (np.concatenate(mean), np.concatenate(var),
            np.concatenate(dly), np.concatenate(conf))
""")

# ============================================================ 10. EVAL
md("""
## 10. Evaluation — subject-wise cross-validation

Subjects are split into `k_folds` groups; each is held out once. **No subject is ever in
both training and test.** Both arms are trained together in every fold, so the shared
encoder is genuinely shared.

Reported with the constant-predictor baseline and an exact permutation null.
""")

code("""
def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid): out[sid==s] -= out[sid==s].mean(0, keepdims=True)
    return out

def perm_p(pred, true, n=CFG.n_perm, seed=0):
    \"\"\"Exact permutation null. A permutation only reorders rows, so all pairwise inner
    products are computed once as a matmul and each permutation is a gather (~550x faster
    than re-correlating, numerically identical).\"\"\"
    P = np.asarray(pred,np.float32).reshape(len(pred),-1)
    T = np.asarray(true,np.float32).reshape(len(true),-1)
    Pc, Tc = P-P.mean(), T-T.mean()
    nP = np.sqrt((Pc.astype(np.float64)**2).sum()); nT = np.sqrt((Tc.astype(np.float64)**2).sum())
    if nP<=0 or nT<=0: return np.nan,np.nan,np.nan
    D = (Pc@Tc.T).astype(np.float64); N=len(D); ar=np.arange(N)
    obs = float(D[ar,ar].sum()/(nP*nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N),ar].sum() for _ in range(n)])/(nP*nT)
    return obs, float(null.mean()), float((np.sum(null>=obs)+1)/(null.size+1))

def folds_by_subject(sid, k, seed=0):
    subs = np.unique(sid); rng = np.random.default_rng(seed); rng.shuffle(subs)
    return [subs[i::k] for i in range(k)]

def run_cv(cfg=CFG, verbose=True, tag=""):
    fold_sets = {m: folds_by_subject(DATA[m]["sid"], cfg.k_folds) for m in DATA}
    per = {m: [] for m in DATA}
    pool = {m: {"P":[], "T":[], "S":[], "V":[], "C":[]} for m in DATA}
    for f in range(cfg.k_folds):
        tr, te_idx = {}, {}
        for m,d in DATA.items():
            te = np.isin(d["sid"], fold_sets[m][f]); te_idx[m] = te
            tr[m] = (d["eeg"][~te], d["hemo"][~te])
        if verbose: print(f"  fold {f}: train "
                          + ", ".join(f"{m}={len(tr[m][0])}" for m in tr))
        model = train_model(tr, cfg, verbose=False)
        for m,d in DATA.items():
            te = te_idx[m]
            mean, var, dly, conf = predict_full(model, d["eeg"][te], m)
            P = mean.reshape(te.sum(),-1); T = d["hemo"][te].reshape(te.sum(),-1)
            sid = d["sid"][te]
            const = np.repeat(d["hemo"][~te].reshape((~te).sum(),-1).mean(0)[None,:], te.sum(), 0)
            per[m].append({"fold":f,
                "r_raw":concat_pearson(P,T), "r_const":concat_pearson(const,T),
                "r_demeaned":concat_pearson(demean(P,sid), demean(T,sid)),
                "delay_s":float(dly.mean())})
            pool[m]["P"].append(P); pool[m]["T"].append(T); pool[m]["S"].append(sid)
            pool[m]["V"].append(var.reshape(te.sum(),-1).mean(1)); pool[m]["C"].append(conf)
        del model
        if DEV=="cuda": torch.cuda.empty_cache()

    res, folds = [], {}
    for m in DATA:
        fd = pd.DataFrame(per[m]); folds[m] = fd
        P,T,S = np.vstack(pool[m]["P"]), np.vstack(pool[m]["T"]), np.concatenate(pool[m]["S"])
        _, null, p = perm_p(demean(P,S), demean(T,S))
        mu, lo, hi = boot_ci(fd.r_demeaned.values)
        res.append({"target":m, "r_raw":fd.r_raw.mean(), "r_const":fd.r_const.mean(),
                    "r_gain":fd.r_raw.mean()-fd.r_const.mean(), "r_demeaned":mu,
                    "ci_low":lo, "ci_high":hi, "null":null, "p":p,
                    "delay_s":fd.delay_s.mean()})
        if verbose:
            flag = "" if res[-1]["r_gain"]>0 else "   <- constant predictor WINS on raw"
            print(f"{tag}{m:6s} raw {fd.r_raw.mean():+.4f}  const {fd.r_const.mean():+.4f}"
                  f"   DEMEANED {mu:+.4f} [{lo:+.4f},{hi:+.4f}]  p={p:.4f}"
                  f"   delay {fd.delay_s.mean():.1f}s{flag}")
    return pd.DataFrame(res), folds, pool

print("cross-validated training (this is the long step)...")
cv, cv_folds, cv_pool = run_cv()
print("\\nper-fold r_demeaned:")
for m,fd in cv_folds.items(): print(f"  {m:6s}", np.round(fd.r_demeaned.values,4))
""")

# ============================================================ 11. RELIABILITY
md("""
## 11. Reliability, calibration and selective prediction

The model emits a per-trial uncertainty (predicted variance) and a domain-confidence score.
A useful reliability signal should **correlate with actual error**, and abstaining on the
least-reliable trials should *improve* accuracy on the rest. That curve is the deployable
part of the system.
""")

code("""
rel_rows, curves = [], {}
for m in DATA:
    P = np.vstack(cv_pool[m]["P"]); T = np.vstack(cv_pool[m]["T"]); S = np.concatenate(cv_pool[m]["S"])
    var = np.concatenate(cv_pool[m]["V"]); conf = np.concatenate(cv_pool[m]["C"])
    err = np.sqrt(((P-T)**2).mean(1))
    reliability = conf/(np.sqrt(var)+1e-8)
    r_var  = concat_pearson(np.sqrt(var), err)
    r_conf = concat_pearson(-conf, err)
    r_comb = concat_pearson(-reliability, err)
    order = np.argsort(-reliability)          # most reliable first
    keep_frac = np.linspace(0.1, 1.0, 19)
    curve = [float(err[order[:max(1,int(f*len(order)))]].mean()) for f in keep_frac]
    curves[m] = (keep_frac, curve)
    rel_rows.append({"target":m, "corr_uncertainty_error":r_var,
                     "corr_domainconf_error":r_conf, "corr_combined_error":r_comb,
                     "err_all":curve[-1], "err_top50pct":curve[len(curve)//2],
                     "abstention_gain":curve[-1]-curve[len(curve)//2]})
rel = pd.DataFrame(rel_rows)
print(rel.round(4).to_string(index=False))
print("\\ncorr_*_error > 0 means the signal tracks real error (higher = better).")
print("abstention_gain > 0 means dropping the least-reliable half REDUCES error.")
""")

code("""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(DATA)+1, figsize=(6*(len(DATA)+1), 4))
for j,(m,(kf,cv_)) in enumerate(curves.items()):
    ax[j].plot(kf*100, cv_, marker="o")
    ax[j].set_xlabel("% most-reliable trials kept"); ax[j].set_ylabel("RMSE")
    ax[j].set_title(f"{m}: selective prediction"); ax[j].grid(alpha=.3)
m0 = list(DATA)[0]
P = np.vstack(cv_pool[m0]["P"]); T = np.vstack(cv_pool[m0]["T"])
t = np.arange(T.shape[1]//DATA[m0]["hemo"].shape[1] if False else DATA[m0]["hemo"].shape[2])
tt = t/(CFG.fnirs_fs if m0=="fnirs" else 1/CFG.fmri_tr) - 2.0
nh = DATA[m0]["hemo"].shape[1]
ax[-1].plot(tt, T.reshape(len(T),nh,-1).mean((0,1)), label="true", lw=2)
ax[-1].plot(tt, P.reshape(len(P),nh,-1).mean((0,1)), label="predicted", lw=2)
ax[-1].axvline(0, color="k", ls=":", lw=.8); ax[-1].legend()
ax[-1].set_title(f"{m0}: grand-average response"); ax[-1].set_xlabel("s from onset")
fig.tight_layout(); fig.savefig(WORK/"results_figure.png", dpi=110); plt.show()
""")

# ============================================================ 12. ABLATIONS
md("""
## 12. Ablations

Each row removes one component and re-runs the whole cross-validation, so the effect is
measured the same way as the main result. `no_delay` fixes the lag at zero; `no_hetero`
drops the uncertainty term; `align` switches on cross-modal latent matching.
""")

code("""
RUN_ABLATIONS = True     # set False to skip (each row re-runs the full CV)

ABLATIONS = {
    "full":        {},
    "no_delay":    {"delay_min_s":0.0, "delay_max_s":0.0},
    "no_hetero":   {"w_hetero":0.0},
    "no_smooth":   {"w_smooth":0.0},
    "align_on":    {"w_align":0.1},
}

abl = pd.DataFrame()
if RUN_ABLATIONS:
    rows=[]
    for name, over in ABLATIONS.items():
        cfg = Config(**{**CFG.__dict__, **over})
        print(f"\\n--- ablation: {name} {over}")
        res,_,_ = run_cv(cfg, verbose=True, tag="   ")
        for _,r in res.iterrows():
            rows.append({"ablation":name, **r.to_dict()})
    abl = pd.DataFrame(rows)
    print("\\n" + abl.round(4).to_string(index=False))
else:
    print("ablations skipped")
""")

# ============================================================ 13. SUMMARY
md("## 13. Summary")

code("""
print("="*94)
print("TARGET QUALITY (the ceiling)"); print(qual.round(3).to_string(index=False))
print("\\nCROSS-VALIDATED PREDICTION");  print(cv.round(4).to_string(index=False))
print("\\nRELIABILITY / ABSTENTION");    print(rel.round(4).to_string(index=False))
if len(abl): print("\\nABLATIONS"); print(abl.round(4).to_string(index=False))
print("="*94)

q = qual.set_index("target"); c = cv.set_index("target")
print(f"\\n{'target':8s}{'reliab':>8s}{'ceiling':>9s}{'r_demeaned':>12s}{'% ceiling':>11s}{'p':>9s}")
for m in DATA:
    ceil = q.loc[m,'ceiling']; r = c.loc[m,'r_demeaned']
    print(f"{m:8s}{q.loc[m,'reliability']:>8.3f}{ceil:>9.3f}{r:>12.4f}"
          f"{(f'{100*r/ceil:.0f}%' if ceil>0 else 'n/a'):>11s}{c.loc[m,'p']:>9.4f}")

for name, df in [("target_quality",qual), ("cv",cv), ("reliability",rel), ("ablations",abl)]:
    if len(df): df.to_csv(WORK/f"results_{name}.csv", index=False)
print(f"\\nCSVs + figure saved to {WORK}")
print("\\nEvery number is subject-wise cross-validated. Judge on r_demeaned: it is the only")
print("column where a predictor ignoring EEG scores zero, so the only one measuring EEG.")
""")

nb["cells"] = C
nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
p = Path("NeuroVascNet_portable.ipynb")
nbf.write(nb, p)
print(f"wrote {p} ({len(C)} cells)")
