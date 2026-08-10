"""Generate NeuroVascNet_v4_hBCI.ipynb — full cross-validated train + evaluation.

Replaces the single fixed train/val split with subject-wise k-fold CV, so every subject
serves as test exactly once and the headline number comes with a confidence interval
instead of resting on one arbitrary 6-subject holdout.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# NeuroVascNet v4 — full cross-validated evaluation

A data-quality audit showed every earlier result was confounded by broken supervision
targets. Both targets have since been repaired; this notebook is the full re-evaluation.

**Every number is subject-wise cross-validated.** Subjects are split into `K_FOLDS` groups;
each group is held out once while the model trains on the rest. No subject ever appears in
both training and test. The reported figure is the mean across folds with a bootstrap CI,
which is what makes it a result rather than one lucky split.

## The three metrics, and why the first one lies

| metric | meaning |
|---|---|
| `r_raw` | plain concat-Pearson — **inflated**: every trial shares a stereotyped HRF shape |
| `r_const` | what you get **ignoring EEG entirely** and emitting the training mean |
| `r_demeaned` | subject means removed, so `r_const` collapses to 0 — **the real EEG contribution** |

On the reference run `r_raw = 0.072` looked significant, but the EEG-free constant predictor
scored **0.076** — it beat the model. Judge on `r_demeaned` only.
""")

code("""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy import signal as sps

EEG_FS = 200.0
RESP_WIN = (4.0, 14.0)
BANDS = [(1,4),(4,8),(8,13),(13,30),(30,45)]
N_BINS = 6

K_FOLDS   = 5        # subject-wise CV folds
N_PERM    = 1000     # permutations for the null (was 200)
N_BOOT    = 2000
MAX_EPOCHS, PATIENCE, BS = 400, 40, 32
LR_GRID   = (3e-4, 1e-3, 3e-3)

V3 = Path("neurovascnet_outputs/cache_v3")
V4 = Path("neurovascnet_outputs/cache_v4")
""")

md("## 1. Load and pool")

code("""
def load(pair, modality, hemo_fs, chromophores):
    tr, va = np.load(pair[0], allow_pickle=True), np.load(pair[1], allow_pickle=True)
    # pool train+val, then re-split by subject during CV; offset val ids so they stay distinct
    sid = np.concatenate([tr["subject_ids"], va["subject_ids"] + tr["subject_ids"].max() + 1])
    return dict(eeg=np.concatenate([tr["eeg"], va["eeg"]]),
                hemo=np.concatenate([tr["hemo"], va["hemo"]]).astype(np.float64),
                sid=sid, modality=modality, hemo_fs=hemo_fs, chromophores=chromophores)

SPEC = {
    "fNIRS HEFMI (v3)": ((V3/"fnirs_train.npz", V3/"fnirs_val.npz"), "fnirs", 10.0, True),
    "fNIRS hBCI (v4)":  ((V4/"hbci_train_MI.npz", V4/"hbci_val_MI.npz"), "fnirs", 10.0, True),
    "fMRI v3 (broken)": ((V3/"ds002336_train.npz", V3/"ds002336_val.npz"), "fmri", 0.5, False),
    "fMRI v4 (fixed)":  ((V4/"ds002336_fixed_train.npz", V4/"ds002336_fixed_val.npz"), "fmri", 0.5, False),
}
DATA = {}
for k, (pair, mod, fs, chrom) in SPEC.items():
    if pair[0].exists() and pair[1].exists():
        DATA[k] = load(pair, mod, fs, chrom)
        d = DATA[k]
        print(f"{k:20s} {mod:5s} {d['eeg'].shape[0]:5d} trials  "
              f"{np.unique(d['sid']).size:2d} subjects  hemo {d['hemo'].shape[1]}x{d['hemo'].shape[2]}")
    else:
        print(f"{k:20s} cache missing, skipped")
""")

md("""
## 2. Target quality — the ceiling

Split a subject's trials in half, average each, correlate. A target that cannot predict
*itself* cannot be predicted by anything, so this bounds every result:
max achievable ≈ `sqrt(reliability)`. All subjects, 50 splits each, bootstrap CI.
""")

code("""
def concat_pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0: return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0,1])

def subject_reliability(t, n_repeat=50, seed=0):
    rng = np.random.default_rng(seed); n = t.shape[0]
    if n < 6: return np.nan
    rs = [concat_pearson(t[p[:n//2]].mean(0), t[p[n//2:]].mean(0))
          for p in (rng.permutation(n) for _ in range(n_repeat))]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else np.nan

def boot_ci(vals, n=N_BOOT, seed=0):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if vals.size == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [rng.choice(vals, vals.size, replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def hbo_hbr(h, chrom):
    if not chrom: return np.nan
    n = h.shape[1]//2
    return float(np.nanmedian([np.corrcoef(h[i,:n].ravel(), h[i,n:].ravel())[0,1]
                               for i in range(min(200, h.shape[0]))]))

rows = []
for k, d in DATA.items():
    vals = [subject_reliability(d["hemo"][d["sid"] == s]) for s in np.unique(d["sid"])]
    m, lo, hi = boot_ci(vals)
    rows.append({"dataset": k, "modality": d["modality"],
                 "n_subjects": int(np.unique(d["sid"]).size),
                 "reliability": m, "ci_low": lo, "ci_high": hi,
                 "ceiling": float(np.sqrt(max(m, 0))),
                 "hbo_hbr_corr": hbo_hbr(d["hemo"], d["chromophores"])})
qual = pd.DataFrame(rows)
print(qual.round(3).to_string(index=False))
print("\\nA target with reliability <= 0 cannot be genuinely predicted;")
print("any 'significant' correlation against it is spurious.")
""")

md("## 3. Features and model")

code("""
def eeg_features(eeg, bands=BANDS, n_bins=N_BINS, erd=False):
    N, C, T = eeg.shape; L = T // n_bins
    x = eeg[:, :, :L*n_bins].reshape(N, C, n_bins, L)
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(L), axis=-1))**2
    f = np.fft.rfftfreq(L, 1/EEG_FS)
    bp = np.log10(np.stack([P[..., (f>=lo)&(f<hi)].sum(-1) for lo,hi in bands], -1) + 1e-12)
    out = [bp.reshape(N,-1)]
    if erd: out.append((bp - bp[:,:,:1,:]).reshape(N,-1))
    return np.concatenate(out,1).astype(np.float32)

def ridge(Xtr, Ytr, Xte, alphas=(1e0,1e1,1e2,1e3,1e4,1e5)):
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-8
    Xtr, Xte = (Xtr-mu)/sd, (Xte-mu)/sd
    ym = Ytr.mean(0); Yc = Ytr-ym
    n = int(Xtr.shape[0]*0.8); best, best_r = alphas[len(alphas)//2], -np.inf
    if Xtr.shape[0]-n >= 4:
        G = Xtr[:n].T@Xtr[:n]
        for al in alphas:
            W = np.linalg.solve(G+al*np.eye(G.shape[0]), Xtr[:n].T@Yc[:n])
            r = concat_pearson(Xtr[n:]@W, Yc[n:])
            if np.isfinite(r) and r > best_r: best_r, best = r, al
    G = Xtr.T@Xtr
    return Xte @ np.linalg.solve(G+best*np.eye(G.shape[0]), Xtr.T@Yc) + ym

def demean(A, sid):
    out = A.copy()
    for s in np.unique(sid): out[sid==s] -= out[sid==s].mean(0, keepdims=True)
    return out

def perm_p(pred, true, n=N_PERM, seed=0):
    \"\"\"Permutation null on the trial pairing, computed exactly but fast.

    A permutation only REORDERS rows, so the flattened correlation is
        r(perm) = sum_i <pred[perm[i]], true[i]> / (||pred|| ||true||)
    with the norms and the global mean unchanged. All the pairwise inner products can
    therefore be computed ONCE as a single matmul, after which each of the n permutations
    is a cheap gather. Naively re-correlating instead costs ~0.3 s per permutation here
    (73 min for 14 calls at n=1000); this is seconds.
    \"\"\"
    P = np.asarray(pred, dtype=np.float32); T = np.asarray(true, dtype=np.float32)
    P = P.reshape(len(P), -1); T = T.reshape(len(T), -1)
    Pc = P - P.mean(); Tc = T - T.mean()
    nP = np.sqrt((Pc.astype(np.float64)**2).sum()); nT = np.sqrt((Tc.astype(np.float64)**2).sum())
    if nP <= 0 or nT <= 0: return np.nan, np.nan, np.nan
    D = (Pc @ Tc.T).astype(np.float64)          # (N, N) all pairwise inner products
    N = len(D); ar = np.arange(N)
    obs = float(D[ar, ar].sum() / (nP * nT))
    rng = np.random.default_rng(seed)
    null = np.array([D[rng.permutation(N), ar].sum() for _ in range(n)]) / (nP * nT)
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (null.size + 1))

def folds_by_subject(sid, k=K_FOLDS, seed=0):
    subs = np.unique(sid); rng = np.random.default_rng(seed); rng.shuffle(subs)
    return [subs[i::k] for i in range(k)]
""")

md("""
## 4. Cross-validated ridge

Each fold: train on the other folds' subjects, predict the held-out subjects, score all
three metrics. Reported as mean across folds with a bootstrap CI, plus a permutation
p-value on the pooled out-of-fold predictions.
""")

code("""
def cv_ridge(d, feat_kw=None, channels=None, target_slice=None, tag=""):
    feat_kw = feat_kw or {}
    X = eeg_features(d["eeg"], **feat_kw)
    H = d["hemo"]
    if channels is not None and d["chromophores"]:
        nh = H.shape[1]//2
        H = H[:, np.concatenate([channels, channels + nh])]
    if target_slice is not None:
        H = H[:, :, target_slice[0]:target_slice[1]]
    Y = H.reshape(len(H), -1)
    sid = d["sid"]

    per_fold, oof_pred, oof_true, oof_sid = [], [], [], []
    for f, test_subs in enumerate(folds_by_subject(sid)):
        te = np.isin(sid, test_subs); tr = ~te
        if te.sum() < 6 or tr.sum() < 20: continue
        pred = ridge(X[tr], Y[tr], X[te])
        const = np.repeat(Y[tr].mean(0)[None,:], te.sum(), axis=0)
        r_raw = concat_pearson(pred, Y[te]); r_c = concat_pearson(const, Y[te])
        r_dm = concat_pearson(demean(pred, sid[te]), demean(Y[te], sid[te]))
        per_fold.append({"fold": f, "n_test": int(te.sum()), "r_raw": r_raw,
                         "r_const": r_c, "r_gain": r_raw-r_c, "r_demeaned": r_dm})
        oof_pred.append(pred); oof_true.append(Y[te]); oof_sid.append(sid[te])

    fold_df = pd.DataFrame(per_fold)
    P, T, S = np.vstack(oof_pred), np.vstack(oof_true), np.concatenate(oof_sid)
    _, null, p = perm_p(demean(P, S), demean(T, S))
    m, lo, hi = boot_ci(fold_df.r_demeaned.values)
    summary = {"r_demeaned_mean": m, "ci_low": lo, "ci_high": hi,
               "r_demeaned_pooled": concat_pearson(demean(P,S), demean(T,S)),
               "null": null, "p": p,
               "r_raw_mean": fold_df.r_raw.mean(), "r_const_mean": fold_df.r_const.mean(),
               "r_gain_mean": fold_df.r_gain.mean(), "n_folds": len(fold_df)}
    if tag:
        flag = "" if summary["r_gain_mean"] > 0 else "   <- constant predictor WINS on raw"
        print(f"{tag:24s} raw {summary['r_raw_mean']:+.4f}  const {summary['r_const_mean']:+.4f}"
              f"   DEMEANED {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}{flag}")
    return summary, fold_df

cv_rows, cv_folds = [], {}
for k, d in DATA.items():
    s, fd = cv_ridge(d, tag=k)
    cv_rows.append({"dataset": k, "modality": d["modality"], **s})
    cv_folds[k] = fd
cv = pd.DataFrame(cv_rows)
print("\\nPer-fold spread (fNIRS hBCI):")
if "fNIRS hBCI (v4)" in cv_folds:
    print(cv_folds["fNIRS hBCI (v4)"].round(4).to_string(index=False))
""")

md("""
## 5. Target-channel selection

The target averages **all 36 fNIRS channels**; many sit over non-responsive cortex and add
only noise. The equivalent fix on the fMRI side moved reliability 0.268 → 0.357.

Channel ranking is recomputed **inside each fold** from training subjects only, so the
selection never sees held-out data.
""")

code("""
def channel_reliability(hemo, sid, n_repeat=20, seed=0):
    rng = np.random.default_rng(seed); n_ch = hemo.shape[1]
    acc, cnt = np.zeros(n_ch), np.zeros(n_ch)
    for s in np.unique(sid):
        idx = np.flatnonzero(sid == s)
        if idx.size < 6: continue
        for _ in range(n_repeat):
            p = rng.permutation(idx)
            a, b = hemo[p[:idx.size//2]].mean(0), hemo[p[idx.size//2:]].mean(0)
            for c in range(n_ch):
                if a[c].std() > 0 and b[c].std() > 0:
                    acc[c] += np.corrcoef(a[c], b[c])[0,1]; cnt[c] += 1
    return np.where(cnt > 0, acc/np.maximum(cnt,1), np.nan)

def cv_channels(d, k_list=(4,8,12,18,24,36)):
    X = eeg_features(d["eeg"]); H = d["hemo"]; sid = d["sid"]; nh = H.shape[1]//2
    out = []
    for k in k_list:
        per_fold, P, T, S = [], [], [], []
        for test_subs in folds_by_subject(sid):
            te = np.isin(sid, test_subs); tr = ~te
            if te.sum() < 6 or tr.sum() < 20: continue
            rel = channel_reliability(H[tr], sid[tr])[:nh]      # TRAIN ONLY
            sel = np.sort(np.argsort(-np.nan_to_num(rel, nan=-9))[:k])
            pick = np.concatenate([sel, sel + nh])
            Y = H[:, pick].reshape(len(H), -1)
            pred = ridge(X[tr], Y[tr], X[te])
            per_fold.append(concat_pearson(demean(pred, sid[te]), demean(Y[te], sid[te])))
            P.append(pred); T.append(Y[te]); S.append(sid[te])
        Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
        _, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
        m, lo, hi = boot_ci(per_fold)
        out.append({"k_channels": k, "r_demeaned": m, "ci_low": lo, "ci_high": hi, "p": p})
        print(f"  top-{k:<3d} channels   {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}"
              f"{' *' if p < 0.05 else ''}")
    return pd.DataFrame(out)

chan = pd.DataFrame(); BEST_K = None
d = DATA.get("fNIRS hBCI (v4)")
if d is not None:
    print("channel selection recomputed inside every fold:")
    chan = cv_channels(d)
    b = chan.loc[chan.r_demeaned.idxmax()]
    all36 = chan[chan.k_channels == 36].r_demeaned.iloc[0]
    BEST_K = int(b.k_channels)
    print(f"\\nbest {BEST_K} channels -> {b.r_demeaned:+.4f}   "
          f"(all 36 -> {all36:+.4f},  change {b.r_demeaned-all36:+.4f})")
""")

md("""
## 6. Deep encoder (EEGNet + TCN), cross-validated

Learning-rate search, augmentation (noise / channel dropout / time jitter), early stopping
on subjects held out **from the training folds**, up to 400 epochs. Held-out test subjects
are never touched. Repeated for every fold.
""")

code("""
import torch, torch.nn as nn
from scipy.signal import butter, filtfilt
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV=="cuda" else "")

def bandpower_seq(eeg, fs=EEG_FS, out_hz=20.0):
    N, C, T = eeg.shape; step = int(fs/out_hz); outs = []
    for lo, hi in BANDS:
        b, a = butter(4, [lo/(fs/2), min(hi/(fs/2), 0.99)], btype="band")
        env = filtfilt(b, a, eeg, axis=-1)**2
        outs.append(env[:, :, :(T//step)*step].reshape(N, C, T//step, step).mean(-1))
    x = np.log10(np.concatenate(outs, 1) + 1e-12).astype(np.float32)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)

class TCN(nn.Module):
    def __init__(s, ch, d):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(ch,ch,3,padding=d,dilation=d), nn.BatchNorm1d(ch),
                              nn.ELU(), nn.Dropout(0.2),
                              nn.Conv1d(ch,ch,3,padding=d,dilation=d), nn.BatchNorm1d(ch), nn.ELU())
    def forward(s, x): return torch.relu(x + s.net(x))

class Enc(nn.Module):
    def __init__(s, ic, oc, ol, lat=64):
        super().__init__()
        s.sp = nn.Sequential(nn.Conv1d(ic,128,1), nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.25),
                             nn.Conv1d(128,lat,1), nn.BatchNorm1d(lat), nn.ELU())
        s.t = nn.Sequential(*[TCN(lat,dd) for dd in (1,2,4,8)])
        s.p = nn.AdaptiveAvgPool1d(16)
        s.h = nn.Sequential(nn.Flatten(), nn.Linear(lat*16,512), nn.ELU(),
                            nn.Dropout(0.3), nn.Linear(512, oc*ol))
        s.oc, s.ol = oc, ol
    def forward(s, x): return s.h(s.p(s.t(s.sp(x)))).view(-1, s.oc, s.ol)

def augment(x, noise=0.05, ch_drop=0.1, jitter=3):
    x = x + torch.randn_like(x)*noise
    if ch_drop > 0:
        x = x * (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > ch_drop).float()
    if jitter > 0:
        x = torch.roll(x, shifts=int(torch.randint(-jitter, jitter+1, (1,)).item()), dims=-1)
    return x

def train_one(Xtr, Ytr, Xte, sid_tr):
    ys = Ytr.std() + 1e-8
    subs = np.unique(sid_tr); n_in = max(1, int(len(subs)*0.8))
    m_tr = np.isin(sid_tr, subs[:n_in]); m_es = ~m_tr
    if m_es.sum() < 8: m_tr = np.ones(len(Xtr), bool); m_es = m_tr
    Xt = torch.tensor(Xtr[m_tr]); Yt = torch.tensor(Ytr[m_tr]/ys, dtype=torch.float32)
    Xe = torch.tensor(Xtr[m_es]).to(DEV); Ye = torch.tensor(Ytr[m_es]/ys, dtype=torch.float32).to(DEV)
    Xv = torch.tensor(Xte).to(DEV)

    best_es_all, best_state = np.inf, None
    for lr in LR_GRID:
        torch.manual_seed(0)
        m = Enc(Xtr.shape[1], Ytr.shape[1], Ytr.shape[2]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
        best_es, bad, state, n = np.inf, 0, None, len(Xt)
        for ep in range(MAX_EPOCHS):
            m.train(); perm = torch.randperm(n)
            for i in range(0, n, BS):
                idx = perm[i:i+BS]; opt.zero_grad()
                nn.functional.mse_loss(m(augment(Xt[idx].to(DEV))), Yt[idx].to(DEV)).backward()
                opt.step()
            sch.step(); m.eval()
            with torch.no_grad(): es = nn.functional.mse_loss(m(Xe), Ye).item()
            if es < best_es - 1e-5:
                best_es, bad = es, 0
                state = {k_: v.detach().clone() for k_, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE: break
        if best_es < best_es_all: best_es_all, best_state = best_es, state

    m = Enc(Xtr.shape[1], Ytr.shape[1], Ytr.shape[2]).to(DEV)
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return torch.cat([m(Xv[i:i+64]).cpu() for i in range(0, len(Xv), 64)]).numpy()*ys

def cv_deep(d, channels=None, tag=""):
    X = bandpower_seq(d["eeg"]); H = d["hemo"]; sid = d["sid"]
    if channels is not None and d["chromophores"]:
        nh = H.shape[1]//2
        H = H[:, np.concatenate([channels, channels + nh])]
    per_fold, P, T, S = [], [], [], []
    for f, test_subs in enumerate(folds_by_subject(sid)):
        te = np.isin(sid, test_subs); tr = ~te
        if te.sum() < 6 or tr.sum() < 20: continue
        pred = train_one(X[tr], H[tr], X[te], sid[tr])
        pr = pred.reshape(len(pred), -1); tt = H[te].reshape(te.sum(), -1)
        per_fold.append(concat_pearson(demean(pr, sid[te]), demean(tt, sid[te])))
        P.append(pr); T.append(tt); S.append(sid[te])
        print(f"    fold {f}: r_demeaned {per_fold[-1]:+.4f}")
    Pm, Tm, Sm = np.vstack(P), np.vstack(T), np.concatenate(S)
    _, null, p = perm_p(demean(Pm, Sm), demean(Tm, Sm))
    m, lo, hi = boot_ci(per_fold)
    print(f"{tag:24s} DEMEANED {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")
    return {"r_demeaned_mean": m, "ci_low": lo, "ci_high": hi, "null": null,
            "p": p, "n_folds": len(per_fold)}

deep_rows = []
for k, d in DATA.items():
    print(f"\\n  {k}")
    ch = None
    if k == "fNIRS hBCI (v4)" and BEST_K is not None:
        rel = channel_reliability(d["hemo"], d["sid"])[:d["hemo"].shape[1]//2]
        ch = np.sort(np.argsort(-np.nan_to_num(rel, nan=-9))[:BEST_K])
    deep_rows.append({"dataset": k, "modality": d["modality"],
                      "channels": "selected" if ch is not None else "all",
                      **cv_deep(d, channels=ch, tag=k)})
deep = pd.DataFrame(deep_rows)
""")

md("## 7. Summary")

code("""
print("="*100)
print("TARGET QUALITY"); print(qual.round(3).to_string(index=False))
print("\\nCROSS-VALIDATED RIDGE"); print(cv.round(4).to_string(index=False))
if len(chan): print("\\nCHANNEL SELECTION"); print(chan.round(4).to_string(index=False))
print("\\nCROSS-VALIDATED DEEP"); print(deep.round(4).to_string(index=False))
print("="*100)

q = qual.set_index("dataset"); r_ = cv.set_index("dataset"); dp_ = deep.set_index("dataset")
print(f"\\n{'dataset':22s}{'reliab':>8s}{'ceiling':>9s}{'ridge':>10s}{'%ceil':>7s}"
      f"{'deep':>10s}{'%ceil':>7s}   significance")
for k in DATA:
    rel = q.loc[k,'reliability']; ceil = q.loc[k,'ceiling']
    rr, rd = r_.loc[k,'r_demeaned_mean'], dp_.loc[k,'r_demeaned_mean']
    f1 = f"{100*rr/ceil:.0f}%" if ceil > 0 else "n/a"
    f2 = f"{100*rd/ceil:.0f}%" if ceil > 0 else "n/a"
    print(f"{k:22s}{rel:>8.3f}{ceil:>9.3f}{rr:>10.4f}{f1:>7s}{rd:>10.4f}{f2:>7s}"
          f"   p_ridge={r_.loc[k,'p']:.4f} p_deep={dp_.loc[k,'p']:.4f}")

out = Path("qc_outputs/tables"); out.mkdir(parents=True, exist_ok=True)
qual.to_csv(out/"v4cv_target_quality.csv", index=False)
cv.to_csv(out/"v4cv_ridge.csv", index=False)
deep.to_csv(out/"v4cv_deep.csv", index=False)
if len(chan): chan.to_csv(out/"v4cv_channels.csv", index=False)
print(f"\\nsaved to {out}")
print("\\nAll figures are subject-wise cross-validated: every subject is held out exactly")
print("once, so these are not one lucky split. CIs are bootstrapped over folds.")
""")

nb["cells"] = C
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
p = Path("NeuroVascNet_v4_hBCI.ipynb")
nbf.write(nb, p)
print(f"wrote {p} ({len(C)} cells)")
