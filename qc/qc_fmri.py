"""fMRI quality control.

Motion statistics and censored volumes, tSNR, usable task volumes, ROI
coverage against the Harvard-Oxford parcellation actually used by the model,
the event design matrix, and event-locked ROI averages. The ROI mask used for
every subject is saved so the parcellation is auditable per recording.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from qc import metrics as M
from qc.config import (ARRAY_DIR, ATLAS_NAME, DATASETS, FIG_DIR, MOTOR_ROI_LABELS,
                       N_ROIS, clean_path)
from qc.rules import FMRI_RULES, RuleHit, band, evaluate

_ATLAS_CACHE = {}


def _slug(row) -> str:
    return f"{row['dataset']}_{row['subject']}_{row['session']}_{row['task']}_fmri"


def get_atlas():
    """Harvard-Oxford cortical, 48 ROIs -- the same atlas the model targets."""
    if "atlas" not in _ATLAS_CACHE:
        from nilearn.datasets import fetch_atlas_harvard_oxford
        _ATLAS_CACHE["atlas"] = fetch_atlas_harvard_oxford(ATLAS_NAME)
    return _ATLAS_CACHE["atlas"]


# =========================================================================
def framewise_displacement(motion: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Power et al. FD from 6 motion parameters (translations mm, rotations rad)."""
    m = np.asarray(motion, dtype=float).copy()
    if m.shape[1] < 6:
        return np.array([])
    m[:, 3:6] *= radius              # rotations -> arc length in mm
    d = np.abs(np.diff(m, axis=0))
    return np.concatenate([[0.0], d.sum(axis=1)])


def read_head_motion(path) -> np.ndarray | None:
    p = clean_path(path)
    if p is None or not p.exists():
        return None
    import pandas as pd
    df = pd.read_csv(p, sep="\t")
    df = df.select_dtypes("number").dropna(axis=1, how="all")
    return df.to_numpy() if df.shape[1] >= 6 else None


# =========================================================================
def qc_fmri_recording(row: dict, save_arrays=True, save_figs=True) -> dict:
    import nibabel as nib

    ds = row["dataset"]
    path = Path(row["primary_path"])
    slug = _slug(row)
    tr = DATASETS[ds].get("documented_tr", 2.0)

    r: dict = {
        "dataset": ds, "subject": row["subject"], "session": row["session"],
        "task": row["task"], "data_role": row["data_role"], "path": str(path),
    }

    img = nib.load(str(path))
    shape = img.shape
    zooms = img.header.get_zooms()
    r["dim_x"], r["dim_y"], r["dim_z"] = shape[0], shape[1], shape[2]
    r["n_volumes"] = int(shape[3]) if len(shape) > 3 else 1
    r["voxel_size_mm"] = "x".join(f"{z:.2f}" for z in zooms[:3])
    r["tr_s"] = float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else tr
    r["documented_tr_s"] = tr
    r["duration_s"] = r["n_volumes"] * r["tr_s"]

    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        r["qc_decision"], r["qc_reason"] = "FAIL", "not a 4-D BOLD series"
        return r

    # ---- tSNR ------------------------------------------------------------
    mean_v = data.mean(axis=3)
    sd_v = data.std(axis=3)
    brain = mean_v > np.percentile(mean_v, 60)     # simple intensity mask
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(sd_v > 0, mean_v / sd_v, np.nan)
    r["median_tsnr"] = float(np.nanmedian(tsnr[brain]))
    r["mean_tsnr"] = float(np.nanmean(tsnr[brain]))
    r["n_brain_voxels"] = int(brain.sum())
    r["mean_signal"] = float(mean_v[brain].mean())
    r["nan_fraction"] = float(np.mean(~np.isfinite(data[..., 0])))

    # ---- motion / censoring ---------------------------------------------
    motion = read_head_motion(row.get("aux_path"))
    if motion is not None and motion.shape[0] > 1:
        fd = framewise_displacement(motion)
        r["motion_source"] = "provider head_motion tsv"
        r["mean_fd_mm"] = float(np.mean(fd))
        r["median_fd_mm"] = float(np.median(fd))
        r["max_fd_mm"] = float(np.max(fd))
        r["mean_dvars"] = np.nan          # keep both motion schemas identical
        cens = fd > FMRI_RULES["fd_censor_mm"]
        r["n_censored_volumes"] = int(cens.sum())
        r["censored_fraction"] = float(cens.mean())
        r["n_usable_volumes"] = int(len(fd) - cens.sum())
        r["max_abs_translation_mm"] = float(np.abs(motion[:, :3]).max())
        r["max_abs_rotation_rad"] = float(np.abs(motion[:, 3:6]).max())
    else:
        # No provider motion file: derive a DVARS-based proxy so the recording
        # still gets a censoring decision rather than silently passing.
        dv = np.sqrt(np.nanmean(np.diff(data[brain], axis=1) ** 2, axis=0))
        dv = np.concatenate([[0.0], dv])
        med = np.median(dv[1:]) if dv.size > 1 else 0.0
        thr = med * 2.0 if med > 0 else np.inf
        cens = dv > thr
        r["motion_source"] = "DVARS proxy (no motion file supplied)"
        r["mean_fd_mm"] = np.nan
        r["median_fd_mm"] = np.nan
        r["max_fd_mm"] = np.nan
        r["mean_dvars"] = float(np.mean(dv[1:])) if dv.size > 1 else np.nan
        r["n_censored_volumes"] = int(cens.sum())
        r["censored_fraction"] = float(cens.mean())
        r["n_usable_volumes"] = int(r["n_volumes"] - cens.sum())
        r["max_abs_translation_mm"] = np.nan
        r["max_abs_rotation_rad"] = np.nan

    # ---- ROI coverage ----------------------------------------------------
    roi = _roi_timeseries(img, data, slug, save_arrays)
    r.update(roi["summary"])

    # ---- events / design matrix -----------------------------------------
    ev = _events(row, r["n_volumes"], r["tr_s"])
    r.update(ev["summary"])

    # ---- event-locked ROI response --------------------------------------
    locked = _event_locked_roi(roi.get("ts"), ev.get("onsets"), ev.get("conditions"),
                               r["tr_s"], roi.get("labels"))
    r.update(locked["summary"])

    if save_arrays and roi.get("ts") is not None:
        out = ARRAY_DIR / "fmri_roi"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{slug}.npz",
                            roi_timeseries=roi["ts"].astype(np.float32),
                            roi_labels=np.array(roi.get("labels", []), dtype=object),
                            roi_voxel_counts=roi.get("voxel_counts", np.array([])),
                            design_matrix=ev.get("design", np.array([])),
                            condition_names=np.array(ev.get("condition_names", []),
                                                     dtype=object))
    if save_figs:
        _figure_fmri(row, tsnr, brain, roi, ev, locked, slug, r)

    r.update(_decide_fmri(r))
    return r


# =========================================================================
def _roi_timeseries(img, data, slug, save_mask=True):
    """Extract the 48 Harvard-Oxford ROI time series + coverage bookkeeping."""
    from nilearn.image import resample_to_img
    import nibabel as nib

    try:
        atlas = get_atlas()
        amap = atlas.maps if hasattr(atlas, "maps") else atlas["maps"]
        labels = list(atlas.labels if hasattr(atlas, "labels") else atlas["labels"])
        ref = nib.Nifti1Image(data[..., 0], img.affine, img.header)
        res = resample_to_img(amap, ref, interpolation="nearest", force_resample=True,
                              copy_header=True)
        lab = np.asarray(res.dataobj).astype(int)
    except Exception as exc:
        # Same key set as the success path, so the CSV schema does not shift.
        return {"summary": {"n_rois": N_ROIS, "n_rois_covered": 0,
                            "roi_coverage": np.nan, "min_roi_voxels": 0,
                            "median_roi_voxels": np.nan, "empty_roi_idx": "",
                            "roi_mask_path": "", "roi_error": str(exc)[:160]},
                "ts": None, "labels": [], "voxel_counts": np.array([])}

    n_t = data.shape[3]
    ts = np.zeros((n_t, N_ROIS), dtype=np.float32)
    counts = np.zeros(N_ROIS, dtype=int)
    for k in range(1, N_ROIS + 1):
        m = lab == k
        counts[k - 1] = int(m.sum())
        if counts[k - 1] >= FMRI_RULES["min_voxels_per_roi"]:
            ts[:, k - 1] = data[m].mean(axis=0)

    covered = (counts >= FMRI_RULES["min_voxels_per_roi"]) & (ts.std(axis=0) > 0)
    mask_path = ""
    if save_mask:
        d = ARRAY_DIR / "fmri_roi_masks"
        d.mkdir(parents=True, exist_ok=True)
        mask_path = str(d / f"{slug}_roimask.nii.gz")
        nib.save(nib.Nifti1Image(lab.astype(np.int16), img.affine), mask_path)

    names = [str(x) for x in labels[1:N_ROIS + 1]] if len(labels) > N_ROIS else []
    return {
        "summary": {
            "n_rois": N_ROIS,
            "n_rois_covered": int(covered.sum()),
            "roi_coverage": float(covered.mean()),
            "min_roi_voxels": int(counts.min()),
            "median_roi_voxels": float(np.median(counts)),
            "empty_roi_idx": ";".join(map(str, np.flatnonzero(~covered).tolist())),
            "roi_mask_path": mask_path,
            "roi_error": "",
        },
        "ts": ts, "labels": names, "voxel_counts": counts,
    }


# =========================================================================
def _events(row, n_vol, tr):
    """Read the events file and build a boxcar design matrix."""
    import pandas as pd

    p = clean_path(row.get("events_path"))
    empty = {"summary": {"n_events": 0, "n_conditions": 0, "condition_names_str": "",
                         "event_coverage_frac": np.nan, "events_source": "none"},
             "onsets": None, "conditions": None, "design": np.array([]),
             "condition_names": []}
    if p is None or not p.exists():
        return empty
    try:
        if p.suffix == ".tsv":
            df = pd.read_csv(p, sep="\t")
        else:
            return empty
    except Exception:
        return empty
    if "onset" not in df.columns:
        return empty

    df = df.dropna(subset=["onset"])
    cond_col = "trial_type" if "trial_type" in df.columns else None
    names = sorted(df[cond_col].astype(str).unique()) if cond_col else ["all"]
    design = np.zeros((n_vol, len(names)), dtype=np.float32)
    t_vol = np.arange(n_vol) * tr
    for j, nm in enumerate(names):
        sub = df if cond_col is None else df[df[cond_col].astype(str) == nm]
        for _, e in sub.iterrows():
            dur = float(e.get("duration", 0) or 0) or tr
            design[(t_vol >= float(e["onset"])) &
                   (t_vol < float(e["onset"]) + dur), j] = 1.0

    onsets = df["onset"].to_numpy(dtype=float)
    span = (onsets.max() - onsets.min()) if onsets.size > 1 else 0.0
    return {
        "summary": {
            "n_events": int(len(df)),
            "n_conditions": len(names),
            "condition_names_str": ";".join(names)[:200],
            "event_coverage_frac": float(span / (n_vol * tr)) if n_vol * tr > 0 else np.nan,
            "events_source": Path(p).name,
        },
        "onsets": onsets,
        "conditions": df[cond_col].astype(str).to_numpy() if cond_col else None,
        "design": design, "condition_names": names,
    }


# =========================================================================
def _event_locked_roi(ts, onsets, conditions, tr, labels):
    """Event-locked ROI averages, reported for the motor/sensorimotor subset."""
    blank = {"summary": {"motor_roi_evoked_peak": np.nan,
                         "motor_roi_evoked_snr": np.nan,
                         "n_motor_rois": 0, "event_locked_source": "unavailable"}}
    if ts is None or onsets is None or len(onsets) == 0 or not labels:
        return blank

    motor_idx = [i for i, nm in enumerate(labels)
                 if any(k.lower() in nm.lower() for k in MOTOR_ROI_LABELS)]
    if not motor_idx:
        return blank

    x = ts[:, motor_idx].T                       # (roi, time)
    mu = x.mean(axis=1, keepdims=True)
    x = (x - mu) / np.where(mu != 0, np.abs(mu), 1.0) * 100.0   # percent signal change
    ep = M.epoch(x, 1.0 / tr, onsets, -2.0, 16.0)
    if ep.shape[0] == 0:
        return blank
    m, ci = M.mean_ci(ep, axis=0)
    t = np.arange(m.shape[1]) * tr - 2.0
    pk = np.argmin(np.abs(t - 6.0))
    peak = float(np.nanmean(m[:, pk]))
    cih = float(np.nanmean(ci[:, pk]))
    return {"summary": {
        "motor_roi_evoked_peak": peak,
        "motor_roi_evoked_snr": float(abs(peak) / cih) if cih > 0 else np.nan,
        "n_motor_rois": len(motor_idx),
        "event_locked_source": "motor ROI subset, percent signal change",
    }}


# =========================================================================
def _decide_fmri(r: dict) -> dict:
    hits: list[RuleHit] = []
    R = FMRI_RULES

    if np.isfinite(r.get("mean_fd_mm", np.nan)):
        hits.append(band(r["mean_fd_mm"], R["warn_mean_fd_mm"], R["max_mean_fd_mm"],
                         "mean_fd_mm", "excessive head motion"))
    hits.append(band(r.get("censored_fraction"), R["warn_censored_fraction"],
                     R["max_censored_fraction"], "censored_fraction",
                     "too many volumes censored for motion"))
    hits.append(band(r.get("median_tsnr"), R["warn_median_tsnr"], R["min_median_tsnr"],
                     "median_tsnr", "temporal SNR too low to carry BOLD signal",
                     higher_is_worse=False))
    hits.append(band(r.get("n_usable_volumes"), R["warn_usable_volumes"],
                     R["min_usable_volumes"], "n_usable_volumes",
                     "too few usable volumes for 30 s windows",
                     higher_is_worse=False))
    hits.append(band(r.get("roi_coverage"), R["warn_roi_coverage"], R["min_roi_coverage"],
                     "roi_coverage", "atlas ROIs missing from the field of view",
                     higher_is_worse=False))
    if r.get("n_events", 0) == 0:
        hits.append(RuleHit("n_events", "WARN", 0, 1,
                            "no events file: recording cannot be epoched by condition"))
    if abs(r.get("tr_s", 0) - r.get("documented_tr_s", 0)) > 0.01:
        hits.append(RuleHit("tr_s", "WARN", r.get("tr_s"), r.get("documented_tr_s"),
                            "TR disagrees with the documented value"))

    hits = [h for h in hits if h is not None]
    decision, reason = evaluate(hits)
    return {"qc_decision": decision, "qc_reason": reason, "n_rule_hits": len(hits)}


# =========================================================================
def _figure_fmri(row, tsnr, brain, roi, ev, locked, slug, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = FIG_DIR / row["dataset"] / "fmri"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle(f"{slug}   [{r.get('qc_decision','QC')}]", fontsize=10)

    z = tsnr.shape[2] // 2
    im = ax[0, 0].imshow(np.rot90(np.nan_to_num(tsnr[:, :, z])), cmap="viridis",
                         vmin=0, vmax=np.nanpercentile(tsnr[brain], 95))
    ax[0, 0].set_title(f"tSNR, mid-slice (median {r['median_tsnr']:.1f})", fontsize=9)
    ax[0, 0].axis("off")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046)

    if roi.get("ts") is not None:
        ax[0, 1].imshow(roi["ts"].T, aspect="auto", cmap="RdBu_r",
                        vmin=np.nanpercentile(roi["ts"], 2),
                        vmax=np.nanpercentile(roi["ts"], 98))
        ax[0, 1].set_title(f"ROI time series ({r.get('n_rois_covered',0)}/"
                           f"{r.get('n_rois',48)} covered)", fontsize=9)
        ax[0, 1].set_xlabel("volume"); ax[0, 1].set_ylabel("ROI")

    d = ev.get("design")
    if d is not None and np.size(d):
        ax[1, 0].imshow(d.T, aspect="auto", cmap="Greys", interpolation="nearest")
        ax[1, 0].set_yticks(range(len(ev["condition_names"])))
        ax[1, 0].set_yticklabels(ev["condition_names"], fontsize=6)
        ax[1, 0].set_title(f"event design matrix ({r.get('n_events',0)} events)",
                           fontsize=9)
        ax[1, 0].set_xlabel("volume")
    else:
        ax[1, 0].text(0.5, 0.5, "no events file", ha="center")

    fd_txt = (f"mean FD {r['mean_fd_mm']:.3f} mm" if np.isfinite(r.get("mean_fd_mm", np.nan))
              else r.get("motion_source", ""))
    ax[1, 1].bar(["censored", "usable"],
                 [r.get("n_censored_volumes", 0), r.get("n_usable_volumes", 0)],
                 color=["firebrick", "seagreen"])
    ax[1, 1].set_title(f"volumes: {fd_txt}\nmotor ROI evoked peak "
                       f"{locked['summary']['motor_roi_evoked_peak']:.3f}% "
                       f"(SNR {locked['summary']['motor_roi_evoked_snr']:.2f})",
                       fontsize=9)

    fig.tight_layout()
    fig.savefig(out / f"{slug}.png", dpi=85)
    plt.close(fig)
