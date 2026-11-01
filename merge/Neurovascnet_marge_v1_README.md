# Neurovascnet_marge_v1: what is in the notebook, and what to check before running it

**Notebook:** `Neurovascnet_marge_v1.ipynb` (47 cells, 27 code)
**Plain-Python copy for reviewing:** `Neurovascnet_marge_v1_export.py` (same code, cells marked `# %% [code cell K | notebook cell N]`)
**Built from:** `NeuroVascNet_FINAL_muntasir.ipynb` (base) + `NeuroVascNet_v3_1_maisha_RUNNABLE_v2_3_IMPROVED1.ipynb`, following `NeuroVascNet_Merge_Plan.md`, then corrected after `CODEX_AUDIT_REPORT.md`.

**Status:** every cell runs top to bottom on synthetic data in both encoder settings. It has **not** been run on the real datasets. Section 4 lists what must be checked on the machine that has them.

---

## 1. What the notebook does

It predicts brain blood-flow signals (fNIRS HbO/HbR and fMRI BOLD) from EEG alone, and tests five claims.

**Dataset roles** (Muntasir's, chosen from target quality before any model was trained):

| Role | Dataset | People |
|---|---|---|
| Train fNIRS | TU Berlin hBCI, motor imagery + mental arithmetic | 29 |
| Train fMRI | OpenNeuro ds002336 | 10 |
| External fNIRS | HEFMI-ICH | 37 |
| External fMRI | OpenNeuro ds002739 | 24 |
| Clinical comparison | HEFMI-ICH (17 healthy, 20 ICH) | 37 |

**The five claims:**
1. Adding fMRI supervision improves fNIRS prediction (vs fNIRS alone and vs shuffled fMRI)
2. The model transfers to unseen datasets
3. A neurovascular delay exists (3a, measured without the model) and the model recovers it (3b)
4. Blood-flow training gives a better EEG representation for other tasks
5. The confidence score identifies bad predictions, and the "refuse to answer" rule works

All decisions (roles, primary encoder, metric, seeds, thresholds) are written in **cell 0 before running**. Do not change them after seeing full-run results.

---

## 2. Features, by section

"Source" says where each part comes from: **M** = Muntasir, **S** = Maisha, **New** = added during the merge, **Audit** = added after the Codex audit.

### 2.1 Setup and run protocol

| Feature | Source |
|---|---|
| One settings cell with a path per dataset and role switches | M |
| New switches: `ENCODER` (`trunk` / `eegnet_tcn`), `EEG_INPUT` (`raw` / `bandpower`), `EEG_FILTER`, `DS2336_TASKS`, `BALANCED_SUBJECTS`, `RUN_SWEEPS`, `USE_CACHE` | New |
| 5-seed protocol (1337, 20260726, 7, 42, 2718), one seed per kernel run; the next missing seed is chosen automatically | S |
| Folds and splits stay fixed (seed 1337); each run's seed changes only model training | New |
| Sweeps and learned-delay heads run only in the first two seeds, to save time | New |
| **Run fingerprint:** every switch, setting and path is hashed into the output folder name, so runs with different settings are never mixed or aggregated | Audit |
| A seed only counts as finished if its fingerprint matches and the claim guard passed | Audit |
| Numeric warnings shown once; cosmetic warnings silenced | S |

### 2.2 Data loading

| Feature | Source |
|---|---|
| Loaders for all four datasets: hBCI cue-locked −2 to +28 s with own Beer-Lambert conversion; HEFMI epochs; ds002336 aligned to scanner triggers; ds002739 locked to stimulus events | M |
| EEG band-pass 0.5–45 Hz + 50 Hz notch on every dataset | S (fix of a gap in M) |
| ds002739 electrode names read from `electrode_info.elp`; runs with fewer than 8 recognised names are rejected | S + Audit |
| **ds002739 EEG windows are now 30 s.** The original loader took 6,000 samples at 1,000 Hz, i.e. only 6 s | New (bug fix) |
| Only standard 10-05 electrodes kept; channel order checked across runs; names must match the array | New + Audit |
| ds002336 loader stops if the EEG is not 200 Hz | Audit |
| Recording IDs for hBCI and HEFMI (needed for the delay test-retest) | New |
| `DS2336_TASKS` switch to drop neurofeedback runs | New |
| Data cache keyed on the loader code, helper functions and data settings. Unchanged code reloads in seconds; any change rebuilds. If the code can't be read, the cache switches off with a warning | S + Audit |
| Channel report: channels per dataset, empty channels, and whether each external montage maps to training by **spherical spline** or **by index** | S + New |

### 2.3 Data-quality checks (before any model)

| Feature | Source |
|---|---|
| Target reliability (split-half), ceiling, HbO/HbR sign | M |
| EEG↔blood-flow timing check per dataset (PASS/WARN by share of people peaking at 2–8 s) | S |
| GSR A/B test on training fNIRS (GSR stays off, as declared) | S |

### 2.4 Statistics

| Feature | Source |
|---|---|
| **Subject-level tests everywhere:** mean per-subject r, within-subject pairing null, sign-flip test, subject bootstrap CI. They replace the old trial-level p-values | S (fix of M) |
| Demeaning by subject × task, plus an extra row by subject × task × class | New |
| MAE and RMSE | S |
| Plain-language reading of each result: "detectable", "detectable, negligible size" or "not distinguishable from the null" | New |

### 2.5 Model and training

| Feature | Source |
|---|---|
| Muntasir's conv trunk (primary encoder) | M |
| Maisha's EEGNet + TCN encoder with per-dataset input normalisation (secondary) | S |
| Fixed band-power front end (4 bands at 20 Hz) inside the model, so electrode interpolation still works on raw EEG | New (Maisha's idea) |
| Joint training: both modalities in one backward pass per step | M |
| Optional subject-balanced sampling | S |
| Batch-wise prediction that applies the electrode-interpolation matrix on the fly (no multi-GB copies) | New |

### 2.6 The five claims

| Claim | What is tested | Source |
|---|---|---|
| Internal CV | Subject-wise 5-fold CV; r against a predictor that ignores EEG; subject-level p; MAE | M + new statistics |
| **1** | Single vs joint vs joint-with-shuffled-fMRI, paired per subject; `w_fmri` grid and trunk-width sweep as **sensitivity only** (never used to re-pick the setting) | M (sweep made non-selecting) |
| **2** | External datasets after spherical-spline electrode interpolation; few-shot calibration curve; a dataset counts only if detectable **and** above the practical size (r ≥ 0.05) | M + New |
| **3a** | Delay measured without the model on hBCI (primary) and HEFMI, with test-retest ICC and a **shuffled-trial null** (EEG of one trial paired with HbO of another) | New (fix of M) |
| **3b** | Learned delay head, bounded vs unbounded, with the pretrained model frozen and a real correlation at each lag | M + Audit |
| **4** | Frozen probe, shuffled control, random init, band-power, fine-tuning from supervised vs scratch, depth read-out. **Each probe fold uses an encoder that never saw that fold's people** | M + Audit |
| **5** | Three uncertainty signals → learned gate → temperature → isotonic. Fixed error tolerance. **Nested calibration** (the scored half never enters any model used for fitting). Locked "refuse to answer" threshold. External test on HEFMI. Deployment function that returns NaN below the threshold | S + New + Audit |

### 2.7 Supporting analyses

| Feature | Source |
|---|---|
| Same-person probe: earlier 70% of each person's trials train, later 30% test, with a shuffled null | S |
| Leave-one-subject-out ridge baseline | S |
| 4-number endpoint (HbO mean, HbR mean, HbO peak, HbR trough) and personalisation with a **no-EEG control** (the person's own average) | S, now on hBCI's 29 people |
| Amplitude evaluation (between-subject, within-subject, cross-subject) | S |
| Leakage comparison: trial split vs honest subject split | M |
| ICH vs controls with artefact checks and Holm correction | M |

### 2.8 Outputs and reproducibility

| Feature | Source |
|---|---|
| File of every prediction (subject, recording, task, label, fold, seed, per-trial r) + compressed predictions | S |
| Summary JSON per seed with full settings, scorecard and claim guard | S + New |
| Checkpoint with weights, full calibrator, reference statistics and model shapes (load with `torch.load(..., weights_only=False)`) | S + Audit |
| 5-seed aggregator (mean, SD, min, max; marks results that don't depend on the seed) | S |
| Summary figure | New |

---

## 3. Requirements on the running machine

| Item | Needed |
|---|---|
| Python | 3.12 (Muntasir's run used 3.12.0) |
| Packages | numpy **< 2** (1.26.4 was used), torch with CUDA, scipy, scikit-learn, pandas, matplotlib, mne, nilearn, nibabel. Cell 1 installs missing ones |
| GPU | Muntasir's run used an RTX 3060 Ti (8 GB). EEGNet + TCN on **raw** EEG is not recommended on 8 GB; use it with `bandpower` |
| RAM | 32 GB. ds002739 EEG alone is about 6.5 GB in memory |
| Disk | About **15 GB free** for the data cache (`neurovascnet_merge_v1_cache/`) plus outputs |
| Internet | Needed once for nilearn's Harvard-Oxford atlas, unless it is already in `nilearn_data` |
| Datasets | Each path in the settings cell must contain: HEFMI `epo_data/`, hBCI `EEG data/` (and `NRI data/NIRS_01-29/`), ds002336 `derivatives/`, ds002739 `sub-01/` |

---

## 4. Checklist before and during the first run

### Step 1: settings
- [ ] Set every `PATH_...` line in the settings cell to the dataset folders on that machine.
- [ ] Leave `QUICK_TEST = True` for the first run.
- [ ] Confirm the primary switches: `ENCODER = "trunk"`, `EEG_INPUT = "raw"`, `EEG_FILTER = True`, all six `DS2336_TASKS`, `BALANCED_SUBJECTS = False`.
- [ ] The path cell must print `[ok  ]` for all five roles.

### Step 2: after the loading cell (section 3), before any training

Paste this into a new cell right after the loading cell and check the printed values against the expectations below.

```python
for role, d in DATA.items():
    print(role, d["name"], d["eeg"].shape, d["eeg"].dtype, f"{d['eeg'].nbytes / 1e9:.1f} GB",
          d["hemo"].shape, "people:", len(np.unique(d["subject_ids"])))
    names = d.get("ch_names")
    print("   electrode names:", None if names is None else len(names), (names or [])[:10])
    assert names is None or len(names) == d["eeg"].shape[1], "names do not match EEG rows"

h = DATA["train_fnirs"]
assert len({len(h[k]) for k in ("eeg", "hemo", "labels", "subject_ids", "task_ids", "rec_ids")}) == 1
print(pd.crosstab(h["subject_ids"], h["task_ids"]))   # trials per person: MI (0) and MA (1)
print(pd.crosstab(h["task_ids"], h["labels"]))        # MA labels must sit above MI labels
print(pd.Series(h["rec_ids"]).value_counts().sort_index())
```

| Check | Expected | If not |
|---|---|---|
| hBCI | 29 people, about 1,740 MI + 1,740 MA trials (about 60 + 60 per person); EEG (N, 30, 6000), target (N, 72, 300) | A person with 0 MI or 0 MA trials means a loader problem; fix before continuing |
| hBCI labels | MI labels 0–1, MA labels 2–3 (never overlapping) | Stop: the class demeaning and Claim 4 would be wrong |
| HEFMI | 37 people, about 2,940 trials; 32 electrode names | |
| ds002336 | 10 people, about 384 trials; about 60 electrode names; no run rejected for the "not 200 Hz" check | If runs are rejected, read the loader's audit table |
| ds002739 | 24 people; about 60 electrode names that look like Fp1, Fz, C3 …; target (N, 5, 15) | Trial count may be a little lower than Muntasir's 4,678, because 30 s windows need 30 s of EEG after the event. If many runs are rejected for "only N named electrodes", inspect `electrode_info.elp` |
| Memory | Total EEG about 12 GB; RAM still free | If close to 32 GB, lower `MAX_FMRI_TRIALS_PER_RUN` (declare the change) |

### Step 3: channel report (section 3)
- [ ] `maps_to_training_montage` says **spherical spline** for both `external_fnirs` and `external_fmri`. **BY INDEX** means electrode names were not recognised; fix that before trusting Claim 2.
- [ ] `zero_channels` is 0 for every dataset.

### Step 4: data-quality section (section 4)
- [ ] Target reliability is close to Muntasir's saved values (hBCI ≈ 0.50, ds002336 ≈ 0.70, HEFMI ≈ 0.08, ds002739 ≈ 0.19). Large differences mean the data or preprocessing changed.
- [ ] Timing table: write down PASS/WARN per dataset. WARN on hBCI restricts timing and delay claims; the claim guard will list it.
- [ ] GSR line prints a removed share and a uniformity value (report only; GSR stays off).

### Step 5: a timing sanity check on ds002739 (from the Codex audit)
The loader assumes the event onsets, the EEG start and the first fMRI volume share the same time zero. Before trusting Claim 2's fMRI result, print for two or three runs:
- the EEG sampling rate (`m.fd`), EEG duration and number of BOLD volumes;
- the first and last event onsets;
- `onset - round(onset / 2.0) * 2.0` for a few events, the gap between the EEG start and the nearest fMRI volume (up to 1 s).

If the dataset documentation says the EEG starts at a different time from the scan, the onsets need an offset.

### Step 6: quick run to the end
- [ ] Run All with `QUICK_TEST = True`. Every section should finish and print a scorecard. **The numbers are not reportable.**
- [ ] Check GPU memory after section 8: `print(torch.cuda.max_memory_allocated() / 1e9, "GB")`.

### Step 7: reproduce Muntasir's headline before the real run
The merged notebook adds EEG filtering, which can change results. To confirm nothing else broke:
- [ ] Set `QUICK_TEST = False`, `EEG_FILTER = False`, keep everything else, and run **up to the end of section 8 only**.
- [ ] The line `trial-pooled r (old estimator)` for hBCI should be close to **0.158** (within about ±0.01), and ds002336 close to **0.064**.
- [ ] If it is far off, stop and find the cause before the full run. Then set `EEG_FILTER = True` again.

### Step 8: the full run
- [ ] `QUICK_TEST = False`, primary switches as in cell 0.
- [ ] Restart the kernel and **Run All once per seed** (5 times). The settings cell picks the next missing seed; never run two seeds in one kernel.
- [ ] After the 5th seed the last cell prints `FIVE-SEED PROTOCOL COMPLETE` and writes `multi_seed_summary.csv`.
- [ ] Optional sensitivity runs, each as its own 5-seed set (they get their own folder automatically):
  - without neurofeedback: `DS2336_TASKS = ("motorloc", "MIpre", "MIpost")`
  - secondary encoder: `ENCODER = "eegnet_tcn"`, `EEG_INPUT = "bandpower"`

**Time:** each full seed trains about 56 models (CV 10, Claim 1 arms 25, final model 1, Claim 4 fold encoders 6, Claim 5 folds and nested folds 13, leakage ablation 1). The first two seeds also run the sweeps and delay heads, about 71 models in total. Expect it to take several times longer than Muntasir's original single run, so plan for overnight runs.

---

## 5. Known limitations (declared, not bugs)

- **ds002739 uses the 5 motor brain regions.** The fMRI head only predicts the regions it was trained on. ds002739 is a visual task, so its result is a cross-task stress test.
- **`w_fmri` stays at 1.0**; the grid is reported, never used to pick a value.
- **EEGNet + TCN** uses per-dataset normalisation only at the input layer.
- **The "refuse to answer" threshold** is set on the temperature-scaled score; the isotonic score is step-shaped and can't hit the declared 75% coverage.
- **ICC:** the delay ICC is a one-way ICC. Muntasir's original `delay_icc` (kept for continuity) calls its similar formula ICC(2,1).
- **Personalisation r** is computed over trials × 4 targets. A predictor that is constant across trials can still score r > 0; that is why the no-EEG control is printed next to it, and only the difference ("eeg_adds") is interpreted.
- **Left out on purpose:**
  - model-based ICH classification
  - Maisha's reference-lag delay check
  - the capacity test
- **The claim guard** checks Claim 1's consistency and lists restrictions. It does **not** certify the loaders or the absence of leakage; Section 4's checks do that.
- **Claim 4 numbers are not comparable with Muntasir's saved ones**: his encoder had seen the probe's test people; the merged notebook's does not.
