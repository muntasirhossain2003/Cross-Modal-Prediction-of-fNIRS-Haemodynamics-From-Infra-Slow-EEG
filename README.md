# NeuroVascNet

**Predicting fNIRS and fMRI haemodynamics from infra-slow scalp EEG.**

This repository holds the code, analysis plan and result files behind the manuscript

> R. Jany, M. Hossain, M. Rahman. *Cross-Modal Prediction of fNIRS Haemodynamics From Infra-Slow EEG:
> Ridge Regression, Shared Low-Rank Space and Two-Run Calibration.*

The datasets are public but are not stored here (see [Data](#data)).

## Main findings

| Question | Answer | Where |
|---|---|---|
| Where is the EEG information about haemodynamics? | Below 0.5 Hz. A standard 0.5–45 Hz band-pass cut cross-person HbO prediction from r = 0.158 to 0.019 in hBCI. Infra-slow EEG beat band power in three cohorts (r = 0.135–0.203 against 0.006–0.045). | `analysis/run_bands.py`, `run_p2_full.py`, `replicate_mw.py`, `replicate_finemi.py` |
| Can one EEG representation serve fNIRS and fMRI? | Yes. A rank-32 shared space predicted both in unseen people with no loss for either (non-inferiority margin −0.02). fMRI supervision did not improve fNIRS. | `analysis/claim1_*.py` |
| Does short calibration enable EEG-only monitoring? | Yes. After two paired runs the EEG-only model beat each person's mean response by 0.059–0.176, and the calibration score predicted later accuracy (Spearman ρ = 0.56–0.90). | `analysis/calibration.py`, `calib_reliability.py`, `nvc_more.py` |
| Is there a neurovascular delay? | Band power led local HbO by about 6 s in hBCI (29 of 29 people). The kernel shape repeated within people; the precise delay repeated only in hBCI. | `analysis/nvc_bandpower.py`, `nvc_residual.py`, `nvc_kernel.py` |
| Do models transfer between laboratories? | Within a laboratory, yes. Frozen models did not transfer to another laboratory under any harmonisation tried. | `analysis/external_*.py`, `nvc_transfer*.py`, `harmonised_transfer.py` |
| Does it hold in a third laboratory? | Not established: only band-power coupling passed there. | `analysis/nvc_mendeley.py`, `nvc_method_check.py` |

All pass rules were written into a dated plan before each test, and Bonferroni correction counts every attempt
([`docs/FINAL_PLAN.md`](docs/FINAL_PLAN.md), [`docs/TEST_LEDGER.md`](docs/TEST_LEDGER.md)).

## Repository layout

```text
pipeline/   Neurovascnet_marge_v1.ipynb   loads all datasets, builds the caches, deep-model comparator
            NeuroVascNet_Results.ipynb    every current result, read from analysis/results/
            NeuroVascNet_Full.ipynb       all analyses as plain functions, with a switch to rerun any of them
analysis/   one script per test (named after the plan step, e.g. P5.1); results/ holds the saved JSON results
paper/      make_figures.py and the other make_*.py scripts that draw the paper figures; figures/
qc/         data-quality audit of every dataset (signal quality, units, synchronisation)
docs/       FINAL_PLAN.md (dated analysis plan), TEST_LEDGER.md (every test and its multiplicity family),
            RESULTS_SUMMARY.md, data-quality audit, reports/, lectures/, presentation/, data_notes/
archive/    earlier notebooks (v3 to v6), run logs, old scripts and outputs, kept for the record
```

Folders that stay on your machine and are ignored by git: `dataset/` (raw data),
`pipeline/neurovascnet_merge_v1_cache/` (about 60 GB of epoch arrays) and `pipeline/_nb_run/` (rerun output).

## Data

Put each dataset in `dataset/` at the repository root, in a folder with the name below. To keep the data
elsewhere, set the environment variable `NVN_DATA` to that folder.

| Cohort | People | Folder in `dataset/` | Source |
|---|---|---|---|
| hBCI (TU Berlin): motor imagery, mental arithmetic | 29 | `TU Berlin hBCI EEG+FNRI` | Shin et al. 2017, IEEE TNSRE; <http://doc.ml.tu-berlin.de/hBCI> |
| MW (TU Berlin): n-back | 26 | `TU Berlin EEG + NIRS` | Shin et al. 2018, Sci Data; <http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS> |
| FineMI: multi-joint upper-limb motor imagery | 18 | `Multi-joint Upper-Limb EEG+FNIR/FineMI/FineMI` | Yi et al. 2025, Sci Data 12:953 |
| Learning capability (third laboratory) | 30 (15 used) | `mendeley Simultaneous EEG-fNIRS Data on Learning Capability` | Mendeley Data, doi:10.17632/tsfs9fhn5y.1 |
| EEG–fMRI motor-imagery neurofeedback | 10 + 17 | `OpenNeuro ds002336`, `OpenNeuro ds002338` | Lioi et al. 2020, Sci Data; <https://openneuro.org> |

HEFMI-ICH and OpenNeuro ds002739 are read by the pipeline notebook but were excluded from the paper, because
their published EEG has no power below 0.5 Hz.

## Setup

Python 3.12 was used. Install the packages with

```bash
pip install -r requirements.txt
```

A CUDA GPU is needed only for the deep comparator in the pipeline notebook; every ridge analysis runs on a CPU.

## Reproducing the results

1. **See the results without the data.** Open `pipeline/NeuroVascNet_Results.ipynb`. It reads the saved files in
   `analysis/results/` and needs no dataset.
2. **Build the caches.** Run `pipeline/Neurovascnet_marge_v1.ipynb` with the datasets in place. It writes
   `pipeline/neurovascnet_merge_v1_cache/`. Then build the extra cohorts:
   `python analysis/build_finemi.py` and `python analysis/build_ds002338.py`.
3. **Rerun an analysis.** Each script in `analysis/` writes its result JSON next to itself, for example
   `python analysis/calibration.py` or `python analysis/claim1_fresh.py`. The docstring at the top of each
   script names its plan step and any modes or environment variables (for example
   `FNIRS_COHORT=finemi python analysis/claim1_fresh.py`). `pipeline/NeuroVascNet_Full.ipynb` runs the same
   analyses from one place: set `RUN = {"calibration"}`, or `RUN = ALL` for everything (a day or more).
4. **Draw the figures.** `python paper/make_figures.py` redraws the result figures from the saved JSON files.
   `paper/make_signal_figure.py` (recorded signals, needs the cache), `make_architecture_figure.py` and
   `make_pipeline_figure.py` draw the remaining paper figures.

### Paper figures and tables

| Paper | Script | Output |
|---|---|---|
| Figure 1, analysis pipeline | `paper/make_pipeline_figure.py` | |
| Figure 2, recorded signals | `paper/make_signal_figure.py` | |
| Figure 3, model architecture | `paper/make_architecture_figure.py` | |
| Figure 4, two methods | `paper/make_figures.py` | `figures/fig16_method_check` |
| Figure 5, person-level reliability | `paper/make_figures.py` | `figures/fig14_reliability_people` |
| Figure 6, lag curves | `paper/make_figures.py` | `figures/fig2_coupling_lag` |
| Figure 7, kernel reproducibility | `paper/make_figures.py` | `figures/fig3_kernel_repro` |
| Figure 8, kernel transfer | `paper/make_figures.py` | `figures/fig4_kernel_transfer` |
| Figure 9, strengthening attempts | `paper/make_figures.py` | `figures/fig18_improvement` |
| Tables 2–5 | `pipeline/NeuroVascNet_Results.ipynb` | read from `analysis/results/` |

## Notes

- The paper's numbers come from the JSON files in `analysis/results/` (and `analysis/` for the most recent
  rounds). The few values that `make_figures.py` types in name their source section in `docs/RESULTS_SUMMARY.md`.
- FineMI failed an internal quality audit before it was adopted (median scalp coupling index 0.023), so its
  results are treated as provisional; see `docs/NeuroVascNet_Data_Quality_Audit.md`.
- Files in `archive/` are the development history. Their paths still point at the old layout and they are not
  maintained.

## Authors

Rafsan Jany (University of Science and Technology and Korea Institute of Oriental Medicine, Daejeon), corresponding
author; Muntasir Hossain and Maisha Rahman (Islamic University of Technology, Gazipur).
