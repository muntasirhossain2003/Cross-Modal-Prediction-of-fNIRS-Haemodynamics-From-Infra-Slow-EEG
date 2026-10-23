"""Builds lectures/Lecture_C_Reference_General_Knowledge.pdf.

General background on neuroimaging data and preprocessing (Part C companion).
Run from the repo root:  python scripts/build_lecture_part_c_reference.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps

from lecture_common import (AQUA, BLUE, GRAY, INK, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, K, M, V,
                            fig, qa_bank, table, title_block)

D = Doc("lecture_c_ref", "Lecture_C_Reference_General_Knowledge.pdf", "NeuroVascNet course · Lecture C reference")


def f_aliasing():
    t = np.linspace(0, 1, 2000)
    f_true = 9
    fs = 10
    ts = np.arange(0, 1.0001, 1 / fs)
    fig, a = plt.subplots(figsize=(W, 2.2))
    a.plot(t, np.sin(2 * np.pi * f_true * t), color=GRAY, lw=1, label="real signal: 9 Hz")
    a.plot(ts, np.sin(2 * np.pi * f_true * ts), "o", color=ORANGE, ms=6, label="samples taken at 10 Hz")
    a.plot(t, np.sin(2 * np.pi * (f_true - fs) * t), color=ORANGE, lw=2, ls="--", label="what the samples suggest: 1 Hz")
    a.set_yticks([]); a.set_xlabel("time (s)"); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8.3)
    a.set_title("Aliasing: sampling too slowly invents a slow wave that was never there", fontsize=10)
    D.save(fig, "rc1_alias")


def f_filters():
    fs = 100
    t = np.arange(0, 6, 1 / fs)
    x = np.zeros_like(t); x[(t > 2) & (t < 2.6)] = 1.0
    b, a_ = sps.butter(3, 3 / (fs / 2))
    causal = sps.lfilter(b, a_, x)
    zero = sps.filtfilt(b, a_, x)
    fig, a = plt.subplots(figsize=(W, 2.3))
    a.plot(t, x, color=GRAY, lw=1.2, label="original event")
    a.plot(t, causal, color=ORANGE, lw=2, label="causal filter (one pass): delayed")
    a.plot(t, zero, color=BLUE, lw=2, label="zero-phase filter (forward + backward): not delayed")
    a.set_xlim(1.5, 3.8); a.set_yticks([]); a.set_xlabel("time (s)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8.1)
    a.set_title("Filtering shifts timing unless it is zero-phase", fontsize=10)
    D.save(fig, "rc2_filters")


def f_leak():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.25})
    for a, title, ok in [(ax[0], "Leaky: normalise with ALL data, then split", False),
                         (ax[1], "Correct: split first, normalise with TRAINING data only", True)]:
        a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis("off"); a.set_title(title, fontsize=9.5)
        from lecture_common import box, arrow
        if not ok:
            box(a, 0.2, 2.6, 4.2, 0.9, "all people (train + test)", fc="#fdeceb", ec=RED, fs=8.5)
            box(a, 5.0, 2.6, 4.6, 0.9, "compute mean / std, scale", fc="#f3f5f8", ec=GRAY, fs=8.5)
            box(a, 0.2, 0.6, 4.2, 0.9, "train", fs=8.5); box(a, 5.0, 0.6, 4.6, 0.9, "test (already 'seen')", fc="#fdeceb", ec=RED, fs=8.5)
            arrow(a, (4.4, 3.05), (5.0, 3.05)); arrow(a, (7.3, 2.6), (7.3, 1.5)); arrow(a, (7.3, 2.6), (2.3, 1.5))
        else:
            box(a, 0.2, 2.6, 4.2, 0.9, "train people", fc="#e8f6f0", ec=AQUA, fs=8.5)
            box(a, 5.0, 2.6, 4.6, 0.9, "mean / std from train only", fc="#f3f5f8", ec=GRAY, fs=8.5)
            box(a, 5.0, 0.6, 4.6, 0.9, "apply same numbers to test", fc="#e8f6f0", ec=AQUA, fs=8.5)
            box(a, 0.2, 0.6, 4.2, 0.9, "test people (untouched)", fs=8.5)
            arrow(a, (4.4, 3.05), (5.0, 3.05)); arrow(a, (7.3, 2.6), (7.3, 1.5)); arrow(a, (4.4, 1.05), (5.0, 1.05))
    D.save(fig, "rc3_leak")


for f in [f_aliasing, f_filters, f_leak]:
    f()

QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "Formats, sharing and documentation"
qa(S, "What is BIDS?", "The Brain Imaging Data Structure: a community standard for organising neuroimaging files and metadata in folders with fixed names (sub-01/ses-01/eeg/…, plus JSON and TSV sidecars). Datasets on OpenNeuro follow it, so tools can read them automatically. There are BIDS extensions for EEG, MEG, fNIRS, iEEG and PET.")
qa(S, "Which file formats will I meet?", "EEG: EDF/BDF, BrainVision (.vhdr/.vmrk/.eeg), EEGLAB (.set), FIF. fNIRS: .nirs (Homer), SNIRF (the newer standard, HDF5-based). MRI: NIfTI (.nii/.nii.gz), DICOM from scanners. Many datasets also ship MATLAB .mat files with custom structures.")
qa(S, "What are the FAIR principles?", "Data should be Findable, Accessible, Interoperable and Reusable: persistent identifiers, open or clearly governed access, standard formats and metadata, and clear licences and documentation.")
qa(S, "Where do public neuroimaging datasets live?", "OpenNeuro (BIDS datasets, especially MRI and EEG), PhysioNet (physiological signals), figshare, Zenodo and institutional repositories, plus large consortia such as the Human Connectome Project and UK Biobank, which require applications.")
qa(S, "What ethical issues apply to shared brain data?", "Participants must have consented to sharing; data must be de-identified (names removed, MRI faces 'defaced' because faces can be reconstructed); clinical data may carry extra restrictions; and data-use agreements must be respected.")
qa(S, "Why check that data match their documentation?", "Files get renamed, versions change, and folders are mislabelled. Counting subjects, reading task names and checking sampling rates against the paper catches problems — such as a wrong dataset in the right folder — before they contaminate results.")

S = "Signals and sampling"
qa(S, "What is the Nyquist rule?", "To represent a frequency faithfully you must sample at more than twice that frequency. At 200 Hz you can represent up to 100 Hz. Frequencies above the Nyquist limit fold back as false low frequencies (aliasing).")
qa(S, "What is aliasing and how is it prevented?", "A fast signal sampled too slowly appears as a slower one that does not exist. It is prevented by applying a low-pass (anti-aliasing) filter below the Nyquist frequency before downsampling; good resampling functions do this automatically.")
qa(S, "What is the difference between FIR and IIR filters?", "FIR (finite impulse response) filters have a finite set of weights, are always stable and can have exactly linear phase, but need many coefficients for sharp cut-offs. IIR filters (such as Butterworth) use feedback, are efficient and sharp, but distort phase and can be unstable at extreme settings.")
qa(S, "What is zero-phase filtering?", "Running a filter forwards and then backwards over the data (filtfilt), which cancels the time shift the filter introduces. It is standard for offline analysis but impossible in real time, because it uses future samples.")
qa(S, "What is a notch filter?", "A filter that removes a narrow band around one frequency, typically mains interference at 50 or 60 Hz and its harmonics.")
qa(S, "Why can filtering create artifacts?", "Sharp filters ring (oscillate) around sudden events and can smear activity in time; high-pass filters with too high a cut-off distort slow responses such as ERPs or haemodynamic responses. Filter choices should match the signal of interest.")

S = "EEG preprocessing"
qa(S, "What are the usual EEG preprocessing steps?", "Load and check channels; filter (high-pass ~0.1–1 Hz, low-pass, notch); identify and interpolate bad channels; re-reference; remove artifacts (ICA, regression or rejection); epoch around events; baseline-correct; reject bad epochs. The order matters and should be reported.")
qa(S, "How does ICA remove artifacts?", "Independent component analysis unmixes the channels into statistically independent sources. Eye blinks, heartbeat and muscle often appear as separate components with characteristic shapes and scalp maps, which can be removed before projecting back to channels. Tools such as ICLabel help classify components.")
qa(S, "How are bad EEG channels repaired?", "By interpolation from neighbouring electrodes, usually with spherical splines (Perrin and colleagues, 1989), which use the electrodes' 3-D positions on a sphere fitted to the head.")
qa(S, "What is re-referencing and which reference is best?", "Expressing all channels relative to a new reference, such as the average of all electrodes, linked mastoids or a Laplacian. No reference is universally best; the average reference is common with dense montages. The choice changes waveforms, so it must be reported.")
qa(S, "Why baseline-correct epochs?", "To remove slow offsets so each epoch shows changes relative to a pre-stimulus period. The baseline window must contain no task-related activity.")
qa(S, "What is the difference between epoch rejection and artifact correction?", "Rejection throws away contaminated trials (simple, but loses data and can bias if one condition has more artifacts). Correction removes the artifact from the signal (keeps data, but may remove some brain signal).")

S = "fNIRS preprocessing"
qa(S, "What are the usual fNIRS preprocessing steps?", "Check signal quality (scalp coupling index, SNR) and prune bad channels; convert intensity to optical density; correct motion artifacts; filter (commonly ~0.01–0.2 Hz); convert to HbO and HbR with the modified Beer–Lambert law; regress short-separation channels; then block-average or fit a GLM.")
qa(S, "How are motion artifacts corrected in fNIRS?", "Common methods include spline interpolation of detected segments, wavelet filtering, correlation-based signal improvement (CBSI, which exploits the expected negative HbO/HbR correlation), and temporal derivative distribution repair (TDDR).")
qa(S, "Why does pre-converted fNIRS data limit what you can check?", "Once light intensities are converted to HbO and HbR, the two wavelengths are mixed, so measures that need raw light, such as the true scalp coupling index, can no longer be computed. You also cannot change the conversion parameters (DPF, distance).")
qa(S, "What units are fNIRS concentrations in?", "Changes are usually reported in µM (micromolar), sometimes mM or M, and sometimes multiplied by the pathlength (mM·mm) when the DPF is not applied. Always check and harmonise units before comparing datasets.")
qa(S, "What is the SNIRF format?", "The Shared Near Infrared Spectroscopy Format: an HDF5-based standard for fNIRS data, replacing the older .nirs format, and supported by BIDS-NIRS.")

S = "fMRI preprocessing"
qa(S, "What does fMRIPrep do?", "A widely used, standardised pipeline that performs motion correction, slice-timing correction, distortion correction, registration to the person's anatomy and to a standard template, and computes confound time series. Its outputs are reproducible and well documented.")
qa(S, "What are MNI space and brain atlases?", "MNI space is a standard brain template (from the Montreal Neurological Institute) into which individual brains are warped so locations can be compared. Atlases such as Harvard–Oxford, AAL or Schaefer divide that space into labelled regions for ROI analysis.")
qa(S, "What are tSNR, FD and DVARS?", "tSNR (temporal signal-to-noise ratio) is a voxel's mean divided by its standard deviation over time. Framewise displacement (FD) summarises head movement between volumes. DVARS measures how much the whole image changes from one volume to the next. High FD or DVARS flags volumes to censor or model.")
qa(S, "What are confound regressors?", "Time series of nuisance signals — motion parameters, white-matter and CSF signals (CompCor), physiological recordings (RETROICOR), drift terms — included in the model so their variance is not mistaken for neural activity.")
qa(S, "Why detrend fMRI time series?", "Scanner hardware drifts slowly, adding trends that are unrelated to the brain but can create spurious correlations and inflate reliability. Polynomial or high-pass detrending removes them.")

S = "Simultaneous recordings and alignment"
qa(S, "How is EEG synchronised with MRI or fNIRS?", "Through shared triggers: the scanner sends a pulse at every volume (or slice), which the EEG amplifier records; stimulus software sends event markers to all devices. Clock drift between devices is corrected using these triggers.")
qa(S, "What is AAS and what is OBS?", "Average artifact subtraction builds a template of the repeating gradient artifact by averaging many occurrences and subtracts it (Allen and colleagues, 2000). Optimal basis sets (OBS) use principal component analysis to capture artifact variation that a single template misses, and are often used for the ballistocardiogram.")
qa(S, "How do you combine datasets with different electrode layouts?", "Options include restricting to shared electrodes, interpolating every dataset onto a common layout with spherical splines, projecting to source space, or giving the model a dataset-specific input layer. Matching by name fails when layouts use different systems (10-20 vs 10-5).")

S = "Normalisation, leakage and splitting"
qa(S, "What are the common normalisation choices?", "Per trial (each window scaled by its own statistics), per subject (by that subject's statistics), or global (statistics from the training set). Per-trial and per-subject scaling remove between-person amplitude differences; per-subject scaling at test time is a form of adaptation that should be reported.")
qa(S, "What is data leakage in preprocessing?", "When information from the test data influences anything fitted during training: computing normalisation statistics, choosing filters, selecting channels or response windows using all data. It inflates performance and is often invisible in the code.")
qa(S, "Why must splits be made at the subject level?", "Because trials from one person share person-specific patterns. If the same person appears in training and test sets, a model can succeed by recognising the person, not by learning a general relationship.")
qa(S, "What is a stratified split?", "A split that keeps important proportions — patients and controls, classes, sites — similar in each part, so the test set is representative and results are not confounded with group.")
qa(S, "What is class or dataset imbalance and how is it handled?", "When one class or dataset has far more examples than another. It can be handled by reweighting the loss, oversampling or undersampling, or balanced batch sampling; the choice affects what the model learns.")


body = title_block("NeuroVascNet course · Lecture C · Reference", "Neuroimaging Data and Preprocessing: General Knowledge",
                   "Formats, sampling, filtering, the standard pipelines for EEG, fNIRS and fMRI, alignment and leakage",
                   "Companion to “The Data” · Read Lecture C first") + f"""
<div class="box">
<h3>How to use this document</h3>
<p>Lecture C described our four datasets and pipeline. This reference covers what any methods reviewer expects you to know about brain data in general. Sections 1–6 are reading; Section 7 is a bank of {len(QA)} questions with model answers. When asked "why did you do step X?", a good answer names <b>the problem the step solves</b>, <b>the standard way of doing it</b>, <b>the parameter you chose</b> and <b>what could go wrong</b>.</p>
</div>

<h2>1. Data standards and sharing</h2>
<p>Most brain datasets you will use were recorded by someone else. Standards make that possible. <b>BIDS</b> (Brain Imaging Data Structure) defines folder names and metadata files so that any tool can find subjects, sessions, tasks and events automatically; OpenNeuro hosts BIDS datasets. Formats differ by modality:</p>
{table(1, "Common file formats.",
       ["Modality", "Formats", "Notes"],
       [["EEG", "EDF/BDF, BrainVision (.vhdr .vmrk .eeg), EEGLAB .set, FIF", "BrainVision is common for EEG-fMRI"],
        ["fNIRS", ".nirs (Homer), SNIRF", "SNIRF is the current standard"],
        ["MRI", "DICOM (from the scanner), NIfTI (.nii)", "NIfTI is used for analysis"],
        ["Anything", "MATLAB .mat, HDF5, CSV/TSV", "custom structures need documentation"]],
       widths=["16%", "52%", "32%"])}
<p>Good data follow the <b>FAIR</b> principles (findable, accessible, interoperable, reusable), and ethical sharing requires consent, de-identification and, for MRI, removal of the face from structural images.</p>
{M("<p>Assuming the folder name tells you what the data are. Count subjects, read task labels and check sampling rates against the original paper. Mismatches between data and documentation are common and silently invalidate results.</p>")}

<h2>2. Sampling and filtering</h2>
<h3>2.1 Sampling</h3>
<p>A digital recording is a series of snapshots taken at a fixed <b>sampling rate</b>. The <b>Nyquist rule</b> says you must sample at more than twice the highest frequency you want to keep. If you sample too slowly, fast activity disguises itself as slow activity — <b>aliasing</b> — which no later processing can undo.</p>
{fig("rc1_alias", 1, "Aliasing.", "A 9 Hz rhythm sampled at only 10 Hz produces samples that look exactly like a 1 Hz wave. An anti-aliasing low-pass filter before downsampling prevents this.")}
<p>Changing the rate is <b>resampling</b>. Downsampling must first remove frequencies above the new Nyquist limit; standard functions do this, but hand-written "take every tenth sample" code does not.</p>

<h3>2.2 Filtering</h3>
<p>Filters keep some frequencies and remove others: <b>high-pass</b> removes slow drift, <b>low-pass</b> removes fast noise, <b>band-pass</b> keeps a range, and <b>notch</b> removes one frequency such as 50 Hz mains. <b>FIR</b> filters are stable and can have linear phase; <b>IIR</b> filters such as the Butterworth are compact and sharp but distort phase. Running a filter forwards and backwards (<b>zero-phase</b> filtering) cancels its time shift, which matters whenever timing matters — as it does when relating EEG to a delayed blood response.</p>
{fig("rc2_filters", 2, "Causal versus zero-phase filtering.", "A one-pass (causal) filter delays and smears an event; a forward–backward (zero-phase) filter keeps it in place. Zero-phase filtering uses future samples, so it is only possible offline.")}
{K("<p>Every filter is a trade-off. Too aggressive a high-pass filter removes part of a slow haemodynamic response; too wide a pass-band keeps physiological noise. Choose cut-offs from the frequency content of the signal you care about, and report them.</p>")}

<h2>3. Standard pipelines by modality</h2>
{table(2, "Typical preprocessing steps.",
       ["EEG", "fNIRS", "fMRI"],
       [["check channels, positions", "check quality (SCI, SNR); prune channels", "convert DICOM to NIfTI"],
        ["filter (high-pass, low-pass, notch)", "intensity → optical density", "motion correction"],
        ["detect and interpolate bad channels", "motion artifact correction", "slice-timing correction"],
        ["re-reference", "band-pass filter", "distortion correction, registration"],
        ["artifact removal (ICA, regression)", "optical density → HbO/HbR (MBLL)", "normalise to template (MNI)"],
        ["epoch around events", "short-separation regression", "smoothing (voxel analyses)"],
        ["baseline correction, epoch rejection", "block average or GLM", "confound regression, GLM or ROI extraction"]],
       widths=["33%", "34%", "33%"])}
<p>Widely used software includes <b>MNE-Python</b>, <b>EEGLAB</b> and <b>FieldTrip</b> for EEG; <b>Homer3</b>, <b>MNE-NIRS</b> and the <b>NIRS Brain AnalyzIR toolbox</b> for fNIRS; and <b>fMRIPrep</b>, <b>SPM</b>, <b>FSL</b>, <b>AFNI</b> and <b>nilearn</b> for fMRI. Using a standard pipeline makes results easier to reproduce and review.</p>
{V("<p><b>“Why didn't you use fMRIPrep?”</b> A fair answer: the fMRI data were reduced to region averages from an atlas, with drift removal and trigger-based alignment, which suits a small region-level target; a full fMRIPrep run with confound regression would be the more standard choice and is a reasonable improvement. Acknowledge the trade-off rather than defend the choice as perfect.</p>")}

<h2>4. Artifacts and quality metrics</h2>
<p>Every modality has characteristic contaminants and standard ways to detect them:</p>
{table(3, "Quality metrics by modality.",
       ["Modality", "Metric", "Meaning"],
       [["EEG", "impedance; line-noise power; flat or noisy channels; amplitude thresholds", "electrode contact and recording health"],
        ["EEG (in MRI)", "residual gradient power; peak amplitude", "how well scanner artifacts were removed"],
        ["fNIRS", "scalp coupling index; coefficient of variation; SNR", "optode contact; signal stability"],
        ["fNIRS", "HbO/HbR correlation", "neural (negative) vs systemic (positive) dominance"],
        ["fMRI", "tSNR; framewise displacement; DVARS", "signal stability; head motion"],
        ["All", "split-half or test–retest reliability of the target", "upper bound on achievable prediction"]],
       widths=["16%", "48%", "36%"])}
<p>Quality thresholds should be set in advance, but automatic rules also need sanity checks. A threshold designed for raw data can misfire on data already filtered by the provider, and a rule that excludes almost every subject is probably wrong rather than the data.</p>

<h2>5. Simultaneous recordings</h2>
<p>When EEG is recorded with fMRI or fNIRS, the devices must be <b>synchronised</b>. The usual method is triggers: the MRI scanner sends a pulse with each volume, and the stimulus computer sends event markers to all devices. Clocks drift slightly, so alignment should be based on these shared markers rather than assumed start times. Inside the scanner, EEG needs <b>gradient artifact</b> removal (average artifact subtraction, Allen 2000) and <b>ballistocardiogram</b> removal (template subtraction or optimal basis sets), with hardware precautions against heating.</p>
<p>Combining datasets recorded with different electrode layouts requires either restricting to shared electrodes, interpolating onto a common layout (spherical splines; Perrin and colleagues, 1989), working in source space, or giving the model a separate input layer for each layout.</p>

<h2>6. Normalisation, leakage and splits</h2>
<p>Normalisation removes scale differences between people and devices, but every statistic used to normalise must come only from data the model is allowed to see. Computing means and standard deviations over the whole dataset before splitting lets test-set information leak into training.</p>
{fig("rc3_leak", 3, "Leakage through preprocessing.", "Left: statistics computed on all people, including test people, quietly inform the training. Right: statistics are computed on training people only and then applied unchanged to the test people.")}
<p>The same applies to every other choice made from data: selecting channels, choosing a response window, tuning filters or picking hyperparameters. These must be decided on training (or separate validation) data. Splits must be made at the <b>subject</b> level and should be <b>stratified</b> by important groups such as patients and controls. Imbalanced datasets need balanced sampling or loss weighting so that the larger dataset does not dominate.</p>
{A("marking your own exam", "<p>If a student is allowed to see the exam questions while revising, a high score proves little. Preprocessing that uses test data is the same: the model has, indirectly, seen the questions. The leak is usually small and accidental, which is what makes it dangerous.</p>")}

<h2>7. Question bank with model answers</h2>
{qa_bank(QA)}

<h2>8. Where to read more</h2>
<ul class="refs">
<li>Gorgolewski K. J. et al. (2016). The brain imaging data structure (BIDS). <i>Scientific Data</i>.</li>
<li>Pernet C. R. et al. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. <i>Scientific Data</i>.</li>
<li>Esteban O. et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. <i>Nature Methods</i>.</li>
<li>Gramfort A. et al. (2013). MEG and EEG data analysis with MNE-Python. <i>Frontiers in Neuroscience</i>.</li>
<li>Perrin F. et al. (1989). Spherical splines for scalp potential and current density mapping. <i>Electroencephalography and Clinical Neurophysiology</i>.</li>
<li>Allen P. J., Josephs O., Turner R. (2000). A method for removing imaging artifact from continuous EEG recorded during functional MRI. <i>NeuroImage</i>.</li>
<li>Huppert T. J. et al. (2009). HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain. <i>Applied Optics</i>.</li>
<li>Widmann A., Schröger E., Maess B. (2015). Digital filter design for electrophysiological data – a practical approach. <i>Journal of Neuroscience Methods</i>.</li>
<li>Shin J. et al. (2017). Open access dataset for EEG+NIRS single-trial classification. <i>IEEE Transactions on Neural Systems and Rehabilitation Engineering</i>.</li>
</ul>
"""

D.render(body, "Lecture C Reference: General Knowledge")
