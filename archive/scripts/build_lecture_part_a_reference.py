"""Builds lectures/Lecture_A_Reference_General_Knowledge.pdf.

General background on brain signals and neuroimaging (Part A companion), written
to let the reader answer questions from anyone, not only about this project.
Run from the repo root:  python scripts/build_lecture_part_a_reference.py
"""
import subprocess
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "report_build" / "lecture_a_ref"
FIG = BUILD / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUTDIR = ROOT / "lectures"
OUTDIR.mkdir(exist_ok=True)
PDF = OUTDIR / "Lecture_A_Reference_General_Knowledge.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, LIGHT, INK, MUTED, GRID = "#a8a6a0", "#e4e2dc", "#1a1a1a", "#55534f", "#e8e7e3"
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 10, "svg.fonttype": "none",
    "axes.edgecolor": "#bdbbb5", "axes.labelcolor": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
})
W = 6.9


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def hrf(t, peak=6.0, under=16.0, ratio=1 / 6):
    t = np.clip(t, 0, None)
    h = t ** (peak - 1) * np.exp(-t) / factorial(int(peak - 1)) - ratio * t ** (under - 1) * np.exp(-t) / factorial(int(under - 1))
    return h / h.max()


# ---------------------------------------------------------------- figures
def f_spacetime():
    fig, a = plt.subplots(figsize=(W, 3.6))
    # (name, t_lo, t_hi [s], s_lo, s_hi [mm], invasive)
    m = [("Single neurons", 1e-4, 1e-2, 0.01, 0.08, True), ("Local field potential", 1e-3, 1e-1, 0.12, 0.9, True),
         ("ECoG (on brain surface)", 1e-3, 1e-1, 1.1, 4.5, True), ("MEG", 1e-3, 1e-1, 5.5, 25, False),
         ("EEG", 1e-3, 1e-1, 30, 100, False), ("fNIRS", 1, 10, 10, 30, False), ("fMRI", 1, 10, 1, 5, False),
         ("PET", 60, 1000, 4, 10, False)]
    for n, t0, t1, s0, s1, inv in m:
        col = ORANGE if inv else BLUE
        a.add_patch(Rectangle((t0, s0), t1 - t0, s1 - s0, fc=col, alpha=0.18, ec=col, lw=1.4))
        a.text(np.sqrt(t0 * t1), np.sqrt(s0 * s1), n, ha="center", va="center", fontsize=8.6, color=INK, weight="bold")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(5e-5, 5e3); a.set_ylim(5e-3, 300)
    a.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    a.set_xticklabels(["0.1 ms", "1 ms", "10 ms", "0.1 s", "1 s", "10 s", "100 s", "1000 s"])
    a.set_yticks([0.01, 0.1, 1, 10, 100]); a.set_yticklabels(["10 µm", "0.1 mm", "1 mm", "1 cm", "10 cm"])
    a.set_xlabel("temporal resolution (how quickly it can follow changes)  →  slower")
    a.set_ylabel("spatial resolution  →  blurrier")
    a.text(40, 150, "■ non-invasive", color=BLUE, fontsize=9); a.text(40, 85, "■ invasive (surgery needed)", color=ORANGE, fontsize=9)
    a.text(3e3, 7e-3, "ideal method would sit in the bottom-left corner", ha="right", fontsize=8.5, color=MUTED, style="italic")
    save(fig, "r1_spacetime")

def f_history():
    ev = [(1875, "Caton: electrical\nactivity in\nanimal brains"), (1890, "Roy & Sherrington:\nbrain regulates its\nown blood supply"),
          (1929, "Berger publishes\nfirst human EEG"), (1958, "10-20 electrode\nsystem\nstandardised"),
          (1977, "Jöbsis: near-\ninfrared light\nthrough the head"), (1986, "Fox & Raichle:\nflow rises more\nthan oxygen use"),
          (1990, "Ogawa:\nBOLD contrast\ndiscovered"), (1992, "First human\nBOLD fMRI"), (1993, "First\nfunctional\nfNIRS studies"),
          (2000, "Allen: MRI\nartifact removal\nfrom EEG"), (2001, "Logothetis:\nBOLD tracks\nsynaptic activity")]
    fig, a = plt.subplots(figsize=(W, 2.9))
    n = len(ev)
    a.plot([-0.4, n - 0.6], [0, 0], color=INK, lw=1.4)
    for i, (y, txt) in enumerate(ev):
        up = i % 2 == 0
        h = 1 if up else -1
        a.plot([i, i], [0, h * 0.35], color=GRAY, lw=1)
        a.scatter(i, 0, s=34, color=BLUE, zorder=3)
        a.text(i, h * 0.42, f"{y}", ha="center", va="bottom" if up else "top", fontsize=9, weight="bold", color=BLUE)
        a.text(i, h * 0.72, txt, ha="center", va="bottom" if up else "top", fontsize=7.6)
    a.set_xlim(-0.6, n - 0.4); a.set_ylim(-2.1, 2.1); a.axis("off")
    save(fig, "r2_history")

def f_dipoles():
    fig, a = plt.subplots(figsize=(W, 2.6))
    a.set_xlim(0, 10); a.set_ylim(-2.3, 1.4); a.axis("off")
    x = np.linspace(0, 10, 500)
    y = -0.9 * np.cos(2 * np.pi * x / 5) ** 3 - 0.3
    a.fill_between(x, y, -2.3, color="#f1e1ea")
    a.plot(x, y, color="#b98aa1", lw=2)
    a.axhline(1.05, color="#cfc9bd", lw=6)
    a.text(9.95, 1.18, "scalp sensors", ha="right", fontsize=8.5, color=MUTED)
    a.text(0.1, -2.15, "cortex (folded)", fontsize=8.5, color=MUTED)

    def dip(p, q, c):
        a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=c, lw=2.6))
    for xc in (2.5, 7.5):
        dip((xc, -0.15), (xc, 0.45), BLUE)
    dip((3.15, -0.62), (3.72, -0.62), ORANGE)
    dip((6.85, -0.62), (6.28, -0.62), ORANGE)
    a.text(2.5, -1.15, "gyrus crest: currents point\nstraight up (radial)", ha="center", fontsize=8.5, color=BLUE)
    a.text(5.0, -1.75, "sulcus walls: currents point\nsideways (tangential)", ha="center", fontsize=8.5, color=ORANGE)
    a.set_title("EEG sees radial sources best; MEG sees only tangential ones", fontsize=10)
    save(fig, "r3_dipoles")

def f_designs():
    t = np.linspace(0, 90, 1800)
    dt = t[1] - t[0]
    k = hrf(np.arange(0, 30, dt))
    block = (((t % 40) < 20) & (t < 80)).astype(float)
    ev = np.zeros_like(t)
    for e in [4, 18, 27, 45, 52, 70]:
        ev[int(e / dt)] = 1
    rb = np.convolve(block, k)[:len(t)]; rb /= rb.max()
    re = np.convolve(ev, k)[:len(t)]; re /= re.max()
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.1), sharex=True, gridspec_kw={"hspace": 0.45})
    a = ax[0]
    a.fill_between(t, 0, block * 1.15, color=YELLOW, alpha=0.18, step="mid", label="task on")
    a.plot(t, rb, color=RED, lw=2, label="blood response")
    a.set_title("Block design: 20 s task, 20 s rest", fontsize=10); a.set_yticks([]); a.legend(loc="upper right", ncol=2)
    a = ax[1]
    for e in [4, 18, 27, 45, 52, 70]:
        a.axvline(e, color=YELLOW, lw=1.6)
    a.plot(t, re, color=RED, lw=2)
    a.set_title("Event-related design: brief events at irregular ('jittered') times", fontsize=10); a.set_yticks([])
    a.set_xlabel("time (s)")
    save(fig, "r4_designs")


def f_balloon():
    t = np.linspace(-2, 35, 800)
    stim = ((t >= 0) & (t < 10)).astype(float)
    dt = t[1] - t[0]
    def lp(x, tau):
        y = np.zeros_like(x)
        for i in range(1, len(x)):
            y[i] = y[i - 1] + dt / tau * (x[i] - y[i - 1])
        return y
    cbf = lp(lp(stim, 1.5), 1.2)
    cbv = lp(cbf, 7.0)
    cbf /= cbf.max(); cbv = cbv / cbv.max() * 0.45
    cmro2 = cbf * 0.35
    bold = cbf - 0.6 * cbv - cmro2 * 0.5
    bold = bold / bold.max()
    fig, a = plt.subplots(figsize=(W, 2.6))
    a.axvspan(0, 10, color=YELLOW, alpha=0.12)
    a.plot(t, cbf, color=RED, lw=2, label="blood flow (CBF): fast up, fast down")
    a.plot(t, cbv, color=VIOLET, lw=2, label="blood volume (CBV): slow, lingers")
    a.plot(t, cmro2, color=GRAY, lw=2, label="oxygen use (CMRO₂): small rise")
    a.plot(t, bold, color=AQUA, lw=2.4, label="BOLD / HbO-like result")
    a.axhline(0, color=MUTED, lw=0.8)
    a.annotate("undershoot: flow is back to\nnormal but swollen veins still\nhold extra deoxygenated blood", xy=(19, bold[np.argmin(abs(t - 19))]),
               xytext=(20, 0.5), fontsize=8.4, arrowprops=dict(arrowstyle="->", color=MUTED))
    a.set_yticks([0]); a.set_xlabel("time (s); shaded = 10 s of activity")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8.4)
    a.text(0.3, 1.05, "activity", color="#8a6a00", fontsize=8.5)
    save(fig, "r5_balloon")


for fn in [f_spacetime, f_history, f_dipoles, f_designs, f_balloon]:
    fn()


# ---------------------------------------------------------------- text helpers
def fig(name, num, title, body):
    return f'<figure><img src="figures/{name}.svg"><figcaption><b>Figure {num}. {title}</b> {body}</figcaption></figure>'


def callout(kind, title, body):
    return f'<div class="call {kind}"><div class="ctitle">{title}</div>{body}</div>'


A = lambda t, b: callout("analogy", "Analogy · " + t, b)
K = lambda b: callout("key", "Key idea", b)
M = lambda b: callout("myth", "Common mistake", b)
V = lambda b: callout("viva", "If someone asks you…", b)


def table(num, title, head, rows, widths=None):
    cg = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><div class="tcap"><b>Table {num}.</b> {title}</div><table><colgroup>{cg}</colgroup><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


QA = []


def qa(section, q, a):
    QA.append((section, q, a))


# ---------------------------------------------------------------- question bank
S1 = "Foundations"
qa(S1, "What is neurovascular coupling?", "The process by which local neural activity causes a local increase in blood flow. Active neurons and astrocytes release signalling molecules (nitric oxide, prostaglandins, potassium ions and others) that relax the muscle around nearby arterioles and capillaries, so more blood flows to the active region within a few seconds.")
qa(S1, "Why does the brain need a constant blood supply?", "It uses about 20% of the body's oxygen and glucose while weighing about 2% of the body, and it stores almost no energy. Most of that energy is spent restoring ion gradients after synaptic signalling. If blood flow stops, consciousness is lost within seconds and damage begins within minutes.")
qa(S1, "What is the difference between HbO, HbR and HbT?", "HbO is haemoglobin carrying oxygen, HbR is haemoglobin that has released it, and HbT (total haemoglobin) is their sum. HbT is a proxy for blood volume; the HbO/HbR balance reflects oxygenation.")
qa(S1, "Why does an active region end up with MORE oxygenated blood?", "Blood flow rises much more (typically 20–60%) than oxygen consumption (typically 5–30%). The surplus fresh blood washes out deoxygenated haemoglobin, so HbO rises and HbR falls. This mismatch, first shown by Fox and Raichle in 1986, is what makes BOLD and fNIRS work.")
qa(S1, "What is the difference between structural and functional imaging?", "Structural imaging (CT, standard MRI) shows anatomy: what the brain looks like. Functional imaging (fMRI, fNIRS, PET, EEG, MEG) shows activity: what the brain is doing over time. A tumour is found with structural imaging; which area handles language is found with functional imaging.")
qa(S1, "Invasive vs non-invasive recording?", "Invasive methods (single-unit electrodes, ECoG) require surgery and record directly from or on the brain, giving excellent signal quality, but are only used in patients who need surgery anyway (for example epilepsy). Non-invasive methods (EEG, MEG, fNIRS, fMRI) are safe for anyone but measure through the skull or indirectly.")

S2 = "EEG and MEG"
qa(S2, "What exactly does EEG measure?", "Voltage differences on the scalp produced mainly by synchronised postsynaptic currents in large populations of cortical pyramidal neurons aligned in parallel. It does not measure single action potentials; those are too brief and too small to sum at the scalp.")
qa(S2, "Why is EEG's temporal resolution so good?", "Electrical fields spread through tissue essentially instantly, so the scalp voltage changes at the same moment as the neural currents. With sampling at 250–1000 Hz, EEG follows events at the millisecond scale.")
qa(S2, "Why is EEG's spatial resolution so poor?", "The skull conducts electricity poorly and smears the field across the scalp (volume conduction), so each electrode sums activity from a wide area. Several to roughly 20 square centimetres of synchronised cortex are typically needed to produce a visible scalp signal.")
qa(S2, "What is the inverse problem?", "Working out which brain sources produced a given scalp pattern. It is ill-posed: infinitely many source configurations can produce the same scalp recording, so source localisation needs extra assumptions (head model, smoothness, number of sources) and is never unique.")
qa(S2, "Why does the reference electrode matter?", "EEG voltage is always measured relative to a reference, so changing the reference changes every channel's waveform. Common choices are a single electrode, linked mastoids, or the average of all electrodes. Results should always state the reference.")
qa(S2, "What is the 10-20 system?", "The international standard (Jasper, 1958) for placing electrodes at proportional distances (10% or 20%) between skull landmarks: nasion, inion and the ears. Letters give the region (F, C, P, O, T), odd numbers are left, even right, z is the midline. Extended 10-10 and 10-5 systems add more positions.")
qa(S2, "What are ERPs and how do they differ from oscillations?", "Event-related potentials are small, time-locked voltage deflections (for example P300, N170, mismatch negativity) revealed by averaging many trials. Oscillations are ongoing rhythms whose power or phase changes with events. ERPs are phase-locked to the event; induced oscillatory changes such as ERD are not, so they disappear in a plain average and need time-frequency analysis.")
qa(S2, "What are ERD and ERS?", "Event-related desynchronisation (ERD) is a decrease in band power, for example mu and beta over the motor cortex during movement or motor imagery. Event-related synchronisation (ERS) is an increase, such as the 'beta rebound' just after a movement ends. The terms were popularised by Pfurtscheller.")
qa(S2, "What are the main EEG artifacts and how are they removed?", "Eye blinks and movements, muscle activity, heartbeat, sweat, electrode pops and mains line noise. Remedies include filtering, notch filters, rejecting bad segments, regression against eye channels, and independent component analysis (ICA), which separates the recording into components so artifact components can be removed.")
qa(S2, "EEG vs MEG?", "Both measure the same neural currents with millisecond resolution. MEG measures the magnetic fields, which pass through the skull without smearing, giving better localisation, but MEG is blind to radially oriented sources, needs a shielded room and costs far more. EEG is cheap, portable and sees both orientations.")
qa(S2, "What is EEG used for clinically?", "Diagnosing and monitoring epilepsy, sleep studies (polysomnography), assessing coma and encephalopathy, monitoring anaesthesia depth, and in some countries as part of brain-death testing.")
qa(S2, "What is a brain–computer interface (BCI) and how does EEG fit?", "A system that turns brain signals into commands. Common EEG paradigms are motor imagery (ERD over motor cortex), the P300 speller and steady-state visually evoked potentials (SSVEP). Hybrid BCIs combine EEG with fNIRS to improve accuracy.")

S3 = "fNIRS"
qa(S3, "How does fNIRS work in one sentence?", "It shines near-infrared light into the head, measures how much returns a few centimetres away at two or more wavelengths, and converts changes in absorption into changes in HbO and HbR using the modified Beer–Lambert law.")
qa(S3, "Why near-infrared light?", "Between about 650 and 950 nm, skin, bone and water absorb relatively little (the optical window), while haemoglobin absorbs strongly and differently in its two forms. Visible light is absorbed too quickly; longer infrared is absorbed by water.")
qa(S3, "What is the modified Beer–Lambert law?", "ΔOD = ε · Δc · d · DPF: the change in optical density equals the extinction coefficient times the change in concentration times the source–detector distance times the differential pathlength factor. With two wavelengths you solve for ΔHbO and ΔHbR. It gives changes, not absolute concentrations.")
qa(S3, "What is the DPF and why does it matter?", "The differential pathlength factor corrects for light travelling a longer, zigzag path than the straight source–detector distance (typically 5–7 times). It depends on age, wavelength and tissue; errors in it scale the concentration estimates, and it also explains why comparing amplitudes across people or patient groups is risky.")
qa(S3, "How deep does fNIRS see?", "Roughly half the source–detector distance. With the standard 3 cm separation in adults, light reaches about 1.5–2 cm below the scalp, which covers only the outer few millimetres of cortex. Deep structures such as the hippocampus or thalamus are invisible.")
qa(S3, "What are continuous-wave, frequency-domain and time-domain fNIRS?", "Continuous-wave systems (the most common and cheapest) measure only light intensity, so give relative changes. Frequency-domain systems modulate the light and measure phase shift; time-domain systems send picosecond pulses and time the photons. The latter two can estimate absolute optical properties and depth-resolved information.")
qa(S3, "What is a short-separation channel?", "A detector placed about 8 mm from the source, so its light path stays in the scalp and skull. It measures superficial and systemic signals, which can be regressed out of the normal channels. It is now considered best practice.")
qa(S3, "Main sources of error in fNIRS?", "Scalp blood flow, systemic physiology (heartbeat, breathing, Mayer waves, task-evoked blood pressure changes), motion artifacts, poor optode–scalp contact, hair (especially dark, dense hair) and skin pigmentation, which absorb light and reduce signal quality.")
qa(S3, "Why is fNIRS popular for infants and real-world studies?", "It is silent, wearable, tolerant of moderate movement, needs no confinement in a scanner and is cheap to run. It is used in infant research, walking or driving studies, social interaction ('hyperscanning' of two people) and bedside clinical monitoring.")
qa(S3, "fNIRS vs fMRI?", "Both measure the haemodynamic response. fMRI covers the whole brain at millimetre resolution but needs a scanner, keeps the subject still and measures mostly HbR. fNIRS sees only the surface at centimetre resolution, but is portable, quiet, cheaper, samples faster and measures HbO and HbR separately.")
qa(S3, "How do you know if an fNIRS channel is good?", "Common checks are the scalp coupling index (whether the heartbeat is clearly visible at both wavelengths; if not, the optode is not touching the skin properly), signal-to-noise ratio, coefficient of variation, and whether HbO and HbR show the expected opposite changes.")

S4 = "fMRI"
qa(S4, "Does fMRI measure neural activity?", "No. It measures the BOLD signal, a consequence of blood oxygenation changes that follow neural activity via neurovascular coupling. It is an indirect, delayed and blurred proxy.")
qa(S4, "What is BOLD and why does it increase with activity?", "Blood-oxygen-level-dependent contrast. Deoxyhaemoglobin is paramagnetic and shortens the T2* relaxation time, darkening the image. Activity brings an oversupply of oxygenated blood, reducing deoxyhaemoglobin, so the T2*-weighted signal rises, usually by 0.5–5% depending on field strength.")
qa(S4, "Does BOLD reflect spiking or synaptic activity?", "Both contribute, but Logothetis and colleagues (2001) showed BOLD correlates best with local field potentials, which mainly reflect synaptic input and local processing, rather than output spiking. So BOLD can rise in a region that receives input even if its output firing changes little.")
qa(S4, "What do 1.5 T, 3 T and 7 T mean?", "The magnet's field strength in tesla. Earth's field is about 50 µT, so 3 T is about 60,000 times stronger. Higher fields give more signal and larger BOLD effects but more distortion and safety constraints. 3 T is the research standard; 7 T is used for high-resolution studies.")
qa(S4, "What is a TR, and what is a voxel?", "TR (repetition time) is the time to acquire one brain volume, typically 0.5–3 s. A voxel is a 3D pixel, typically 2–3 mm wide at 3 T. One volume contains on the order of 100,000 voxels.")
qa(S4, "What is the GLM in fMRI analysis?", "The general linear model: you build predicted time courses by convolving the task timing with the HRF, then fit them to each voxel's signal with regression. The fitted weights, compared between conditions (contrasts), give the activation maps.")
qa(S4, "What is the multiple comparisons problem (the 'dead salmon' study)?", "Testing 100,000 voxels at p < 0.05 produces thousands of false positives by chance. Bennett and colleagues showed apparent 'activation' in a dead salmon when no correction was applied. Corrections such as family-wise error control or false-discovery rate are essential.")
qa(S4, "What is resting-state fMRI and the default mode network?", "Recording while the person does nothing in particular. Slow BOLD fluctuations are correlated between distant regions (functional connectivity, Biswal 1995). The default mode network (Raichle 2001) is a set of regions more active at rest and during internal thought than during external tasks.")
qa(S4, "How do you record EEG inside an MRI scanner?", "With MR-compatible electrodes, carbon or copper cabling with current-limiting resistors and amplifiers inside the room. Two big artifacts must be removed: the gradient artifact from switching magnetic gradients (removed by average artifact subtraction, Allen 2000) and the ballistocardiogram from pulse-related motion in the field. Safety checks prevent heating of leads.")
qa(S4, "fMRI safety issues?", "No ionising radiation, but ferromagnetic objects can become projectiles, implants may be unsafe, the scanner is very loud (above 100 dB, so hearing protection is required), radio-frequency energy heats tissue slightly and the bore can be claustrophobic.")

S5 = "The haemodynamic response"
qa(S5, "What is the HRF?", "The haemodynamic response function: the blood signal produced by a brief burst of neural activity. It starts after 1–2 s, peaks at about 4–6 s, returns towards baseline by about 10–12 s, dips below baseline (post-stimulus undershoot) and recovers by about 25–30 s.")
qa(S5, "Why is the response delayed?", "Because it depends on a chain of slow events: signalling molecules must be released, smooth muscle must relax, vessels must dilate and blood must travel. Electrical signalling takes milliseconds; vascular changes take seconds.")
qa(S5, "Is the HRF the same everywhere and for everyone?", "No. Its timing and shape vary between brain regions, between people (peak latency can differ by a second or more) and with age, disease, medication and caffeine. Analyses often use a canonical HRF plus derivatives, or estimate the HRF flexibly (FIR models), to cope with this.")
qa(S5, "What causes the post-stimulus undershoot?", "The balloon model (Buxton 1998) explains it: after activity, blood flow returns to baseline quickly but venous blood volume returns slowly, so for a while there is extra deoxygenated blood in the region, lowering BOLD. Prolonged elevated oxygen metabolism may also contribute.")
qa(S5, "What is the initial dip?", "A small, brief decrease in oxygenation (rise in HbR) in the first 1–2 seconds, thought to reflect oxygen consumption before the flow increase arrives. It is small, hard to detect and still debated.")
qa(S5, "Is the haemodynamic response linear?", "Approximately, for stimuli longer than a few seconds: responses to successive events add (linear time-invariant model). For very brief or closely spaced events it becomes non-linear; the second response is smaller (refractory effect).")
qa(S5, "Block vs event-related designs?", "Block designs alternate long task and rest periods (e.g., 20 s each); they give large, reliable signals and are efficient for detecting activation. Event-related designs present brief events, often with jittered timing; they allow estimating the response shape and mixing trial types, but each response is smaller.")
qa(S5, "What is negative BOLD?", "A decrease below baseline in some regions during a task, often reflecting reduced neural activity (deactivation), but sometimes a vascular 'steal' of blood towards nearby active areas. Its interpretation requires care.")

S6 = "Noise and confounds"
qa(S6, "What physiological signals contaminate haemodynamic data?", "Cardiac pulsation (~1 Hz), respiration (~0.2–0.3 Hz), Mayer waves in blood pressure (~0.1 Hz) and very slow drifts (<0.04 Hz). In fNIRS, scalp blood flow adds to all of these.")
qa(S6, "Why can't you just filter physiological noise out?", "Mayer waves and slow drifts overlap with the frequency of typical task responses. Worse, tasks can change heart rate, blood pressure and breathing in time with the task, producing systemic signals that look exactly like a brain response.")
qa(S6, "How do you tell a neural from a systemic fNIRS signal?", "A neural response is local and shows HbO increasing while HbR decreases. A systemic response appears across many channels, including short channels, with HbO and HbR moving together.")
qa(S6, "What is global signal regression and why is it controversial?", "Removing the average signal across all channels or voxels. It removes global physiological noise but also removes any genuine widespread response and, in fMRI, can create artificial negative correlations (Murphy 2009). It is a trade-off, not a neutral cleaning step.")
qa(S6, "What does motion do to each modality?", "In EEG it causes large artifacts from electrode movement and muscle. In fNIRS it causes spikes and baseline shifts when optodes shift. In fMRI it misaligns voxels between volumes and changes the signal, and correlates of motion can mimic functional connectivity.")

S7 = "Combining modalities"
qa(S7, "Why combine EEG with fNIRS or fMRI?", "They are complementary: EEG gives timing, haemodynamics gives location (fMRI) or a second, independent signal (fNIRS). Combined, they give a fuller picture of when and where, and in BCIs they improve classification.")
qa(S7, "What is EEG-informed fMRI?", "Using features of the simultaneously recorded EEG (for example alpha power or spike times) as regressors, convolved with the HRF, to find brain regions whose BOLD signal tracks them. It is widely used to map epileptic spikes and the networks behind EEG rhythms.")
qa(S7, "Can EEG predict fMRI or fNIRS signals?", "Partly. 'EEG fingerprint' methods (for example Meir-Hasson and colleagues, 2014) learned to predict activity in a specific deep region from EEG, and newer deep-learning methods reconstruct some fMRI signals. Limits are inter-subject variability in coupling and HRF, EEG's weakness for deep sources, small paired datasets and the indirect relationship between electrical and blood signals.")
qa(S7, "Why is neurovascular coupling a problem when studying patients?", "Coupling can be altered by stroke, haemorrhage, small-vessel disease, dementia, ageing, medication and anaesthesia, and is immature in newborns. A changed haemodynamic signal in patients may therefore reflect altered vessels rather than altered neural activity, so haemodynamic findings in clinical groups need extra controls.")
qa(S7, "What is the neurovascular unit?", "The functional group of cells that controls local blood flow: neurons, astrocytes, vascular smooth muscle cells, pericytes around capillaries and endothelial cells lining the vessels. It also maintains the blood–brain barrier.")


# ---------------------------------------------------------------- document
def qa_html():
    out, cur, n = [], None, 0
    for sec, q, a in QA:
        if sec != cur:
            out.append(f"<h3>{sec}</h3>")
            cur = sec
        n += 1
        out.append(f'<div class="qa"><div class="q">Q{n}. {q}</div><div class="a">{a}</div></div>')
    return "\n".join(out)


html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Lecture A Reference: General Knowledge</title>
<style>
@page {{ size: A4; margin: 22mm 21mm 20mm 21mm;
  @bottom-center {{ content: counter(page); font-family: Calibri, sans-serif; font-size: 9pt; color: #777; }}
  @top-right {{ content: "NeuroVascNet course · Lecture A reference"; font-family: Calibri, sans-serif; font-size: 8.5pt; color: #999; }} }}
@page :first {{ @top-right {{ content: none; }} @bottom-center {{ content: none; }} }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: Cambria, Georgia, serif; font-size: 10.5pt; line-height: 1.48; color: #1a1a1a; margin: 0; }}
h1, h2, h3, .ctitle, figcaption, table, .kicker, .meta, .q, .tcap {{ font-family: Calibri, "Segoe UI", sans-serif; }}
.kicker {{ font-size: 10pt; letter-spacing: .08em; text-transform: uppercase; color: {BLUE}; font-weight: 700; margin-top: 6mm; }}
h1 {{ font-size: 25pt; line-height: 1.12; margin: 2pt 0 4pt; }}
.subtitle {{ font-size: 12.5pt; color: #444; font-style: italic; margin-bottom: 10pt; }}
.meta {{ font-size: 9.5pt; color: #555; border-top: 1.5px solid #222; border-bottom: .6px solid #bbb; padding: 5pt 0; margin-bottom: 12pt; }}
h2 {{ font-size: 16pt; margin: 20pt 0 6pt; padding-bottom: 2pt; border-bottom: .8px solid #ccc; break-after: avoid; }}
h3 {{ font-size: 12pt; margin: 13pt 0 3pt; break-after: avoid; color: #222; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }}
ul, ol {{ margin: 0 0 7pt; padding-left: 17pt; }} li {{ margin-bottom: 2.5pt; }}
figure {{ margin: 8pt 0 12pt; break-inside: avoid; }} figure img {{ width: 100%; display: block; }}
figcaption {{ font-size: 9pt; line-height: 1.35; color: #333; margin-top: 4pt; }}
.call {{ border-left: 3.5px solid; padding: 6pt 11pt 3pt; margin: 8pt 0 11pt; break-inside: avoid; border-radius: 0 4px 4px 0; }}
.call p:last-child {{ margin-bottom: 4pt; }}
.ctitle {{ font-weight: 700; font-size: 9.5pt; letter-spacing: .03em; text-transform: uppercase; margin-bottom: 3pt; }}
.analogy {{ background: #fff7e8; border-color: {YELLOW}; }} .analogy .ctitle {{ color: #8a5a00; }}
.key {{ background: #f6f2fb; border-color: {VIOLET}; }} .key .ctitle {{ color: {VIOLET}; }}
.myth {{ background: #fdf0ec; border-color: {ORANGE}; }} .myth .ctitle {{ color: #b2431b; }}
.viva {{ background: #eef6f1; border-color: {AQUA}; }} .viva .ctitle {{ color: #0f6b4a; }}
.howto {{ background: #f4f6f9; padding: 8pt 12pt 3pt; margin-bottom: 10pt; break-inside: avoid; }}
.tbl {{ break-inside: avoid; margin: 6pt 0 12pt; }}
.tcap {{ font-size: 9pt; color: #333; margin-bottom: 3pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.9pt; line-height: 1.33; table-layout: fixed; }}
thead th {{ border-top: 1.3px solid #222; border-bottom: .7px solid #222; text-align: left; padding: 3pt 5pt; vertical-align: bottom; }}
td {{ padding: 2.8pt 5pt; vertical-align: top; border-bottom: .4px solid #e2e2e2; }}
tbody tr:last-child td {{ border-bottom: 1.3px solid #222; }}
.qa {{ break-inside: avoid; margin: 0 0 7pt; }}
.q {{ font-weight: 700; font-size: 10pt; color: #1d5aa6; margin-bottom: 1pt; }}
.a {{ text-align: justify; }}
.refs li {{ margin-bottom: 4pt; }}
</style></head><body>

<div class="kicker">NeuroVascNet course · Lecture A · Reference</div>
<h1>Brain Signals and Neuroimaging: General Knowledge</h1>
<div class="subtitle">The wider field behind Lecture A, with a bank of questions and model answers</div>
<div class="meta">Companion to "The Brain's Two Signals" · Read Lecture A first · Use this to answer questions from anyone: supervisors, examiners, reviewers, colleagues</div>

<div class="howto">
<h3 style="margin-top:0">How to use this document</h3>
<p>Lecture A taught the ideas through our project. This document steps back and covers what any researcher in the field is expected to know, whether or not it appears in our work. Sections 1–7 are reference reading. Section 8 lists common mistakes people make when talking about these methods. Section 9 is a bank of {{NQ}} questions with model answers, grouped by topic; it is the part to revise before a meeting or viva.</p>
<p style="margin-bottom:4pt">When you face an unfamiliar question, a reliable structure is: <b>what is measured</b> → <b>the physical principle</b> → <b>strengths</b> → <b>limitations</b> → <b>a typical number</b> → <b>a caveat</b>. Almost every question in this area can be answered well with those six steps.</p>
</div>

<h2>1. The landscape of brain measurement</h2>
<p>Every method for measuring the living brain is a compromise between three things: how finely it resolves <b>space</b>, how finely it resolves <b>time</b>, and how <b>invasive</b> it is. No method is good at all three. Figure 1 is the standard way to show this trade-off.</p>
{fig("r1_spacetime", 1, "The space–time map of brain measurement methods.", "Approximate ranges. Each box spans the spatial and temporal resolution a method typically achieves. Electrical methods (EEG, MEG, ECoG) are fast; haemodynamic methods (fNIRS, fMRI, PET) are slow because blood is slow. Good spatial detail without surgery requires fMRI.")}
<p>Methods fall into families according to what they detect:</p>
{table(1, "Brain measurement methods by what they detect.",
       ["Family", "Method", "What it detects", "Typical use"],
       [["Electrical", "EEG", "scalp voltages from synchronised cortical currents", "clinic, sleep, epilepsy, BCI, cognition"],
        ["", "ECoG / intracranial EEG", "voltages on or inside the brain (surgery)", "epilepsy surgery planning, high-performance BCI"],
        ["", "Single-unit / LFP", "individual neurons / local populations", "animal research, implanted BCIs"],
        ["Magnetic", "MEG", "magnetic fields of the same currents as EEG", "source localisation, epilepsy, cognition"],
        ["Haemodynamic", "fMRI (BOLD)", "blood oxygenation via magnetic effect of HbR", "brain mapping, connectivity, clinical planning"],
        ["", "fNIRS", "HbO and HbR via light absorption", "infants, naturalistic tasks, bedside, BCI"],
        ["", "ASL MRI", "cerebral blood flow directly", "perfusion, clinical"],
        ["Metabolic / chemical", "PET", "radioactive tracers (glucose use, receptors, amyloid)", "dementia, oncology, neurochemistry"],
        ["Structural", "CT, structural MRI, DTI", "anatomy, white-matter tracts", "diagnosis, anatomy, surgery"]],
       widths=["16%", "22%", "35%", "27%"])}
{K("<p>Electrical methods are fast because they measure the neurons' own currents. Haemodynamic methods are slow because they measure blood, which responds over seconds. This single fact explains most of the differences between EEG and fNIRS or fMRI, and it is the reason a combined system is attractive.</p>")}

<h2>2. Brain basics you will be asked about</h2>
<h3>2.1 Cells and energy</h3>
<p>The human brain weighs about 1.3–1.4 kg and contains about 86 billion neurons and a similar number of glial (support) cells, connected by roughly 100 trillion synapses. Neurons communicate with <b>action potentials</b> (all-or-nothing electrical pulses lasting about 1 ms) and with <b>postsynaptic potentials</b> (graded voltage changes lasting tens of milliseconds where synapses release neurotransmitters such as glutamate, which excites, or GABA, which inhibits). The slower postsynaptic potentials, summed across many aligned neurons, are what EEG mostly measures.</p>
<p>The brain runs on about 20 watts, about 20% of the body's resting energy, almost entirely from glucose burned with oxygen. Most of this energy is spent restoring ion gradients after synaptic signalling (Attwell and Laughlin, 2001). Average cerebral blood flow is about 50 mL per 100 g of tissue per minute, higher in grey matter (where neuron bodies and synapses are) than in white matter (the insulated cables).</p>

<h3>2.2 Anatomy and the motor system</h3>
<p>The outer layer, the <b>cortex</b>, is a sheet of grey matter 2–4 mm thick, heavily folded into ridges (<b>gyri</b>) and grooves (<b>sulci</b>). Each hemisphere has four lobes:</p>
{table(2, "The lobes and their main functions.",
       ["Lobe", "Main functions", "EEG electrodes over it"],
       [["Frontal", "planning, decisions, working memory, movement (motor cortex at its back edge), speech production (left)", "Fp, AF, F, FC"],
        ["Parietal", "body sensation (somatosensory cortex), spatial attention, integrating senses", "CP, P"],
        ["Temporal", "hearing, language comprehension (left), memory (with hippocampus deep inside)", "T, FT, TP"],
        ["Occipital", "vision", "PO, O"]],
       widths=["14%", "62%", "24%"])}
<p>The <b>primary motor cortex</b> (Brodmann area 4) runs along the precentral gyrus, just in front of the central sulcus; the <b>primary somatosensory cortex</b> (areas 1–3) runs just behind it. In front of the motor cortex lie the <b>premotor cortex</b> and the <b>supplementary motor area</b> (area 6), which plan movements. Body parts are mapped in order along the motor strip (the "homunculus"): legs near the midline at the top of the head, hands and face further out to the sides. That is why hand movements are recorded best at C3 and C4, and foot movements near Cz. Each hemisphere controls the opposite side of the body.</p>
<p><b>Motor imagery</b>, imagining a movement without doing it, activates the premotor and supplementary motor areas strongly and the primary motor cortex more weakly than real movement. It produces clear mu/beta ERD in EEG, which makes it the standard paradigm for motor BCIs.</p>

<h2>3. Neurovascular coupling in depth</h2>
<h3>3.1 The neurovascular unit</h3>
<p>Blood flow is controlled locally by the <b>neurovascular unit</b>: neurons, <b>astrocytes</b> (star-shaped glial cells whose processes touch both synapses and blood vessels), the <b>smooth muscle</b> wrapped around arterioles, <b>pericytes</b> wrapped around capillaries, and the <b>endothelial cells</b> lining the vessels. Together they sense activity and adjust vessel diameter. They also maintain the <b>blood–brain barrier</b>, which controls what passes from blood into brain tissue.</p>

<h3>3.2 Three quantities: flow, volume and oxygen use</h3>
{table(3, "The three physiological quantities behind every haemodynamic signal.",
       ["Quantity", "Meaning", "Typical change with activity", "Measured by"],
       [["CBF (cerebral blood flow)", "how fast blood passes through tissue", "+20% to +60%", "ASL MRI, PET"],
        ["CBV (cerebral blood volume)", "how much blood is in the tissue", "+5% to +30%, slow to return", "fNIRS HbT, special MRI"],
        ["CMRO<sub>2</sub> (oxygen metabolism)", "how much oxygen the tissue burns", "+5% to +30%", "calibrated fMRI, PET"]],
       widths=["27%", "29%", "24%", "20%"])}
<p>BOLD and fNIRS do not measure any single one of these. They measure the <i>combined</i> result: oxygenation depends on how much oxygen arrives (CBF), how much is used (CMRO<sub>2</sub>) and how much blood is present (CBV). That is why the same neural change can produce different haemodynamic signals in different people or diseases.</p>

<h3>3.3 What triggers the flow increase?</h3>
<p>A natural assumption is that tissue runs short of oxygen and the vessels respond to the shortage. Research since the 2000s shows that coupling is largely <b>feedforward</b>: the flow increase is triggered directly by signalling linked to synaptic activity, mainly glutamate release acting through neurons and astrocytes, which produce vasodilators such as nitric oxide, prostaglandins and potassium ions. Flow increases <i>before</i> any oxygen shortage could develop. This is why flow overshoots demand (Fox and Raichle, 1986) and why haemodynamic signals track synaptic input more than neuronal output (Logothetis et al., 2001).</p>
{A("ordering stock from the till, not the shelf", "<p>A well-run shop does not wait until the shelf is empty before ordering more stock; it orders automatically every time an item is scanned at the till. Neurovascular coupling works like the till: synaptic activity itself triggers the delivery order, rather than an empty shelf (low oxygen). And the order is generous, so the shelf ends up fuller than before.</p>")}

<h3>3.4 The balloon model and the shape of the response</h3>
{fig("r5_balloon", 2, "Why the response has an undershoot.", "Schematic of the balloon model (Buxton, Wong and Frank, 1998). Flow rises and falls quickly; veins inflate like a balloon and deflate slowly. After activity stops, the still-swollen veins hold extra deoxygenated blood, so BOLD (and HbO) dip below baseline before recovering.")}

<h3>3.5 When coupling changes</h3>
<p>Coupling is not fixed. It weakens or changes with <b>ageing</b> (stiffer vessels), <b>stroke</b>, <b>haemorrhage</b>, <b>small-vessel disease</b>, <b>Alzheimer's disease</b>, <b>hypertension</b>, <b>anaesthesia</b> and some drugs. <b>Caffeine</b> constricts vessels and changes baseline flow. In <b>newborns</b> coupling is immature and responses can even be inverted. The practical lesson: in patient groups, a different haemodynamic response does not automatically mean different neural activity.</p>
{V("<p><b>“So is the blood response a reliable measure of brain activity?”</b> It is a reliable <i>indirect</i> measure in healthy adults, because coupling is consistent and well characterised. It becomes less reliable when the vessels themselves are affected — in disease, ageing or under drugs — so in those groups it should be combined with a direct measure such as EEG or with vascular controls.</p>")}

<h2>4. EEG and MEG in depth</h2>
<h3>4.1 Where the signal comes from</h3>
<p>Pyramidal neurons in the cortex are aligned perpendicular to its surface, like trees in a forest. When many receive synaptic input together, their currents form a <b>dipole</b>, a source with positive and negative ends. Thousands of square millimetres of synchronised cortex add their dipoles into a field large enough to reach the scalp. Because the cortex is folded, the orientation of these dipoles depends on where they sit (Figure 3), which is the key difference between EEG and MEG.</p>
{fig("r3_dipoles", 3, "Source orientation decides what EEG and MEG can see.", "Schematic. On the crest of a gyrus, currents point straight out towards the scalp (radial); EEG sees these best and MEG not at all. In the walls of a sulcus, currents point sideways (tangential); MEG sees these well.")}

<h3>4.2 Recording</h3>
<p>Electrodes (silver/silver-chloride is the standard) are attached with conductive gel to bring skin–electrode impedance down, traditionally below about 5–10 kΩ; dry and high-impedance systems trade some quality for convenience. Typical systems have 19 (clinical), 32, 64, 128 or 256 channels, sampled at 250–1000 Hz (up to 5000 Hz in an MRI scanner to capture the artifact). Every voltage is relative to a <b>reference</b>; re-referencing to the average of all electrodes is common in research. Recordings are band-pass filtered (for example 0.5–45 Hz) and notch-filtered at the mains frequency (50 Hz in Europe and Asia, 60 Hz in the Americas).</p>

<h3>4.3 Two ways to analyse EEG</h3>
<p><b>Event-related potentials (ERPs)</b> average many trials time-locked to an event so that random activity cancels and a small consistent waveform remains. Famous components include the P300 (attention to a rare target, about 300 ms), the N170 (faces) and the mismatch negativity (detecting a change in sound). <b>Oscillatory analysis</b> asks how the power of rhythms changes over time and frequency, using the Fourier transform or wavelets. Motor imagery is analysed this way because ERD is not phase-locked and would vanish in a simple average.</p>
{table(4, "EEG rhythms in more detail.",
       ["Band", "Range", "Where and when", "Notes"],
       [["Delta", "0.5–4 Hz", "deep sleep; brain injury when awake", "large amplitude; abnormal in awake adults"],
        ["Theta", "4–8 Hz", "drowsiness; frontal midline during memory and concentration", "prominent in children"],
        ["Alpha", "8–13 Hz", "occipital, eyes closed and relaxed", "the 'Berger rhythm'; blocked by opening the eyes"],
        ["Mu", "8–13 Hz", "sensorimotor cortex at rest", "blocked by movement, imagery and watching movement"],
        ["Beta", "13–30 Hz", "frontal and motor, alert and active", "post-movement 'beta rebound' (ERS)"],
        ["Gamma", "30–100 Hz", "local processing, attention, binding", "easily contaminated by muscle activity"],
        ["Sleep features", "—", "spindles (11–16 Hz), K-complexes, slow waves", "used to score sleep stages"]],
       widths=["14%", "13%", "41%", "32%"])}

<h3>4.4 MEG</h3>
<p>Magnetoencephalography measures the magnetic fields produced by the same neural currents. They are extremely weak — femtotesla, roughly a hundred million times smaller than Earth's field — and traditionally require superconducting sensors (SQUIDs) cooled with liquid helium inside a magnetically shielded room. Magnetic fields are not distorted by the skull, so MEG localises sources better than EEG, but it cannot see radial sources. Newer optically pumped magnetometers (OPM-MEG) are wearable and may make MEG more practical.</p>

<h2>5. fNIRS in depth</h2>
<h3>5.1 Instruments and geometry</h3>
<p>An fNIRS cap holds <b>sources</b> (lasers or LEDs) and <b>detectors</b> (photodiodes). Each source–detector pair within range forms a <b>channel</b>, and the pattern of channels is the <b>montage</b>. Adults use about 3 cm separations; infants about 2–2.5 cm because their heads are thinner. <b>High-density diffuse optical tomography (HD-DOT)</b> uses many overlapping channels at several distances to reconstruct images with better resolution. Sampling rates of 5–50 Hz are typical.</p>
{table(5, "Types of fNIRS instrument.",
       ["Type", "What it measures", "Gives", "Comment"],
       [["Continuous wave (CW)", "light intensity only", "relative changes in HbO and HbR", "most common, cheapest, wearable"],
        ["Frequency domain (FD)", "intensity and phase of modulated light", "absolute absorption and scattering", "more complex"],
        ["Time domain (TD)", "arrival time of photons from picosecond pulses", "absolute values, depth information", "most informative, most expensive"]],
       widths=["22%", "30%", "28%", "20%"])}

<h3>5.2 From light to haemoglobin</h3>
<p>The <b>modified Beer–Lambert law</b> relates the change in optical density (ΔOD) at each wavelength to concentration changes: ΔOD = ε · Δc · d · DPF, where ε is the extinction coefficient (how strongly the molecule absorbs that wavelength), d is the source–detector distance and DPF is the differential pathlength factor (about 5–7, age-dependent). Measuring at two or more wavelengths gives a solvable system for ΔHbO and ΔHbR. Results are <i>changes</i> from a baseline, usually in micromolar (µM), and a partial-volume effect means the true brain change is underestimated because only part of the light path is in the brain.</p>
<p>A related clinical device, the <b>cerebral oximeter</b>, uses spatially resolved spectroscopy to estimate absolute tissue oxygen saturation (StO<sub>2</sub>) and is used to monitor patients during cardiac surgery.</p>

<h3>5.3 Signal quality and analysis</h3>
<p>Good practice checks each channel before analysis. The <b>scalp coupling index</b> tests whether the heartbeat is visible at both wavelengths; a missing heartbeat means the optode is not coupled to the skin. Motion artifacts are corrected with methods such as spline interpolation, wavelet filtering or temporal derivative distribution repair (TDDR). Systemic and scalp signals are removed with short-separation regression. Analysis then uses block averaging or a GLM, as in fMRI. Common software includes Homer3, MNE-NIRS and the NIRS Brain AnalyzIR toolbox.</p>
{M("<p>Ignoring hair and skin. Dark, dense hair and darker skin pigmentation absorb more light, lowering signal quality. Studies that exclude poor channels without reporting it can quietly under-represent some participants. Reporting channel exclusion rates is now recommended.</p>")}

<h2>6. fMRI in depth</h2>
<h3>6.1 How MRI makes images</h3>
<p>Hydrogen nuclei (protons, mostly in water) behave like tiny compass needles. In a strong magnetic field they align; a radio-frequency pulse tips them over; as they realign they emit a radio signal whose decay depends on the local tissue. The decay rates are called <b>T1</b>, <b>T2</b> and <b>T2*</b>. fMRI uses fast <b>echo-planar imaging (EPI)</b> sensitive to T2*, which is shortened by deoxyhaemoglobin. Gradient coils add position-dependent fields so the signal can be located in space; switching them rapidly is what makes the scanner loud and what creates the gradient artifact in simultaneous EEG.</p>

<h3>6.2 Analysis pipeline</h3>
<ol>
<li><b>Preprocessing</b>: correct for head motion, for the different timing of slices, align to the person's anatomy, warp to a standard template (for example MNI space) and smooth.</li>
<li><b>Modelling</b>: a GLM with task regressors convolved with the HRF, plus nuisance regressors (motion, physiological noise; RETROICOR and CompCor are common methods).</li>
<li><b>Inference</b>: statistical maps, corrected for multiple comparisons, and group analysis across participants.</li>
<li><b>Alternatives</b>: region-of-interest (ROI) averages, functional connectivity, and machine-learning decoding (multivoxel pattern analysis).</li>
</ol>
{M("<p>Reading colour on an fMRI map as 'amount of brain activity'. The colours are usually statistics (t or z values): how confidently a voxel's signal matched the model, not how hard the neurons worked. A small but very consistent change can appear brighter than a large, noisy one.</p>")}

<h2>7. The haemodynamic response and experimental design</h2>
<p>For a brief event, the canonical HRF starts after 1–2 s, peaks at 4–6 s, returns to baseline by about 10–12 s and undershoots before recovering by 25–30 s. In fNIRS, HbO follows this shape, HbR follows an inverted, smaller and slightly later version, and HbT (blood volume) follows a smoother version. The response is approximately <b>linear</b> for events longer than a few seconds, so overlapping responses add; the standard analysis convolves the stimulus timing with the HRF. Because the shape varies between regions and people, analyses often include the HRF's time and dispersion derivatives, or estimate the shape freely with finite impulse response (FIR) models.</p>
{fig("r4_designs", 4, "Block and event-related designs.", "Schematic. A block design sums many responses into a large plateau that is easy to detect. An event-related design gives smaller individual responses, but allows the response shape to be estimated and different trial types to be mixed; jittered timing lets overlapping responses be separated.")}

<h2>8. Common mistakes and misconceptions</h2>
{table(6, "Things people often get wrong.",
       ["Mistaken belief", "Correction"],
       [["fMRI and fNIRS measure neural activity.", "They measure blood oxygenation, an indirect and delayed consequence of neural activity."],
        ["Blood rushes in because tissue is starved of oxygen.", "Coupling is mainly feedforward from synaptic signalling; flow rises before any shortage."],
        ["Active tissue has less oxygen.", "It has more oxygenated blood, because supply overshoots demand. HbR falls."],
        ["EEG records single neurons or individual thoughts.", "It records summed activity of millions of aligned neurons; it cannot read specific thoughts."],
        ["More EEG electrodes means precise localisation.", "Density helps, but the inverse problem has no unique solution; localisation always relies on assumptions."],
        ["fNIRS can see the whole brain.", "It sees only the outer few millimetres of cortex."],
        ["fNIRS gives absolute haemoglobin concentrations.", "Continuous-wave fNIRS gives relative changes, scaled by an assumed DPF."],
        ["BOLD has physical units.", "BOLD is a relative signal (percent change from baseline), not an absolute quantity."],
        ["Brighter fMRI colours mean more activity.", "They usually mean a stronger statistic, not a larger response."],
        ["Filtering removes physiological noise.", "Noise that shares the task's frequency or timing cannot be filtered out."],
        ["Global signal regression is a harmless clean-up.", "It also removes genuine widespread responses and can create artificial negative correlations."],
        ["The HRF is the same for everyone.", "Its timing and shape vary with region, person, age, disease and drugs."],
        ["Motor imagery equals real movement.", "It engages planning areas strongly but primary motor cortex more weakly, with no muscle output."],
        ["We only use 10% of our brains.", "Practically all of the brain is active; imaging shows activity everywhere, varying in level."]],
       widths=["42%", "58%"])}

<h2>9. Question bank with model answers</h2>
<p>The answers are written at the depth expected in a viva or a supervisor meeting: correct, compact and with the key number or name where one exists. Practise saying them aloud in your own words rather than memorising them.</p>
{qa_html()}

<h2>10. Key numbers</h2>
{table(7, "Numbers worth remembering (typical values; exact figures vary between studies).",
       ["Quantity", "Typical value"],
       [["Neurons in the human brain", "about 86 billion"],
        ["Brain share of body mass / of resting oxygen use", "about 2% / about 20% (≈ 20 W)"],
        ["Average cerebral blood flow", "about 50 mL per 100 g per minute"],
        ["Cortex thickness", "2–4 mm"],
        ["Action potential / postsynaptic potential duration", "about 1 ms / tens of milliseconds"],
        ["EEG amplitude at the scalp", "10–100 µV"],
        ["MEG field strength", "femtotesla (≈ 10<sup>8</sup> times weaker than Earth's field)"],
        ["EEG sampling rate", "250–1000 Hz (up to 5000 Hz in MRI)"],
        ["fNIRS wavelengths / separation / depth", "650–950 nm / ≈ 3 cm in adults / ≈ 1.5–2 cm below scalp"],
        ["fNIRS sampling rate", "5–50 Hz"],
        ["Differential pathlength factor", "about 5–7"],
        ["MRI field strengths", "1.5 T, 3 T (research standard), 7 T; Earth ≈ 50 µT"],
        ["fMRI TR / voxel size", "0.5–3 s / 2–3 mm at 3 T"],
        ["BOLD signal change", "about 0.5–5%"],
        ["Flow / oxygen-use increase during activity", "+20–60% / +5–30%"],
        ["HRF onset / peak / undershoot / recovery", "1–2 s / 4–6 s / ≈ 10–20 s / ≈ 25–30 s"],
        ["Heartbeat / breathing / Mayer waves", "≈ 1 Hz / ≈ 0.2–0.3 Hz / ≈ 0.1 Hz"],
        ["Scanner noise", "above 100 dB"]],
       widths=["55%", "45%"])}

<h2>11. A short history</h2>
{fig("r2_history", 5, "Milestones in measuring the brain's electrical and blood signals.", "The idea that the brain controls its own blood supply is more than a century old; the tools to measure it non-invasively in humans arrived in the late twentieth century.")}
<p>Two earlier stories are worth knowing. In the 1880s the Italian physiologist Angelo Mosso observed that brain pulsations increased during mental effort in patients with skull defects, and built a balancing table to detect blood shifting towards the head during thought — arguably the first functional brain imaging experiment. And Hans Berger, who recorded the first human EEG, named the 10 Hz rhythm he saw with eyes closed the "alpha wave"; it is still sometimes called the Berger rhythm.</p>

<h2>12. Where to read more</h2>
<ul class="refs">
<li>Iadecola C. (2017). The neurovascular unit coming of age: a journey through neurovascular coupling in health and disease. <i>Neuron</i>.</li>
<li>Attwell D., Laughlin S. B. (2001). An energy budget for signaling in the grey matter of the brain. <i>Journal of Cerebral Blood Flow and Metabolism</i>.</li>
<li>Logothetis N. K. et al. (2001). Neurophysiological investigation of the basis of the fMRI signal. <i>Nature</i>. And Logothetis (2008), What we can do and what we cannot do with fMRI, <i>Nature</i>.</li>
<li>Buxton R. B., Wong E. C., Frank L. R. (1998). Dynamics of blood flow and oxygenation changes during brain activation: the balloon model. <i>Magnetic Resonance in Medicine</i>.</li>
<li>Pfurtscheller G., Lopes da Silva F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. <i>Clinical Neurophysiology</i>.</li>
<li>Scholkmann F. et al. (2014). A review on continuous wave functional near-infrared spectroscopy and imaging instrumentation and methodology. <i>NeuroImage</i>.</li>
<li>Tachtsidis I., Scholkmann F. (2016). False positives and false negatives in functional near-infrared spectroscopy: issues, challenges, and the way forward. <i>Neurophotonics</i>.</li>
<li>Pinti P. et al. (2020). The present and future use of functional near-infrared spectroscopy (fNIRS) for cognitive neuroscience. <i>Annals of the New York Academy of Sciences</i>.</li>
<li>Allen P. J., Josephs O., Turner R. (2000). A method for removing imaging artifact from continuous EEG recorded during functional MRI. <i>NeuroImage</i>.</li>
<li>Books: Huettel, Song and McCarthy, <i>Functional Magnetic Resonance Imaging</i>; Cohen, <i>Analyzing Neural Time Series Data</i>; Buzsáki, <i>Rhythms of the Brain</i>; Nunez and Srinivasan, <i>Electric Fields of the Brain</i>.</li>
</ul>
</body></html>"""

html = html.replace("{NQ}", str(len(QA)))
(BUILD / "lecture_a_ref.html").write_text(html, encoding="utf-8")
tmp = BUILD / "lecture_a_ref.pdf"
tmp.unlink(missing_ok=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={tmp}", (BUILD / "lecture_a_ref.html").as_uri()], check=True, capture_output=True)
if not tmp.exists():
    raise SystemExit("Chrome did not produce a PDF")
try:
    PDF.write_bytes(tmp.read_bytes())
    print("wrote", PDF, "questions:", len(QA))
except PermissionError:
    print(f"{PDF.name} is open in another program; new version left at {tmp}")
