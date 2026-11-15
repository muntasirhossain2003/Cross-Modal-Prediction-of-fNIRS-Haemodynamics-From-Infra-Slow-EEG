"""Builds lectures/Lecture_D_The_Model.pdf (teaching notes, Part D).

Architecture details follow scripts/build_v6_notebook.py (SharedEncoder,
HemoHead, compute_loss, train_joint, DelayHead, mc_dropout) and Maisha's v2.3
notebook. Figures are schematic. Run:  python scripts/build_lecture_part_d.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, LIGHT, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, E, K, M,
                            P, Q, arrow, box, fig, hrf, table, title_block)

D = Doc("lecture_d", "Lecture_D_The_Model.pdf", "NeuroVascNet course · Lecture D")


# ---------------------------------------------------------------- figures
def f_neuron():
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ins = [("x₁ = 0.8", 3.3, "w₁ = 0.5"), ("x₂ = −0.2", 2.0, "w₂ = 1.5"), ("x₃ = 0.4", 0.7, "w₃ = −1.0")]
    for lab, y, wl in ins:
        box(ax, 0.1, y - 0.3, 1.5, 0.6, lab, fs=8.8)
        arrow(ax, (1.6, y), (4.0, 2.0))
        ax.text(1.7, y + 0.12, wl, fontsize=8.5, color=ORANGE)
    ax.add_patch(plt.Circle((4.6, 2.0), 0.6, fc="#fff4e5", ec=YELLOW, lw=1.5))
    ax.text(4.6, 2.0, "Σ + b", ha="center", va="center", fontsize=10, weight="bold")
    box(ax, 5.8, 1.6, 1.7, 0.8, "activation\n(e.g. GELU)", fc="#f6f2fb", ec=VIOLET, fs=8.6)
    box(ax, 8.2, 1.6, 1.6, 0.8, "output", fc=BLUE, ec=BLUE, color="white", weight="bold")
    arrow(ax, (5.2, 2.0), (5.8, 2.0)); arrow(ax, (7.5, 2.0), (8.2, 2.0))
    ax.text(6.9, 0.55, "sum = 0.8×0.5 + (−0.2)×1.5 + 0.4×(−1.0) + b", ha="center", fontsize=8.6, color=MUTED)
    D.save(fig, "d1_neuron")


def f_descent():
    x = np.linspace(-3, 3, 300)
    yv = 0.35 * x ** 2 + 0.25 * np.sin(3 * x) + 0.4
    fig, a = plt.subplots(figsize=(W, 2.4))
    a.plot(x, yv, color=INK, lw=2)
    pts = [-2.6, -1.9, -1.25, -0.8, -0.5, -0.38]
    for i, p in enumerate(pts):
        v = 0.35 * p ** 2 + 0.25 * np.sin(3 * p) + 0.4
        a.scatter(p, v, s=50, color=ORANGE, zorder=3, edgecolor="white", lw=1.2)
        if i < len(pts) - 1:
            q = pts[i + 1]; vq = 0.35 * q ** 2 + 0.25 * np.sin(3 * q) + 0.4
            a.annotate("", xy=(q, vq + 0.05), xytext=(p, v + 0.05), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    a.text(-2.6, 3.35, "start: random settings,\nhigh loss", fontsize=8.5, color=MUTED)
    a.text(-0.2, 0.05, "low loss", fontsize=8.5, color=MUTED)
    a.set_xlabel("one of the model's settings (a weight)"); a.set_ylabel("loss (how wrong)")
    a.set_yticks([]); a.set_xticks([])
    D.save(fig, "d2_descent")


def f_overfit():
    ep = np.arange(1, 61)
    tr = 1.0 * np.exp(-ep / 12) + 0.08
    va = 0.75 * np.exp(-ep / 10) + 0.35 + 0.006 * np.clip(ep - 18, 0, None)
    fig, a = plt.subplots(figsize=(W, 2.3))
    a.plot(ep, tr, color=BLUE, lw=2, label="training loss")
    a.plot(ep, va, color=ORANGE, lw=2, label="validation loss (new people)")
    a.axvline(ep[np.argmin(va)], color=MUTED, ls="--", lw=1)
    a.text(ep[np.argmin(va)] + 1, 0.95, "best point to stop", fontsize=8.5, color=MUTED)
    a.text(42, 0.2, "memorising\nthe training set", fontsize=8.5, color=BLUE)
    a.set_xlabel("epochs (passes through the training data)"); a.set_yticks([]); a.legend(loc="upper right")
    D.save(fig, "d3_overfit")


def f_conv():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4, 400)
    x = np.sin(2 * np.pi * 1.0 * t) * (t > 1.5) * (t < 3.0) + 0.25 * rng.standard_normal(len(t))
    k = np.sin(2 * np.pi * 1.0 * np.linspace(0, 1, 100)) * np.hanning(100)
    out = np.convolve(x, k[::-1], "same") / 20
    fig, ax = plt.subplots(3, 1, figsize=(W, 3.0), sharex=True, gridspec_kw={"hspace": 0.45})
    ax[0].plot(t, x, color=GRAY, lw=1); ax[0].set_title("input signal (one EEG channel)", fontsize=9.5)
    ax[0].axvspan(0.6, 1.6, color=YELLOW, alpha=0.25); ax[0].text(1.1, 1.3, "the filter slides along", fontsize=8.3, ha="center")
    ax[1].plot(np.linspace(0.6, 1.6, 100), k, color=VIOLET, lw=2); ax[1].set_title("learned filter (a small template)", fontsize=9.5)
    ax[2].plot(t, out, color=BLUE, lw=2); ax[2].set_title("output: high wherever the input looks like the template", fontsize=9.5)
    for a in ax:
        a.set_yticks([])
    ax[2].set_xlabel("time (s)")
    D.save(fig, "d4_conv")


def f_arch():
    fig, ax = plt.subplots(figsize=(W, 3.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
    stages = [("EEG window", "channels × 6000\n(200 Hz)", BLUE, "white"),
              ("adapter", "1×1 convolution\n→ 64 × 6000", "#f3f5f8", INK),
              ("conv 1", "kernel 25, stride 5\n→ 64 × 1200 (40 Hz)", "#fff4e5", INK),
              ("conv 2", "kernel 15, stride 4\n→ 64 × 300 (10 Hz)", "#fff4e5", INK),
              ("conv 3", "kernel 9, stride 3\n→ 64 × 100 (3.3 Hz)", "#fff4e5", INK),
              ("pool", "average to 16 steps\n→ 64 × 16 = 1,024", "#fff4e5", INK)]
    for i, (n, sub, fc, tc) in enumerate(stages):
        y = 6.4 - i * 1.02
        box(ax, 0.2, y, 1.6, 0.8, n, fc=fc, ec=BLUE if i == 0 else (YELLOW if i > 1 else GRAY), color=tc, weight="bold", fs=8.8)
        ax.text(2.0, y + 0.4, sub, va="center", fontsize=8.3, color=MUTED)
        if i < len(stages) - 1:
            arrow(ax, (1.0, y), (1.0, y - 0.22))
    ax.add_patch(plt.Rectangle((0.05, 1.12), 4.35, 4.25, fill=False, ec=YELLOW, lw=1.2, ls="--"))
    ax.text(0.1, 0.72, "SHARED TRUNK (dashed box; each conv is followed by BatchNorm + GELU)", fontsize=8.3, color="#8a6a00", ha="left", weight="bold")
    box(ax, 5.0, 3.0, 1.8, 1.1, "1,024 features\n+ 8-number\nsubject embedding", fc="#f6f2fb", ec=VIOLET, fs=8.2)
    arrow(ax, (4.4, 1.7), (5.0, 3.0), rad=0.0)
    box(ax, 7.5, 4.6, 2.35, 1.3, "fNIRS head\n256 hidden, dropout 0.3\n→ mean + log-variance\n(HbO/HbR × 300)", fc="#fdeceb", ec=RED, fs=7.9)
    box(ax, 7.5, 2.75, 2.35, 1.3, "fMRI head\n256 hidden, dropout 0.3\n→ mean + log-variance\n(regions × 15)", fc="#e8f6f0", ec=AQUA, fs=7.9)
    box(ax, 7.5, 0.9, 2.35, 1.3, "delay head\n(separate test)\n→ one lag in seconds", fc="#fff4e5", ec=YELLOW, fs=7.9)
    arrow(ax, (6.8, 3.7), (7.5, 5.2)); arrow(ax, (6.8, 3.5), (7.5, 3.4)); arrow(ax, (6.8, 3.2), (7.5, 1.6))
    D.save(fig, "d5_arch")


def f_pearson():
    t = np.linspace(-2, 28, 300)
    truth = hrf(t - 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    a = ax[0]
    a.plot(t, truth, color=INK, lw=2, label="measured")
    a.plot(t, 0.35 * truth, color=BLUE, lw=2, label="prediction A")
    a.set_title("A: right shape, wrong size\nPearson loss ≈ 0, squared error large", fontsize=9.5); a.legend(fontsize=8); a.set_yticks([0])
    a = ax[1]
    a.plot(t, truth, color=INK, lw=2, label="measured")
    a.plot(t, np.full_like(t, truth.mean()) + 0.05 * np.sin(t), color=ORANGE, lw=2, label="prediction B")
    a.set_title("B: flat line at the right level\nsquared error moderate, Pearson loss ≈ 1", fontsize=9.5); a.legend(fontsize=8)
    for a in ax:
        a.axhline(0, color=MUTED, lw=0.7); a.set_xlabel("time (s)")
    D.save(fig, "d6_pearson")


def f_nll():
    s = np.linspace(0.05, 2.5, 300)
    fig, a = plt.subplots(figsize=(W, 2.3))
    for e, c in [(0.3, BLUE), (1.0, ORANGE)]:
        L = e ** 2 / s ** 2 + np.log(s ** 2)
        a.plot(s, L, color=c, lw=2, label=f"actual error = {e}")
        a.scatter(e, L[np.argmin(abs(s - e))], color=c, s=40, zorder=3)
    a.set_ylim(-3, 5); a.set_xlabel("uncertainty the model claims (σ)"); a.set_ylabel("loss")
    a.legend(loc="upper right")
    a.text(0.75, -2.85, "the loss is lowest when claimed uncertainty\nmatches the actual error (dots)", fontsize=8.5, color=MUTED)
    D.save(fig, "d7_nll")


def f_joint():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    box(ax, 0.1, 2.9, 1.9, 0.9, "batch of 64\nfNIRS trials", fc="#fdeceb", ec=RED, fs=8.5)
    box(ax, 0.1, 0.6, 1.9, 0.9, "batch of 64\nfMRI trials", fc="#e8f6f0", ec=AQUA, fs=8.5)
    box(ax, 2.8, 1.6, 1.8, 1.2, "shared\ntrunk", fc="#fff4e5", ec=YELLOW, weight="bold")
    box(ax, 5.3, 2.9, 1.7, 0.9, "fNIRS loss", fc="#fdeceb", ec=RED, fs=8.5)
    box(ax, 5.3, 0.6, 1.7, 0.9, "fMRI loss × w", fc="#e8f6f0", ec=AQUA, fs=8.5)
    box(ax, 7.6, 1.6, 2.2, 1.2, "add, then ONE\nbackward pass\nand update", fc="#f3f5f8", ec=GRAY, fs=8.5)
    arrow(ax, (2.0, 3.35), (2.8, 2.5)); arrow(ax, (2.0, 1.05), (2.8, 1.9))
    arrow(ax, (4.6, 2.5), (5.3, 3.35)); arrow(ax, (4.6, 1.9), (5.3, 1.05))
    arrow(ax, (7.0, 3.35), (7.6, 2.5)); arrow(ax, (7.0, 1.05), (7.6, 1.9))
    arrow(ax, (8.7, 1.6), (3.7, 1.55), color=ORANGE, rad=0.35)
    ax.text(6.2, 0.05, "the trunk is corrected by both errors at once", fontsize=8.5, color=ORANGE, ha="center")
    D.save(fig, "d8_joint")


def f_mc():
    t = np.linspace(-2, 28, 300)
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    for a, spread, title in [(ax[0], 0.04, "Familiar input: passes agree\n→ high confidence"), (ax[1], 0.25, "Unfamiliar input: passes disagree\n→ low confidence")]:
        for _ in range(12):
            a.plot(t, hrf(t - 0.5 + rng.normal(0, spread * 6)) * (1 + rng.normal(0, spread)), color=BLUE, lw=0.9, alpha=0.55)
        a.set_title(title, fontsize=9.5); a.set_yticks([]); a.set_xlabel("time (s)")
    D.save(fig, "d9_mc")


for f in [f_neuron, f_descent, f_overfit, f_conv, f_arch, f_pearson, f_nll, f_joint, f_mc]:
    f()


body = title_block("NeuroVascNet course · Lecture D", "The Model",
                   "Neural networks from first principles, then every part of the NeuroVascNet architecture and how it is trained",
                   "Part D of 7 · Builds on Lectures A–C · About 100 minutes of reading · No prior machine-learning knowledge assumed") + f"""
<div class="box">
<h3>By the end of this lecture you should be able to</h3>
<ul>
<li>explain what a neural network is, how it learns from examples, and why it can overfit;</li>
<li>explain convolution, stride, pooling, normalisation, activation, dropout and embeddings in plain language;</li>
<li>trace one EEG window through the final model, with the shape of the data at every stage;</li>
<li>explain each part of the loss function and why it was chosen;</li>
<li>describe how one network is trained on two datasets at once, and how the delay and reliability outputs are produced.</li>
</ul>
</div>

<h2>1. What a neural network is</h2>
<p>A neural network is a mathematical function with a very large number of adjustable settings. It takes numbers in (here, an EEG window) and produces numbers out (a predicted blood signal). Training means adjusting the settings until the outputs are close to the correct answers on many examples.</p>
<h3>1.1 The artificial neuron</h3>
<p>The basic unit is the artificial neuron. It multiplies each input by a <b>weight</b>, adds the results together with a <b>bias</b>, and passes the sum through a simple non-linear <b>activation function</b>. Weights and biases are the settings that training adjusts; collectively they are called <b>parameters</b>.</p>
{fig("d1_neuron", 1, "One artificial neuron.", "Each input is multiplied by a weight, the products are summed with a bias, and the result passes through an activation function. A network is many of these neurons arranged in layers, each layer's outputs becoming the next layer's inputs.")}
<p>The activation function matters because without it, stacking layers would still only produce a straight-line (linear) relationship. Our model uses <b>GELU</b>, a smooth function that lets small negative values partly through and passes positive values almost unchanged. With activations, a deep stack of layers can represent very complicated relationships.</p>
{A("a mixing desk", "<p>Picture a sound engineer's mixing desk with thousands of sliders. Each slider (a weight) controls how much one input contributes to one output. Nobody sets them by hand; training moves every slider a little at a time until the music coming out sounds right. A trained network is just a desk whose sliders have been set well.</p>")}

<h3>1.2 How the network learns</h3>
<p>Training repeats four steps many thousands of times:</p>
<ol>
<li><b>Forward pass</b>: feed a small group of examples (a <b>batch</b>; ours has 64 trials) through the network to get predictions.</li>
<li><b>Loss</b>: compute a single number saying how wrong the predictions are (Section 4).</li>
<li><b>Backward pass (backpropagation)</b>: compute, for every parameter, which direction of change would reduce the loss. This set of directions is the <b>gradient</b>.</li>
<li><b>Update</b>: move every parameter a small step in that direction. The step size is the <b>learning rate</b> (ours: 0.001).</li>
</ol>
<p>One pass through all training examples is an <b>epoch</b>; the final model trains for 30 epochs. The rule that turns gradients into updates is the <b>optimiser</b>; we use <b>AdamW</b>, a widely used optimiser that adapts the step size for each parameter and includes <b>weight decay</b>, a gentle pull of every weight towards zero that discourages extreme settings.</p>
{fig("d2_descent", 2, "Gradient descent.", "Schematic. The loss depends on the model's settings. Starting from random settings, each update moves downhill, step by step, towards settings that make the loss small.")}
{A("a hiker in fog", "<p>A hiker in thick fog wants to reach the valley floor. They cannot see the valley, but they can feel which way the ground slopes under their feet, so they take a step downhill, feel again, and repeat. The slope underfoot is the gradient, the step length is the learning rate, and the valley is a set of good parameters. In fog you can end up in a small dip rather than the deepest valley, which is why results can differ slightly between training runs with different random starting points (seeds).</p>")}

<h3>1.3 Overfitting and how to fight it</h3>
<p>A network with enough parameters can memorise its training examples, including their noise, instead of learning the general relationship. This is <b>overfitting</b>: the loss on training data keeps falling while the loss on new data stops improving and starts to rise.</p>
{fig("d3_overfit", 3, "Overfitting.", "Schematic. Training loss keeps falling; loss on new people falls at first, then rises as the network starts memorising. The gap between the curves is the warning sign.")}
<p>The standard defences, all used in the project, are:</p>
<ul>
<li><b>Dropout</b>: during training, randomly switch off a fraction of neurons (30% in our heads) so the network cannot rely on any single path.</li>
<li><b>Weight decay</b>: penalise large weights.</li>
<li><b>Batch normalisation</b>: rescale each layer's activity within each batch, which stabilises and speeds up training.</li>
<li><b>Small models and limited training</b>: fewer parameters and epochs give less room to memorise.</li>
<li><b>Honest validation</b>: always measure on people the network has never seen (Lecture E).</li>
</ul>
{P("<p>Overfitting is a constant danger here because the data are small: 3,480 fNIRS trials from 29 people and only 384 fMRI trials from 10 people, against a network with millions of parameters (most of them in the output heads, which must produce thousands of numbers per window). The early v3.2 experiment found fMRI training correlation of 0.41 against validation 0.007 — a textbook overfitting gap.</p>")}

<h2>2. Building blocks for signals</h2>
<h3>2.1 Convolution</h3>
<p>A <b>convolutional layer</b> slides a short learned pattern, called a <b>filter</b> or <b>kernel</b>, along the signal and measures at each position how well the signal matches it. The output is high wherever the signal resembles the pattern. Each layer learns many filters at once (64 in our trunk), each looking for a different pattern, such as a burst of 10 Hz rhythm or a slow drift.</p>
{fig("d4_conv", 4, "What a convolution does.", "Schematic. A learned template (middle) slides along the input (top). The output (bottom) rises wherever the input contains something template-like. The network learns the templates itself during training.")}
{A("a template on a transparency", "<p>Draw a small wave on a transparent sheet and slide it along a long printed recording. Wherever the drawing lines up with the recording, you mark a high score. A convolutional layer does this with dozens of different drawings at once — and learns which drawings are worth using.</p>")}
<p>Convolution suits EEG because the same pattern (say, a rhythm dropping in power) means the same thing whenever it occurs, and it needs far fewer parameters than connecting every time point to every neuron.</p>

<h3>2.2 Stride and pooling: summarising time</h3>
<p>The <b>stride</b> is how far the filter jumps between positions. A stride of 5 keeps one output for every 5 input samples, so the signal becomes five times shorter; this is also a reduction in effective sampling rate. <b>Pooling</b> summarises stretches of the signal, for example by averaging. Together they turn a long, fast signal into a short summary of slower patterns — appropriate when the target, the blood response, is slow.</p>

<h3>2.3 Other pieces</h3>
<ul>
<li><b>1×1 convolution</b>: a filter one sample wide. It does not look along time; it mixes channels at each moment, like re-weighting the electrodes. Our per-dataset adapters are 1×1 convolutions.</li>
<li><b>Linear (fully connected) layer</b>: every input connected to every output; used in the heads.</li>
<li><b>Embedding</b>: a small learned list of numbers attached to an identity, here one per training subject (Section 3.3).</li>
<li><b>Flatten</b>: unroll a table of numbers into one long list so that a linear layer can read it.</li>
</ul>

<h3>2.4 Input representations used in the project</h3>
<p>Two ways of presenting EEG to the network were used. The early pipeline and Maisha's pipeline fed <b>band-power envelopes</b> (Lecture A): four frequency bands for each of 32 electrodes, giving 128 channels at 20 samples per second. Their encoder combined <b>EEGNet</b>, a compact architecture designed for EEG that learns temporal filters and then spatial (across-electrode) filters, with a <b>temporal convolutional network (TCN)</b>, which uses stacked convolutions with growing gaps (dilations) to see long stretches of time. The final v6 model instead feeds the <b>z-scored raw EEG</b> at 200 Hz and lets the first convolution learn its own filters.</p>

<h2>3. The final model, stage by stage</h2>
{fig("d5_arch", 5, "The final (v6) NeuroVascNet architecture, with real shapes.", "An EEG window enters the adapter for its dataset, passes through three convolutional stages and a pooling step (the shared trunk), is joined with a subject embedding and is read by the head for its modality. Each head predicts a mean and a log-variance for every channel and time point.")}
{table(1, "Shape of the data at each stage (final model).",
       ["Stage", "Operation", "Output shape", "Effective time resolution"],
       [["Input", "z-scored EEG window", "channels × 6,000", "200 Hz (5 ms)"],
        ["Adapter", "1×1 convolution, one per dataset", "64 × 6,000", "200 Hz"],
        ["Conv 1", "kernel 25, stride 5, BatchNorm, GELU", "64 × 1,200", "40 Hz"],
        ["Conv 2", "kernel 15, stride 4, BatchNorm, GELU", "64 × 300", "10 Hz"],
        ["Conv 3", "kernel 9, stride 3, BatchNorm, GELU", "64 × 100", "3.3 Hz"],
        ["Pool", "adaptive average to 16 steps, flatten", "1,024", "≈ 0.5 Hz"],
        ["+ subject embedding", "8 numbers appended", "1,032", "—"],
        ["fNIRS head", "linear 256, GELU, dropout 0.3, two outputs", "2 × (HbO/HbR channels × 300)", "10 Hz"],
        ["fMRI head", "same design", "2 × (regions × 15)", "0.5 Hz"]],
       widths=["17%", "38%", "25%", "20%"])}

<h3>3.1 The adapters</h3>
<p>Each dataset has a different number and layout of electrodes (Lecture C). A small <b>adapter</b> per dataset maps its channels into a common set of 64 internal channels. Only the adapters differ between datasets; everything after them is shared. This is how one network can read EEG from four different caps.</p>

<h3>3.2 The shared trunk</h3>
<p>Three convolutional stages progressively shorten the signal (6,000 → 1,200 → 300 → 100 time steps) while 64 filters per stage look for patterns at each time scale. A final pooling step averages down to 16 time steps, giving 64 × 16 = 1,024 numbers: the <b>latent summary</b> of the 30-second window. The trunk is the only part trained by both fNIRS and fMRI errors, so it is where any cross-modal benefit would have to appear.</p>
{P("<p>An option in the code adds a narrow <b>bottleneck</b> layer after the trunk (16 to 1,024 numbers), forcing both heads to share a smaller summary. The reasoning: with 1,024 numbers each head might use its own private part, so the two supervisions never interact. The bottleneck sweep (Lecture F) tested this; at every size, joint training was still no better than fNIRS alone.</p>")}

<h3>3.3 The subject embedding — and why test subjects get the average</h3>
<p>Each training subject has 8 learned numbers, an <b>embedding</b>, appended to the latent summary. It lets the network absorb person-specific offsets (one person's responses are generally larger, another's later) without distorting the shared trunk. The embeddings start at zero.</p>
<p>The critical detail is what happens for a person <i>not</i> seen in training. They have no embedding, so the model uses the <b>average of all training embeddings</b>. This guarantees that a held-out person is treated as "a typical person", and the model cannot cheat by recognising who they are. A control run with the embeddings switched off checks that any benefit is not coming from subject adaptation.</p>

<h3>3.4 The heads and their two outputs</h3>
<p>Each modality has its own <b>head</b>: a linear layer of 256 neurons with GELU and 30% dropout, followed by two output layers. One gives the predicted <b>mean</b> for every channel and time point. The other gives a predicted <b>log-variance</b> — how uncertain the model is about each value. The log-variance starts at zero (equal uncertainty everywhere) and is limited to a sensible range. Predicting the logarithm keeps the variance positive and numerically stable.</p>

<h2>4. The loss function</h2>
<p>The loss is a weighted sum of three terms:</p>
<div class="formula">total loss = 1.0 × Pearson loss + 0.1 × Gaussian NLL + 0.05 × smoothness</div>

<h3>4.1 Pearson loss: match the shape</h3>
<p>The main term is <b>1 − correlation</b> between predicted and measured time courses, computed for each channel and averaged. Correlation ignores overall size and offset, so this term rewards getting the <i>shape</i> and <i>timing</i> right regardless of scale.</p>
{fig("d6_pearson", 6, "Why a correlation-based loss.", "Schematic. Prediction A has the right shape but the wrong size: correlation-based loss is near zero, squared error is large. Prediction B is a flat line at the right average level: squared error is moderate, but correlation-based loss is near its maximum. The correlation loss refuses to reward lazy flat predictions.")}
<p>The choice was deliberate. Amplitudes differ wildly between datasets (a 2,600-fold unit difference) and are dominated by systemic drift, which the project did not want the model to fit. A squared-error loss would also let the network score reasonably by predicting a flat average — a known failure of regression on noisy targets.</p>

<h3>4.2 Gaussian negative log-likelihood: learn the uncertainty</h3>
<p>The second term trains the log-variance output. For each value it is (error² ÷ variance) + log(variance). Claiming tiny uncertainty is punished when the error is large; claiming huge uncertainty is punished by the log term. The loss is lowest when the claimed uncertainty matches the actual error, so the head learns where it tends to be wrong.</p>
{fig("d7_nll", 7, "How the uncertainty output learns.", "For a given actual error, the loss is plotted against the uncertainty the model claims. The minimum (dot) sits exactly where the claimed uncertainty equals the actual error, for both a small and a large error.")}

<h3>4.3 Smoothness</h3>
<p>The third term penalises large jumps between neighbouring time points of the prediction. Blood signals are slow and smooth, so this nudges predictions to look physiological.</p>

<h2>5. Training on two datasets at once</h2>
{fig("d8_joint", 8, "One joint training step.", "At every step the model processes one batch from each dataset, computes each modality's loss, adds them (the fMRI loss multiplied by a weight w) and performs a single backward pass, so the shared trunk is corrected by both errors at once.")}
<p>Three design decisions in the training loop matter:</p>
<ul>
<li><b>Both losses in one update.</b> Taking a separate update for each dataset in turn would make the second update partly undo the first — interference rather than joint learning.</li>
<li><b>The smaller dataset cycles.</b> An epoch is as long as the larger dataset (about 55 batches of fNIRS against 6 of fMRI); the fMRI batches are reused in a loop so every step contains both. Otherwise the last 49 steps of each epoch would be fNIRS only and the model would drift back to a single modality.</li>
<li><b>The fMRI arm is weighted.</b> The weight w controls how strongly the small fMRI dataset pulls on the shared trunk; values from 0.1 to 1.0 were tested.</li>
</ul>
{table(2, "Training settings (final model).",
       ["Setting", "Value", "Why"],
       [["Optimiser", "AdamW, learning rate 0.001, weight decay 0.0001", "standard, robust default"],
        ["Batch size", "64 trials per dataset per step", "balances noise and memory"],
        ["Epochs", "30", "enough to converge on this data size without long overfitting"],
        ["Loss weights", "Pearson 1.0, NLL 0.1, smoothness 0.05, fMRI arm 1.0", "shape first; uncertainty and smoothness as helpers"],
        ["Cross-validation", "5 folds, by subject", "every person is tested once by a model that never saw them"],
        ["Seeds", "fixed and re-set per run", "results reproducible and comparable between arms"]],
       widths=["22%", "43%", "35%"])}
{K("<p>Single-modality and cross-modal models are produced by the <b>same function</b>: pass it one dataset and it trains an fNIRS-only model; pass it two and it trains the joint model. Identical code, budget and settings make the comparison in Claim 1 fair.</p>")}

<h2>6. The delay module</h2>
<p>The delay head is a small network on top of the trunk's summary that outputs one number, the estimated lag in seconds. In its original form it is <b>bounded</b>:</p>
<div class="formula">delay = d<sub>min</sub> + (d<sub>max</sub> − d<sub>min</sub>) × sigmoid(network output)</div>
<p>The sigmoid squashes any number into the range 0 to 1, so the delay can only fall between d<sub>min</sub> and d<sub>max</sub> (for example 2 and 8 s). It is trained to match the lag at which the predicted and measured blood signals line up best — and that target lag is itself clamped to the same window.</p>
{M("<p>Reading a bounded output as a measurement. If the output is forced into 2–8 s, finding it inside 2–8 s says nothing about physiology. The project therefore also trained the head with a different window (0–20 s) and with no bound at all. A real measurement would stay put; this one moved with the window (2.0 s, then 0.07 s), showing it reported the bounds rather than the brain.</p>")}

<h2>7. The reliability layer</h2>
<p>Reliability is built from three ingredients:</p>
<ul>
<li><b>Predicted variance</b> from the heads' log-variance output: the model's estimate of noise in the data (aleatoric uncertainty).</li>
<li><b>Monte Carlo dropout</b>: dropout is kept switched on at test time and each window is passed through the network 20 times. Because different neurons drop out each time, the 20 predictions differ; their spread measures the model's own uncertainty (epistemic uncertainty).</li>
<li>In the earlier and Maisha pipelines, a <b>domain-confidence</b> score: how far the window's latent summary lies from the training data.</li>
</ul>
{fig("d9_mc", 9, "Monte Carlo dropout.", "Schematic. Each line is one of 20 passes with different neurons dropped. For familiar inputs the passes agree; for unfamiliar ones they disagree. The spread becomes a confidence score.")}
<p>The raw score is then <b>calibrated</b>. In the final model, <b>Platt scaling</b> (a logistic curve mapping score to probability) is fitted on one half of the data and judged on the other half, so it cannot simply memorise. Maisha's pipeline used temperature scaling followed by isotonic regression, fitted on separate calibration subjects, and then <b>locked an abstention threshold</b> chosen to answer 75% of calibration cases. Lecture F shows how these performed.</p>

<h2>8. How Maisha's model differs</h2>
{table(3, "Main design differences between the two pipelines.",
       ["Aspect", "Final v6 (Muntasir)", "v3.1-v2.3 (Maisha)"],
       [["EEG input", "z-scored raw EEG, 200 Hz", "band-power envelopes, 128 channels at 20 Hz"],
        ["Encoder", "adapter + 3 strided convolutions", "EEGNet front-end + TCN"],
        ["Dataset handling", "per-dataset adapter; subject embedding", "per-dataset batch normalisation; subject-balanced sampling"],
        ["Main target", "full waveforms (correlation loss)", "compact response measures; waveform secondary"],
        ["Cross-modal alignment", "none (joint loss only)", "stage-2 alignment tested, then disabled (no benefit)"],
        ["Delay", "bounded and free variants tested", "learned delay disabled; fixed 5 s used"],
        ["Reliability", "MC dropout + variance, Platt scaling", "learned gate, temperature + isotonic, locked threshold"]],
       widths=["22%", "39%", "39%"])}
{Q(["What do the weights in a network represent, and what does training do to them?",
    "Why does a network need non-linear activation functions?",
    "In your own words, what is a gradient and what is the learning rate?",
    "What is the warning sign of overfitting, and name three defences used in this project.",
    "Why is convolution a natural choice for EEG?",
    "Follow the time axis through the trunk: how many time steps are left after each stage?",
    "Why does a held-out person receive the average subject embedding?",
    "Why was a correlation-based loss chosen over squared error?",
    "What does the Gaussian NLL term teach the network?",
    "Why are both datasets' losses added before a single update rather than updated in turn?",
    "Why can a bounded delay output not be taken as evidence of physiology?",
    "How does Monte Carlo dropout produce a measure of uncertainty?"])}

<h2>Summary</h2>
<ul>
<li>A neural network is a function with many adjustable weights. It learns by repeatedly predicting, measuring the loss, computing the gradient by backpropagation and stepping downhill (AdamW, learning rate 0.001, batches of 64, 30 epochs).</li>
<li>Small data make <b>overfitting</b> the central risk; dropout, weight decay, batch normalisation, small models and honest validation are the defences.</li>
<li>Convolutions slide learned templates along the signal; stride and pooling shorten time to match the slow target.</li>
<li>The final model: a <b>per-dataset adapter</b>, a <b>shared trunk</b> of three strided convolutions and pooling (6,000 → 1,024 numbers), an 8-number <b>subject embedding</b> (averaged for new people), and one <b>head per modality</b> outputting a mean and a log-variance.</li>
<li>The loss combines <b>correlation</b> (shape), <b>Gaussian NLL</b> (uncertainty) and <b>smoothness</b>.</li>
<li>Joint training adds both datasets' losses before one update, cycles the smaller dataset and weights the fMRI arm; the same function trains single and joint models for a fair comparison.</li>
<li>The <b>delay</b> head is bounded by a sigmoid, which makes its output uninformative unless tested against other windows. <b>Reliability</b> combines predicted variance and Monte Carlo dropout, then calibration.</li>
</ul>
<h3>Coming next: Lecture E</h3>
<p>Lecture E is about evaluation: how to measure whether any of this works without fooling yourself — correlations, null tests, confidence intervals, and the many ways a model can look good for the wrong reasons.</p>

<h2>Glossary</h2>
{table(4, "Terms introduced in this lecture.",
       ["Term", "Meaning"],
       [["Parameter / weight / bias", "the adjustable numbers inside a network"],
        ["Activation function (GELU)", "the non-linear function applied to each neuron's sum"],
        ["Forward / backward pass", "computing predictions / computing gradients"],
        ["Gradient", "the direction each parameter should change to reduce the loss"],
        ["Learning rate", "the size of each update step"],
        ["Batch / epoch", "a group of examples processed together / one pass over all training data"],
        ["Optimiser (AdamW)", "the rule that turns gradients into parameter updates"],
        ["Overfitting", "memorising training data instead of learning the general relationship"],
        ["Dropout", "randomly switching off neurons during training"],
        ["Weight decay", "a penalty that keeps weights small"],
        ["Batch normalisation", "rescaling activity within each batch to stabilise training"],
        ["Convolution / kernel", "sliding a learned template along a signal / that template"],
        ["Stride / pooling", "jumping several samples between positions / summarising stretches"],
        ["Adapter", "a small dataset-specific input layer"],
        ["Trunk / head", "the shared part of the network / an output-specific part"],
        ["Embedding", "a learned vector attached to an identity (here a subject)"],
        ["Pearson loss", "one minus the correlation between prediction and target"],
        ["Gaussian NLL", "a loss that trains a predicted variance to match actual errors"],
        ["Sigmoid", "a function squashing any number into the range 0 to 1"],
        ["Monte Carlo dropout", "keeping dropout on at test time to measure prediction spread"],
        ["Platt scaling", "fitting a logistic curve to turn scores into calibrated probabilities"],
        ["Seed", "the starting point of the random number generator"]],
       widths=["30%", "70%"])}

<h2>Answers to "Pause and think"</h2>
<ol class="answers">
<li>Weights set how strongly each input influences each neuron. Training adjusts them, step by step, so the network's outputs come closer to the correct answers.</li>
<li>Without them, any stack of layers collapses into one linear function and cannot represent complicated relationships.</li>
<li>The gradient tells each parameter which way to move to reduce the loss; the learning rate says how big a step to take in that direction.</li>
<li>Training loss keeps falling while validation loss on new people stalls or rises. Defences used: dropout, weight decay, batch normalisation, a small model, limited epochs and subject-level validation.</li>
<li>The same pattern means the same thing wherever it occurs in time, and convolution reuses one small set of weights across all positions, needing few parameters.</li>
<li>6,000 → 1,200 → 300 → 100, then pooled to 16.</li>
<li>So the model treats them as a typical person and cannot use their identity; any success must come from their EEG.</li>
<li>It rewards correct shape and timing regardless of scale, ignores the large unit and drift differences between datasets, and does not reward flat average predictions.</li>
<li>To predict, for each value, how large its error is likely to be: it is rewarded when claimed uncertainty matches actual error.</li>
<li>Separate updates partly undo each other; adding the losses gives one update that serves both, which is genuine joint training.</li>
<li>The sigmoid forces every output into the window, so landing inside it is guaranteed. Only stability across different windows, or agreement with an independent measurement, would show physiology.</li>
<li>Different neurons are dropped on each of 20 passes, so predictions vary; small variation means the model is consistent (confident), large variation means it is unsure.</li>
</ol>
"""

D.render(body, "Lecture D: The Model")
