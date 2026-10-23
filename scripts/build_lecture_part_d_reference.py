"""Builds lectures/Lecture_D_Reference_General_Knowledge.pdf.

General deep-learning and EEG-modelling background (Part D companion).
Run from the repo root:  python scripts/build_lecture_part_d_reference.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, K, M, V,
                            arrow, box, fig, qa_bank, table, title_block)

D = Doc("lecture_d_ref", "Lecture_D_Reference_General_Knowledge.pdf", "NeuroVascNet course · Lecture D reference")


def f_fit():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 18)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, len(x))
    xs = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, deg, title in [(ax[0], 1, "Underfitting: too simple"), (ax[1], 3, "Good fit"), (ax[2], 11, "Overfitting: memorises noise")]:
        c = np.polyfit(x, y, deg)
        a.scatter(x, y, s=14, color=GRAY, zorder=3)
        a.plot(xs, np.clip(np.polyval(c, xs), -1.9, 1.9), color=[ORANGE, AQUA, RED][[1, 3, 11].index(deg)], lw=2)
        a.set_title(title, fontsize=9.5); a.set_xticks([]); a.set_yticks([]); a.set_ylim(-2, 2)
    D.save(fig, "rd1_fit")


def f_lr():
    ep = np.arange(0, 50)
    fig, a = plt.subplots(figsize=(W, 2.2))
    a.plot(ep, 1.0 - 0.012 * ep, color=GRAY, lw=2, label="too small: slow progress")
    a.plot(ep, 0.15 + 0.85 * np.exp(-ep / 6), color=AQUA, lw=2, label="good")
    a.plot(ep, 0.6 + 0.35 * np.abs(np.sin(ep * 1.3)) + 0.004 * ep, color=RED, lw=2, label="too large: unstable")
    a.set_xlabel("training steps"); a.set_ylabel("loss"); a.set_yticks([]); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8.3)
    a.set_title("The learning rate is the most important single setting", fontsize=10)
    D.save(fig, "rd2_lr")


def f_families():
    fig, ax = plt.subplots(1, 4, figsize=(W, 2.1), gridspec_kw={"wspace": 0.1})
    for a in ax:
        a.set_xlim(0, 4); a.set_ylim(0, 3.3); a.axis("off")
    a = ax[0]; a.set_title("MLP", fontsize=9.5)
    for i in range(3):
        for j in range(3):
            a.plot([0.8, 3.2], [0.6 + i, 0.6 + j], color=GRAY, lw=0.6)
    for i in range(3):
        a.add_patch(plt.Circle((0.8, 0.6 + i), 0.18, color=BLUE)); a.add_patch(plt.Circle((3.2, 0.6 + i), 0.18, color=ORANGE))
    a.text(2, 0.05, "every input to every unit", ha="center", fontsize=7.6, color=MUTED)
    a = ax[1]; a.set_title("CNN", fontsize=9.5)
    xs = np.linspace(0.3, 3.7, 12)
    for xx in xs:
        a.add_patch(plt.Circle((xx, 0.8), 0.08, color=BLUE))
    a.add_patch(plt.Rectangle((1.0, 0.6), 0.95, 0.4, fill=False, ec=ORANGE, lw=1.6))
    a.add_patch(plt.Circle((1.47, 2.2), 0.12, color=ORANGE)); arrow(a, (1.47, 1.0), (1.47, 2.05), color=ORANGE)
    a.text(2, 0.05, "local window slides", ha="center", fontsize=7.6, color=MUTED)
    a = ax[2]; a.set_title("RNN / LSTM", fontsize=9.5)
    for i in range(4):
        box(a, 0.2 + i * 0.95, 1.4, 0.7, 0.6, "", fc="#fff4e5", ec=YELLOW)
        if i < 3:
            arrow(a, (0.9 + i * 0.95, 1.7), (1.15 + i * 0.95, 1.7))
        arrow(a, (0.55 + i * 0.95, 0.7), (0.55 + i * 0.95, 1.4))
    a.text(2, 0.05, "memory carried step by step", ha="center", fontsize=7.6, color=MUTED)
    a = ax[3]; a.set_title("Transformer", fontsize=9.5)
    pts = [(0.5, 1.0), (1.5, 1.0), (2.5, 1.0), (3.5, 1.0)]
    for p in pts:
        a.add_patch(plt.Circle(p, 0.14, color=BLUE))
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            a.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-", color=VIOLET, lw=0.9, connectionstyle="arc3,rad=-0.5"))
    a.text(2, 0.05, "every step attends to every step", ha="center", fontsize=7.6, color=MUTED)
    D.save(fig, "rd3_families")


def f_receptive():
    fig, a = plt.subplots(figsize=(W, 2.3))
    a.set_xlim(-0.5, 16.5); a.set_ylim(-0.6, 3.6); a.axis("off")
    for layer, dil, col in [(0, 0, GRAY), (1, 1, BLUE), (2, 2, AQUA), (3, 4, ORANGE)]:
        for i in range(17):
            a.add_patch(plt.Circle((i, layer), 0.13, color=col if layer else GRAY, alpha=1 if layer else 0.6))
    top = 16
    # trace dependencies of the top-right unit
    frontier = {top}
    for layer, dil in [(3, 4), (2, 2), (1, 1)]:
        nxt = set()
        for f in frontier:
            for d in (0, dil):
                if f - d >= 0:
                    a.plot([f, f - d], [layer, layer - 1], color=INK, lw=0.9)
                    nxt.add(f - d)
        frontier = nxt
    for layer, lab in [(0, "input"), (1, "dilation 1"), (2, "dilation 2"), (3, "dilation 4")]:
        a.text(-0.4, layer, lab, ha="right", va="center", fontsize=8.3)
    a.text(8, -0.55, "one output at the top sees 8 input steps after three layers; doubling the gaps doubles the reach", ha="center", fontsize=8.3, color=MUTED)
    D.save(fig, "rd4_dilation")


for f in [f_fit, f_lr, f_families, f_receptive]:
    f()

QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "Foundations"
qa(S, "What is the difference between a model, its parameters and its hyperparameters?", "The model is the architecture (the structure of the function). Parameters are the weights learned from data. Hyperparameters are choices made before training — learning rate, batch size, number of layers, dropout rate — and are tuned on validation data, never on test data.")
qa(S, "What is backpropagation?", "An efficient way to compute the gradient of the loss with respect to every parameter by applying the chain rule backwards from the output through each layer. It makes training deep networks feasible.")
qa(S, "What is the bias–variance trade-off?", "Simple models are biased (they underfit, missing real structure); flexible models have high variance (they overfit, fitting noise). Good models sit between, and more data lets you afford more flexibility.")
qa(S, "What is stochastic gradient descent, and why use mini-batches?", "Gradient descent that estimates the gradient from a small random batch rather than the whole dataset. It is much faster per step, and the noise in the estimate can help escape poor solutions.")
qa(S, "What do Adam and AdamW do?", "Adam adapts the step size for each parameter using running averages of the gradient and its square. AdamW is Adam with weight decay applied correctly (decoupled from the adaptive step), which generalises better and is now the usual default.")
qa(S, "What is a learning-rate schedule?", "A rule that changes the learning rate during training, for example warming up from a small value, then decaying it (step, cosine or plateau-based). Large early steps make fast progress; small late steps settle into a good solution.")
qa(S, "Why does initialisation matter?", "Weights that start too large or too small make signals and gradients explode or vanish through many layers. Schemes such as He or Xavier initialisation scale starting weights by layer size to keep signals stable.")
qa(S, "What are vanishing and exploding gradients, and how are they handled?", "In deep networks gradients can shrink towards zero or grow uncontrollably as they pass backwards through many layers. Remedies include careful initialisation, normalisation layers, residual (skip) connections, gating in LSTMs and gradient clipping.")

S = "Architectures"
qa(S, "When would you use an MLP, a CNN, an RNN or a Transformer?", "MLPs suit fixed-length feature vectors. CNNs suit data with local, repeating structure in time or space. RNNs and LSTMs process sequences step by step with memory. Transformers use attention to relate every position to every other, and excel with large datasets; with small datasets they overfit easily.")
qa(S, "What is a receptive field?", "The span of input that can influence one output unit. Stacking convolutions, using strides or pooling, and dilation all enlarge it. For a slow haemodynamic target the receptive field must cover seconds of EEG.")
qa(S, "What is a temporal convolutional network (TCN)?", "A stack of one-dimensional convolutions with increasing dilation (gaps between kernel taps), usually causal and with residual connections, giving a very long receptive field with few layers. It is a common alternative to recurrent networks for sequences.")
qa(S, "What is EEGNet?", "A compact convolutional network for EEG (Lawhern and colleagues, 2018): a temporal convolution learns frequency filters, a depthwise convolution learns spatial filters per frequency band (like common spatial patterns), and a separable convolution combines them. It has few parameters, so it works on small EEG datasets.")
qa(S, "What are ShallowConvNet and DeepConvNet?", "Two reference EEG architectures (Schirrmeister and colleagues, 2017). ShallowConvNet mimics filter-bank common spatial patterns: temporal and spatial convolution, squaring, pooling and a log. DeepConvNet is a deeper, more generic stack. Both are standard baselines for EEG decoding.")
qa(S, "What is a residual connection?", "A shortcut that adds a block's input to its output, so the block learns only a correction. It eases training of deep networks (He and colleagues, 2016).")
qa(S, "What is attention?", "A mechanism that computes, for each position, a weighted average of all positions, with weights based on learned similarity. It lets a model focus on the most relevant parts of the input. Transformers are built from attention layers.")
qa(S, "What are EEG foundation models?", "Large networks pre-trained on thousands of hours of unlabelled EEG with self-supervised objectives (for example masked-signal reconstruction), then fine-tuned for specific tasks. Examples include BENDR and LaBraM. They aim to overcome small labelled datasets.")

S = "Losses and outputs"
qa(S, "Which regression losses are common?", "Mean squared error (penalises large errors strongly), mean absolute error (robust to outliers), Huber loss (a compromise), correlation-based losses (shape regardless of scale) and negative log-likelihood (when predicting a distribution).")
qa(S, "When is a correlation loss better than mean squared error?", "When the scale or offset of the target is arbitrary or unreliable (different units, drifts, devices) and the shape over time is what matters. Its drawback is that it says nothing about amplitude, so amplitude must be handled separately if needed.")
qa(S, "What is heteroscedastic regression?", "Predicting both a value and its noise level for each input, trained with the Gaussian negative log-likelihood. The model learns where its predictions are less precise.")
qa(S, "How do you balance several losses?", "With fixed weights tuned on validation data, by normalising each loss's scale, or adaptively — for example weighting by learned uncertainty (Kendall and colleagues, 2018) or balancing gradient sizes (GradNorm). Badly balanced losses let one task dominate.")
qa(S, "What is gradient conflict in multi-task learning?", "When two tasks' gradients point in opposing directions for the shared parameters, so improving one harms the other. Methods such as PCGrad project away the conflicting component. Conflict is one cause of negative transfer.")

S = "Regularisation and normalisation"
qa(S, "How does dropout work, and why does it help?", "During training, each neuron is switched off with some probability, so the network cannot depend on particular co-adapted neurons. It acts like training many thinned networks and averaging them. At test time it is normally switched off.")
qa(S, "BatchNorm versus LayerNorm?", "Batch normalisation rescales each feature using statistics across the batch; it speeds training but depends on batch composition and behaves differently in training and testing. Layer normalisation rescales across features within each example and is independent of the batch, which is why Transformers use it.")
qa(S, "What data augmentation is used for EEG?", "Adding noise, small time shifts, amplitude scaling, channel dropout, frequency shifts, and mixing examples. Augmentations must preserve the label; for a time-locked target, time shifts must be applied to input and target together or not at all.")
qa(S, "What is early stopping?", "Stopping training when the validation loss stops improving, keeping the best model. The validation data must be separate from the final test data, or the test result becomes optimistic.")
qa(S, "How do parameter count and data size relate?", "There is no fixed rule, but a model with many more parameters than independent examples can memorise them. With small neural datasets, compact architectures, strong regularisation, pre-training or classical models usually work better than very large networks.")

S = "Practice"
qa(S, "What is nested cross-validation?", "An outer loop of train/test splits to estimate performance, with an inner loop inside each training set to tune hyperparameters. It prevents the test folds from influencing hyperparameter choices.")
qa(S, "Why fix random seeds, and is that enough for reproducibility?", "Seeds make runs repeatable, but results should be reported over several seeds because one seed can be lucky. Full reproducibility also needs fixed data versions, software versions, deterministic GPU settings and released code.")
qa(S, "What classical baselines should a deep EEG model beat?", "For motor imagery: common spatial patterns with linear discriminant analysis, filter-bank CSP and Riemannian-geometry classifiers. For regression: ridge regression on band-power features. If a deep model does not beat these, its complexity is not justified.")
qa(S, "How can you see what a network has learned?", "Saliency or gradient maps, occlusion tests, inspecting learned filters, and probing intermediate layers with simple classifiers. All interpretations should be checked against controls such as randomly initialised networks.")
qa(S, "What is fine-tuning versus feature extraction?", "Feature extraction freezes a pre-trained network and trains only a new head on top. Fine-tuning updates some or all pre-trained weights for the new task. Fine-tuning usually performs better with enough data but can overfit or forget with little.")
qa(S, "Why can a randomly initialised network still produce useful features?", "Random convolutions with pooling compute many random filtered summaries of the input, some of which correlate with useful properties such as band power. This is why representation-learning claims must be compared against a random-network control.")


body = title_block("NeuroVascNet course · Lecture D · Reference", "Deep Learning for Brain Signals: General Knowledge",
                   "Architectures, training, losses, regularisation and good practice, with model answers",
                   "Companion to “The Model” · Read Lecture D first") + f"""
<div class="box">
<h3>How to use this document</h3>
<p>Lecture D explained our network. This reference covers the wider deep-learning toolkit that reviewers will compare it with, especially for EEG. Section 7 is a bank of {len(QA)} questions with model answers. When asked "why this architecture?", answer with <b>the property of the data</b> it matches, <b>the alternative</b> you considered, the <b>data-size constraint</b>, and the <b>baseline</b> it must beat.</p>
</div>

<h2>1. Learning and generalisation</h2>
<p>A model generalises if it performs well on data it has not seen. Too simple a model <b>underfits</b>; too flexible a model <b>overfits</b>. The balance between them is the <b>bias–variance trade-off</b>, and the amount of data decides where it lies: more independent examples let you use a more flexible model safely.</p>
{fig("rd1_fit", 1, "Underfitting, a good fit and overfitting.", "The same noisy points fitted with a straight line (too simple), a moderate curve (good) and a very flexible curve (which passes through the noise and would fail on new points).")}
<p>Training uses <b>stochastic gradient descent</b> on mini-batches, with gradients computed by <b>backpropagation</b>. Modern training almost always uses an adaptive optimiser such as <b>Adam</b> or <b>AdamW</b> with a <b>learning-rate schedule</b>. The learning rate is the setting most likely to make or break training.</p>
{fig("rd2_lr", 2, "Effect of the learning rate.", "Schematic loss curves. Too small: training crawls. Too large: the loss jumps around or diverges. A good rate falls quickly and settles.")}

<h2>2. Architecture families</h2>
{fig("rd3_families", 3, "Four families of neural network.", "MLPs connect everything to everything; CNNs slide local filters; recurrent networks carry a memory along the sequence; Transformers let every position attend to every other.")}
{table(1, "Which family suits which data.",
       ["Family", "Strength", "Weakness", "EEG examples"],
       [["MLP", "simple; good on fixed feature vectors", "ignores order; many parameters on raw signals", "classifiers on band-power features"],
        ["CNN", "local patterns, few parameters, shift-tolerant", "limited reach without depth or dilation", "EEGNet, ShallowConvNet, DeepConvNet"],
        ["RNN / LSTM / GRU", "memory across time", "slow, harder to train on long sequences", "sleep staging, continuous decoding"],
        ["TCN", "long reach via dilation; parallel", "fixed receptive field", "EEG–haemodynamic models, sequence regression"],
        ["Transformer", "global context via attention", "data-hungry; overfits small sets", "EEG Conformer, foundation models"]],
       widths=["16%", "29%", "29%", "26%"])}
<p>The <b>receptive field</b> — how much input one output can see — matters a great deal for EEG-to-haemodynamics, because the blood response depends on several seconds of activity. Strides, pooling and <b>dilation</b> enlarge it efficiently.</p>
{fig("rd4_dilation", 4, "Dilated convolutions.", "Each layer skips more inputs than the last (gaps of 1, 2, 4). The top output depends on 8 input steps after only three layers; a TCN uses this to see long stretches of time cheaply.")}
<h3>EEG-specific architectures</h3>
<p><b>EEGNet</b> (Lawhern and colleagues, 2018) separates temporal filtering (learning frequency bands) from spatial filtering (learning electrode combinations) and has very few parameters. <b>ShallowConvNet</b> and <b>DeepConvNet</b> (Schirrmeister and colleagues, 2017) are standard reference models; ShallowConvNet mirrors the classic filter-bank common-spatial-patterns pipeline. Recent work adds attention (EEG Conformer) and large self-supervised <b>foundation models</b> (for example BENDR, LaBraM) pre-trained on thousands of hours of EEG. For motor imagery, a deep model should always be compared against <b>common spatial patterns with linear discriminant analysis</b> and <b>Riemannian</b> classifiers, which remain strong.</p>
{V("<p><b>“Why not use a Transformer?”</b> With a few thousand trials from a few dozen people, attention models overfit easily and offer little advantage; compact convolutional models are the standard choice at this data scale. A pre-trained EEG foundation model would be the natural way to bring attention in.</p>")}

<h2>3. Losses</h2>
{table(2, "Regression losses and when to use them.",
       ["Loss", "Behaviour", "Use when"],
       [["Mean squared error", "punishes large errors heavily", "scale is meaningful and noise is roughly Gaussian"],
        ["Mean absolute error", "robust to outliers", "occasional large artifacts"],
        ["Huber", "squared near zero, absolute far away", "a compromise between the two"],
        ["Correlation (1 − r)", "ignores scale and offset", "shape and timing matter, scale is unreliable"],
        ["Gaussian NLL", "trains a predicted variance too", "you need per-prediction uncertainty"]],
       widths=["24%", "36%", "40%"])}
<p>With several losses or tasks, their <b>balance</b> matters. Weights can be fixed and tuned on validation data, scaled to comparable magnitudes, or set adaptively by learned uncertainty (Kendall and colleagues, 2018) or gradient-balancing methods (GradNorm). When tasks' gradients point in opposite directions — <b>gradient conflict</b> — methods such as PCGrad can reduce interference.</p>

<h2>4. Regularisation and normalisation</h2>
<ul>
<li><b>Dropout</b> randomly removes neurons during training; <b>weight decay</b> keeps weights small; <b>early stopping</b> ends training at the best validation point.</li>
<li><b>Data augmentation</b> for EEG: noise, scaling, channel dropout, small shifts, mixing trials — always preserving the target.</li>
<li><b>Batch normalisation</b> normalises across the batch (fast training, but depends on batch makeup); <b>layer normalisation</b> normalises within each example (used in Transformers). Separate normalisation statistics per dataset or domain can absorb dataset differences.</li>
<li><b>Residual connections</b> and careful <b>initialisation</b> keep gradients healthy in deep networks.</li>
</ul>
{M("<p>Choosing hyperparameters, early-stopping points or architectures by looking at test-set results. Every such choice must use validation data inside the training set — ideally with nested cross-validation — or the reported performance is optimistic.</p>")}

<h2>5. Transfer and representation learning</h2>
<p>Networks can be <b>pre-trained</b> on a large dataset or a self-supervised task, then reused by <b>feature extraction</b> (freeze and add a head) or <b>fine-tuning</b> (update the weights). Representations are judged with <b>linear probes</b> and must be compared with a <b>randomly initialised network</b>, because random convolutional features are often surprisingly informative. Interpretation tools (saliency maps, occlusion, filter inspection, layer probes) suggest what a network uses but need the same controls.</p>
{A("judging a cook by the dish, not the kitchen", "<p>A shiny kitchen (a complex architecture) does not guarantee good food. Judge the dish against simple, honest comparisons: what a basic recipe (a classical baseline) produces, and what a cook throwing random ingredients together (a random network) produces. Only a clear improvement over both shows the kitchen earned its cost.</p>")}

<h2>6. Good practice checklist</h2>
<ul>
<li>Compare against classical baselines and a random-network control.</li>
<li>Tune on validation data only; use nested cross-validation for small datasets.</li>
<li>Report results over several random seeds with their spread.</li>
<li>Match training budgets when comparing two models.</li>
<li>Report parameter counts, training settings and the exact data splits.</li>
<li>Release code and fix software and data versions for reproducibility.</li>
</ul>

<h2>7. Question bank with model answers</h2>
{qa_bank(QA)}

<h2>8. Where to read more</h2>
<ul class="refs">
<li>Goodfellow I., Bengio Y., Courville A. (2016). <i>Deep Learning</i>. MIT Press (free online).</li>
<li>Lawhern V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. <i>Journal of Neural Engineering</i>.</li>
<li>Schirrmeister R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. <i>Human Brain Mapping</i>.</li>
<li>Bai S., Kolter J. Z., Koltun V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. <i>arXiv</i>.</li>
<li>Roy Y. et al. (2019). Deep learning-based electroencephalography analysis: a systematic review. <i>Journal of Neural Engineering</i>.</li>
<li>Loshchilov I., Hutter F. (2019). Decoupled weight decay regularization. <i>ICLR</i>.</li>
<li>Ioffe S., Szegedy C. (2015). Batch normalization. <i>ICML</i>.</li>
<li>Srivastava N. et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. <i>JMLR</i>.</li>
<li>Kendall A., Gal Y., Cipolla R. (2018). Multi-task learning using uncertainty to weigh losses. <i>CVPR</i>.</li>
<li>Yu T. et al. (2020). Gradient surgery for multi-task learning (PCGrad). <i>NeurIPS</i>.</li>
</ul>
"""

D.render(body, "Lecture D Reference: General Knowledge")
