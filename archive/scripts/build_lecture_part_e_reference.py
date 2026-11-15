"""Builds lectures/Lecture_E_Reference_General_Knowledge.pdf.

General statistics and evaluation background (Part E companion).
Run from the repo root:  python scripts/build_lecture_part_e_reference.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from lecture_common import (AQUA, BLUE, GRAY, INK, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, K, M, V,
                            fig, qa_bank, table, title_block)

D = Doc("lecture_e_ref", "Lecture_E_Reference_General_Knowledge.pdf", "NeuroVascNet course · Lecture E reference")


def f_errors():
    x = np.linspace(-4, 7, 500)
    h0 = stats.norm.pdf(x, 0, 1); h1 = stats.norm.pdf(x, 2.5, 1)
    crit = 1.645
    fig, a = plt.subplots(figsize=(W, 2.4))
    a.plot(x, h0, color=GRAY, lw=2, label="no real effect (null)")
    a.plot(x, h1, color=BLUE, lw=2, label="real effect")
    a.fill_between(x, 0, h0, where=x >= crit, color=RED, alpha=0.35, label="false positive (type I, α)")
    a.fill_between(x, 0, h1, where=x < crit, color=ORANGE, alpha=0.3, label="missed effect (type II, β)")
    a.axvline(crit, color=INK, ls="--", lw=1); a.text(crit + 0.1, 0.28,"decision\nthreshold", fontsize=8.3)
    a.set_yticks([]); a.set_xlabel("test statistic"); a.legend(loc="upper right", fontsize=8)
    a.set_title("Power = 1 − β: the chance of detecting a real effect", fontsize=10)
    D.save(fig, "re1_errors")


def f_smalln():
    rng = np.random.default_rng(0)
    ns = [10, 20, 30, 50, 100, 200, 500]
    fig, a = plt.subplots(figsize=(W, 2.4))
    for n in ns:
        est = [np.corrcoef(*(lambda x: (x, 0.2 * x + np.sqrt(1 - 0.04) * rng.standard_normal(n)))(rng.standard_normal(n)))[0, 1] for _ in range(300)]
        lo, hi = np.percentile(est, [2.5, 97.5])
        a.plot([n, n], [lo, hi], color=BLUE, lw=3)
        a.scatter(n, np.mean(est), color=INK, s=18, zorder=3)
    a.axhline(0.2, color=ORANGE, ls="--", lw=1.2); a.text(12, 0.24, "true r = 0.2", fontsize=8.5, color=ORANGE, ha="left")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xscale("log"); a.set_xticks(ns); a.set_xticklabels(ns)
    a.set_xlabel("number of independent observations (e.g. subjects)"); a.set_ylabel("estimated r")
    a.set_title("Small samples give wide error bars", fontsize=10)
    D.save(fig, "re2_smalln")


def f_multiple():
    m = np.array([1, 5, 10, 20, 50, 100])
    fig, a = plt.subplots(figsize=(W, 2.2))
    a.plot(m, 1 - 0.95 ** m, color=RED, lw=2, marker="o", ms=5, label="no correction")
    a.plot(m, np.full_like(m, 0.05, dtype=float), color=AQUA, lw=2, marker="o", ms=5, label="Bonferroni / Holm (family-wise)")
    a.set_xscale("log"); a.set_xticks(m); a.set_xticklabels(m)
    a.set_xlabel("number of independent tests at p < 0.05"); a.set_ylabel("chance of ≥ 1 false positive")
    a.legend(loc="center right", fontsize=8.3); a.set_ylim(0, 1.05)
    D.save(fig, "re3_multiple")


for f in [f_errors, f_smalln, f_multiple]:
    f()

QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "Metrics"
qa(S, "Pearson versus Spearman correlation?", "Pearson measures linear agreement and is sensitive to outliers. Spearman correlates ranks, measuring any monotonic relationship and resisting outliers. For time courses with a few artifacts, Spearman or robust alternatives are worth reporting alongside Pearson.")
qa(S, "What is R², and can it be negative?", "The coefficient of determination: 1 minus the ratio of the model's squared error to the variance of the target. Out of sample it can be negative, meaning the model is worse than predicting the mean. It is not the same as the square of a correlation, which ignores scale and offset.")
qa(S, "MSE, RMSE or MAE?", "Mean squared error penalises large errors heavily; its square root (RMSE) is in the target's units. Mean absolute error is more robust to outliers. All depend on the target's scale, so they are hard to compare across datasets; correlation is scale-free.")
qa(S, "What is chance level for a classifier?", "For balanced classes, 1 divided by the number of classes. But with small test sets, random classifiers often exceed it by chance; Combrisson and Jerbi (2015) show that the threshold for significance with few trials can be well above nominal chance. Use permutation tests or binomial tests instead of comparing with the nominal value.")
qa(S, "What are balanced accuracy and AUC?", "Balanced accuracy averages accuracy within each class, so it is not inflated by imbalanced classes. The area under the ROC curve (AUC) measures how well scores rank positives above negatives across all thresholds; 0.5 is chance.")
qa(S, "How is calibration measured?", "With reliability diagrams, the expected calibration error (average gap between confidence and accuracy across bins) and proper scoring rules such as the Brier score or log-likelihood. Selective prediction is judged with risk–coverage curves and their area (AURC).")

S = "Hypothesis testing"
qa(S, "What exactly is a p-value?", "The probability, assuming the null hypothesis is true, of a result at least as extreme as the one observed. It is not the probability that the null hypothesis is true, and not the probability that the result will replicate.")
qa(S, "What are type I and type II errors, and what is power?", "A type I error is a false positive (rejecting a true null; rate α, usually 0.05). A type II error is a missed real effect (rate β). Power, 1 − β, is the chance of detecting an effect of a given size; it grows with sample size and effect size.")
qa(S, "Why is low power a problem even when a result is significant?", "Underpowered studies detect only effects that happen to look large in their sample, so significant results from them overestimate the true effect and replicate poorly (Button and colleagues, 2013, for neuroscience).")
qa(S, "Parametric or non-parametric tests?", "Parametric tests (t-test, ANOVA) assume distributions such as normality. Non-parametric tests (Wilcoxon signed-rank, Mann–Whitney, permutation tests) make fewer assumptions and suit small samples and unusual distributions.")
qa(S, "Why are permutation tests popular in neuroimaging?", "They build the null distribution from the data themselves by shuffling labels or pairings, making few assumptions and handling complex statistics and dependencies naturally (Nichols and Holmes, 2002). The shuffling must respect the data's structure, for example shuffling within or between subjects as appropriate.")
qa(S, "What is pseudo-replication?", "Treating non-independent observations (trials from the same person, voxels from the same brain) as independent, which inflates sample size and produces false significance. The remedy is to analyse at the level of independent units or use mixed-effects models.")
qa(S, "What is a mixed-effects model?", "A regression with fixed effects (the conditions of interest) and random effects (variation between subjects, sessions or items). It uses all trials while correctly accounting for their grouping, and is the standard tool for repeated-measures data.")

S = "Uncertainty and effect size"
qa(S, "How should a 95% confidence interval be interpreted?", "If the study were repeated many times, 95% of such intervals would contain the true value. Practically, it shows the range of effect sizes compatible with the data; an interval that includes zero and also large values means the study cannot tell.")
qa(S, "What is the bootstrap?", "A method (Efron, 1979) that estimates the variability of a statistic by repeatedly resampling the data with replacement and recomputing it. Resampling must be done at the level of independent units — subjects, not trials.")
qa(S, "Which effect sizes are common?", "Cohen's d (difference in means divided by the standard deviation), correlation r and r², odds ratios, and percentage of a ceiling. Report them with confidence intervals; a p-value alone conveys neither size nor precision.")
qa(S, "Why do small samples give unreliable cross-validation estimates?", "With few test subjects, the estimate varies widely from split to split; Varoquaux (2018) showed error bars of around ±10% accuracy at typical neuroimaging sample sizes. Report the spread across folds and seeds, not just the mean.")

S = "Multiple comparisons"
qa(S, "Why correct for multiple comparisons?", "Each test at p < 0.05 has a 5% false-positive chance; with 20 independent tests, at least one false positive is expected about 64% of the time.")
qa(S, "Bonferroni, Holm or FDR?", "Bonferroni divides α by the number of tests (simple, conservative). Holm applies the same idea step by step and is always at least as powerful. False-discovery-rate control (Benjamini–Hochberg) limits the expected proportion of false positives among discoveries, giving more power when many tests are run, as in voxel-wise analyses.")
qa(S, "What is the garden of forking paths?", "Even without deliberate p-hacking, analysis choices made after seeing the data (which subjects, which filter, which window, which metric) create many possible analyses, inflating false positives (Gelman and Loken). Pre-registration and pre-declared decision rules close the paths.")
qa(S, "What are p-hacking and HARKing?", "P-hacking is trying analyses until one is significant. HARKing is Hypothesising After the Results are Known: presenting a post-hoc finding as if it had been predicted. Both make published results unreliable.")

S = "Validation design and leakage"
qa(S, "What forms of data leakage occur in neuroimaging machine learning?", "Same subject in training and test sets; preprocessing or feature selection fitted on all data; hyperparameters tuned on the test set; temporal leakage when overlapping windows straddle the split; and duplicated recordings. Kapoor and Narayanan (2023) document leakage behind many irreproducible machine-learning results.")
qa(S, "What is double dipping (circular analysis)?", "Using the same data to select something (voxels, channels, a time window) and then to test it, which guarantees a biased positive result (Kriegeskorte and colleagues, 2009). Selection and testing must use independent data.")
qa(S, "What makes a good baseline?", "One that could achieve the result without the ingredient you are claiming: a mean predictor, a shuffled-label model, a simpler model on standard features, or your model without the new component at the same budget. Beating only weak or untrained baselines is not evidence.")
qa(S, "Why test externally, and what does a drop mean?", "External data reveal whether a model depends on the quirks of one study. A drop can come from different people, devices, montages, tasks or target quality; good reporting separates these where possible and states which could not be separated.")

S = "Reliability"
qa(S, "What is test–retest reliability and the ICC?", "How consistent a measure is when repeated. The intraclass correlation (Shrout and Fleiss, 1979) compares variance between people with total variance; values above about 0.75 are usually called good, 0.5–0.75 moderate.")
qa(S, "What is the Spearman–Brown formula?", "It predicts the reliability of a measure when its length changes. Split-half reliability uses half the trials, so the full-length reliability is higher: r_full = 2r_half ÷ (1 + r_half).")
qa(S, "What is a noise ceiling?", "The best performance any model could achieve given the noise in the target, estimated from how well the target agrees with itself (for example across repeated trials or between subjects). Model performance is often reported as a fraction of the ceiling.")
qa(S, "What is attenuation, and can you correct for it?", "Measurement noise in two variables lowers (attenuates) their observed correlation. Spearman's correction divides the observed correlation by the square root of the product of the two reliabilities; corrected values should be reported alongside, not instead of, raw values.")

S = "Open science"
qa(S, "What is pre-registration, and what is a registered report?", "Pre-registration records hypotheses and analysis plans before data are analysed. A registered report goes further: the journal reviews and accepts the plan before results exist, so publication does not depend on the outcome.")
qa(S, "What should a methods section report so others can trust and reproduce the work?", "Data sources and versions, inclusion and exclusion criteria, every preprocessing parameter, model details and training settings, the exact split procedure, metrics, statistical tests, corrections, number of seeds, and code availability. Community checklists such as COBIDAS give detailed lists for MRI and EEG.")


body = title_block("NeuroVascNet course · Lecture E · Reference", "Statistics and Evaluation: General Knowledge",
                   "Metrics, hypothesis tests, uncertainty, multiple comparisons, leakage and reliability, with model answers",
                   "Companion to “Honest Evaluation” · Read Lecture E first") + f"""
<div class="box">
<h3>How to use this document</h3>
<p>Evaluation is where examiners and reviewers probe hardest. This reference covers the statistics and validation ideas they assume. Section 7 is a bank of {len(QA)} questions with model answers. For any "how do you know?" question, answer with four parts: <b>the unit of analysis</b>, <b>the comparison or baseline</b>, <b>the test and its result</b>, and <b>the size of the effect relative to a ceiling</b>.</p>
</div>

<h2>1. Choosing a metric</h2>
{table(1, "Common metrics.",
       ["Task", "Metric", "Watch out for"],
       [["Regression / signals", "Pearson r, Spearman ρ", "ignore scale; Pearson sensitive to outliers"],
        ["", "R² (out of sample)", "can be negative; not the same as r²"],
        ["", "MSE, RMSE, MAE", "depend on units; hard to compare across datasets"],
        ["Classification", "accuracy, balanced accuracy", "chance level is higher than nominal with few trials"],
        ["", "AUC", "measures ranking, not calibration"],
        ["Uncertainty", "ECE, Brier score, risk–coverage", "fit and evaluate calibration on separate data"],
        ["Reliability", "split-half, ICC", "Spearman–Brown correction for test length"]],
       widths=["22%", "34%", "44%"])}

<h2>2. Hypothesis tests</h2>
<p>A test compares the data with what a <b>null hypothesis</b> (no effect) would produce. The <b>p-value</b> is how often the null would give a result at least this extreme; it is <i>not</i> the probability that the null is true. Two kinds of mistakes are possible, and their balance depends on the threshold and the sample size.</p>
{fig("re1_errors", 1, "Two kinds of error.", "Schematic. If there is no effect, results beyond the threshold are false positives (red, rate α). If there is a real effect, results below the threshold are missed (orange, rate β). Power, 1 − β, rises with more data and larger effects.")}
<p><b>Parametric</b> tests (t-tests, ANOVA) assume distributions; <b>non-parametric</b> tests (Wilcoxon, Mann–Whitney) and <b>permutation tests</b> assume less. Permutation tests are popular in neuroimaging because the null distribution comes from the data, but the shuffling must respect the data's structure. Treating dependent observations as independent — trials from one person, neighbouring voxels — is <b>pseudo-replication</b>; <b>mixed-effects models</b> handle grouped data properly.</p>
{M("<p>Reading p = 0.04 as \"96% sure the effect is real\". A p-value says nothing about the probability of the hypothesis, and with a small sample a significant result is likely to overestimate the true effect.</p>")}

<h2>3. Uncertainty and sample size</h2>
<p>Every estimate has uncertainty, best shown as a <b>confidence interval</b> and an <b>effect size</b>. The <b>bootstrap</b> estimates intervals by resampling independent units. Small samples give wide intervals, which is a central problem in neuroimaging machine learning: with tens of subjects, cross-validated performance can vary by about ±10 percentage points between splits (Varoquaux, 2018).</p>
{fig("re2_smalln", 2, "Estimates from small samples are unstable.", "Simulation: the true correlation is 0.2. Bars show where 95% of estimates fall at each sample size. With 10–30 independent units the estimate can range from about zero to more than double the truth.")}
{K("<p>Low statistical power does not only cause missed effects. Among the significant results an underpowered study produces, effects are systematically inflated, which is why small neuroscience studies often fail to replicate (Button and colleagues, 2013).</p>")}

<h2>4. Many tests at once</h2>
{fig("re3_multiple", 3, "Why corrections are needed.", "Without correction, the chance of at least one false positive grows rapidly with the number of independent tests. Family-wise corrections such as Bonferroni or Holm keep it at 5%.")}
<p><b>Bonferroni</b> divides α by the number of tests; <b>Holm</b> does so step by step and is uniformly more powerful; <b>false-discovery-rate</b> control (Benjamini–Hochberg) bounds the expected proportion of false discoveries and suits large numbers of tests. Less visible multiplicity comes from flexible analysis choices — the <b>garden of forking paths</b> — and from p-hacking and HARKing. <b>Pre-registration</b> and <b>registered reports</b> are the main defences.</p>

<h2>5. Validation design and leakage</h2>
<p>Most inflated machine-learning results trace back to <b>leakage</b>: information from the test data reaching the training process. In neuroimaging the main routes are:</p>
<ul>
<li>the same <b>subject</b> in training and test sets;</li>
<li><b>preprocessing, normalisation or feature selection</b> fitted on all data;</li>
<li><b>hyperparameters or stopping points</b> chosen on the test set;</li>
<li><b>overlapping windows</b> straddling the split;</li>
<li><b>double dipping</b>: selecting channels, voxels or windows on the same data used to test them (Kriegeskorte and colleagues, 2009).</li>
</ul>
<p>Good designs use subject-level splits, <b>nested cross-validation</b> for tuning, several seeds, strong baselines and, where possible, a fully external test set.</p>
{A("tasting your own cooking", "<p>A chef who adjusts a recipe while tasting from the same pot that will be judged cannot claim an unbiased verdict. Tuning, selecting and testing on the same data is tasting from the judged pot; independent data are a second, untouched pot.</p>")}

<h2>6. Reliability and ceilings</h2>
<p><b>Reliability</b> asks whether a measure agrees with itself. <b>Split-half</b> reliability compares two halves of the trials; the <b>Spearman–Brown</b> formula adjusts it to full length; the <b>intraclass correlation</b> (ICC) measures consistency across sessions or raters. Noise in a target sets a <b>noise ceiling</b> on any model's performance, and noise in both variables <b>attenuates</b> their correlation (Spearman's correction estimates the noise-free value). Reporting performance as a fraction of the ceiling makes results comparable across datasets of different quality.</p>
{V("<p><b>“Why report the ceiling at all?”</b> Because a correlation of 0.16 means very different things against a ceiling of 0.29 and against a ceiling of 0.71. The ceiling separates 'the model is weak' from 'the target is noisy', which lead to opposite next steps.</p>")}

<h2>7. Question bank with model answers</h2>
{qa_bank(QA)}

<h2>8. Where to read more</h2>
<ul class="refs">
<li>Nichols T. E., Holmes A. P. (2002). Nonparametric permutation tests for functional neuroimaging. <i>Human Brain Mapping</i>.</li>
<li>Efron B., Tibshirani R. (1993). <i>An Introduction to the Bootstrap</i>. Chapman &amp; Hall.</li>
<li>Button K. S. et al. (2013). Power failure: why small sample size undermines the reliability of neuroscience. <i>Nature Reviews Neuroscience</i>.</li>
<li>Kriegeskorte N. et al. (2009). Circular analysis in systems neuroscience: the dangers of double dipping. <i>Nature Neuroscience</i>.</li>
<li>Varoquaux G. (2018). Cross-validation failure: small sample sizes lead to large error bars. <i>NeuroImage</i>.</li>
<li>Poldrack R. A., Huckins G., Varoquaux G. (2020). Establishment of best practices for evidence for prediction. <i>JAMA Psychiatry</i>.</li>
<li>Kapoor S., Narayanan A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. <i>Patterns</i>.</li>
<li>Combrisson E., Jerbi K. (2015). Exceeding chance level by chance. <i>Journal of Neuroscience Methods</i>.</li>
<li>Benjamini Y., Hochberg Y. (1995). Controlling the false discovery rate. <i>Journal of the Royal Statistical Society B</i>.</li>
<li>Shrout P. E., Fleiss J. L. (1979). Intraclass correlations: uses in assessing rater reliability. <i>Psychological Bulletin</i>.</li>
<li>Gelman A., Loken E. (2014). The statistical crisis in science. <i>American Scientist</i>.</li>
</ul>
"""

D.render(body, "Lecture E Reference: General Knowledge")
