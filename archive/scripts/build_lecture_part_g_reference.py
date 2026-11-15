"""Builds lectures/Lecture_G_Reference_General_Knowledge.pdf.

General background on communicating research (Part G companion): paper
structure, figures, choosing venues, peer review, open science, authorship,
presenting and examinations. Run:  python scripts/build_lecture_part_g_reference.py
"""
import matplotlib.pyplot as plt

from lecture_common import (AQUA, BLUE, GRAY, INK, MUTED, ORANGE, RED, VIOLET, W, YELLOW, A, Doc, K, M, V,
                            arrow, box, fig, qa_bank, table, title_block)

D = Doc("lecture_g_ref", "Lecture_G_Reference_General_Knowledge.pdf", "NeuroVascNet course · Lecture G reference")


def f_review():
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [("Submit", BLUE), ("Editor\nscreens", AQUA), ("Peer\nreview", YELLOW), ("Decision", ORANGE), ("Revise &\nrespond", VIOLET), ("Accept /\npublish", GRAY)]
    for i, (n, c) in enumerate(steps):
        x = 0.1 + i * 1.65
        box(ax, x, 1.7, 1.4, 1.0, n, fc=c, ec=c, color="white", weight="bold", fs=8.8)
        if i < 5:
            arrow(ax, (x + 1.42, 2.2), (x + 1.63, 2.2))
    ax.annotate("", xy=(4.1, 1.65), xytext=(7.4, 1.65),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5, connectionstyle="arc3,rad=-0.45",
                                shrinkA=0, shrinkB=0))
    ax.text(5.75, 0.35, "major / minor revision loops back to review", ha="center", fontsize=8.4, color=ORANGE)
    ax.text(3.0, 3.2, "desk rejection possible here", ha="center", fontsize=8.2, color=MUTED)
    arrow(ax, (3.0, 3.05), (2.45, 2.75), color=MUTED)
    D.save(fig, "rg1_review")


f_review()

QA = []
def qa(s, q, a): QA.append((s, q, a))

S = "Writing the paper"
qa(S, "What is the IMRaD structure?", "Introduction (why the question matters and what is unknown), Methods (enough detail to reproduce), Results (what was found, without interpretation), and Discussion (what it means, limitations, implications). Most empirical journals use it.")
qa(S, "How should an abstract be built?", "Five to seven sentences: context, gap, what was done, main quantitative results (with effect sizes), and the take-home message. It must stand alone and contain no claim absent from the paper.")
qa(S, "What makes a good title?", "It states the finding or the question precisely and avoids hype. Titles that state the result ('Cross-modal supervision does not improve EEG-based haemodynamic prediction') are increasingly common for clear findings, including negative ones.")
qa(S, "How should the introduction be organised?", "Funnel from the broad problem to the specific gap, cite the most relevant prior work fairly, and end with the aim and a short preview of the approach. Avoid overstating novelty.")
qa(S, "What belongs in the methods section?", "Data sources and versions, inclusion and exclusion, preprocessing parameters, model architecture and training settings, splitting and evaluation protocol, statistical tests, multiple-comparison handling, seeds, software versions and code availability.")
qa(S, "What makes a good discussion?", "Summarise the main findings, interpret them against prior work, explain unexpected results, state limitations with their likely effect, and end with concrete implications or next steps. Do not introduce new results.")

S = "Figures and reporting"
qa(S, "What makes a good scientific figure?", "One message per figure, stated in the caption's first sentence; readable labels; consistent colours across figures; colour-blind-safe palettes; individual data points where possible rather than only bars; and uncertainty shown.")
qa(S, "Why show individual data points?", "Bar charts of means hide distributions and outliers. Showing each subject's value (dot plots, paired lines) lets readers judge consistency, which is especially important with small samples.")
qa(S, "What reporting checklists exist?", "COBIDAS for MRI and MEEG, guidance for fNIRS reporting from the Society for fNIRS, TRIPOD for prediction models, and machine-learning reproducibility checklists at major conferences. They list what readers need to evaluate and reproduce the work.")
qa(S, "How should statistics be reported?", "Give the test, the statistic, degrees of freedom or permutation count, exact p-value, effect size with confidence interval, the unit of analysis (subjects) and any correction applied.")

S = "Publishing"
qa(S, "How do you choose a journal?", "By audience and scope first: where do the people who should read it publish and read? Then consider article types, open-access options and costs, typical review time, and data and code policies. Read a few recent papers from the journal to judge fit.")
qa(S, "What is a preprint?", "A version of the paper posted publicly (for example on bioRxiv or arXiv) before or during peer review. It establishes priority, lets the community give feedback, and is accepted by most journals.")
qa(S, "What happens in peer review?", "An editor screens the submission and may reject it without review (desk rejection). Otherwise two or three experts review it and recommend acceptance, minor or major revision, or rejection. Authors revise and respond point by point; this can repeat.")
qa(S, "How do you respond to reviewers?", "Thank them briefly, quote each comment, answer it directly, describe exactly what changed and where, and keep a respectful tone even when you disagree. Disagreement is acceptable if reasoned and supported by evidence.")
qa(S, "What is open access?", "Publication that is free to read. Gold open access is published openly by the journal (often with a fee); green open access is a freely available copy, such as an accepted manuscript in a repository.")
qa(S, "What are data and code availability statements?", "Statements saying where the data and code can be accessed and under what conditions. Many journals require them; sharing code (for example on GitHub with an archived release on Zenodo) greatly strengthens credibility.")

S = "Integrity and authorship"
qa(S, "Who qualifies as an author?", "Under common guidelines (ICMJE), people who made substantial contributions to the work, helped draft or revise it, approved the final version and accept accountability. The CRediT taxonomy records each person's specific roles (conceptualisation, software, analysis, writing and so on).")
qa(S, "What counts as research misconduct?", "Fabrication, falsification and plagiarism. Questionable practices — selective reporting, undisclosed analysis flexibility, p-hacking, HARKing — are not always misconduct but damage reliability and should be avoided.")
qa(S, "Why acknowledge the original dataset creators?", "Because their work made yours possible, licences often require citation, and correct attribution lets readers find the source and its documentation.")

S = "Presenting and examinations"
qa(S, "How should a research talk be structured?", "One clear question, the minimum background to understand it, the key methods, three or four main results each with one figure, limitations, and a take-home slide. Rehearse to time.")
qa(S, "What is a viva (thesis defence)?", "An oral examination in which examiners question the candidate about the thesis: its contribution, methods, results, limitations and context. They test understanding and ownership of the work more than memory of details.")
qa(S, "How do you handle a question you cannot answer?", "Say so honestly, explain how you would find out, or answer the closest question you can and say that is what you are doing. Bluffing is far more damaging than admitting a gap.")
qa(S, "How do you handle a hostile or critical question?", "Restate it to show you understood, acknowledge any valid part, then give your evidence. Stay calm and specific; examiners often test how you respond to criticism rather than expecting you to have no weaknesses.")
qa(S, "What are examiners most likely to ask about?", "What is new, why the methods were chosen, how alternative explanations were excluded, what the limitations are, how the work relates to the literature, and what you would do next or differently.")


body = title_block("NeuroVascNet course · Lecture G · Reference", "Communicating Research: General Knowledge",
                   "Writing papers, figures, publishing and peer review, integrity, presenting and vivas, with model answers",
                   "Companion to “Using the Results” · Read Lecture G first") + f"""
<div class="box">
<h3>How to use this document</h3>
<p>Lecture G showed how to frame and defend this project. This reference covers the general craft of communicating research: how papers are structured, what makes figures and statistics trustworthy, how publishing and peer review work, authorship and integrity, and how to present and handle an examination. Section 6 is a bank of {len(QA)} questions with model answers.</p>
</div>

<h2>1. Structure of a research paper</h2>
{table(1, "The IMRaD structure.",
       ["Section", "Purpose", "Common mistake"],
       [["Title and abstract", "state the question and main findings precisely", "promising more than the results deliver"],
        ["Introduction", "from broad problem to specific gap and aim", "a literature survey with no clear gap"],
        ["Methods", "enough detail for someone else to reproduce", "missing preprocessing, split or seed details"],
        ["Results", "what was found, with effect sizes and uncertainty", "interpretation mixed in; p-values without effect sizes"],
        ["Discussion", "meaning, comparison with prior work, limitations, implications", "new results; vague limitations; overclaiming"]],
       widths=["20%", "42%", "38%"])}
{K("<p>Write the figures and their one-sentence messages first. If you cannot state each figure's message in one sentence, the analysis or the story is not ready.</p>")}

<h2>2. Figures and reporting</h2>
<p>A good figure makes one point, stated in the first sentence of its caption. Show individual data points (each subject) rather than bare bars, show uncertainty, use consistent and colour-blind-safe colours, and label axes in words. Report statistics with the test, the statistic, the exact p-value, an effect size with its confidence interval, the unit of analysis and any correction. Community checklists — COBIDAS for MRI and EEG/MEG, fNIRS reporting guidance, TRIPOD for prediction models, and conference machine-learning checklists — list what readers need.</p>
{M("<p>A bar chart of two group means with error bars and a star. It hides how many people there were, how consistent the effect was and whether a few outliers drive it. Plot the people.</p>")}

<h2>3. Publishing</h2>
<p>Choose a journal by its audience first: where do the people who should read the work publish? Then check article types, open-access policy and costs, review times and data/code requirements. Many authors post a <b>preprint</b> at submission, which most journals accept.</p>
{fig("rg1_review", 1, "The peer-review process.", "The editor may reject without review. Otherwise experts review the paper and the editor decides; revisions are answered point by point and may be reviewed again.")}
<p>When responding to reviewers, quote each comment, answer it directly, say exactly what changed and where, and stay courteous even in disagreement. A reasoned, evidence-based disagreement is acceptable; ignoring a comment is not.</p>
{A("a conversation with a careful colleague", "<p>Treat reviewers as careful colleagues who have spent hours on your work and found its weak spots for you. Most comments either improve the paper or reveal where the writing was unclear — and the second kind is just as useful.</p>")}

<h2>4. Open science, integrity and authorship</h2>
<ul>
<li><b>Data and code</b>: state where they are and on what terms; archive a release of the code with a permanent identifier.</li>
<li><b>Pre-registration</b> and <b>registered reports</b> separate planned from exploratory analyses.</li>
<li><b>Authorship</b> follows contribution and accountability (ICMJE); the <b>CRediT</b> taxonomy records each person's roles.</li>
<li><b>Integrity</b>: fabrication, falsification and plagiarism are misconduct; selective reporting and undisclosed flexibility are questionable practices to avoid.</li>
<li><b>Attribution</b>: cite the creators of public datasets and software you used.</li>
</ul>

<h2>5. Presenting and examinations</h2>
<p>A talk should pose one clear question, give only the background needed, show three or four results each with a single figure, state limitations and end with a take-home message. In a <b>viva</b>, examiners test understanding and ownership: why each choice was made, how alternatives were excluded, what the limits are and what comes next.</p>
{V("<p><b>“What would you do differently if you started again?”</b> A strong answer shows you learned from the work: 'Measure target reliability and the HbO–HbR sign for every candidate dataset before building any model, and decide the evaluation protocol, controls and decision rules in advance.'</p>")}
{table(2, "Handling difficult questions.",
       ["Situation", "Good response"],
       [["You don't know", "say so, then explain how you would find out or what you would expect and why"],
        ["The question is unclear", "restate it in your own words and check before answering"],
        ["A valid criticism", "agree, then describe what you did about it or how it limits the claim"],
        ["A hostile tone", "stay calm, acknowledge the valid part, answer with evidence"],
        ["A question outside your scope", "say where your work stops, and connect it to what you do know"]],
       widths=["30%", "70%"])}

<h2>6. Question bank with model answers</h2>
{qa_bank(QA)}

<h2>7. Where to read more</h2>
<ul class="refs">
<li>Gopen G. D., Swan J. A. (1990). The science of scientific writing. <i>American Scientist</i>.</li>
<li>Rougier N. P., Droettboom M., Bourne P. E. (2014). Ten simple rules for better figures. <i>PLoS Computational Biology</i>.</li>
<li>Weissgerber T. L. et al. (2015). Beyond bar and line graphs: time for a new data presentation paradigm. <i>PLoS Biology</i>.</li>
<li>Nichols T. E. et al. (2017). Best practices in data analysis and sharing in neuroimaging using MRI (COBIDAS). <i>Nature Neuroscience</i>.</li>
<li>Pernet C. et al. (2020). Issues and recommendations from the OHBM COBIDAS MEEG committee. <i>Nature Neuroscience</i>.</li>
<li>Yücel M. A. et al. (2021). Best practices for fNIRS publications. <i>Neurophotonics</i>.</li>
<li>Collins G. S. et al. (2015). Transparent reporting of a multivariable prediction model (TRIPOD). <i>BMJ</i>.</li>
<li>Noble W. S. (2017). Ten simple rules for writing a response to reviewers. <i>PLoS Computational Biology</i>.</li>
<li>Brand A. et al. (2015). Beyond authorship: attribution, contribution, collaboration, and credit (CRediT). <i>Learned Publishing</i>.</li>
</ul>
"""

D.render(body, "Lecture G Reference: General Knowledge")
