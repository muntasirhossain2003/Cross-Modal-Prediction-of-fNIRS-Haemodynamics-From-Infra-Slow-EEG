# NeuroVascNet — complete presentation guide

**Written for someone who has never seen this project.** No neuroscience or
machine-learning background assumed. Read Parts 1–3 once and you will understand
every number in the deck.

- **Deck:** `NeuroVascNet_Presentation.tex` — 33 content slides + title + outline
- **Figures:** `figures/` — 7 figures, all generated from real result files
- **Companion:** this file

## Where to start

| If you are… | Read |
|---|---|
| **New to the project** | Parts 1–3 (about 15 min). Then everything makes sense |
| **Presenting it** | Parts 3, 4, 5, then Part 10 for timing |
| **Being questioned on it** | Parts 5, 7, 8 |
| **Just need a number** | Part 9 |
| **Stuck on a word** | Part 11 (glossary) |

**Contents**

0. How to build the slides
1. The science, from zero
2. Every term in the deck, explained
3. What the project actually claimed
4. Slide by slide
5. Why claims 2 and 3 fail ← *the part that gets questioned*
6. The leakage story
7. Corrections in this deck
8. Hard questions, prepared answers
9. Where every number comes from
10. Delivering the talk
11. Glossary

---

# PART 0 — How to build the slides

```bash
python make_presentation_figures.py       # creates figures/*.pdf and *.png
pdflatex NeuroVascNet_Presentation.tex    # run TWICE (second pass fills the contents page)
```

`pdflatex` is **not installed on this machine.** Easiest route:

1. Go to overleaf.com → New Project → Upload Project
2. Upload `NeuroVascNet_Presentation.tex` **and the whole `figures/` folder**
3. Press Recompile twice

If a figure is missing, the PDF still builds but shows a grey box where the
image should be. That means the `figures/` folder wasn't uploaded.

---

# PART 1 — The science, from zero

## 1.1 The two kinds of brain signal

When part of your brain becomes active, two separate things happen:

**Electrical (fast, milliseconds).** Neurons fire. We measure this with **EEG** —
electrodes on the scalp. EEG is cheap, portable, and you can wear it walking
around. Its weakness is that it can't tell you *where* in the brain the activity
came from — the skull blurs everything.

**Blood flow (slow, seconds).** Active neurons need oxygen, so blood flow to that
region increases a few seconds later. This is called the **haemodynamic
response**. Two machines measure it:

| | What it is | Cost / practicality |
|---|---|---|
| **fNIRS** | Shines infrared light through the skull, measures how much comes back | Moderate; a cap you wear |
| **fMRI** | A large magnet imaging blood oxygenation throughout the brain | Very expensive, immobile, ~£500/hour |

Blood-flow methods tell you **where** precisely. EEG tells you **when**
precisely. Neither gives both.

## 1.2 What HbO and HbR mean (you need this for the clinical result)

Blood carries haemoglobin, the molecule that holds oxygen.

- **HbO** = oxygenated haemoglobin (carrying oxygen)
- **HbR** = deoxygenated haemoglobin (oxygen delivered)

**The key fact:** when a brain region genuinely activates, fresh oxygen-rich
blood floods in. HbO goes **UP** and HbR goes **DOWN**. They move in *opposite*
directions.

So if you measure HbO and HbR and find they move **together** (both up, both
down), you are **not** looking at brain activity. You are looking at something
affecting the whole head at once — heartbeat, breathing, blood pressure, blood
flow in the scalp. This is called **systemic** signal, and it is noise for us.

> This single fact explains a large part of the project. Our original dataset had
> HbO and HbR correlated at **+0.59** — moving strongly together. We had been
> trying to predict noise.

## 1.3 What we were trying to build

> Train a model on people who were recorded with **both** EEG and a blood-flow
> machine at the same time. Then, at test time, give it **EEG only** and have it
> predict what the blood-flow signal would have been.

If it worked, a clinic with a £200 EEG cap could get information that today needs
a £2M MRI scanner.

We also wanted the model to **know when it doesn't know** — to say "I'm not
confident, don't trust this one" rather than guessing. That's the *reliability /
abstention layer*.

## 1.4 The one rule that matters most: cross-subject

There are two ways to test a model like this:

**Within-subject (easy).** Train on some of Alice's data, test on the rest of
Alice's data. Works better, but useless — you'd need to put every new patient in
an MRI scanner first, which defeats the entire purpose.

**Cross-subject (hard, and the only meaningful test).** Train on Alice, Bob and
Carol. Test on **Dave, who the model has never seen.** This is what a real clinic
needs, and it is what this project insisted on throughout.

---

# PART 2 — Every term in the deck, explained

## 2.1 Correlation (*r*)

A number from −1 to +1 measuring how well a prediction matches reality.

| *r* | Meaning |
|---|---|
| 1.0 | perfect prediction |
| 0.0 | no relationship at all — you might as well guess |
| −1.0 | perfectly backwards |

**Our headline result is r ≈ 0.04.** That is small, and the deck says so plainly.
Parts 4 and 5 explain why it is still a real, interesting result.

## 2.2 p-value and the permutation null

*r* = 0.04 is small — so how do we know it isn't just luck?

We **deliberately break** the data: shuffle it so each EEG recording is paired
with the **wrong** person's blood-flow signal. Any correlation now is pure
chance. Do that 1000 times and you learn what "luck" looks like.

The **p-value** is the fraction of those 1000 shuffles that scored as well as our
real result.

- **p = 0.001** → only 1 shuffle in 1000 did as well. Real.
- **p = 0.579** → 58% of random shuffles did as well. Meaningless.

`p = 0.001` is the smallest value possible with 1000 shuffles — it means "none of
them beat us."

> **Technical contribution.** Doing this naively is slow. We found that shuffling
> only *reorders rows*, so all the pairwise comparisons can be computed **once**
> as a single matrix multiplication, and each shuffle becomes a cheap lookup.
> **559× faster**, and verified identical to 10 decimal places. This matters
> because it makes the honest test cheap enough that other groups will actually
> run it.

## 2.3 The constant baseline — the most important idea in the deck

**Read this twice. Almost every conclusion depends on it.**

Suppose you must predict people's height, and you know nothing about anyone. Best
strategy: guess the average height every time. You'd score surprisingly well —
not because you understand height, but because people cluster near the average.

The same trap exists here. Each person's blood-flow signal has a characteristic
shape. A model can learn *"this recording is from Alice, and Alice's signal
usually looks like this"* — scoring well while knowing **nothing** about the
brain.

So we always run a **constant predictor**: a fake model that **ignores the EEG
completely** and always outputs the average. Whatever it scores is the amount you
get for free, with no brain understanding at all.

**Our finding:**

| | *r* |
|---|---|
| Our real model | 0.072 |
| **Fake model that ignores EEG** | **0.076** |

**The fake model wins.** Our model's raw score was not measuring neurovascular
coupling — it was mostly measuring *"which person is this"*.

## 2.4 Demeaning — the fix

If the problem is that each person has a characteristic average, **subtract each
person's average** before scoring. Now the constant predictor scores **exactly
zero** — it has nothing left to exploit.

Whatever survives is genuinely from the EEG. **After demeaning our model scores
0.038–0.053, and the fake model scores 0.** That's the honest number, and it is
what the deck reports.

## 2.5 The ceiling — why a small number can still be a good number

**If a measurement can't reproduce itself, nothing can predict it.**

Test: take one person's recordings, split them randomly into two halves, average
each half, and compare the two halves. If the signal is real, the halves should
look the same. This is **split-half reliability**.

- reliability 1.0 = perfectly repeatable
- reliability 0.0 = pure noise
- reliability **negative** = the two halves *disagree* — something is badly wrong

Maths says a model can never beat **√reliability**. This is the **ceiling**.

| Dataset | Reliability | Ceiling |
|---|---|---|
| HEFMI (original fNIRS) | 0.207 | 0.455 |
| TU Berlin hBCI (our main set) | 0.440 | 0.663 |
| fMRI, before our fix | **−0.639** | **0** |
| fMRI, after our fix | +0.638 | 0.799 |

That **−0.639** is the most dramatic number in the project. A *negative*
reliability means the two halves of the same recording pointed in opposite
directions — the signal was misaligned in time. **No model could ever have
worked on it.** We were not failing at modelling; we were predicting a
scrambled target.

## 2.6 Leakage — the mistake that invalidates everything

If the same person appears in both the training set and the test set, the model
has effectively seen the answers. Results become meaningless.

This sounds easy to avoid, but see Part 6 — it is genuinely easy to get wrong,
and we got it wrong once ourselves.

---

# PART 3 — What the project actually claimed

This is the part most people get wrong, so be clear about it.

**We did NOT promise "we will predict blood flow accurately."** The project
blueprint made **five specific structural claims** about how the model should
behave. Those five claims are the target. The deck tests each one.

| # | The claim, in plain English | Result |
|---|---|---|
| 1 | Training on *both* blood-flow types beats training on one | ✅ **holds** |
| 2 | It still works on data it has never seen | ❌ **fails** |
| 3 | It learns the real time-lag between electrical and blood signals | ❌ **fails** |
| 4 | The trained encoder helps other EEG tasks | ⚠️ **partial** |
| 5 | It knows when it is unreliable | ✅ **holds** |

**2 hold, 1 partial, 2 fail.**

> **A previous internal summary said "all five achieved, only claim 4 partial."
> That was wrong.** Part 5 explains exactly how the two failing claims came to be
> mis-scored — and the explanation is genuinely interesting, not just an error.

---

# PART 4 — Slide by slide

Slides 1–2 are the title and the contents page.

### Slide 3 — Executive summary
The whole story on one slide. If someone only sees one slide, this is it.
It gives the scorecard (2 hold / 1 partial / 2 fail), the clinical result, and
the honest statement that the correlation is small.

### Slide 4 — What we set out to do
The goal, and the three self-imposed rules: cross-subject, always report a
no-EEG baseline, freeze the rules before looking at results.

**Say:** "These three rules are unusual, and they're why our numbers are smaller
than published ones. Same data, relaxed rules, we'd report 0.084 instead of
0.038."

### Slide 5 — The data
5 datasets, 126 people total. Two things to stress:

- The fNIRS files give us **raw light intensities**, not haemoglobin. We convert
  them ourselves using standard physics (Beer–Lambert law). Not doing this was an
  early mistake.
- One fMRI dataset has **scanner trigger pulses** marking each brain image. Using
  them to line up EEG and fMRI exactly is what flipped that target from −0.639 to
  +0.638.

### Slide 6 — How to read every number
Ceiling, constant baseline, demeaning, permutation null — the four ideas from
Part 2. **Don't skip it.** Everything after it is uninterpretable without it.

### Slide 7 — The five claims (figure `fig6`)
Four panels, one per claim (claim 1 has its own slide). The slide carries a
"How to read this" box. Walk through it left→right, top→bottom.

### Slide 8 — Claim 1 ✅ HOLDS
Training on both blood-flow types beats one: **+0.0113**, and crucially the
confidence interval `[0.0049, 0.0176]` **doesn't include zero**, and the result
came out the same direction on **3 out of 3** random seeds.

**Say:** "Small, but it's a paired test — everything else held identical — and it
never flipped sign. At this effect size that's the strongest evidence available."

### Slide 9 — Claim 2 ❌ FAILS
| Test | *r* | *p* |
|---|---|---|
| New **task**, same people | 0.058 | 0.001 |
| New **people** (different study) | 0.009 | 0.223 |
| Reverse direction | 0.003 | 0.355 |

**The catch, and you must say it:** the first row keeps **the same 29 people**.
It changes what they were *doing*, not *who they are*. So it's not a real
generalisation test. On the only test with genuinely new people, the result is
**zero**.

### Slide 10 — Claim 3 ❌ FAILS
The delays look right (2–8 seconds, which is physiologically correct) but this
proves nothing — see Part 5. And every subject got essentially the **same**
delay (spread of 0.011 seconds), when real people genuinely differ.

### Slide 11 — Claim 4 ⚠️ PARTIAL
| Encoder | Accuracy |
|---|---|
| Ours (blood-flow supervised) | 0.6187 |
| Control: trained on scrambled targets | 0.5961 |
| Control: untrained network | 0.5942 |
| **Simple textbook features** | **0.6216** ← wins |

**Good news:** we beat both controls, so the model *is* learning something real
from the blood-flow data.
**Bad news:** we lose to the simplest possible method. A reviewer will ask "why
use your model at all?" and today we have no answer.

**This is the most valuable claim to fix**, because it doesn't need accurate
prediction — only a better representation.

### Slide 12 — Claim 5 ✅ HOLDS
The model's confidence correlates with its actual error at **0.326**, and
discarding low-confidence predictions improves accuracy on the rest by **+0.011**.

**Say:** "Given the prediction is weak, a working 'I don't know' signal is what
makes this safe instead of just wrong. It's the most deployable piece."

### Slide 13 — Scorecard and correction
States openly that an earlier summary claimed all five passed, and why that was
wrong. **Present this confidently** — volunteering it is far stronger than being
caught.

### Slide 14 — WHY claim 3 fails
Shows the two lines of code that constrain the delay, and the table proving the
answer follows our chosen range rather than biology.

**Say:** "The delay module was supposed to learn the lag between electrical and
blood signals. It reported 2–8 seconds, which is biologically correct — but the
code *forces* the output into 2–8 seconds. It can't produce anything else. And
the teaching signal is clipped to the same range. When we widened the allowed
range to 0–20 seconds, the answer moved to 8 seconds. A real biological delay
wouldn't move. So the module never measured anything."

**If asked "how do you know it's not real?":** point at the third row of the
table — 8.05 s. The physiology didn't change between runs; only our constraint
did, and the answer followed the constraint.

### Slide 15 — WHY claim 2 fails
Shows that the external dataset is the *best* one we own and still gives zero,
which rules out the two obvious excuses.

**Say:** "When we test on a completely different study, the score drops to zero.
The natural response is 'that dataset must be worse' — but it isn't: it has the
highest reliability of any fNIRS data we have, better than what we trained on.
The second response is 'the sensors must be arranged differently' — they aren't,
they're matched by design. So it isn't the data, it's the model. What it learned
belongs to those particular 29 people."

**The killer detail:** the test that *appears* to pass reuses the same 29 people.
Change the task, it works. Change the people, it collapses.

### Slide 16 — We were predicting a broken target
Four problems: (1) HbO/HbR moving together = systemic noise; (2) the standard
cleanup removes 83% of the real signal too; (3) reliability only 0.207;
(4) a 2600× unit error between datasets.

### Slide 17 — The ceiling (figure `fig1`)
Left: does each target reproduce itself. Right: the resulting hard limit.
Point at the negative bar and explain it — that's the clearest story in the deck.

### Slide 18 — Reliability varies by person (figure `fig5`)
Each dot is one person. Many sit at or **below zero** — for those people the
blood-flow signal isn't measurable at all, so nothing can predict it. Averaging
them in makes a broken target look merely mediocre.

### Slide 19 — What fixed the target
Switching datasets, turning off the harmful cleanup, and the scanner-trigger
alignment. Best configuration: pooled tasks, **r = 0.038, p = 0.001**.

### Slide 20 — The result that settles it (figure `fig3`)
Our score *appears* to improve as we use fewer training people (0.072 → 0.243).
Looks great. But the no-EEG baseline rises **faster** (0.076 → 0.309).

**Say:** "The gain over the baseline is negative at every single setting. Anyone
not running that baseline would have published 0.24 as a success."

### Slide 21 — Every dataset (figure `fig7`)
Left: achieved vs ceiling — we reach only 1–12% everywhere.
Right: gain over the no-EEG baseline — positive in only 4 of 9, and of those four,
one is a rejected dataset, one is a control region where a positive result is a
*warning*, and one is negligible. **Exactly one legitimate positive.**

### Slide 22 — What we ruled out
Eleven things tested and eliminated before concluding the effect is simply small.
This is what makes it a *finding* rather than a *failure*.

### Slides 23–24 — The clinical result (figure `fig4`)
**The strongest positive result in the project.**

17 healthy people vs 20 patients who had a brain haemorrhage, same protocol.
Patients' HbO/HbR is **0.810** vs **0.645** in controls, **p = 0.00085** — i.e.
patients' blood-flow signals are much more systemic and less brain-driven,
consistent with damaged neurovascular coupling.

**Why this stands up:** we also checked whether patients simply moved more or had
weaker signals — four such checks, **all non-significant**. So the difference
isn't an artefact of how the data was collected.

**Say the nulls too:** three of the four brain measures showed **no** difference
(p = 0.097, 0.749, 0.659). Only HbO/HbR did. They're on the slide.

**Best part:** this needs **no accurate prediction**. It's measured directly.

### Slide 25 — The protocol contribution
Removing shortcuts one at a time takes 0.084 → 0.038. **A ~2× inflation from
evaluation method alone**, with no change of model or data.

### Slide 26 — The data-quality audit
Every recording checked; every subject has a saved row and a pass/fail with a
documented reason; **rules frozen before looking at results**
(`RULESET_VERSION = 1.0.0-frozen-2026-07-25`).

**Say:** "Choosing thresholds after seeing results is the reproducibility problem
the field is criticised for. Freezing them costs nothing and anyone can check."

### Slide 27 — The leakage trap (Maisha Rahman)
**Say:** "In the clinical dataset every person's files are named `1_epo.mat`,
`2_epo.mat` inside their own folder. So `1_epo.mat` exists 37 times, belonging to
37 different people. And `Normals/subject1` and `Patients/subject1` are two
different human beings. If we'd split the data by filename, 7 out of 7 validation
people would have been in the training set too — every clinical number would have
been worthless. Maisha caught this and built a crosswalk giving each real person
a unique ID. Verified: zero overlap."

**Add:** "This wasn't hypothetical. We'd already made the same class of mistake —
an ID mix-up once made 23 people look like 46."

### Slide 28 — The second defect the same check exposed
**Say:** "Once people were identified correctly, we checked *which* people ended
up where. The validation set was 7 patients and 0 healthy controls. The IDs are
`Normals_subject1`, `Patients_subject3` — sorted alphabetically, all the healthy
ones come first, so taking the last 20% took only patients. No leakage, but
completely confounded: we'd have been training on healthy brains and validating
only on injured ones. Fixed by holding out 20% within each group, and declared as
an amendment rather than quietly changed — because changing a split after seeing
results is exactly what our own rules forbid."

### Slide 29 — Concerns we state ourselves
Seven weaknesses, listed by us.

**Say:** "These are the things a reviewer would attack, so we're raising them
first." Then read two or three — the small effect, losing to the no-EEG baseline
on raw score, and the retracted anatomical result — and move on.

**Do not apologise while presenting this slide.** Volunteering weaknesses reads
as confidence; hedging reads as doubt. The retracted result is the strongest item
here: we found it ourselves at n=6, re-tested at n=24, and withdrew it.

### Slides 30–32 — Anticipated questions
Eleven likely questions with prepared answers, three slides.

**In the talk:** say "we've prepared answers to the obvious questions" and move
on — don't read them. They exist so that during discussion you can turn to the
relevant slide instead of improvising.

**Have these three ready without looking:** why it's a contribution if only 2 of
5 claims hold; why the no-EEG baseline beating the model is the *point*; why
r = 0.04 is meaningful against a measured ceiling. Full wording in Part 8.

### Slide 33 — What this work delivers
Five contributions: (1) five claims tested under frozen rules with the failures
diagnosed; (2) a *measured* ceiling rather than an assumed one; (3) a protocol
that removes a ~2× inflation, plus the 559× faster exact test; (4) a frozen,
auditable data-quality pipeline; (5) the clinical finding.

**Say:** "The result isn't 'we predicted blood flow well'. It's that we can say
precisely how much is predictable, why the usual numbers are inflated, and we
have one clinical finding that stands on its own."

### Slide 34 — Status and next steps
**Say:** "Everything on the left is done. The most valuable next step is claim 4,
because it's the only one that doesn't need accurate prediction — just a
representation that beats textbook features. Second is removing the clamp on the
delay module so it's free to be wrong; only then is it a measurement."

### Slide 35 — Contributions
Who did what. Muntasir: pipeline, models, evaluation protocol, audit, clinical
analysis. Maisha Rahman: the biological-subject crosswalk and the split-leakage
identification.

Closes by noting every figure is generated from the audited tables — none are
illustrative — and every slide names its source file.

---

# PART 5 — Why claims 2 and 3 fail (learn these two)

If your supervisor asks one hard question, it will be this. "The script says so"
is not an answer.

## 5.1 Claim 3 — the delay question was rigged without anyone noticing

**Background.** Blood flow lags electrical activity by roughly 2–8 seconds. We
built a module to *learn* that lag per person. If it recovers sensible,
person-specific lags, the model has learned real physiology.

**Reported result:** delays came out at 2–8 seconds. Looks like a pass.

**The problem.** Two lines of code:

```python
DELAY_MIN, DELAY_MAX = 2.0, 8.0

delay   = DELAY_MIN + (DELAY_MAX - DELAY_MIN) * sigmoid(...)   # line 96
tgt_lag = empirical_lag(...).clamp(DELAY_MIN, DELAY_MAX)       # line 216
```

Line 96 **mathematically forces** the output between 2 and 8. The model *cannot*
produce 1 second or 15 seconds — the formula won't allow it.

Line 216 does the same to the *teaching signal*: whatever value we train it
toward is also clipped into 2–8.

**So:** we forced the answer into the "correct" range, trained it toward a target
we'd also forced into that range, then announced that the answer was in the
correct range. **That's circular.** It's like grading an exam where every
possible answer is marked correct.

**The proof.** Re-run allowing a different range:

| Range we allowed | Delay it "found" | Position within the range |
|---|---|---|
| 2–8 s | 3.90 s | 32% |
| 0–12 s | 4.50 s | 37% |
| 0–20 s | 8.05 s | 40% |

A real biological delay is a **fixed number of seconds**. It should read ~4 s in
all three rows regardless of what we allow. Instead **the seconds double** while
the *position within the range* barely moves. The model is reporting the range we
chose, not the biology.

**Why it never learned per-person differences.** The teaching signal is computed
by comparing the model's *own* prediction to reality — but that prediction is
only r ≈ 0.05, essentially noise. A lag measured from noise is noise; clipped and
averaged, it collapses to one constant for everybody. Hence the 0.011 s spread.
The code comment on line 193 says this teaching signal was *added* to fix an
earlier collapse. It collapsed anyway.

> **This is worth publishing on its own.** *Any* delay module built this way will
> produce "physiologically plausible" delays automatically and look successful.
> Other groups almost certainly have this in their code right now.

**Say it in one sentence:** *"The delay was constrained to the biologically
correct range by the code itself, so getting a biologically correct answer proved
nothing — and when we widened the constraint, the answer moved with it."*

## 5.2 Claim 2 — what the model learned doesn't apply to new people

**The test.** Train on one study, then evaluate on a completely different study
with different people. Score drops from 0.073 to **−0.001** (p = 0.58) — nothing.

**The obvious excuses, and why both are wrong:**

| Excuse | Why it fails |
|---|---|
| "The new dataset is poor quality" | It has the **highest reliability of any fNIRS set we have (0.645)** — *better* than the 0.440 set we trained on |
| "The sensors are arranged differently" | They're **matched**: 30 channels / 72 targets on both sides, by design of the comparison |

So it isn't the data. It's the model.

**And the arm that looked like a pass isn't one.** The "cross-task" test scoring
0.058 uses **the same 29 people** in both halves — the loader just picks
different task-runs from the same folders. It changes what they were doing, not
who they were.

**The conclusion:** when the *people* stay the same, the model transfers fine.
When the *people* change, it collapses. So what it learned is specific to that
group of individuals — not a general physiological rule.

**Say it in one sentence:** *"On the cleanest dataset we own, with matched
sensors, a completely new group of people gives exactly zero — so what the model
learned belongs to those particular 29 people, not to human physiology."*

> This is arguably the single most important negative result in the project: the
> subject-independent signal we set out to find isn't merely small — it doesn't
> survive a change of cohort at all.

---

# PART 6 — The leakage story

## 6.1 Maisha Rahman's finding

In the clinical dataset, every person's file is named `1_epo.mat`, `2_epo.mat`
and so on **inside their own folder**. So:

- `1_epo.mat` appears **37 times**
- belonging to **37 completely different people**

And `Normals/subject1` and `Patients/subject1` are **two different human
beings** who happen to share a folder name.

**If we had split the data by filename**, 7 out of 7 validation people would have
appeared in **both** training and testing. The entire holdout leaks, and every
clinical number becomes worthless.

**Fixed** with a verified crosswalk giving each real person a unique ID, splitting
on *person* before anything else. Verified: **zero overlap**.

**This is not hypothetical — we had already made this exact mistake.** An ID
mix-up once made 23 people look like 46, inflating a result from 0.0427 to
0.0454. That's why the check was worth running.

## 6.2 The second defect the same check found

Once people were correctly identified, we checked *which* people ended up where:

| | |
|---|---|
| Training | 17 healthy + 13 patients |
| Validation | **7 patients, 0 healthy** |

**Cause:** IDs look like `Normals_subject1` / `Patients_subject3`. Sorted
alphabetically, every `Normals_` comes before every `Patients_`, so taking the
last 20% took **only patients**.

**Why it matters:** the split had no leakage, but it was completely confounded —
we trained mostly on healthy brains and validated only on injured ones, so no
validation number could be separated from "is this a patient?".

**Fixed** by holding out ~20% *within each group*, declared openly as a protocol
amendment rather than quietly changed — because changing a split after seeing
results is exactly what our own frozen-rules policy forbids.

---

# PART 7 — Corrections in this deck

Every one is stated openly on a slide. Volunteering them is far stronger than
being caught.

| Previously said | Actually |
|---|---|
| "All 5 claims achieved, claim 4 partial" | 2 hold, 1 partial, **2 fail** |
| Claim 2 passes with a 34% drop | That was the same-people test. New people: 102% drop, p = 0.58 |
| Claim 3 passes (delays 2–8 s) | The range is hard-coded; the script itself prints `DEGENERATE: delay collapsed to a constant` |
| r = 0.084 | That was one lucky split. Done properly: **0.038–0.053** |
| Visual/motor brain regions behave differently (n=6) | At n=24 both identical. **Retracted** |
| Clinical: strength, lag and reliability all differ | **Only HbO/HbR** differs. Other three: p = 0.097, 0.749, 0.659 |
| Reliability 0.90 | That was a *different* metric (SCI). Reliability is 0.440 |

**Both failing-claim verdicts are printed by our own analysis script.** They were
simply never carried into the summary.

---

# PART 8 — Hard questions, prepared answers

**"Only 2 of 5 claims hold. What's the contribution?"**
All five were *tested* under rules frozen in advance, and the two failures are
*diagnosed*, not hidden. Knowing which component fails and why is more useful to
other groups than untested claims that appear to pass.

**"A model that ignores EEG beats yours."**
On the raw score, yes — and that's the point we're making. Once each person's
average is removed, the fake model scores exactly 0 and ours retains 0.038–0.053
at p = 0.001. Without that comparison we'd have published 0.24 as a success.

**"r = 0.04 is basically nothing."**
It's 1–12% of a ceiling we *measured* rather than assumed. The finding is that
the person-independent signal is small — which tells the field to stop tuning
architectures and fix the targets.

**"Why did you change datasets halfway through?"**
The audit showed the first target was dominated by systemic noise (HbO/HbR
positive where physiology demands negative) with reliability 0.207. Both are
reported; the rules were frozen before the switch.

**"Could the patient result just be the injury changing the optics?"**
We can't fully exclude it. We tested movement, signal strength (both types) and
trial count — all non-significant — which removes the usual confounds, but a
haemorrhage does change the skull's optical properties. It's listed as a
limitation, with lesion-side analysis as the follow-up.

**"Only 1 of 4 clinical measures is significant — cherry-picking?"**
No. All four were declared primary in advance and **all four are reported**. The
significant one survives correction for testing four things. The three nulls are
on the slide.

**"Can anyone reproduce this?"**
`NeuroVascNet_v5.ipynb` is fully self-contained — no other notebook, no project
files. Set one path variable and run top to bottom.

**"What would you do next?"**
Claim 4. It's the only claim that doesn't require accurate prediction — just a
representation that beats textbook features. And remove the clamp from the delay
module so it's *free to be wrong*; only then is it a measurement.

---

# PART 9 — Where every number comes from

All files are in `qc_outputs/tables/`.

| Topic | File |
|---|---|
| Claims 2, 3, 5 | `blueprint_claims.csv` |
| Claims 3 and 4, detail | `claims_3_4.csv` |
| Claim 1, multi-seed checks | `confirmation.csv` |
| Transfer between tasks/datasets | `cross_analyses.csv` |
| Master results, every dataset | `final_evaluation.csv` |
| Reliability and ceilings | `v4_target_quality.csv` |
| Reliability per person | `reliability_per_subject.csv` |
| Protocol ablation | `demeaned_eval.csv` |
| Training-size sweep | `crosssubject_sweep.csv` |
| Clinical | `clinical_per_subject.csv`, `clinical_group_comparison.csv` |

`make_presentation_figures.py` builds every figure from these files. **No figure
is illustrative or hand-drawn** — if a file were missing, the script skips that
figure and says so rather than inventing data.

---

# PART 10 — Delivering the talk

## 10.1 Timing

**Full version — 35 to 40 minutes**

| Section | Slides | Minutes |
|---|---|---|
| Executive summary | 3 | 2 |
| Goal, data, how to read numbers | 4–6 | 5 |
| The five claims | 7–13 | 10 |
| Why 2 and 3 fail | 14–15 | 5 |
| Why prediction is small | 16–22 | 7 |
| Clinical result | 23–24 | 4 |
| Methodology and leakage | 25–28 | 4 |
| Concerns, delivers, next steps | 29, 33–35 | 3 |

**Short version — 15 minutes.** Show slides **3, 6, 7, 14, 15, 20, 23, 33**.
That's the summary, how to read numbers, the scorecard, both failure mechanisms,
the baseline result, the clinical finding, and the contributions. Skip everything
else; it's there for questions.

**Five minutes.** Slide 3, then slide 23. The scorecard and the clinical result.

## 10.2 The three things that must land

If the audience remembers nothing else:

1. **A model that ignores EEG entirely scored higher than ours** (0.076 vs
   0.072). That's why every number in this project is reported next to a
   baseline, and why our honest figure is 0.038 rather than 0.084.
2. **The delay module was constrained to the right answer by its own code**, so
   getting the right answer proved nothing — and widening the constraint moved
   the answer.
3. **The clinical finding stands on its own** (p = 0.00085) and needs no accurate
   prediction at all.

## 10.3 If you are asked something you don't know

Say so, then say where the answer lives. Every slide names its source file, and
Part 9 maps every topic to a CSV. *"I don't have that in my head — it's in
`final_evaluation.csv`, I can pull it up"* is a perfectly good answer and far
better than guessing.

**Do not invent a number under pressure.** The entire credibility of this deck
rests on every figure tracing back to a file.

## 10.4 Tone

The instinct with a small result is to apologise for it or oversell it. Do
neither.

- **Don't apologise.** The small number is a *measured* bound, and the deck shows
  the measurement. That's a finding.
- **Don't oversell.** Never say "we predicted blood flow from EEG" — say "we
  measured how much of it is predictable across people, and it's small."
- **Volunteer the corrections.** Slide 13 admits an earlier summary was wrong.
  Presenting that confidently is the single strongest credibility move available;
  being caught on it later is the weakest position.

## 10.5 Likely first question from a supervisor

Almost certainly: **"Why do claims 2 and 3 fail?"**

"The script says so" is not an answer. Part 5 has both mechanisms with a
one-sentence version of each. Learn those two sentences.

---

# PART 11 — Glossary

| Term | Meaning |
|---|---|
| **EEG** | Electrodes on the scalp measuring electrical brain activity. Cheap, portable, poor at locating things |
| **fNIRS** | Infrared light through the skull measuring blood oxygenation |
| **fMRI** | Magnet-based scanner imaging blood oxygenation. Expensive and immobile |
| **HbO / HbR** | Oxygen-carrying / oxygen-depleted haemoglobin. Real brain activity moves them in *opposite* directions |
| **Haemodynamic response** | The blood-flow increase following neural activity, a few seconds later |
| **Systemic signal** | Heartbeat/breathing/blood-pressure effects across the whole head — noise for us |
| **Cross-subject** | Testing on people never seen in training. The only meaningful test here |
| **Leakage** | The same person appearing in both training and testing. Invalidates results |
| **r (correlation)** | −1 to +1; how well prediction matches reality |
| **p-value** | Probability of seeing this result by pure chance. Lower is better; 0.001 is our floor |
| **Permutation null** | Deliberately mispairing the data 1000× to see what luck looks like |
| **Constant baseline** | A fake model ignoring EEG entirely, predicting the average. What our model must beat |
| **Demeaning** | Subtracting each person's average, so the fake model scores exactly 0 |
| **Split-half reliability** | Does a measurement reproduce itself? Bounds what any model can achieve |
| **Ceiling** | √reliability — the hard maximum for any model on that target |
| **Beer–Lambert law** | Physics converting raw light readings into HbO/HbR |
| **GSR (global signal regression)** | Standard cleanup for systemic noise. Here it removed 83% of the real signal too |
| **ICH** | Intracerebral haemorrhage — bleeding within the brain. Our patient group |
| **R128 trigger** | A pulse the MRI emits per image, letting us align EEG and fMRI exactly |
| **MC-dropout** | Running the model several times with parts randomly disabled; disagreement = uncertainty |
| **Abstention** | The model declining to answer when unconfident |
| **Frozen ruleset** | Exclusion rules fixed *before* seeing results, so they can't be tuned to flatter us |
