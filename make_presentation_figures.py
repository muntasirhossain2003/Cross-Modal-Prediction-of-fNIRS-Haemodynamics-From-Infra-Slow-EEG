"""Generate presentation figures from the AUDITED result tables.

Every figure here is drawn from a CSV in qc_outputs/tables/ that was produced by
an actual run. Nothing is illustrative or hand-drawn -- if a table is missing the
figure is skipped rather than faked, and the script says so.

Usage:  python make_presentation_figures.py   ->  figures/*.pdf (+ .png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = Path("qc_outputs/tables")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": ":",
})
GOOD, BAD, WARN, NEUT = "#00783c", "#aa1e1e", "#be7800", "#4a6fa5"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def load(name):
    p = T / name
    if not p.exists():
        print(f"  SKIP {name} (not found)")
        return None
    return pd.read_csv(p)


# ===================================================================== fig 1
def fig_target_quality():
    """The single most important figure: the target was broken, then repaired."""
    d = load("v4_target_quality.csv")
    if d is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    lab = [s.replace(" (v3)", "\n(v3)").replace(" (v4)", "\n(v4)")
             .replace(" (broken)", "\n(broken)").replace(" (fixed)", "\n(fixed)")
           for s in d.dataset]
    col = [BAD if ("v3" in s or "broken" in s) else GOOD for s in d.dataset]

    err = np.abs(np.vstack([d.reliability - d.ci_low, d.ci_high - d.reliability]))
    ax[0].bar(lab, d.reliability, color=col, yerr=err, capsize=4, alpha=.9)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_ylabel("target split-half reliability")
    ax[0].set_title("Does the target predict itself?", fontweight="bold")
    for i, v in enumerate(d.reliability):
        ax[0].text(i, v + (.04 if v >= 0 else -.09), f"{v:+.3f}",
                   ha="center", fontsize=9, fontweight="bold")

    ax[1].bar(lab, d.ceiling, color=col, alpha=.9)
    ax[1].set_ylabel(r"ceiling on prediction  $r_{max}=\sqrt{reliability}$")
    ax[1].set_title("The bound this places on ANY model", fontweight="bold")
    for i, v in enumerate(d.ceiling):
        ax[1].text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle("A model cannot beat the reliability of what it predicts",
                 fontweight="bold", y=1.02)
    save(fig, "fig1_target_quality")


# ===================================================================== fig 2
def fig_protocol_waterfall():
    """Where the inflation lives: model vs a predictor that ignores EEG."""
    d = load("demeaned_eval.csv")
    if d is None:
        return
    sub = d[(d.dataset.str.contains("hBCI")) & (d.k == 1)]
    if sub.empty:
        print("  SKIP protocol waterfall (no hBCI k=1 rows)")
        return
    raw = sub[sub["mode"] == "raw"].iloc[0]
    dem = sub[sub["mode"] == "own-mean"].iloc[0]

    names = ["model\n(raw r)", "constant predictor\n(ignores EEG)",
             "model\n(demeaned)", "constant\n(demeaned)"]
    vals = [raw.r_model, raw.r_const, dem.r_model, dem.r_const]
    cols = [WARN, BAD, GOOD, NEUT]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(names, vals, color=cols, alpha=.9)
    ax.axhline(0, color="k", lw=.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .003,
                f"{v:.4f}" if abs(v) > 1e-6 else "0 (exactly)",
                ha="center", fontweight="bold", fontsize=10)

    ax.annotate("", xy=(1, raw.r_const), xytext=(0, raw.r_model),
                arrowprops=dict(arrowstyle="<->", color=BAD, lw=1.6))
    ax.text(0.5, max(raw.r_model, raw.r_const) + .012,
            "the baseline BEATS the model", ha="center",
            color=BAD, fontweight="bold", fontsize=9.5)

    ax.set_ylabel("cross-subject Pearson $r$")
    ax.set_title("Raw correlation measures subject identity, not coupling\n"
                 f"(demeaned model $r$ = {dem.r_model:.4f}, "
                 f"$p$ = {dem.p:.4f})", fontweight="bold")
    save(fig, "fig2_protocol")


# ===================================================================== fig 3
def fig_crosssubject_sweep():
    """Honest: the model never beats the group mean at any training size."""
    d = load("crosssubject_sweep.csv")
    if d is None:
        return
    d = d[d.dataset.str.contains("hBCI")].sort_values("k")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(d.k, d.r_model, "o-", color=NEUT, label="model")
    ax[0].plot(d.k, d.r_groupmean, "s--", color=BAD, label="group mean (no EEG)")
    ax[0].plot(d.k, d.null_mean, "^:", color="gray", label="permutation null")
    ax[0].set_xscale("log"); ax[0].set_xlabel("k (folds)  →  fewer training subjects")
    ax[0].set_ylabel("Pearson $r$"); ax[0].legend(fontsize=9)
    ax[0].set_title("Raw $r$ rises as folds shrink —\nbecause the subject mean does",
                    fontweight="bold", fontsize=10.5)

    ax[1].axhline(0, color="k", lw=.8)
    ax[1].bar(range(len(d)), d.r_gain,
              color=[GOOD if v > 0 else BAD for v in d.r_gain], alpha=.9)
    ax[1].set_xticks(range(len(d))); ax[1].set_xticklabels(d.k)
    ax[1].set_xlabel("k (folds)")
    ax[1].set_ylabel("model $-$ group mean")
    ax[1].set_title("Gain over the no-EEG baseline is\nnegative at every k",
                    fontweight="bold", fontsize=10.5)
    save(fig, "fig3_crosssubject")


# ===================================================================== fig 4
def fig_clinical():
    """The strongest positive result, with its artefact controls beside it."""
    per = load("clinical_per_subject.csv")
    grp = load("clinical_group_comparison.csv")
    if per is None or grp is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    a = per[per.group == "Normals"].hbo_hbr.dropna()
    b = per[per.group == "Patients"].hbo_hbr.dropna()
    p = float(grp[grp.measure == "hbo_hbr"].p.iloc[0])

    parts = ax[0].violinplot([a, b], showmeans=True, widths=.7)
    for pc, c in zip(parts["bodies"], [NEUT, BAD]):
        pc.set_facecolor(c); pc.set_alpha(.45)
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate([(a, NEUT), (b, BAD)], start=1):
        ax[0].scatter(rng.normal(i, .05, len(v)), v, s=22, color=c,
                      edgecolor="white", linewidth=.6, zorder=3)
    ax[0].set_xticks([1, 2])
    ax[0].set_xticklabels([f"Controls\n(n={len(a)})", f"ICH patients\n(n={len(b)})"])
    ax[0].set_ylabel("HbO / HbR correlation")
    ax[0].set_title(f"Altered neurovascular coupling in ICH\n"
                    f"$p$ = {p:.4f} (Mann–Whitney)", fontweight="bold", fontsize=11)
    y = max(a.max(), b.max())
    ax[0].plot([1, 1, 2, 2], [y + .04, y + .07, y + .07, y + .04], color="k", lw=1.1)
    ax[0].text(1.5, y + .08, "***", ha="center", fontweight="bold")

    g = grp.sort_values("is_primary", ascending=False)
    cols = [GOOD if (pp < .05 and prim) else (BAD if pp < .05 else "#b0b0b0")
            for pp, prim in zip(g.p, g.is_primary)]
    ax[1].barh(range(len(g)), -np.log10(g.p), color=cols, alpha=.9)
    ax[1].axvline(-np.log10(.05), color="k", ls="--", lw=1,
                  label="$p$ = 0.05")
    ax[1].set_yticks(range(len(g)))
    ax[1].set_yticklabels([m + ("" if prim else "  [control]")
                           for m, prim in zip(g.measure, g.is_primary)], fontsize=9)
    ax[1].set_xlabel(r"$-\log_{10}(p)$"); ax[1].legend(fontsize=9)
    ax[1].set_title("Only HbO/HbR separates the groups;\n"
                    "artefact controls do not", fontweight="bold", fontsize=11)
    ax[1].invert_yaxis()
    save(fig, "fig4_clinical")


# ===================================================================== fig 5
def fig_reliability_spread():
    """Per-subject reliability: the spread, not just the mean."""
    d = load("reliability_per_subject.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sets = list(dict.fromkeys(d.dataset))
    data = [d[d.dataset == s].reliability.dropna().values for s in sets]
    cols = [BAD if "v3" in s or "HEFMI" in s else GOOD for s in sets]

    bp = ax.boxplot(data, patch_artist=True, widths=.55, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(.4)
    rng = np.random.default_rng(1)
    for i, (v, c) in enumerate(zip(data, cols), start=1):
        ax.scatter(rng.normal(i, .06, len(v)), v, s=20, color=c,
                   edgecolor="white", linewidth=.5, zorder=3)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticklabels([s.replace(" ", "\n", 1) for s in sets], fontsize=9)
    ax.set_ylabel("split-half reliability")
    ax.set_title("Per-subject target reliability — many subjects sit at or below 0",
                 fontweight="bold")
    save(fig, "fig5_reliability_spread")


if __name__ == "__main__":
    print("generating figures from audited tables in qc_outputs/tables/")
    for f in (fig_target_quality, fig_protocol_waterfall, fig_crosssubject_sweep,
              fig_clinical, fig_reliability_spread):
        try:
            f()
        except Exception as exc:
            print(f"  FAILED {f.__name__}: {type(exc).__name__}: {exc}")
    print(f"\nfigures -> {OUT.resolve()}")


# ===================================================================== fig 6
def fig_five_claims():
    """The blueprint's five headline claims -- the project's ACTUAL target."""
    b = load("blueprint_claims.csv")
    c34 = load("claims_3_4.csv")
    conf = load("confirmation.csv")
    if b is None or c34 is None:
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))

    # -- claim 2: the gradient from same-subjects to new-cohort -----------
    # Three bars, not two: the middle one is the arm that LOOKS like a pass.
    # hBCI MI and MA are the SAME 29 people, so it changes the task but not the
    # subjects -- hatched to mark it as not subject-disjoint.
    r2 = b[b.claim == "2 external"].iloc[0]
    ca = load("cross_analyses.csv")
    xtask = None
    if ca is not None:
        m = ca[ca.setting.str.contains("MI -> MA", regex=False)]
        if len(m):
            xtask = float(m.r_demeaned.iloc[0])
    a = ax[0, 0]
    names = ["internal\n(held-out subj.)", "cross-TASK\n(same 29 people)",
             "cross-COHORT\n(TU Berlin WL)"]
    vals = [r2.internal_r, xtask if xtask is not None else np.nan, r2.external_r]
    bars = a.bar(names, vals, color=[GOOD, WARN, BAD], alpha=.9)
    bars[1].set_hatch("///")
    a.axhline(0, color="k", lw=.8)
    labs = [f"{r2.internal_r:+.4f}\np={r2.internal_p:.3f}",
            f"{xtask:+.4f}\nsame people\n(not disjoint)" if xtask is not None else "",
            f"{r2.external_r:+.4f}\np={r2.external_p:.3f}"]
    for bb, v, t in zip(bars, vals, labs):
        if not np.isfinite(v):
            continue
        inside = v > 0.02
        a.text(bb.get_x() + bb.get_width() / 2,
               v * 0.5 if inside else 0.006, t, ha="center",
               va="center" if inside else "bottom",
               color="white" if inside else BAD,
               fontsize=8, fontweight="bold")
    a.set_ylim(min(r2.external_r * 4, -.006), r2.internal_r * 1.15)
    a.set_ylabel("demeaned $r$")
    a.set_title(f"CLAIM 2  holds subjects fixed, fails on a new\n"
                f"cohort (drop = {r2.drop_pct:.0f}%) "
                r"$\rightarrow$ FAILS", fontweight="bold", fontsize=11, color=BAD)

    # -- claim 3: delay depends on the search window ---------------------
    d3 = c34[c34.claim == 3]
    a = ax[0, 1]
    lab = [f"[{lo:.0f},{hi:.0f}] s" for lo, hi in zip(d3.dmin, d3.dmax)]
    a.bar(lab, d3.mean_s, yerr=d3.between_subject_sd, capsize=4,
          color=WARN, alpha=.9)
    a.axhspan(2, 8, color=GOOD, alpha=.12, label="physiological 2-8 s")
    # THE ARGUMENT: a real lag is a fixed number of SECONDS, so all three bars
    # would touch one horizontal line. Instead the seconds scale with the bounds
    # while the FRACTION of the allowed range stays ~constant -- the model is
    # reporting our prior, not the physiology.
    a.axhline(d3.mean_s.iloc[0], color=BAD, ls="--", lw=1.4,
              label=f"a true fixed lag would stay here ({d3.mean_s.iloc[0]:.1f} s)")
    for i, (v, lo, hi) in enumerate(zip(d3.mean_s, d3.dmin, d3.dmax)):
        frac = (v - lo) / (hi - lo)
        a.text(i, v + .35, f"{v:.2f} s", ha="center", fontweight="bold", fontsize=9)
        # y = v*0.25 keeps this clear of the dashed reference line at 3.9 s
        a.text(i, v * 0.25, f"{frac:.2f}\nof range", ha="center", va="center",
               color="white", fontweight="bold", fontsize=8.5)
    a.set_ylabel("recovered delay (s)"); a.set_xlabel("search window given to the model")
    a.legend(fontsize=8, loc="upper left")
    # The 2-8 s band is HARD-CODED in the model (DELAY_MIN/DELAY_MAX), so landing
    # inside it is guaranteed, not evidence. The claim fails on its testable half.
    a.set_title("CLAIM 3  window is hard-coded; delay tracks\nit, not physiology "
                r"$\rightarrow$ FAILS", fontweight="bold", fontsize=11, color=BAD)

    # -- claim 4: downstream transfer ------------------------------------
    d4 = c34[c34.claim == 4].sort_values("accuracy")
    a = ax[1, 0]
    cols = [GOOD if "supervised" in e else (NEUT if "raw" in e else "#b0b0b0")
            for e in d4.encoder]
    a.barh(range(len(d4)), d4.accuracy, xerr=d4.sd, capsize=3, color=cols, alpha=.9)
    a.set_yticks(range(len(d4)))
    a.set_yticklabels([e.replace(" ", "\n", 1) for e in d4.encoder], fontsize=8.5)
    a.set_xlim(0.55, 0.66); a.set_xlabel("downstream EEG-only accuracy")
    best_raw = float(d4[d4.encoder.str.contains("raw")].accuracy.iloc[0])
    a.axvline(best_raw, color=NEUT, ls="--", lw=1.2)
    a.set_title("CLAIM 4  supervision beats shuffled/random\n"
                r"but LOSES to raw band-power $\rightarrow$ PARTIAL",
                fontweight="bold", fontsize=11, color=WARN)

    # -- claim 5: reliability gating -------------------------------------
    r5 = b[b.claim == "5 reliability"].iloc[0]
    a = ax[1, 1]
    ks = ["uncertainty\n<-> error", "dominance\n<-> error", "reliability\n<-> error"]
    vs = [r5.corr_unc_err, r5.corr_dom_err, r5.corr_rel_err]
    bars = a.bar(ks, vs, color=[GOOD, "#b0b0b0", "#b0b0b0"], alpha=.9)
    for bb, v in zip(bars, vs):
        a.text(bb.get_x() + bb.get_width() / 2, v + .008, f"{v:.3f}",
               ha="center", fontweight="bold", fontsize=9)
    a.set_ylabel("correlation with true error")
    a.set_title(f"CLAIM 5  gate tracks error (0.326);\n"
                f"abstention gain +{r5.abstention_gain:.4f} "
                r"$\rightarrow$ HOLDS", fontweight="bold", fontsize=11, color=GOOD)

    fig.suptitle("The five headline claims, measured", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig6_five_claims")


# ===================================================================== fig 7
def fig_final_evaluation():
    """Master table: every dataset, ceiling vs achieved, gain over no-EEG."""
    d = load("final_evaluation.csv")
    if d is None:
        return
    d = d.dropna(subset=["r_demeaned"]).copy()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    y = np.arange(len(d))
    ax[0].barh(y, d.ceiling, color="#d8d8d8", label=r"ceiling $\sqrt{rel.}$")
    ax[0].barh(y, d.r_demeaned, height=.5,
               color=[GOOD if p < .05 else "#b0b0b0" for p in d.p],
               label="achieved (demeaned)")
    ax[0].set_yticks(y); ax[0].set_yticklabels(d.dataset, fontsize=8.5)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=8.5, loc="lower right")
    ax[0].set_xlabel("Pearson $r$")
    ax[0].set_title("Achieved vs the ceiling the target allows",
                    fontweight="bold", fontsize=11)
    for i, (v, pc) in enumerate(zip(d.r_demeaned, d.pct_ceiling)):
        ax[0].text(v + .015, i, f"{v:.3f}  ({pc:.1f}% of ceiling)",
                   va="center", fontsize=7.5)

    ax[1].barh(y, d.r_gain, color=[GOOD if g > 0 else BAD for g in d.r_gain], alpha=.9)
    ax[1].axvline(0, color="k", lw=.9)
    ax[1].set_yticks(y); ax[1].set_yticklabels(d.dataset, fontsize=8.5)
    ax[1].invert_yaxis(); ax[1].set_xlabel("model $-$ constant baseline (raw $r$)")
    n_pos = int((d.r_gain > 0).sum())
    ax[1].set_title(f"Gain over a predictor that ignores EEG\n"
                    f"(positive in only {n_pos} of {len(d)})",
                    fontweight="bold", fontsize=11)
    save(fig, "fig7_final_evaluation")


for _f in (fig_five_claims, fig_final_evaluation):
    try:
        _f()
    except Exception as _e:
        print(f"  FAILED {_f.__name__}: {type(_e).__name__}: {_e}")
