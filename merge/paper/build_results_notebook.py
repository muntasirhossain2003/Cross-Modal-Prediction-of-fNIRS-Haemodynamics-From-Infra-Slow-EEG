"""Writes ../NeuroVascNet_Results.ipynb (execute it afterwards with nbconvert)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("# NeuroVascNet: current results\n\nAll numbers are read from `analysis_p2_p4/results/`. "
            "Rules and history: `FINAL_PLAN.md`, `TEST_LEDGER.md`."))

C.append(code('''import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, Image, display
from scipy import stats

R, FIG = Path("analysis_p2_p4/results"), Path("paper/figures")
J = lambda f: json.loads((R / f).read_text())
COH = {"hbci": "hBCI", "mental_workload": "Mental workload", "finemi": "FineMI"}
ci = lambda a, b: f"[{a:+.3f}, {b:+.3f}]"
def table(rows, cols=None):
    display(HTML(pd.DataFrame(rows, columns=cols).to_html(index=False, escape=False)))
def fig(name, w=620):
    display(Image(filename=str(FIG / f"{name}.png"), width=w))
def signflip(d, n=200000, seed=0):
    d, g = np.asarray(d), np.random.default_rng(seed)
    return (np.sum((g.choice([-1, 1], size=(n, len(d))) * d).mean(1) >= d.mean()) + 1) / (n + 1)
def spearman_p(rho, n):
    return 2 * stats.t.sf(abs(rho) * np.sqrt((n - 2) / (1 - rho ** 2)), n - 2)
'''))

C.append(md("## Scorecard"))
C.append(code('''c1m, c1f = J("claim1_fresh_mental_workload_results.json"), J("claim1_fresh_finemi_results.json")
ext = J("external_claim2_results.json")["hbci_to_mental_workload"]
t2, bp, sm = J("nvc_transfer2_results.json"), J("nvc_bandpower_results.json"), J("nvc_delay_smooth_results.json")
rep = J("downstream_repr_nlab0_results.json")["hbci_MI_left_right (primary)"]["mean_acc"]
rel = J("calib_reliability_results.json")
cal = t2["calibrated | finemi"]
table([
 ["1. Shared fNIRS + fMRI space", f"fNIRS MW {c1m['fnirs']['joint']['r_subject']:.3f}, FineMI {c1f['fnirs']['joint']['r_subject']:.3f}; fMRI {c1m['fmri']['joint']['r_subject']:.3f}", "<b>Holds</b> (shared-space form). fMRI supervision does not improve fNIRS."],
 ["2. External validation", f"Same lab {ext['r_subject']:+.3f} {ci(ext['ci_low'], ext['ci_high'])}; other lab, calibrated {cal['diff']:+.3f} {ci(*cal['ci'])}", "<b>Holds within lab.</b> Cross-lab: frozen fails, calibrated nominal only."],
 ["3. Delay / coupling", f"hBCI {bp['hbci']['frac_beating_null']:.0%} beat null, delay {bp['hbci']['median_delay_s']:+.0f} s (EEG first); delay repeats {sm['hbci | raw']['agree_2s']:.0%} vs {sm['hbci | raw']['chance']:.0%} chance", "<b>Holds in hBCI.</b> Kernel shape repeats in other cohorts; precise delay does not."],
 ["4. Downstream EEG-only task", f"Virtual haemodynamics {rep['V']:.3f} vs PCA {rep['PC']:.3f}", "<b>Refuted.</b>"],
 ["5. Calibrated reliability", "Spearman " + ", ".join(f"{COH[c]} {rel[c]['spearman_score_vs_minus_error']:.2f}" for c in COH), "<b>Holds per person</b>; fails per trial."],
], ["Claim", "Best result", "Verdict"])'''))

C.append(md("## Founding result: the information is infra-slow"))
C.append(code('''mw, fm = J("replication_results.json"), J("replication_finemi_results.json")
ps = J("person_specific_results_runs.json")["slow"]
def row(name, d):
    a = d["R1"]
    return [name, f"{a['slow_lt0.5']['r_subject']:.3f} {ci(*a['slow_lt0.5']['ci'])}", f"{a['bandpower_1-45']['r_subject']:.3f}",
            f"{a['eog_slow_lt0.5']['r_subject']:.3f}", f"{d['R3']['HbO_local']['r_subject']:.3f}", f"{d['R2']['slow_lt0.5']['identification']:.0%}"]
table([["hBCI (FINAL_PLAN T1, T2, T3)", "0.197 (ridge); 0.155 (deep)", "≈0.02", "14% of all-EEG score", "0.07–0.08", f"{ps['identification']:.0%}"],
       row("Mental workload", mw), row("FineMI", fm)],
      ["Cohort", "Slow EEG &lt; 0.5 Hz", "Band power 1–45 Hz", "EOG", "Local HbO", "Person identification"])
fig("fig1_slow_band", 420)'''))

C.append(md("## Claim 1: one EEG space for fNIRS and fMRI"))
C.append(code('''c1h = J("claim1_fresh_hbci_results.json")
rows = []
for n, d in (("Mental workload (primary)", c1m), ("FineMI", c1f), ("hBCI", c1h)):
    f, g = d["fnirs"]["joint"], d["fmri"]["joint"]
    rows.append([n, f"{f['r_subject']:.3f} {ci(f['ci_low'], f['ci_high'])}", f"{g['r_subject']:.3f} {ci(g['ci_low'], g['ci_high'])}",
                 f"{d['fnirs_joint_minus_specific']['ci90'][0]:+.3f}", f"{d['p31_joint_minus_shuffled_ci95']['mean']:+.4f}", "yes" if d["claim1_shared_space_holds"] else "no"])
table(rows, ["fNIRS cohort", "fNIRS r", "fMRI r", "Cost to fNIRS, 90% CI low (margin −0.02)", "Joint − shuffled fMRI", "Shared space holds"])'''))

C.append(md("## Claim 2: external validation"))
C.append(code('''rev = J("external_claim2_results.json")["mental_workload_to_hbci"]
tr = J("nvc_transfer_results.json")
mwc = t2["calibrated | mental_workload"]
print("Same lab")
table([["hBCI → MW, slow EEG", f"{ext['r_subject']:+.3f}", ci(ext["ci_low"], ext["ci_high"]), f"{ext['p_signflip']:.4f}"],
       ["MW → hBCI, slow EEG", f"{rev['r_subject']:+.3f}", ci(rev["ci_low"], rev["ci_high"]), f"{rev['p_signflip']:.4f}"],
       ["hBCI → MW, band-power kernel", f"{tr['hbci_to_mental_workload']['mean_r']:+.3f}", ci(*tr["hbci_to_mental_workload"]["ci"]), "—"],
       ["hBCI → MW, calibrated gain", f"{mwc['diff']:+.3f}", ci(*mwc["ci"]), "—"]], ["Test", "r", "95% CI", "p"])
print("Other lab: hBCI → FineMI, best first")
h = J("harmonised_transfer_results.json")["hbci_to_finemi | primary: window + montage"]
base = [("Slow EEG, " + n, J(f)["hbci_to_finemi"]) for n, f in (("plain", "external_finemi_results.json"), ("detrend", "external_finemi_detrend_results.json"), ("linked mastoids", "external_finemi_lm_results.json"), ("Euclidean alignment", "external_finemi_ea_results.json"), ("hardware emulation", "external_finemi_hw_results.json"))]
b2 = J("nvc_bands_results.json")
ct = J("calibrated_transfer_results.json")["hbci_to_finemi"]
rows = [("Band-power kernel, calibrated gain (P8.3)", cal["diff"], cal["ci"], "nominal"),
        ("Alpha + beta kernel, calibrated gain (P9.3)", b2["calibrated | finemi | alpha+beta"]["diff"], b2["calibrated | finemi | alpha+beta"]["ci"], "fail"),
        ("Slow EEG, calibrated gain (P6.2)", ct["diff"], ct["ci"], "fail"),
        ("Slow EEG, window + montage (P6.1)", h["r_subject"], [h["ci_low"], h["ci_high"]], "fail")]
rows += [(n, v["r_subject"], [v["ci_low"], v["ci_high"]], "fail") for n, v in base]
rows += [("Band-power kernel, frozen (P7.2)", tr["hbci_to_finemi"]["mean_r"], tr["hbci_to_finemi"]["ci"], "fail (negative)"),
         ("CBSI target, frozen (P8.2)", t2["hbci_to_finemi | cbsi"]["mean_r"], t2["hbci_to_finemi | cbsi"]["ci"], "fail (negative)")]
table([[n, f"{m:+.3f}", ci(*c), v] for n, m, c, v in sorted(rows, key=lambda x: -x[1])], ["Form", "Value", "95% CI", "Outcome"])
print("Why: kernels within and between labs")
ld = J("nvc_labdiff_results.json")
table([["FineMI leave-one-person-out", f"{b2['lopo | finemi']['mean_r']:+.3f}", ci(*b2["lopo | finemi"]["ci"])],
       ["FineMI kernel → hBCI", f"{ld['finemi_to_hbci']['mean_r']:+.3f}", ci(*ld["finemi_to_hbci"]["ci"])]] +
      [[f"Broadband power vs global HbO, {COH[c]}", f"{ld['broadband | ' + c]['mean_r']:+.3f}", ci(*ld["broadband | " + c]["ci"])] for c in COH], ["Diagnostic", "r", "95% CI"])
fig("fig4_kernel_transfer", 560)'''))

C.append(md("## Claim 3: coupling and delay"))
C.append(code('''res, kr, kt = J("nvc_residual_results.json"), J("nvc_kernel_results.json"), J("nvc_kernel_resid_results.json")
rows = []
for c, n in COH.items():
    a, b = sm[f"{c} | raw"], sm[f"{c} | task removed"]
    rows.append([n, f"{bp[c]['frac_beating_null']:.0%}", f"{bp[c]['median_delay_s']:+.1f} s", f"{bp[c]['median_peak_r']:.3f}",
                 f"{kr[c]['wilcoxon_p_greater']:.1e}", f"{kt[c]['wilcoxon_p_greater']:.1e}",
                 f"{a['agree_2s']:.0%} vs {a['chance']:.0%} (p {a['p_agree']:.4f})", f"{b['agree_2s']:.0%} vs {b['chance']:.0%} (p {b['p_agree']:.4f})"])
table(rows, ["Cohort", "Beat null", "Median delay", "Peak local r", "Kernel p, raw", "Kernel p, task removed", "Delay repeats, raw", "Delay repeats, task removed"])
pw = J("nvc_power_results.json")
print("hBCI cut to other cohorts' recording length")
table([[k.replace("hbci matched to ", ""), f"{v['agree_2s']:.0%} vs {v['chance']:.0%}", f"{v['p_agree']:.4f}", "yes" if v["reproducible"] else "no"] for k, v in pw.items() if k.startswith("hbci matched")],
      ["Matched to", "Delay repeats", "p", "Reproducible"])
fig("fig2_coupling_lag", 520); fig("fig3_kernel_repro", 640)'''))

C.append(md("## Claim 4: downstream EEG-only task"))
C.append(code('''d = J("downstream_repr_nlab0_results.json")
table([[k, f"{v['mean_acc']['V']:.3f}", f"{v['mean_acc']['PC']:.3f}", f"{v['mean_acc']['Rp']:.3f}", f"{v['chance']:.3f}", "pass" if v["pass"] else "fail"] for k, v in d.items()],
      ["Task", "Virtual haemodynamics", "PCA", "Random projection", "Chance", "Rule"])'''))

C.append(md("## Claim 5 and deployment: two calibration runs"))
C.append(code('''g, gl = J("calibration_results.json"), J("calibration_results_local.json")
loc = J("calib_reliability_local_results.json")
trl = J("trial_reliability_results.json")
rows = []
for c, n in COH.items():
    a, b = g[c]["gains"]["k2_personal_minus_baseline"], gl[c]["gains"]["k2_personal_minus_baseline"]
    rows.append([n, f"{a['mean']:+.3f} {ci(*a['ci'])}", f"{b['mean']:+.3f} {ci(*b['ci'])}",
                 f"{rel[c]['spearman_score_vs_minus_error']:.2f} {ci(*rel[c]['ci95'])}", f"{loc[c]['spearman_score_vs_minus_error']:.2f}", "pass" if trl[c]["passes"] else "fail"])
table(rows, ["Cohort", "Gain, raw HbO", "Gain, local HbO", "Person reliability ρ", "ρ, local", "Trial reliability"])
fig("fig5_deployment_reliability", 640)'''))

C.append(md("## What survives correction for all attempts"))
C.append(code('''from_attempts = {"Claim 1": 8, "Claim 2 same lab": 3, "Claim 2 other lab": 16, "Claim 3": 13, "Claim 5": 4}
p83 = signflip([v["personal_plus_external"] - v["personal"] for v in cal["people"].values()])
items = [("Claim 1", "Shared space predicts fMRI", c1m["fmri"]["joint"]["p_signflip"]),
         ("Claim 2 same lab", "hBCI → MW, slow EEG", ext["p_signflip"]),
         ("Claim 2 other lab", "hBCI → FineMI, calibrated (P8.3)", p83),
         ("Claim 3", "hBCI kernel repeats", kr["hbci"]["wilcoxon_p_greater"]),
         ("Claim 3", "hBCI delay repeats", sm["hbci | raw"]["p_agree"]),
         ("Claim 3", "FineMI kernel, task removed", kt["finemi"]["wilcoxon_p_greater"]),
         ("Claim 3", "MW kernel, task removed", kt["mental_workload"]["wilcoxon_p_greater"]),
         ("Claim 3", "FineMI kernel, raw", kr["finemi"]["wilcoxon_p_greater"])]
items += [("Claim 5", f"Reliability, {COH[c]}", spearman_p(rel[c]["spearman_score_vs_minus_error"], rel[c]["n"])) for c in COH]
rows = []
for fam, name, p in items:
    n = from_attempts[fam]
    a, b = 0.05 / n, 0.05 / (n * 3)
    rows.append([fam, name, f"{p:.1e}", n, f"{a:.4f}", "yes" if p < a else "<b>no</b>", "yes" if p < b else "<b>no</b>"])
table(rows, ["Family", "Result", "p", "Attempts", "α", "Survives", "Survives if cohorts counted too (×3)"])'''))

C.append(md("## Latest round (P10)"))
C.append(code('''f = R / "nvc_broadband_results.json"
if not f.exists():
    print("P10.1 still running: no result yet.")
else:
    d = J("nvc_broadband_results.json")
    table([[k.split(" | ")[0].replace("hbci_to_", "hBCI → "), k.split(" | ")[1], f"{d[k]['mean_r']:+.3f}", ci(*d[k]["ci"]), f"{d[k]['frac_beating_null']:.0%}", "pass" if d[k]["pass"] else "fail"]
           for k in d if k.startswith("hbci_to_")], ["Direction", "Kernel", "r", "95% CI", "Beat null", "P7.2 rule"])
    table([[k, f"{v['diff']:+.3f}", ci(*v["ci"]), f"{v.get('signflip_p', v.get('signflip_p_removed')):.4f}"] for k, v in d.items() if k.startswith(("paired", "calibrated"))], ["Test", "Diff", "95% CI", "Sign-flip p"])
f = R / "nvc_snr_results.json"
if not f.exists():
    print("P10.2 still running: no result yet.")
else:
    d = J("nvc_snr_results.json")
    table([[k, f"{v['agree_2s']:.0%} vs {v['chance']:.0%}", f"{v['p_agree']:.4f}", "yes" if v["reproducible"] else "no"] for k, v in d.items() if k.startswith("hbci at")], ["Level", "Delay repeats", "p", "Reproducible"])'''))

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
out = Path(__file__).parent.parent / "NeuroVascNet_Results.ipynb"
nbf.write(nb, out)
print("wrote", out)
