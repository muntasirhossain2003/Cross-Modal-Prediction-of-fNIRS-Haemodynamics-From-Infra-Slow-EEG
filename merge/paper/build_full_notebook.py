"""Writes ../NeuroVascNet_Full.ipynb: plain code cells in the layout of the merge notebook.
Pipeline cells are copied from Neurovascnet_marge_v1.ipynb; every analysis script becomes one plain function."""
import ast
import io
import re
import tokenize
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
M = HERE.parent
A = M / "analysis_p2_p4"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def clean(src, indent=0):
    """Remove comments and the module docstring; indent every line that is not inside a multi-line string."""
    lines = src.split("\n")
    drop, cut, inside = set(), {}, set()
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        drop |= set(range(tree.body[0].lineno, tree.body[0].end_lineno + 1))
    fstack = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            if lines[t.start[0] - 1][:t.start[1]].strip():
                cut[t.start[0]] = t.start[1]
            else:
                drop.add(t.start[0])
        elif t.type == tokenize.STRING and t.end[0] > t.start[0]:
            inside |= set(range(t.start[0] + 1, t.end[0] + 1))
        elif t.type == tokenize.FSTRING_START:
            fstack.append(t.start[0])
        elif t.type == tokenize.FSTRING_END:
            first = fstack.pop()
            if t.end[0] > first:
                inside |= set(range(first + 1, t.end[0] + 1))
    out = []
    for n, l in enumerate(lines, 1):
        if n in drop:
            continue
        if n in cut:
            l = l[:cut[n]].rstrip()
        if n not in inside and l.strip():
            l = " " * indent + l
        if not l.strip() and out and not out[-1].strip():
            continue
        out.append(l)
    return "\n".join(out).strip("\n")


def pipeline(cell):
    return "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))


def replace_block(src, start, end, new):
    """Replace the lines from the first matching `start` to the next matching `end`, keeping the indent of `start`."""
    L = src.split("\n")
    i = next((k for k, l in enumerate(L) if re.search(start, l)), None)
    if i is None:
        return src, False
    j = next(k for k in range(i, len(L)) if re.search(end, L[k]))
    ind = L[i][:len(L[i]) - len(L[i].lstrip())]
    return "\n".join(L[:i] + ([ind + new] if new else []) + L[j + 1:]), True


CUT = {"continuous_lag": "rng = np.random.default_rng(SEED)\nres = {}", "nvc_bandpower": "COHORTS = tuple(", "nvc_residual": "if MODE in (",
       "nvc_transfer2": "CAL_SEEDS = [", "nvc_bands": "rng = np.random.default_rng(SEED)", "calibration": "def run_cohort",
       "nvc_delay_smooth": "rng = np.random.default_rng(1337)", "nvc_mendeley": 'if MODE == "coupling":'}


def script(name):
    s = (A / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    mode = re.search(r'^MODE = sys\.argv\[1\] if len\(sys\.argv\) > 1 else "(\w+)"', s, re.M)
    default = mode.group(1) if mode else "all"
    s = s.replace('ARGS = sys.argv[1:] or ["merge"]', 'ARGS = list(ARGS) or ["merge"]')
    s = s.replace('LC["run_pair"] = hook\nBP["run_pair_bp"] = hook', 'LC["HOOK"]["run_pair"] = hook')
    s = re.sub(r'^MODE = sys\.argv.*\n', "", s, flags=re.M)
    s = re.sub(r'^SHARD = .*sys\.argv.*\n', "", s, flags=re.M)
    s = re.sub(r'^sys\.argv = .*\n', "", s, flags=re.M)
    s = re.sub(r'^os\.environ\.setdefault\("(OMP|OPENBLAS|MKL)_NUM_THREADS".*\n', "", s, flags=re.M)
    s = re.sub(r'^os\.environ\["OMP_NUM_THREADS"\].*\n', "", s, flags=re.M)
    s = s.replace("len(sys.argv) > 1 and name not in sys.argv[1:]", "ARGS and name not in ARGS")
    s = s.replace("Path(__file__).with_name(", "NB_OUT.joinpath(").replace("Path(__file__).parent", "NB_OUT")
    s = s.replace("sys.exit(0)", "return")

    if name == "nvc_kernel":
        s, ok = replace_block(s, r"^if RESID:", r"^exec\(src, ns\)", 'ns = nvc_residual(LIB=True) if RESID else nvc_bandpower(LIB=True)')
        assert ok
    for pat, var in ((r'^\s*src = \(HERE / "(\w+)\.py"\)', r"exec\(src.*,\s*(\w+)\)"), (r'^\s*src_d = \(HERE / "(\w+)\.py"\)', r"exec\(src_d.*,\s*(\w+)\)"),
                     (r'^\s*sd = \(HERE / "(\w+)\.py"\)', r"exec\(sd.*,\s*(\w+)\)")):
        m = re.search(pat, s, re.M)
        if m:
            target = re.search(var, s).group(1)
            s, ok = replace_block(s, pat, var, f"{target} = {m.group(1)}(LIB=True)")
            assert ok

    while re.search(r"nbformat\.read\(", s):
        blk = s[s.index("nbformat.read("):]
        s, ok = replace_block(s, r"nbformat\.read\(", r"os\.chdir\(cwd\)", "ns = globals()" if "ns = {" in blk[:blk.index("os.chdir(cwd)")] else ('nbns = inner["ns"]' if "nbns = inner" in blk[:blk.index("os.chdir(cwd)")] else ""))
        assert ok

    if name == "continuous_lag":
        assert "def hbci_people" in s
        s = re.sub(r'(?<!def )\brun_pair\(', 'HOOK["run_pair"](', s)
        s = s.replace("def hbci_people", 'HOOK = {"run_pair": run_pair}\n\n\ndef hbci_people', 1)
    if name == "nvc_bandpower":
        s = s.replace('ns["run_pair"] = run_pair_bp', 'ns["HOOK"]["run_pair"] = run_pair_bp')
        s = re.sub(r'(?<!def )(?<![\w"])run_pair_bp\(', 'ns["HOOK"]["run_pair"](', s)
    if name == "nvc_transfer":
        s = s.replace('NS["run_pair_bp"] = run_pair_tr', 'inner["HOOK"]["run_pair"] = run_pair_tr')
        s = re.sub(r'^inner\["run_pair"\] = run_pair_tr.*\n', "", s, flags=re.M)
    if name in CUT:
        assert CUT[name] in s, (name, CUT[name])
        s = s.replace(CUT[name], "if LIB:\n    return locals()\n\n" + CUT[name], 1)
    for bad in ("__file__", "sys.argv", "sys.exit", "exec("):
        assert bad not in s, (name, bad)
    body = clean(s, 4)
    return f'def {name}(LIB=False, MODE="{default}", SHARD=(0, 1), ARGS=()):\n{body}'


SECTIONS = [
    ("7. Founding result: infra-slow EEG", [("person_specific", ['person_specific(MODE="runs")']), ("replicate_mw", ["replicate_mw()"]),
                                            ("replicate_finemi", ["replicate_finemi()"])]),
    ("8. Claim 1: shared fNIRS + fMRI space", [("claim1_fresh", [f'run_with_env(claim1_fresh, FNIRS_COHORT="{c}")' for c in ("mental_workload", "finemi", "hbci")])]),
    ("9. Claim 2: external validation", [("external_claim2", ["external_claim2()"]),
                                         ("external_finemi", ["external_finemi()"] + [f'run_with_env(external_finemi, {e})' for e in ('DETREND="1"', 'REREF="lm"', 'EA="1"', 'FINEMI_TAG="_hw"')]),
                                         ("harmonised_transfer", ["harmonised_transfer()"]), ("calibrated_transfer", ["calibrated_transfer()"])]),
    ("10. Continuous-recording loaders and band-power coupling", [("continuous_lag", ["continuous_lag()"]), ("nvc_bandpower", ["nvc_bandpower()"])]),
    ("11. Claim 2: coupling-kernel transfer", [("nvc_transfer", ["nvc_transfer()"]), ("nvc_transfer2", ["nvc_transfer2()", 'run_with_env(nvc_transfer2, CAL_SEEDS="1,2,3,4,5")']),
                                               ("nvc_bands", ["nvc_bands()"]), ("nvc_labdiff", ["nvc_labdiff()"]), ("nvc_broadband", ["nvc_broadband()"])]),
    ("12. Claim 3: coupling and delay", [("nvc_residual", ["nvc_residual()"]), ("nvc_kernel", ["nvc_kernel()", 'run_with_env(nvc_kernel, RESID="1")']), ("nvc_delay_smooth", ["nvc_delay_smooth()"]),
                                         ("nvc_power", ['nvc_power(MODE="durations")', 'nvc_power(MODE="shard")', 'nvc_power(MODE="merge")']),
                                         ("nvc_snr", ['nvc_snr(MODE="calibrate")', 'nvc_snr(MODE="pick")', 'nvc_snr(MODE="shard")', 'nvc_snr(MODE="merge")']), ("nvc_woody", ["nvc_woody()"])]),
    ("12b. Confirmation in a third lab (P12)", [("nvc_mendeley", ['nvc_mendeley(MODE="sync")', 'nvc_mendeley(MODE="coupling")', 'nvc_mendeley(MODE="founding")', 'nvc_mendeley(MODE="transfer")', 'nvc_mendeley(MODE="merge")'])]),
    ("12c. Can the third-lab method see the advantage? (P13)", [("nvc_method_check", ['nvc_method_check(ARGS=("hbci", "mental_workload", "finemi", "mendeley"))', "nvc_method_check()"])]),
    ("13. Claim 4: downstream task", [("downstream_repr", ['run_with_env(downstream_repr, N_LAB="0")'])]),
    ("14. Claim 5: calibration and reliability", [("calibration", ["calibration()", 'run_with_env(calibration, TARGET="local")']),
                                                  ("calib_reliability", ["calib_reliability()", 'run_with_env(calib_reliability, TARGET="local")']), ("trial_reliability", ["trial_reliability()"])]),
]

PIPE = nbf.read(M / "Neurovascnet_marge_v1.ipynb", as_version=4)
assert "EEG_FILTER = True" in PIPE.cells[4].source
PIPE.cells[4].source = PIPE.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")
_c6 = PIPE.cells[6].source.split("\n")
_i = next(k for k, l in enumerate(_c6) if l.startswith("    raise FileNotFoundError("))
_j = next(k for k in range(_i, len(_c6)) if _c6[k].rstrip().endswith("join(_lines))"))
_c6[_i:_j + 1] = ["    if RUN or BUILD_CACHES:"] + ["    " + l for l in _c6[_i:_j + 1]]
PIPE.cells[6].source = "\n".join(_c6)

names = [n for _, sc in SECTIONS for n, _ in sc]
defined = set()
for i in (2, 4, 5, 6, 8, 9, 10, 11, 13, 19):
    defined |= {n.name for n in ast.walk(ast.parse(pipeline(PIPE.cells[i]))) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
assert not defined & set(names), defined & set(names)
assert not defined & {"NB_OUT", "RUN", "should", "run_with_env"}

C = [md("""# NeuroVascNet: full analysis (infra-slow protocol)

**Run All shows every current result from the saved files in `analysis_p2_p4/results/`. Nothing is recomputed.**

**To rerun everything:** in section 2 (Settings), change `RUN = set()` to `RUN = ALL`, then Run All.

**To rerun one analysis:** go to the `HOW TO RERUN` block in section 2. Every analysis has a `# RERUN` line under its code with its name.

Open this notebook from the `merge/` folder. Rerunning needs the datasets in `dataset/` and the caches in `merge/neurovascnet_merge_v1_cache/`, neither of which is in git."""),
     md("## 1. Environment"), code(clean(pipeline(PIPE.cells[2]))),
     md("## 2. Settings: `RUN = ALL` is set here"),
     code('''from pathlib import Path

# ======================= HOW TO RERUN =======================
# RUN = set()              show the saved results, recompute nothing (default)
# RUN = {"calibration"}    rerun one analysis (names: see the "# RERUN" lines below each analysis)
# RUN = {"calibration", "nvc_kernel"}    rerun several
# RUN = ALL                rerun everything, one after another (rough estimate: a day or more)
#
# New results are written to merge/_nb_run/ and shown in the tables below.
# The saved results in analysis_p2_p4/results/ are never changed.
#
# PROJECT_ROOT must be the folder that holds dataset/ and merge/.
# Keep BUILD_CACHES = False: True overwrites the 60 GB cache in merge/neurovascnet_merge_v1_cache/, which is not in git.
ALL = {"all"}
RUN = set()
BUILD_CACHES = False
PROJECT_ROOT = Path.cwd().parent
NB_OUT = Path("_nb_run")
NB_OUT.mkdir(exist_ok=True)'''),
     code(clean(pipeline(PIPE.cells[4]))), code(clean(pipeline(PIPE.cells[5]))), code(clean(pipeline(PIPE.cells[6]))),
     code('''import os


def should(name):
    return "all" in RUN or name in RUN


def run_with_env(fn, **env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in old.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)'''),
     md("## 3. Data loading"), code(clean(pipeline(PIPE.cells[8]))), md("### HEFMI"), code(clean(pipeline(PIPE.cells[9]))),
     md("### hBCI"), code(clean(pipeline(PIPE.cells[10]))), md("### ds002336, ds002739"), code(clean(pipeline(PIPE.cells[11]))),
     md("### Load all four (cached)"), code("if BUILD_CACHES:\n" + clean(pipeline(PIPE.cells[13]), 4)),
     md("## 4. FineMI cache"), code(script("build_finemi") + '\n\n\nif BUILD_CACHES:\n    build_finemi()\n    run_with_env(build_finemi, REREF="lm")\n    run_with_env(build_finemi, HWHP="1")'),
     md("## 5. Metrics"), code(clean(pipeline(PIPE.cells[19])))]

src = (HERE / "build_results_notebook.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "build_results_notebook.py")}
exec(src[:src.index("nb = nbf.v4.new_notebook(")], ns)
figs = (HERE / "make_figures.py").read_text(encoding="utf-8")
assert 'HERE = Path(__file__).parent\n' in figs and 'A = HERE.parent / "analysis_p2_p4"\n' in figs
figs = figs.replace('HERE = Path(__file__).parent\n', 'HERE = Path("paper")\n')
figs = figs.replace('A = HERE.parent / "analysis_p2_p4"\n', 'A = NB_OUT if "all" in RUN else Path("analysis_p2_p4")\n')
s = ns["C"][1].source
a = 'where = lambda f: R / f\n'
assert a in s
s = s.replace(a, 'where = lambda f: NB_OUT / f if RUN and (NB_OUT / f).exists() else R / f\n')
figs = re.sub(r"\bCOH\b", "FIG_COH", figs)
figs = re.sub(r"\bOUT\b", "FIG_OUT", figs)
figs = re.sub(r'\nprint\("figures written to".*\n', "\n", figs)
_parts = re.split(r"^# Fig (\d+) .*$", figs, flags=re.M)
_prelude, FIGS = _parts[0], {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}
FNAME = {1: ("fig1_slow_band", 420), 2: ("fig2_coupling_lag", 520), 3: ("fig3_kernel_repro", 640), 4: ("fig4_kernel_transfer", 560), 5: ("fig5_deployment_reliability", 640), 6: ("fig6_slow_bands", 520), 7: ("fig7_person_matrix", 420), 8: ("fig8_calibration_runs", 500), 9: ("fig9_cross_lab", 620), 10: ("fig10_delay_halves", 860), 11: ("fig11_band_sign", 540), 12: ("fig12_downstream", 560), 13: ("fig13_multiplicity", 620), 14: ("fig14_reliability_people", 720), 15: ("fig15_third_lab", 780), 16: ("fig16_method_check", 780)}


def draw(n):
    return f"def draw_fig{n}():\n" + clean(FIGS[n], 4) + f'\n\n\ndraw_fig{n}()\nfig("{FNAME[n][0]}", {FNAME[n][1]})'


s += r"""


def replication(d):
    a = d["R1"]
    table([[k, f"{a[k]['r_subject']:+.3f}", ci(*a[k]["ci"]), f"{a[k]['p_signflip']:.4f}", a[k]["n_positive"]] for k in ("slow_lt0.5", "slow_lt0.1", "slow_0.1-0.5", "bandpower_1-45", "eog_slow_lt0.5")],
          ["EEG features", "r", "95% CI", "p", "People with r > 0"])
    table([[k, f"{v['r_subject']:+.3f}", ci(*v["ci"])] for k, v in d["R3"].items()], ["Target", "r", "95% CI"])
    table([[k, f"{v['diag']:.3f}", f"{v['offdiag']:.3f}", f"{v['diff']:+.3f} {ci(*v['ci'])}", f"{v['identification']:.0%}", f"{v['chance']:.0%}"] for k, v in d["R2"].items()],
          ["Features", "Own model", "Other people's models", "Difference", "Person identified", "Chance"])
"""
_src = {c.source: c for c in ns["C"] if c.cell_type == "code"}
_find = lambda key: next(x for x in _src if key in x)
_founding, _claim1, _claim4, _score, _surv = _find("mw, fm = J("), _find("c1h = J("), _find('d = J("downstream_repr_nlab0'), _find("c1m, c1f = J("), _find("from_attempts =")
_founding = _founding.replace('fig("fig1_slow_band", 420)', draw(1))
assert "draw_fig1()" in _founding

_cal_row = """rows = []
for label, g in (("raw HbO", J("calibration_results.json")), ("local HbO", J("calibration_results_local.json"))):
    for c, n in COH.items():
        for k in range(1, 6):
            v = g[c]["gains"].get(f"k{k}_personal_minus_baseline")
            if v and v.get("n", 0) >= 10:
                rows.append([label, n, k, f"{v['mean']:+.3f}", ci(*v["ci"]), f"{v['n_positive']}/{v['n']}"])
table(rows, ["Target", "Cohort", "Calibration runs", "Gain over own mean", "95% CI", "People better"])"""

RES = {
    "person_specific": ("""d = J("person_specific_results_runs.json")
table([[k, f"{v['diag_mean']:.3f}", f"{v['offdiag_mean']:.3f}", f"{v['diff_mean']:+.3f} {ci(*v['diff_ci'])}", f"{v['identification']:.0%}", f"{v['identification_chance']:.0%}"] for k, v in d.items()],
      ["Features", "Own model", "Other people's models", "Difference", "Person identified", "Chance"])""", "**Result (hBCI).** Prediction with a person's own model against other people's models."),
    "replicate_mw": ('replication(J("replication_results.json"))', "**Result (mental workload).** Signal by EEG band, by haemodynamic target, and how person-specific it is."),
    "replicate_finemi": ('replication(J("replication_finemi_results.json"))\n' + _founding, "**Result (FineMI).** The same three tables, then all three cohorts side by side and Figure 1."),
    "claim1_fresh": ('c1m, c1f = J("claim1_fresh_mental_workload_results.json"), J("claim1_fresh_finemi_results.json")\n' + _claim1, "**Result.** One EEG space predicts both outputs in unseen people. The cost column is the 90% CI lower bound, against a margin of −0.02 fixed in advance."),
    "external_claim2": ("""d = J("external_claim2_results.json")
table([[k.replace("_to_", " → "), f"{v['r_subject']:+.3f}", ci(v["ci_low"], v["ci_high"]), f"{v['p_signflip']:.4f}", f"{v['n_positive']}/{v['n_subjects']}", v["verdict"]] for k, v in d.items()],
      ["Direction", "r", "95% CI", "p", "People with r > 0", "Verdict"])""", "**Result.** Transfer of the slow-EEG model within the TU Berlin lab."),
    "external_finemi": ("""rows = []
for n, f in (("plain", ""), ("detrend", "_detrend"), ("linked mastoids", "_lm"), ("Euclidean alignment", "_ea"), ("hardware emulation", "_hw")):
    v = J(f"external_finemi{f}_results.json")["hbci_to_finemi"]
    rows.append([n, f"{v['r_subject']:+.3f}", ci(v["ci_low"], v["ci_high"]), f"{v['n_positive']}/{v['n_subjects']}"])
table(rows, ["hBCI → FineMI", "r", "95% CI", "People with r > 0"])""", "**Result.** Transfer of the slow-EEG model to the other lab, under each harmonisation."),
    "harmonised_transfer": ("""d = J("harmonised_transfer_results.json")
table([[k, f"{v['r_subject']:+.3f}", ci(v["ci_low"], v["ci_high"]), f"{v['p_signflip']:.4f}", f"{v['n_positive']}/{v['n_subjects']}", "pass" if v["pass"] else "fail"] for k, v in d.items()],
      ["Test", "r", "95% CI", "p", "People with r > 0", "Rule"])""", "**Result.** Transfer after matching the time window and the electrode montage."),
    "calibrated_transfer": ("""d = J("calibrated_transfer_results.json")
table([[k.replace("_to_", " → "), f"{v['personal']:.3f}", f"{v['personal_plus_external']:.3f}", f"{v['diff']:+.4f}", ci(*v["ci"]), f"{v['n_better']}/{v['n']}", "pass" if v["pass"] else "fail"] for k, v in d.items()],
      ["Direction", "Own model", "Own + external", "Gain", "95% CI", "People better", "Rule"])""", "**Result.** Gain from adding the frozen slow-EEG model to a person's own two-run model."),
    "continuous_lag": ("""d = J("continuous_lag_results.json")
table([[COH[c], f"{d[c]['frac_beating_null']:.0%}", f"{d[c]['median_lag_s']:+.1f} s", str(d[c]["iqr_lag_s"]), f"{d[c]['wilcoxon_p_vs_0']:.4f}"] for c in d], ["Cohort", "Beat null", "Median lag (positive = EEG first)", "IQR", "Wilcoxon p"])""", "**Result.** Lag between slow EEG and HbO in the simple single-channel form."),
    "nvc_bandpower": ("""d = J("nvc_bandpower_results.json")
table([[COH[c], f"{d[c]['frac_beating_null']:.0%}", f"{d[c]['median_delay_s']:+.1f} s", str(d[c]["iqr_delay_s"]), f"{d[c]['wilcoxon_p']:.1e}", f"{d[c]['split_half_r']:+.2f}", f"{d[c]['split_half_perm_p']:.3f}", f"{d[c]['median_peak_r']:.3f}"] for c in COH],
      ["Cohort", "Beat null", "Median delay (positive = EEG first)", "IQR", "Wilcoxon p", "Split-half r", "p", "Peak local r"])
""" + draw(2), "**Result and Figure 2.** Band-power coupling per cohort."),
    "nvc_transfer": ("""d = J("nvc_transfer_results.json")
table([["hBCI leave one person out", f"{d['within_hbci_lopo']['mean_r']:+.3f}", "—", f"{d['within_hbci_lopo']['frac_beating_null']:.0%}", "—"]] +
      [[k.replace("_to_", " → "), f"{d[k]['mean_r']:+.3f}", ci(*d[k]["ci"]), f"{d[k]['frac_beating_null']:.0%}", "pass" if d[k]["pass"] else "fail"] for k in ("hbci_to_mental_workload", "hbci_to_finemi", "mental_workload_to_finemi")],
      ["Test", "r", "95% CI", "Beat null", "Rule"])""", "**Result.** Frozen band-power kernel, within and between labs."),
    "nvc_transfer2": ("""d, sd = J("nvc_transfer2_results.json"), J("nvc_transfer2_seeds_results.json")
table([[k, f"{d[k]['mean_r']:+.3f}", ci(*d[k]["ci"]), f"{d[k]['frac_beating_null']:.0%}", "pass" if d[k]["pass"] else "fail"] for k in d if k.startswith("hbci_to_")], ["Frozen kernel", "r", "95% CI", "Beat null", "Rule"])
all_ = {"seed 1337 | " + k.split(" | ")[1]: v for k, v in d.items() if k.startswith("calibrated")}
all_.update({k.split(" | ")[2] + " | " + k.split(" | ")[1]: v for k, v in sd.items()})
table([[k, f"{v['diff']:+.3f}", ci(*v["ci"]), f"{v['n_better']}/{v['n']}", f"{v['frac_negative_external_gain']:.0%}", "pass" if v["pass"] else "fail"] for k, v in all_.items()], ["Calibrated gain", "Gain", "95% CI", "People better", "Negative external gain", "Rule"])""", "**Result.** Frozen kernel with a CBSI target, then the calibrated form over six calibration-run seeds. The FineMI gain is nominal: see the summary at the end."),
    "nvc_bands": ("""d = J("nvc_bands_results.json")
table([[k.replace("hbci_to_", "hBCI → "), f"{v['mean_r']:+.3f}", ci(*v["ci"]), f"{v['frac_beating_null']:.0%}", "pass" if v["pass"] else "fail"] for k, v in d.items() if k.startswith("hbci_to_")] +
      [["FineMI leave one person out", f"{d['lopo | finemi']['mean_r']:+.3f}", ci(*d["lopo | finemi"]["ci"]), f"{d['lopo | finemi']['frac_beating_null']:.0%}", "pass" if d["lopo | finemi"]["pass"] else "fail"]],
      ["Kernel", "r", "95% CI", "Beat null", "Rule"])
table([[k.split(" | ")[1], f"{v['diff']:+.3f}", ci(*v["ci"]), f"{v['n_better']}/{v['n']}", "pass" if v["pass"] else "fail"] for k, v in d.items() if k.startswith("calibrated")], ["Calibrated alpha + beta", "Gain", "95% CI", "People better", "Rule"])
""" + draw(4), "**Result and Figure 4.** Every single band transfers with a negative sign into FineMI, while FineMI people share a kernel of their own."),
    "nvc_labdiff": ("""d = J("nvc_labdiff_results.json")
table([[k.replace("_to_", " → "), f"{v['mean_r']:+.3f}", ci(*v["ci"]), f"{v['n_positive']}/{v['n']}"] for k, v in d.items() if "_to_" in k] +
      [[f"Overall EEG power vs global HbO, {COH[c]}", f"{d['broadband | ' + c]['mean_r']:+.3f}", ci(*d["broadband | " + c]["ci"]), f"{d['broadband | ' + c]['n_positive']}/{d['broadband | ' + c]['n']}"] for c in COH],
      ["Test", "r", "95% CI", "People with r > 0"])""", "**Result.** The kernel flip in the reverse direction, and overall EEG power against global HbO."),
    "nvc_broadband": ("""d = J("nvc_broadband_results.json")
table([[k.split(" | ")[0].replace("hbci_to_", "hBCI → "), k.split(" | ")[1], f"{d[k]['mean_r']:+.3f}", ci(*d[k]["ci"]), f"{d[k]['frac_beating_null']:.0%}", "pass" if d[k]["pass"] else "fail"] for k in d if k.startswith("hbci_to_")], ["Direction", "Kernel", "r", "95% CI", "Beat null", "Rule"])
table([[k, f"{v['diff']:+.3f}", ci(*v["ci"]), f"{v.get('signflip_p', v.get('signflip_p_removed')):.4f}"] for k, v in d.items() if k.startswith(("paired", "calibrated"))], ["Test", "Difference", "95% CI", "Sign-flip p"])""", "**Result.** Removing the broadband term shrinks the FineMI flip but does not remove it."),
    "nvc_residual": ("""d = J("nvc_residual_results.json")
table([[COH[c], f"{d[c]['frac_beating_null']:.0%}", f"{d[c]['median_delay_s']:+.1f} s", f"{d[c]['split_half_r']:+.2f}", f"{d[c]['split_half_perm_p']:.3f}", f"{d[c]['median_peak_r']:.3f}"] for c in COH], ["Cohort", "Beat null", "Median delay", "Split-half r", "p", "Peak local r"])""", "**Result.** Coupling after the cue-locked task response is removed."),
    "nvc_kernel": ("""kr, kt = J("nvc_kernel_results.json"), J("nvc_kernel_resid_results.json")
table([[COH[c], label, f"{d[c]['median_k']:+.3f}", f"{d[c]['median_k_null']:+.3f}", f"{d[c]['median_d']:+.3f}", f"{d[c]['wilcoxon_p_greater']:.1e}", "yes" if d[c]["kernel_reproducible"] else "no"] for c in COH for label, d in (("raw", kr), ("task removed", kt))],
      ["Cohort", "Data", "Median half-to-half r", "Shifted null", "Difference", "Wilcoxon p", "Reproducible"])
""" + draw(3), "**Result and Figure 3.** Does a person's coupling curve repeat between the two halves of the recording?"),
    "nvc_delay_smooth": ("""d = J("nvc_delay_smooth_results.json")
table([[k.split(" | ")[0], k.split(" | ")[1], f"{v['agree_2s']:.0%}", f"{v['chance']:.0%}", f"{v['p_agree']:.4f}", f"{v['median_delay_s']:+.1f} s", "yes" if v["reproducible"] else "no"] for k, v in d.items()], ["Cohort", "Data", "Halves agree within 2 s", "Chance", "p", "Median delay", "Reproducible"])""", "**Result.** The personal delay repeats in hBCI only."),
    "nvc_power": ("""d = J("nvc_power_results.json")
table([[k.replace("hbci matched to ", ""), f"{v['agree_2s']:.0%}", f"{v['chance']:.0%}", f"{v['p_agree']:.4f}", f"{v['wilcoxon_p']:.1e}", "yes" if v["reproducible"] else "no"] for k, v in d.items() if k.startswith("hbci matched")], ["hBCI cut to the length of", "Halves agree within 2 s", "Chance", "p", "Kernel Wilcoxon p", "Delay reproducible"])""", "**Result.** Recording length alone does not explain the delay failures."),
    "nvc_snr": ("""d = J("nvc_snr_results.json")
table([[k.replace("hbci at ", "").replace(" strength", ""), d["calibration"]["c"][k.split()[2]], f"{d['calibration']['median_peak'][k.split()[2]][str(d['calibration']['c'][k.split()[2]])]:.3f}", f"{v['agree_2s']:.0%}", f"{v['chance']:.0%}", f"{v['p_agree']:.4f}", "yes" if v["reproducible"] else "no"] for k, v in d.items() if k.startswith("hbci at")],
      ["hBCI degraded to the strength of", "Noise level c", "Achieved peak r", "Halves agree within 2 s", "Chance", "p", "Delay reproducible"])""", "**Result.** At FineMI's coupling strength the hBCI delay no longer repeats, so the FineMI delay result is inconclusive rather than negative."),
    "downstream_repr": (_claim4, "**Result.** Decoding accuracy from virtual haemodynamics against same-size unsupervised codes."),
    "calibration": (_cal_row, "**Result.** Gain over a person's own mean response, by number of calibration runs."),
    "calib_reliability": ("""rel, loc = J("calib_reliability_results.json"), J("calib_reliability_local_results.json")
table([[COH[c], f"{rel[c]['spearman_score_vs_minus_error']:.2f}", ci(*rel[c]["ci95"]), f"{loc[c]['spearman_score_vs_minus_error']:.2f}", ci(*loc[c]["ci95"]), "pass" if rel[c]["passes"] else "fail"] for c in COH], ["Cohort", "Spearman ρ, raw", "95% CI", "ρ, local", "95% CI", "Rule"])
""" + draw(5), "**Result and Figure 5.** Calibration-run score against later accuracy, with the deployment gain."),
    "trial_reliability": ("""d = J("trial_reliability_results.json")
table([[COH[c], f"{d[c]['ood']['mean']:+.3f}", ci(*d[c]["ood"]["ci"]), f"{d[c]['energy']['mean']:+.3f}", ci(*d[c]["energy"]["ci"]), "pass" if d[c]["passes"] else "fail"] for c in COH], ["Cohort", "Out-of-distribution score", "95% CI", "Predicted-signal energy", "95% CI", "Rule"])""", "**Result.** The declared rule requires hBCI, which fails, so the claim fails at trial level. Mental workload passes on its own."),
}
RES["nvc_mendeley"] = ('d, f_, t_, s_ = J("nvc_mendeley_results.json"), J("nvc_mendeley_founding.json"), J("nvc_mendeley_transfer.json"), J("nvc_mendeley_sync.json")\nprint(f"People passing the sync and length rules: {sum(not v[\'excluded\'] for v in s_.values())} of {len(s_)}")\ntable([["P12.1", "Founding: slow EEG, then band power", f"{f_[\'slow\'][\'mean_r\']:+.3f} {ci(*f_[\'slow\'][\'ci\'])}; {f_[\'bandpower\'][\'mean_r\']:+.3f} {ci(*f_[\'bandpower\'][\'ci\'])}", f"{f_[\'slow\'][\'p\']:.4f}", "pass" if f_["pass"] else "fail"],\n       ["P12.2", "Coupling exists", f"{d[\'P12.2 coupling\'][\'frac_beating_null\']:.0%} beat null; median delay {d[\'P12.2 coupling\'][\'median_delay_s\']:+.1f} s", "—", "pass" if d["P12.2 coupling"]["pass"] else "fail"],\n       ["P12.3", "Kernel repeats", f"median d {d[\'P12.3 kernel\'][\'median_d\']:+.3f}", f"{d[\'P12.3 kernel\'][\'wilcoxon_p\']:.4f}", "pass" if d["P12.3 kernel"]["pass"] else "fail"],\n       ["P12.4", "Personal delay repeats", f"{d[\'P12.4 delay\'][\'agree_2s\']:.0%} vs {d[\'P12.4 delay\'][\'chance\']:.0%} chance", f"{d[\'P12.4 delay\'][\'p_agree\']:.4f}", "pass" if d["P12.4 delay"]["pass"] else "fail"],\n       ["P12.5", "Frozen hBCI kernel", f"{t_[\'frozen\'][\'mean_r\']:+.3f} {ci(*t_[\'frozen\'][\'ci\'])}", f"{t_[\'frozen\'][\'signflip_p\']:.4f}", "pass" if t_["frozen"]["pass_all"] else "fail"],\n       ["P12.6", "Calibrated hBCI kernel", f"{t_[\'calibrated\'][\'diff\']:+.3f} {ci(*t_[\'calibrated\'][\'ci\'])}", f"{t_[\'calibrated\'][\'signflip_p\']:.4f}", "pass" if t_["calibrated"]["pass_all"] else "fail"]],\n      ["Test", "What", "Result", "p", "Rule (α = 0.0083)"])' + "\n" + draw(15), "**Result and Figure 15.** One of six declared tests passes. Coupling exists in the third lab and the kernel does not transfer. P12.1 cannot be read as a non-replication: see section 12c.")
RES["nvc_method_check"] = ("""d = J("nvc_method_check_results.json")
print(f"P12.1 method passes in {d['P13.1_passes_of_3']} of 3 established cohorts. Third-lab rerun differs from P12.1 by at most {d['mendeley_reproduces_P12.1_max_abs_diff']:.0e}.")
table([[{**COH, "mendeley": "Third lab"}[c], k, f"{v[k]['slow']['mean_r']:+.3f} {ci(*v[k]['slow']['ci'])}", f"{v[k]['bandpower']['mean_r']:+.3f}", f"{v[k]['diff']['mean']:+.3f} {ci(*v[k]['diff']['ci'])}", "pass" if v[k]["pass"] else "fail"] for c, v in d["cohorts"].items() for k in ("continuous", "windowed")], ["Cohort", "Method", "Slow EEG", "Band power", "Slow − band power", "Rule (α = 0.0071)"])
""" + draw(16), "**Result and Figure 16.** The P12.1 method recovers the slow-EEG advantage in only one of the three established cohorts, so the third-lab test is uninformative. Windows that are not aligned to cues recover it in all three. The third-lab window row is exploratory.")
RES["nvc_woody"] = ('d = J("nvc_woody_results.json")\ntable([[k, f"{v[\'agree_2s\']:.0%}", f"{v[\'chance\']:.0%}", f"{v[\'p_agree\']:.4f}", f"{v[\'median_delay_s\']:+.1f} s"] for k, v in d.items() if isinstance(v, dict)], ["Data", "Halves agree within 2 s", "Chance", "p", "Median delay"])\nprint("Estimator passes its positive control:", d["estimator_valid"])', "**Result.** The estimator fails its positive control, so its mental-workload and FineMI rows are descriptive only and the personal delay there stays inconclusive.")
for _n, _k in (("replicate_finemi", 6), ("person_specific", 7), ("calibration", 8), ("nvc_bands", 9), ("nvc_snr", 10), ("nvc_broadband", 11), ("downstream_repr", 12), ("calib_reliability", 14)):
    RES[_n] = (RES[_n][0] + "\n" + draw(_k), RES[_n][1] + f" Figure {_k} is shown below.")
assert set(RES) == {n for _, sc in SECTIONS for n, _ in sc}, set(RES) ^ {n for _, sc in SECTIONS for n, _ in sc}

C += [md("## 6. Display helpers"), code(clean(_prelude)), code(s)]
for title, scripts in SECTIONS:
    C.append(md(f"## {title}"))
    for n, calls in scripts:
        run = ("\n\n\n# RERUN: set RUN = {\"" + n + "\"} in section 2\nif should(\"" + n + "\"):\n" + "\n".join("    " + c for c in calls)) if calls else ""
        C.append(code(script(n) + run))
        C.append(code(RES[n][0], metadata={"desc": RES[n][1]}))
C += [md("## 15. Summary of all claims"), code(_score), code('kr, kt = J("nvc_kernel_results.json"), J("nvc_kernel_resid_results.json")\n' + _surv + "\n" + draw(13))]

DESCRIBE = {
    "person_specific": "Tests whether the link between slow EEG and fNIRS is specific to each person. Each person's model is scored on their own held-out run and on every other person. Output: `person_specific_results_runs.json`. Rerun name: `person_specific`.",
    "replicate_mw": "Repeats the founding tests on the mental-workload cohort: which EEG band carries the signal, whether EOG explains it, whether a local haemodynamic part remains, and how person-specific it is. Output: `replication_results.json`. Rerun name: `replicate_mw`.",
    "replicate_finemi": "The same founding tests on FineMI, the cohort from a different lab. Output: `replication_finemi_results.json`. Rerun name: `replicate_finemi`.",
    "claim1_fresh": "Claim 1. Fits one shared linear EEG space (rank 32, fixed in advance) that predicts both fNIRS and fMRI, and tests it on unseen people. It also checks whether real fMRI targets help fNIRS more than shuffled ones. Output: `claim1_fresh_<cohort>_results.json`. Rerun name: `claim1_fresh`.",
    "external_claim2": "Claim 2 within one lab. A slow-EEG ridge model is frozen on hBCI and tested on mental workload, and the reverse. Output: `external_claim2_results.json`. Rerun name: `external_claim2`.",
    "external_finemi": "Claim 2 across labs. The same frozen model is tested on FineMI, with five versions of the data: plain, detrended, linked-mastoid reference, Euclidean alignment and emulated hardware filter. Output: `external_finemi*_results.json`. Rerun name: `external_finemi`.",
    "harmonised_transfer": "Claim 2 across labs, after matching the time window and the electrode montage between cohorts. Output: `harmonised_transfer_results.json`. Rerun name: `harmonised_transfer`.",
    "calibrated_transfer": "Claim 2 across labs, calibrated form: does the frozen slow-EEG model add to a person's own two-run model? Output: `calibrated_transfer_results.json`. Rerun name: `calibrated_transfer`.",
    "continuous_lag": "Loaders for the continuous hBCI and mental-workload recordings, helpers that align EEG with fNIRS at 1 Hz, and the lag between slow EEG and HbO. The analyses below reuse the loaders. Output: `continuous_lag_results.json`. Rerun name: `continuous_lag`.",
    "nvc_bandpower": "Claim 3. Classical neurovascular coupling: lagged ridge from EEG band power (theta to gamma) to local HbO, at lags from −10 to +20 s, with each person compared to their own time-shifted null. Also holds the FineMI loader. Output: `nvc_bandpower_results.json`. Rerun name: `nvc_bandpower`.",
    "nvc_transfer": "Claim 2. A band-power coupling kernel is frozen on hBCI and tested on the other cohorts, plus leave-one-person-out within hBCI. Output: `nvc_transfer_results.json`. Rerun name: `nvc_transfer`.",
    "nvc_transfer2": "Claim 2. The frozen kernel with a CBSI target, and the calibrated form: does it add to a person's own two-run model? The second call repeats the calibrated form over five seeds. Output: `nvc_transfer2_results.json`, `nvc_transfer2_seeds_results.json`. Rerun name: `nvc_transfer2`.",
    "nvc_bands": "Diagnostic for the sign flip into FineMI: kernels restricted to single EEG bands, an alpha + beta kernel, and FineMI leave-one-person-out. Output: `nvc_bands_results.json`. Rerun name: `nvc_bands`.",
    "nvc_labdiff": "Diagnostic: the FineMI kernel applied to TU Berlin data, and the correlation of overall EEG power with global HbO in each cohort. Output: `nvc_labdiff_results.json`. Rerun name: `nvc_labdiff`.",
    "nvc_broadband": "Tests whether the overall EEG power term causes the sign flip, by refitting the kernel with that term removed and with that term alone. Output: `nvc_broadband_results.json`. Rerun name: `nvc_broadband`.",
    "nvc_residual": "Claim 3 beyond the task. The coupling analysis repeated after removing the cue-locked average response, so a shared task response cannot explain it. Output: `nvc_residual_results.json`. Rerun name: `nvc_residual`.",
    "nvc_kernel": "Claim 3. Does a person's coupling curve in the first half of the recording match the second half, more than after time-shifting? The second call uses the task-removed data. Output: `nvc_kernel_results.json`, `nvc_kernel_resid_results.json`. Rerun name: `nvc_kernel`.",
    "nvc_delay_smooth": "Claim 3. Personal delay taken from smoothed half-recording curves, and how often the two halves agree within 2 s. Needs the output of `nvc_kernel`. Output: `nvc_delay_smooth_results.json`. Rerun name: `nvc_delay_smooth`.",
    "nvc_power": "Diagnostic: hBCI recordings cut to the length of the other cohorts, to see whether the personal delay still repeats. Output: `nvc_power_results.json`. Rerun name: `nvc_power`.",
    "nvc_snr": "Diagnostic: hBCI fNIRS degraded with added noise to the coupling strength of the other cohorts, to see whether the personal delay still repeats. Output: `nvc_snr_results.json`. Rerun name: `nvc_snr`.",
    "nvc_woody": "One attempt at a more sensitive delay estimator (template alignment), which must first recover the delay in degraded hBCI. Needs the output of `nvc_kernel` and `nvc_snr`. Output: `nvc_woody_results.json`. Rerun name: `nvc_woody`.",
    "nvc_mendeley": "Confirmation on a cohort from a third lab (Mendeley Learning Capability), not used in the main analysis (an early band-power baseline had been run on it). Nothing is tuned on it: all rules were written down first. Output: `nvc_mendeley_results.json`, `nvc_mendeley_founding.json`, `nvc_mendeley_transfer.json`. Rerun name: `nvc_mendeley`.",
    "nvc_method_check": "Checks the method used in the third-lab test. Runs the P12.1 code on the three established cohorts, and the original window analysis on 30 s windows not aligned to cues, in all four cohorts. Needs `nvc_mendeley_sync.json` and `nvc_mendeley_founding.json` from `nvc_mendeley`. Output: `nvc_method_check_*.json`. Rerun name: `nvc_method_check`.",
    "downstream_repr": "Claim 4. Task decoding from EEG-predicted haemodynamics compared with PCA and random-projection codes of the same size. Output: `downstream_repr_nlab0_results.json`. Rerun name: `downstream_repr`.",
    "calibration": "Deployment test. After `k` calibration runs with EEG and fNIRS together, does a person's EEG-only model beat replaying their own average response? Run on raw and local HbO. Output: `calibration_results.json`, `calibration_results_local.json`. Rerun name: `calibration`.",
    "calib_reliability": "Claim 5, per person. Does a person's score on their two calibration runs predict how well the model works on their later runs? Output: `calib_reliability_results.json`, `calib_reliability_local_results.json`. Rerun name: `calib_reliability`.",
    "trial_reliability": "Claim 5, per trial. Can single unreliable trials be identified in advance? Output: `trial_reliability_results.json`. Rerun name: `trial_reliability`.",
    "build_finemi": "Builds the FineMI epoch cache from the raw files (plain, linked-mastoid and emulated-hardware versions). Runs only when `BUILD_CACHES = True`.",
}
PIPE_TEXT = {2: "Checks the Python packages (installs any that are missing), then sets the random seed and the device.",
             4: "Settings from the merge notebook: dataset paths and pipeline options. `EEG_FILTER = False` keeps the infra-slow band.",
             5: "Builds the configuration object and the run identifiers from those settings.",
             6: "Checks that each dataset folder exists. On a machine without the datasets it prints `[BAD]`; this does not matter when only viewing results.",
             8: "Shared signal helpers: filtering, resampling, montage interpolation, z-scoring, baseline correction and the Beer-Lambert conversion.",
             9: "Loader for HEFMI. This dataset is excluded from the main analysis, because its published EEG has no power below 0.5 Hz.",
             10: "Loader for the hBCI cohort (`load_hbci`), used to build the hBCI epoch cache.",
             11: "Loaders for the fMRI datasets ds002336 and ds002739. ds002739 is excluded from the main analysis for the same reason as HEFMI.",
             13: "Loads all four datasets and writes the shared cache. Runs only when `BUILD_CACHES = True`.",
             19: "Subject-level statistics used throughout: correlations, sign-flip tests, permutation nulls and verdict text."}
OTHER = (("HOW TO RERUN", "**This is the only cell you need to edit.** Leave `RUN = set()` to view the saved results. Change it to `RUN = ALL` to recompute everything."),
         ("def should(name)", "Two small helpers: `should` decides whether an analysis runs, and `run_with_env` runs one with temporary settings."),
         ("def save(fig, name)", "Figure style and helpers. Each figure is drawn under the analysis it belongs to and saved in `paper/figures/`."),
         ("def table(rows", "Helpers that read the result files and draw the tables shown under each analysis."),
         ("c1m, c1f = J(", "The best result for each of the five claims and its verdict."),
         ("mw, fm = J(", "Where the EEG information about haemodynamics lies, per cohort. Slow EEG carries it and 1-45 Hz band power does not."),
         ("c1h = J(", "One EEG space predicts both fNIRS and fMRI in unseen people. The cost column is the 90% CI lower bound, against a margin of −0.02 fixed in advance."),
         ("rev = J(", "Transfer within the TU Berlin lab, then every form tried into FineMI (best first), then the diagnostics for the sign flip."),
         ("res, kr, kt = J(", "Coupling per cohort: share of people beating their null, delay, whether the kernel repeats between halves, and whether the delay repeats."),
         ('d = J("downstream_repr_nlab0', "Decoding accuracy from virtual haemodynamics against PCA and random-projection codes."),
         ("g, gl = J(", "Gain from two calibration runs over a person's own mean response, and how well reliability is known in advance."),
         ("from_attempts =", "Which positive results survive Bonferroni correction across all attempts at the same claim, also when cohorts are counted as separate tests. Details: `TEST_LEDGER.md`."),
         ('nvc_broadband_results.json").exists', "The most recent tests: the broadband term in the sign flip (P10.1) and the coupling-strength diagnostic for the delay (P10.2)."))


def describe(src):
    m = re.match(r"def (\w+)\(LIB=False", src)
    if m:
        return DESCRIBE[m.group(1)]
    for i, txt in PIPE_TEXT.items():
        if src == clean(pipeline(PIPE.cells[i])) or (i == 13 and src.startswith("if BUILD_CACHES:\n    LIMIT_PEOPLE")):
            return txt
    for key, txt in OTHER:
        if key in src:
            return txt
    raise KeyError(src[:80])


_out = []
for c in C:
    if c.cell_type == "code":
        txt = c.metadata.pop("desc", None) or describe(c.source)
        if _out and _out[-1].cell_type == "markdown":
            _out[-1].source += "\n\n" + txt
        else:
            _out.append(md(txt))
    _out.append(c)
C = _out

for c in C:
    if c.cell_type == "code":
        c.source = re.sub(r'r"G:\\research_bio_medical\\([^"]*)"', lambda m: 'f"{PROJECT_ROOT}/' + m.group(1).replace("\\", "/") + '"', c.source)
        assert "research_bio_medical" not in c.source, c.source[:80]

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
out = M / "NeuroVascNet_Full.ipynb"
nbf.write(nb, out)
total = sum(len(c.source.splitlines()) for c in C if c.cell_type == "code")
comments = sum(1 for c in C if c.cell_type == "code" for l in c.source.splitlines() if l.strip().startswith("#"))
print("wrote", out, len(C), "cells;", total, "code lines;", comments, "comment lines")
