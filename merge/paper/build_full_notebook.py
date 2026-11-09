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
       "nvc_delay_smooth": "rng = np.random.default_rng(1337)"}


def script(name):
    s = (A / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    mode = re.search(r'^MODE = sys\.argv\[1\] if len\(sys\.argv\) > 1 else "(\w+)"', s, re.M)
    default = mode.group(1) if mode else "all"
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
        s, ok = replace_block(s, r"nbformat\.read\(", r"os\.chdir\(cwd\)", "ns = globals()" if "ns = {" in blk[:blk.index("os.chdir(cwd)")] else "")
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
    ("10. Continuous-recording loaders and band-power coupling", [("continuous_lag", []), ("nvc_bandpower", ["nvc_bandpower()"])]),
    ("11. Claim 2: coupling-kernel transfer", [("nvc_transfer", ["nvc_transfer()"]), ("nvc_transfer2", ["nvc_transfer2()", 'run_with_env(nvc_transfer2, CAL_SEEDS="1,2,3,4,5")']),
                                               ("nvc_bands", ["nvc_bands()"]), ("nvc_labdiff", ["nvc_labdiff()"]), ("nvc_broadband", ["nvc_broadband()"])]),
    ("12. Claim 3: coupling and delay", [("nvc_residual", ["nvc_residual()"]), ("nvc_kernel", ["nvc_kernel()", 'run_with_env(nvc_kernel, RESID="1")']), ("nvc_delay_smooth", ["nvc_delay_smooth()"]),
                                         ("nvc_power", ['nvc_power(MODE="durations")', 'nvc_power(MODE="shard")', 'nvc_power(MODE="merge")']),
                                         ("nvc_snr", ['nvc_snr(MODE="calibrate")', 'nvc_snr(MODE="pick")', 'nvc_snr(MODE="shard")', 'nvc_snr(MODE="merge")'])]),
    ("13. Claim 4: downstream task", [("downstream_repr", ['run_with_env(downstream_repr, N_LAB="0")'])]),
    ("14. Claim 5: calibration and reliability", [("calibration", ["calibration()", 'run_with_env(calibration, TARGET="local")']),
                                                  ("calib_reliability", ["calib_reliability()", 'run_with_env(calib_reliability, TARGET="local")']), ("trial_reliability", ["trial_reliability()"])]),
]

PIPE = nbf.read(M / "Neurovascnet_marge_v1.ipynb", as_version=4)
assert "EEG_FILTER = True" in PIPE.cells[4].source
PIPE.cells[4].source = PIPE.cells[4].source.replace("EEG_FILTER = True", "EEG_FILTER = False")

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

for title, scripts in SECTIONS:
    C.append(md(f"## {title}"))
    for n, calls in scripts:
        run = ("\n\n\n# RERUN: set RUN = {\"" + n + "\"} in section 2\nif should(\"" + n + "\"):\n" + "\n".join("    " + c for c in calls)) if calls else ""
        C.append(code(script(n) + run))

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
C += [md("## 15. Figures"), code(clean(figs)), md("# Results"), code(s)]
C += ns["C"][2:]

DESCRIBE = {
    "person_specific": "Tests whether the link between slow EEG and fNIRS is specific to each person. Each person's model is scored on their own held-out run and on every other person. Output: `person_specific_results_runs.json`. Rerun name: `person_specific`.",
    "replicate_mw": "Repeats the founding tests on the mental-workload cohort: which EEG band carries the signal, whether EOG explains it, whether a local haemodynamic part remains, and how person-specific it is. Output: `replication_results.json`. Rerun name: `replicate_mw`.",
    "replicate_finemi": "The same founding tests on FineMI, the cohort from a different lab. Output: `replication_finemi_results.json`. Rerun name: `replicate_finemi`.",
    "claim1_fresh": "Claim 1. Fits one shared linear EEG space (rank 32, fixed in advance) that predicts both fNIRS and fMRI, and tests it on unseen people. It also checks whether real fMRI targets help fNIRS more than shuffled ones. Output: `claim1_fresh_<cohort>_results.json`. Rerun name: `claim1_fresh`.",
    "external_claim2": "Claim 2 within one lab. A slow-EEG ridge model is frozen on hBCI and tested on mental workload, and the reverse. Output: `external_claim2_results.json`. Rerun name: `external_claim2`.",
    "external_finemi": "Claim 2 across labs. The same frozen model is tested on FineMI, with five versions of the data: plain, detrended, linked-mastoid reference, Euclidean alignment and emulated hardware filter. Output: `external_finemi*_results.json`. Rerun name: `external_finemi`.",
    "harmonised_transfer": "Claim 2 across labs, after matching the time window and the electrode montage between cohorts. Output: `harmonised_transfer_results.json`. Rerun name: `harmonised_transfer`.",
    "calibrated_transfer": "Claim 2 across labs, calibrated form: does the frozen slow-EEG model add to a person's own two-run model? Output: `calibrated_transfer_results.json`. Rerun name: `calibrated_transfer`.",
    "continuous_lag": "Loaders for the continuous hBCI and mental-workload recordings, and the helpers that align EEG with fNIRS at 1 Hz. The analyses below reuse them. This cell runs nothing by itself.",
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
         ("fig1_slow_band", "Draws the five figures from the result files and saves them in `paper/figures/`."),
         ("def table(rows", "Helpers that read the result files and draw the tables below."),
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
        txt = describe(c.source)
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
