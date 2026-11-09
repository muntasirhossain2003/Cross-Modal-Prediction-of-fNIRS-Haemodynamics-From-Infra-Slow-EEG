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

**To rerun an analysis:** go to the `HOW TO RERUN` block in section 2 (Settings). Every analysis has a `# RERUN` line under its code with its name.

Open this notebook from the `merge/` folder. Rerunning needs the datasets in `dataset/` and the caches in `merge/neurovascnet_merge_v1_cache/`, neither of which is in git."""),
     md("## 1. Environment"), code(clean(pipeline(PIPE.cells[2]))),
     md("## 2. Settings"),
     code('''from pathlib import Path

# ======================= HOW TO RERUN =======================
# RUN = set()              show the saved results, recompute nothing (default)
# RUN = {"calibration"}    rerun one analysis (names: see the "# RERUN" lines below each analysis)
# RUN = {"calibration", "nvc_kernel"}    rerun several
# RUN = {"all"}            rerun everything, one after another (rough estimate: a day or more)
#
# New results are written to merge/_nb_run/ and shown in the tables below.
# The saved results in analysis_p2_p4/results/ are never changed.
#
# PROJECT_ROOT must be the folder that holds dataset/ and merge/.
# Keep BUILD_CACHES = False: True overwrites the 60 GB cache in merge/neurovascnet_merge_v1_cache/, which is not in git.
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
