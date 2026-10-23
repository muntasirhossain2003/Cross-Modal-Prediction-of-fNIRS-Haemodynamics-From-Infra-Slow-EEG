"""Shared styling and helpers for the lecture PDFs (Lectures B onward).

Figures are drawn with matplotlib as SVG, pages are typeset as print HTML and
rendered to PDF by headless Chrome.
"""
import subprocess
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
LECTURES = ROOT / "lectures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, LIGHT, INK, MUTED, GRID = "#a8a6a0", "#e4e2dc", "#1a1a1a", "#55534f", "#e8e7e3"
W = 6.9

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 10, "svg.fonttype": "none",
    "axes.edgecolor": "#bdbbb5", "axes.labelcolor": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
})


class Doc:
    """One lecture document: owns its build folder, figures and output path."""

    def __init__(self, slug, pdf_name, header):
        self.build = ROOT / "report_build" / slug
        self.figdir = self.build / "figures"
        self.figdir.mkdir(parents=True, exist_ok=True)
        LECTURES.mkdir(exist_ok=True)
        self.pdf = LECTURES / pdf_name
        self.header = header

    def save(self, fig, name):
        fig.savefig(self.figdir / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    def render(self, body, title):
        html = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{css(self.header)}</style></head><body>{body}</body></html>'
        src = self.build / "doc.html"
        src.write_text(html, encoding="utf-8")
        tmp = self.build / "doc.pdf"
        tmp.unlink(missing_ok=True)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={tmp}", src.as_uri()], check=True, capture_output=True)
        if not tmp.exists():
            raise SystemExit("Chrome did not produce a PDF")
        try:
            self.pdf.write_bytes(tmp.read_bytes())
            print("wrote", self.pdf)
        except PermissionError:
            print(f"{self.pdf.name} is open in another program; new version left at {tmp}")


# ---------------------------------------------------------------- drawing helpers
def box(ax, x, y, w, h, text, fc="#f3f5f8", ec="#9aa7b8", fs=9.3, color=INK, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)


def arrow(ax, p, q, color=MUTED, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def badge(ax, x, y, n, color=ORANGE):
    ax.add_patch(plt.Circle((x, y), 0.17, color=color, zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=8.5, weight="bold", zorder=6)


def hrf(t, peak=6.0, under=16.0, ratio=1 / 6):
    t = np.clip(t, 0, None)
    h = t ** (peak - 1) * np.exp(-t) / factorial(int(peak - 1)) - ratio * t ** (under - 1) * np.exp(-t) / factorial(int(under - 1))
    return h / h.max()


# ---------------------------------------------------------------- html helpers
def fig(name, num, title, body):
    return f'<figure><img src="figures/{name}.svg"><figcaption><b>Figure {num}. {title}</b> {body}</figcaption></figure>'


def callout(kind, title, body):
    return f'<div class="call {kind}"><div class="ctitle">{title}</div>{body}</div>'


def A(t, b): return callout("analogy", "Analogy · " + t, b)
def E(t, b): return callout("example", "Example · " + t, b)
def P(b): return callout("project", "In our project", b)
def K(b): return callout("key", "Key idea", b)
def M(b): return callout("myth", "Common mistake", b)
def V(b): return callout("viva", "If someone asks you…", b)


def Q(items):
    li = "".join(f"<li>{q}</li>" for q in items)
    return f'<div class="call think"><div class="ctitle">Pause and think</div><ol class="qs">{li}</ol><div class="hint">Try to answer before reading on. Answers are at the end.</div></div>'


def table(num, title, head, rows, widths=None, note=""):
    cg = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    nt = f'<div class="tnote">{note}</div>' if note else ""
    return f'<div class="tbl"><div class="tcap"><b>Table {num}.</b> {title}</div><table><colgroup>{cg}</colgroup><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>{nt}</div>'


def title_block(kicker, title, subtitle, meta):
    return f'<div class="kicker">{kicker}</div><h1>{title}</h1><div class="subtitle">{subtitle}</div><div class="meta">{meta}</div>'


def qa_bank(items):
    out, cur, n = [], None, 0
    for sec, q, a in items:
        if sec != cur:
            out.append(f"<h3>{sec}</h3>")
            cur = sec
        n += 1
        out.append(f'<div class="qa"><div class="q">Q{n}. {q}</div><div class="a">{a}</div></div>')
    return "\n".join(out)


def css(header):
    return f"""
@page {{ size: A4; margin: 22mm 21mm 20mm 21mm;
  @bottom-center {{ content: counter(page); font-family: Calibri, sans-serif; font-size: 9pt; color: #777; }}
  @top-right {{ content: "{header}"; font-family: Calibri, sans-serif; font-size: 8.5pt; color: #999; }} }}
@page :first {{ @top-right {{ content: none; }} @bottom-center {{ content: none; }} }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: Cambria, Georgia, serif; font-size: 10.6pt; line-height: 1.5; color: #1a1a1a; margin: 0; }}
h1, h2, h3, .ctitle, figcaption, table, .kicker, .meta, .hint, .q, .tcap, .tnote {{ font-family: Calibri, "Segoe UI", sans-serif; }}
.kicker {{ font-size: 10pt; letter-spacing: .08em; text-transform: uppercase; color: {BLUE}; font-weight: 700; margin-top: 6mm; }}
h1 {{ font-size: 25pt; line-height: 1.12; margin: 2pt 0 4pt; }}
.subtitle {{ font-size: 12.5pt; color: #444; font-style: italic; margin-bottom: 10pt; }}
.meta {{ font-size: 9.5pt; color: #555; border-top: 1.5px solid #222; border-bottom: .6px solid #bbb; padding: 5pt 0; margin-bottom: 12pt; }}
h2 {{ font-size: 16pt; margin: 20pt 0 6pt; padding-bottom: 2pt; border-bottom: .8px solid #ccc; break-after: avoid; }}
h3 {{ font-size: 12pt; margin: 13pt 0 3pt; break-after: avoid; color: #222; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }}
ul, ol {{ margin: 0 0 7pt; padding-left: 17pt; }} li {{ margin-bottom: 2.5pt; }}
figure {{ margin: 8pt 0 12pt; break-inside: avoid; }} figure img {{ width: 100%; display: block; }}
figcaption {{ font-size: 9pt; line-height: 1.35; color: #333; margin-top: 4pt; }}
.call {{ border-left: 3.5px solid; padding: 6pt 11pt 3pt; margin: 8pt 0 11pt; break-inside: avoid; border-radius: 0 4px 4px 0; }}
.call p:last-child {{ margin-bottom: 4pt; }}
.ctitle {{ font-weight: 700; font-size: 9.5pt; letter-spacing: .03em; text-transform: uppercase; margin-bottom: 3pt; }}
.analogy {{ background: #fff7e8; border-color: {YELLOW}; }} .analogy .ctitle {{ color: #8a5a00; }}
.example {{ background: #eef6f1; border-color: {AQUA}; }} .example .ctitle {{ color: #0f6b4a; }}
.project {{ background: #eef3fb; border-color: {BLUE}; }} .project .ctitle {{ color: #1d5aa6; }}
.key {{ background: #f6f2fb; border-color: {VIOLET}; }} .key .ctitle {{ color: {VIOLET}; }}
.think {{ background: #f6f6f4; border-color: #8a877f; }} .think .ctitle {{ color: #444; }}
.myth {{ background: #fdf0ec; border-color: {ORANGE}; }} .myth .ctitle {{ color: #b2431b; }}
.viva {{ background: #eef6f1; border-color: {AQUA}; }} .viva .ctitle {{ color: #0f6b4a; }}
.hint {{ font-size: 8.5pt; color: #777; margin: 2pt 0 4pt; }}
.box {{ background: #f4f6f9; padding: 8pt 12pt 3pt; margin-bottom: 10pt; break-inside: avoid; }}
.box h3 {{ margin-top: 0; }}
.tbl {{ break-inside: avoid; margin: 6pt 0 12pt; }}
.tcap {{ font-size: 9pt; color: #333; margin-bottom: 3pt; }}
.tnote {{ font-size: 8.3pt; color: #666; margin-top: 2pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.9pt; line-height: 1.33; table-layout: fixed; }}
thead th {{ border-top: 1.3px solid #222; border-bottom: .7px solid #222; text-align: left; padding: 3pt 5pt; vertical-align: bottom; }}
td {{ padding: 2.8pt 5pt; vertical-align: top; border-bottom: .4px solid #e2e2e2; }}
tbody tr:last-child td {{ border-bottom: 1.3px solid #222; }}
.formula {{ text-align: center; font-size: 11.5pt; margin: 6pt 0 8pt; }}
.answers li {{ margin-bottom: 5pt; }}
.qa {{ break-inside: avoid; margin: 0 0 7pt; }}
.q {{ font-weight: 700; font-size: 10pt; color: #1d5aa6; margin-bottom: 1pt; }}
.a {{ text-align: justify; }}
.refs li {{ margin-bottom: 4pt; }}
.yes {{ color: #12754f; font-weight: 700; }} .no {{ color: #b2431b; font-weight: 700; }} .mid {{ color: #8a6a00; font-weight: 700; }}
"""
