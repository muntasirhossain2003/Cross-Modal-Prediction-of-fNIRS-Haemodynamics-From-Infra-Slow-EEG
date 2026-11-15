"""Draws the graphical abstract for the Biomedical Signal Processing and Control manuscript.

Layout follows Maisha Rahman's draft. Canvas 13.0 x 5.2 cm (Elsevier's 1328 x 531 px proportion); 1 unit = 0.1 mm,
so 20 units = 5.7 pt, the smallest text used. Rendered by headless Chrome to bspc/graphical_abstract.pdf (vector) and
bspc/graphical_abstract.tif (600 dpi). Every number comes from the manuscript (Sections 3.1-3.8, Tables 2-5).
"""
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = Path(__file__).parent / "bspc"
W, H = 1300, 520
TEAL, TEAL_D, TEAL_L, TEAL_P, NAVY = "#1f6b7d", "#2b5566", "#e8f1f4", "#9fcbd6", "#16243a"
INK, INK2, MUTED = "#1c2430", "#3f4752", "#6b7280"
ORANGE, GREY, PANEL = "#d9640f", "#9aa1aa", "#eef2f5"
F = "font-family:'Segoe UI',Arial,sans-serif"
S = []


def t(x, y, s, size=20, w=400, c=INK, a="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    S.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{c}" text-anchor="{a}"{st}>{s}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=0, r=8):
    S.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def circle(x, y, r, fill, stroke="none", sw=0):
    S.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x1, y, x2, c=INK2, sw=3):
    S.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 6:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="{sw}"/>'
             f'<path d="M{x2 - 9:.1f},{y - 7:.1f} L{x2:.1f},{y:.1f} L{x2 - 9:.1f},{y + 7:.1f} Z" fill="{c}"/>')


# Title
rect(0, 0, W, 50, TEAL_D, r=0)
t(W / 2, 34, "The missing band: infra-slow EEG predicts haemodynamics, and high-pass filtering removes it", 25, 700, "#ffffff", "middle")

# Study chips
chips = [("people", "88 people", " EEG + fNIRS · 4 cohorts · 3 labs"), ("brain", "27 people", " EEG + fMRI · 2 datasets"),
         ("method", "Method", " ridge · leave-one-out · frozen transfer")]
CWS = [424, 366, 446]
for i, (icon, a, b) in enumerate(chips):
    x, cw = 20 + sum(CWS[:i]) + 12 * i, CWS[i]
    rect(x, 58, cw, 38, PANEL, r=10)
    cx, cy = x + 22, 77
    circle(cx, cy, 14, TEAL)
    if icon == "people":
        circle(cx - 4.5, cy - 4, 3.2, "#ffffff"); circle(cx + 4.5, cy - 4, 3.2, "#ffffff")
        S.append(f'<path d="M{cx - 10},{cy + 7} q5.5,-9 11,0 M{cx - 1},{cy + 7} q5.5,-9 11,0" fill="#ffffff" stroke="#ffffff" stroke-width="1.4"/>')
    elif icon == "brain":
        circle(cx, cy, 8, "none", "#ffffff", 2.6); S.append(f'<line x1="{cx}" y1="{cy - 8}" x2="{cx}" y2="{cy + 8}" stroke="#ffffff" stroke-width="2"/>')
    else:
        for dx, dy in ((-6, -5), (6, -5), (0, 6)):
            circle(cx + dx, cy + dy, 3, "#ffffff")
        S.append(f'<path d="M{cx - 6},{cy - 5} L{cx + 6},{cy - 5} L{cx},{cy + 6} Z" fill="none" stroke="#ffffff" stroke-width="1.4"/>')
    S.append(f'<text x="{x + 42:.1f}" y="84" font-size="20" fill="{INK}"><tspan font-weight="700" fill="{TEAL}">{a}</tspan>{b}</text>')

# Five result cards
CY, CH = 104, 206
kw = (W - 40 - 4 * 10) / 5
titles = ["1  Beats band power", "2  One EEG space", "3  Calibration", "4  Timing, hBCI", "5  Transfer"]
X = [20 + i * (kw + 10) for i in range(5)]
for x, ttl in zip(X, titles):
    rect(x, CY, kw, CH, "#ffffff", TEAL_D, 2.2, 12)
    rect(x + 7, CY + 7, kw - 14, 30, TEAL_D, r=7)
    t(x + 16, CY + 29, ttl, 21, 700, "#ffffff")

# 1. infra-slow against band power (Table 2)
x = X[0]
sc = 92 / 0.21
for j, (lab, s_, b_) in enumerate((("hBCI", 0.197, 0.02), ("MW", 0.135, 0.006), ("FineMI", 0.203, 0.045))):
    y = CY + 46 + j * 44
    t(x + 70, y + 21, lab, 20, 600, INK, "end")
    rect(x + 76, y + 2, s_ * sc, 16, TEAL, r=2)
    t(x + 81 + s_ * sc, y + 17, f"{s_:.3f}", 20, 700, TEAL)
    rect(x + 76, y + 21, max(b_ * sc, 2.5), 12, GREY, r=2)
    t(x + 81 + max(b_ * sc, 2.5), y + 34, f"{b_:g}", 20, 400, MUTED)
rect(x + 14, CY + 182, 12, 12, TEAL, r=2); t(x + 31, CY + 193, "infra-slow", 20, 600, INK)
rect(x + 140, CY + 182, 12, 12, GREY, r=2); t(x + 157, CY + 193, "band", 20, 400, INK2)

# 2. one EEG space (Section 3.2)
x = X[1]
rect(x + 10, CY + 52, 50, 34, TEAL_D, r=6); t(x + 35, CY + 75, "EEG", 20, 700, "#ffffff", "middle")
arrow(x + 62, CY + 69, x + 76, GREY, 5)
rect(x + 78, CY + 46, 76, 46, TEAL, r=6); t(x + 116, CY + 75, "rank 32", 20, 700, "#ffffff", "middle")
arrow(x + 156, CY + 69, x + 170, GREY, 5)
ow = kw - 182
rect(x + 172, CY + 44, ow, 22, NAVY, r=5); t(x + 172 + ow / 2, CY + 61, "fNIRS", 20, 700, "#ffffff", "middle")
rect(x + 172, CY + 72, ow, 22, NAVY, r=5); t(x + 172 + ow / 2, CY + 89, "fMRI", 20, 700, "#ffffff", "middle")
t(x + 12, CY + 136, "p = 0.0001", 34, 800, TEAL)
t(x + 12, CY + 164, "fMRI, 27 unseen people;", 20, 400, INK2)
t(x + 12, CY + 188, "fNIRS r 0.12–0.18", 20, 400, INK2)

# 3. calibration (Table 3)
x = X[2]
t(x + 12, CY + 80, "+0.06 to +0.18", 29, 800, TEAL)
t(x + 12, CY + 104, "gain over the person's", 20, 400, INK2)
t(x + 12, CY + 126, "own mean, 2 runs", 20, 400, INK2)
t(x + 12, CY + 166, "ρ 0.56–0.90", 31, 800, TEAL)
t(x + 12, CY + 190, "forecasts later accuracy", 20, 400, INK2)

# 4. timing (Table 4)
x = X[3]
t(x + 26, CY + 62, "EEG", 20, 700, INK, "middle"); t(x + kw - 26, CY + 62, "HbO", 20, 700, ORANGE, "middle")
S.append(f'<line x1="{x + 26}" y1="{CY + 80}" x2="{x + kw - 26}" y2="{CY + 80}" stroke="#c3cbd4" stroke-width="3.5"/>')
circle(x + 26, CY + 80, 10, TEAL_D); circle(x + kw - 26, CY + 80, 10, ORANGE)
t(x + kw / 2, CY + 110, "+6 s", 25, 800, INK, "middle")
t(x + 12, CY + 148, "29/29 people", 30, 800, TEAL)
t(x + 12, CY + 172, "delay repeats in 66%", 20, 400, INK2)
t(x + 12, CY + 194, "(chance 29%)", 20, 400, INK2)

# 5. transfer (Table 5, Sections 3.5 and 3.8)
x = X[4]
t(x + 12, CY + 64, "Within lab: holds", 22, 700, TEAL)
t(x + 12, CY + 87, "r 0.205 and 0.177", 20, 400, INK)
t(x + 12, CY + 120, "Across labs: fails", 22, 700, ORANGE)
t(x + 12, CY + 143, "none of 16 attempts", 20, 400, INK)
t(x + 12, CY + 176, "3rd lab: unresolved", 22, 700, ORANGE)
t(x + 12, CY + 198, "n = 15, not established", 20, 400, INK)

# Band strip, proportional to log frequency (hBCI deep network; Section 3.1)
BY = 322
t(20, BY + 18, "Deep network, hBCI: cross-person r carried by each EEG band", 21, 700, INK)
lo, hi = math.log10(0.01), math.log10(200)
xpos = lambda f: 20 + 835 * (math.log10(f) - lo) / (hi - lo)
segs = [(0.01, 0.1, "&lt; 0.1 Hz", 0.139, TEAL, "#ffffff", TEAL), (0.1, 0.5, "0.1–0.5 Hz", 0.053, TEAL_P, INK, TEAL),
        (0.5, 45, "0.5–45 Hz", 0.019, ORANGE, "#ffffff", ORANGE), (45, 200, "&gt; 45 Hz", 0.021, GREY, "#ffffff", MUTED)]
for f0, f1, lab, v, c, tc, vc in segs:
    x0, x1 = xpos(f0), xpos(f1)
    rect(x0 + 1, BY + 66, x1 - x0 - 2, 34, c, r=3)
    t((x0 + x1) / 2, BY + 89, lab, 20, 700, tc, "middle")
    t((x0 + x1) / 2, BY + 58, f"{v:.3f}", 29, 800, vc, "middle")
xa, xb, xc = xpos(0.01), xpos(0.5), xpos(45)
S.append(f'<path d="M{xa + 2:.1f},{BY + 106} v7 H{xb - 2:.1f} v-7" fill="none" stroke="{TEAL}" stroke-width="2"/>')
t((xa + xb) / 2, BY + 134, "&lt; 0.5 Hz together: 0.155 (98%)", 20, 700, TEAL, "middle")
t((xb + xc) / 2, BY + 134, "kept by the usual 0.5–45 Hz band-pass", 20, 700, ORANGE, "middle")

# Same network, filtered or not
rect(872, BY, 408, 140, PANEL, r=12)
t(1076, BY + 24, "same network, same people", 20, 400, MUTED, "middle")
t(970, BY + 82, "0.158", 50, 800, TEAL, "middle")
arrow(1050, BY + 66, 1104, NAVY, 7)
t(1184, BY + 82, "0.019", 50, 800, ORANGE, "middle")
t(970, BY + 108, "unfiltered", 20, 600, INK, "middle")
t(1184, BY + 108, "0.5–45 Hz filter", 20, 600, INK, "middle")
t(1076, BY + 132, "routine 0.1–1 Hz high-pass filters attenuate it", 20, 400, MUTED, "middle", italic=True)

# Conclusion
rect(20, 470, W - 40, 42, TEAL_D, r=10)
S.append(f'<text x="{W / 2}" y="497" font-size="21" fill="#ffffff" text-anchor="middle"><tspan font-weight="800">CONCLUSION: </tspan>'
         "infra-slow EEG carries person-specific haemodynamic information; two calibration runs enable EEG-only monitoring</text>")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="13cm" height="5.2cm" style="{F}">'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(S) + "</svg>")
html = ("<!doctype html><html><head><meta charset='utf-8'><style>@page{size:13cm 5.2cm;margin:0}"
        "html,body{margin:0;padding:0;background:#fff}svg{display:block}</style></head><body>" + svg + "</body></html>")
OUT.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as d:
    src = Path(d) / "ga.html"
    src.write_text(html, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT / 'graphical_abstract.pdf'}", src.as_uri()], check=True, capture_output=True)
    png = Path(d) / "ga.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=6.25",
                    "--window-size=492,197", f"--screenshot={png}", src.as_uri()], check=True, capture_output=True)
    im = Image.open(png).convert("RGB")
    im = im.crop((0, 0, min(round(13 / 2.54 * 600), im.width), min(round(5.2 / 2.54 * 600), im.height)))
    im.save(OUT / "graphical_abstract.tif", dpi=(600, 600), compression="tiff_lzw")
print("written", im.size)
