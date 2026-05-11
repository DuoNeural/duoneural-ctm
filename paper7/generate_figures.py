#!/usr/bin/env python3
"""
Paper 7 — Figure Generation
Archon, DuoNeural, 2026-05-11

Generates all figures for paper7.tex from local result JSON files.
Run from the paper7/ directory or pass --results-dir.

Outputs to paper7/figures/:
  fig1_direction_traces.pdf   — 4-model direction trace (magnitude + xfer)
  fig2_crossdomain.pdf        — safety vs political comparison
  fig3_instillation_sweep.pdf — Qwen amplify + Mistral bidirectional sweep
  fig4_2x2_table.pdf          — visual 2×2 alignment matrix
"""

import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results", help="Path to results dir")
args = parser.parse_args()

RESULTS = Path(args.results_dir)
FIGDIR  = Path("figures")
FIGDIR.mkdir(exist_ok=True)

# ── COLOR SCHEME ──────────────────────────────────────────────────────────────
SUPP_COLOR   = "#D6604D"   # red — suppressors
CRYST_COLOR  = "#4393C3"   # blue — crystallizers
SAFE_COLOR   = "#1A9641"   # green — safety domain
POL_COLOR    = "#D6604D"   # red — political domain
NEUTRAL      = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 2.0,
})


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with open(RESULTS / "qwen7b_trace/direction_trace.json") as f:
    qwen7b = json.load(f)["trace"]
with open(RESULTS / "mistral7b_trace/direction_trace.json") as f:
    mistral7b = json.load(f)["trace"]
with open(RESULTS / "mistral_instill/results.json") as f:
    mistral_instill = json.load(f)
with open(RESULTS / "paper7_exp_results/all_results.json") as f:
    paper7 = json.load(f)

# Qwen14B + MistralNeMo traces from embedded data (already known, replicate key points)
# Using the values from ccs_direction_trace_results.json / mistral_direction_trace.json
# Hard-coded from validated results for figure generation
qwen14b_mag = [
    0.0, 0.6, 0.7, 0.7, 0.9, 1.1, 1.2, 1.5, 1.6, 1.8,
    2.0, 2.1, 2.7, 3.1, 3.6, 4.1, 4.5, 5.3, 6.7, 7.8,
    8.2, 10.3, 12.4, 16.5, 18.2, 21.1, 23.3, 25.9, 32.9, 36.9,
    40.3, 50.6, 58.1, 64.7, 81.8, 92.9, 101.0, 108.5, 121.0, 134.0,
    141.2, 150.8, 160.8, 169.3, 181.4, 193.5, 209.2, 229.9, 68.7
]
qwen14b_xfer = [
    0.0, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8,
    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9,
    0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
    0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8
]
mistral_nemo_mag = [
    0.0, 11.3, 11.5, 11.2, 11.8, 11.4, 11.6, 12.0, 11.9, 12.1,
    12.3, 12.0, 12.4, 12.2, 12.5, 12.8, 12.6, 12.9, 13.1, 13.0,
    13.2, 13.4, 13.1, 13.3, 13.5, 13.2, 13.4, 13.6, 13.3, 13.5,
    13.7, 13.4, 13.6, 13.4, 13.5, 13.3, 13.4, 13.2, 13.3, 13.4, 45.9
]
mistral_nemo_xfer = [
    0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0
]

qwen7b_mag   = [r["magnitude"] for r in qwen7b]
qwen7b_xfer  = [r["exp_acc"]   for r in qwen7b]
m7b_mag      = [r["magnitude"] for r in mistral7b]
m7b_xfer     = [r["exp_acc"]   for r in mistral7b]

# Safety domain trace from paper7 exp3
safety_trace = paper7["exp3_cross_domain_safety"]["trace"]
safety_mag   = [r["magnitude"] for r in safety_trace]
safety_xfer  = [r["xfer_acc"]  for r in safety_trace]


# ── FIG 1: DIRECTION TRACES ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
fig.suptitle("Direction Trace: Suppressor vs. Crystallizer", fontsize=14, fontweight="bold", y=1.01)

panel_data = [
    ("Qwen2.5-14B (Suppressor)",  qwen14b_mag,     qwen14b_xfer,     SUPP_COLOR),
    ("Qwen2.5-7B (Weak Suppressor)", qwen7b_mag,   qwen7b_xfer,      SUPP_COLOR + "99"),
    ("Mistral-NeMo-12B (Crystallizer)", mistral_nemo_mag, mistral_nemo_xfer, CRYST_COLOR),
    ("Mistral-7B (Strong Crystallizer)", m7b_mag,  m7b_xfer,         CRYST_COLOR + "99"),
]

for ax, (title, mag, xfer, color) in zip(axes.flat, panel_data):
    layers = list(range(len(mag)))
    ax2 = ax.twinx()
    ax.fill_between(layers, mag, alpha=0.15, color=color)
    ax.plot(layers, mag, color=color, linewidth=2, label="|r| magnitude")
    ax2.plot(layers, xfer, color="black", linewidth=1.5, linestyle="--",
             alpha=0.8, label="xfer acc")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|r| (truth direction magnitude)", color=color)
    ax2.set_ylabel("Transfer accuracy", color="black")
    ax2.set_ylim(-0.05, 1.15)
    ax2.tick_params(axis='y', labelcolor='black')
    ax.tick_params(axis='y', labelcolor=color)
    # Mark final layer
    n = len(mag) - 1
    ax.axvline(x=n, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.axvline(x=n-1, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.annotate("final\nlayer", xy=(n-0.5, max(mag)*0.5),
                ha="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(FIGDIR / "fig1_direction_traces.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig1_direction_traces.png", bbox_inches="tight", dpi=150)
plt.close()
print("[fig1] direction traces saved")


# ── FIG 2: CROSS-DOMAIN ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cross-Domain Generalization: Universal Bottleneck + Alignment-Specific Rotation",
             fontsize=12, fontweight="bold")

# Left: magnitude comparison (political=Qwen14B, safety=paper7 exp3)
layers_pol = list(range(len(qwen14b_mag)))
layers_saf = list(range(len(safety_mag)))

ax1.plot(layers_pol, qwen14b_mag, color=POL_COLOR, linewidth=2.2, label="Political domain (Qwen14B)")
ax1.plot(layers_saf, safety_mag,  color=SAFE_COLOR, linewidth=2.2, label="Safety domain (Qwen14B)")
ax1.axvspan(47, 48, alpha=0.12, color="gray", label="L47→L48 bottleneck")

# Annotate compression
ax1.annotate("", xy=(48, safety_mag[48]), xytext=(47, safety_mag[47]),
             arrowprops=dict(arrowstyle="->", color=SAFE_COLOR, lw=2))
ax1.annotate("", xy=(48, qwen14b_mag[48]), xytext=(47, qwen14b_mag[47]),
             arrowprops=dict(arrowstyle="->", color=POL_COLOR, lw=2))
ax1.text(47.3, (safety_mag[47]+safety_mag[48])/2, "3.35×\ncompress",
         ha="left", fontsize=8, color="gray")

ax1.set_xlabel("Layer"); ax1.set_ylabel("|r| magnitude")
ax1.set_title("Magnitude: Both domains compressed 3.35×")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_xlim(-1, 49)

# Right: xfer comparison
ax2.plot(layers_pol, qwen14b_xfer, color=POL_COLOR, linewidth=2.2,
         label="Political (peaks L47, drops post-bottleneck)")
ax2.plot(layers_saf, safety_xfer,  color=SAFE_COLOR, linewidth=2.2,
         label="Safety (peaks L12, survives bottleneck)")
ax2.axvspan(47, 48, alpha=0.12, color="gray")
ax2.axvline(x=12, color=SAFE_COLOR, linestyle=":", alpha=0.5, linewidth=1.2)
ax2.text(12.2, 1.05, "Safety peak\nL12=1.00", color=SAFE_COLOR, fontsize=8)
ax2.axhline(y=0.80, color=POL_COLOR, linestyle=":", alpha=0.5, linewidth=1.2)
ax2.text(2, 0.78, "Political floor\n@L48=0.80", color=POL_COLOR, fontsize=8)

ax2.set_xlabel("Layer"); ax2.set_ylabel("Transfer accuracy")
ax2.set_title("Transfer Accuracy: Safety survives, Political drops")
ax2.set_ylim(-0.05, 1.15)
ax2.legend(loc="lower right", fontsize=8)
ax2.set_xlim(-1, 49)

plt.tight_layout()
plt.savefig(FIGDIR / "fig2_crossdomain.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig2_crossdomain.png", bbox_inches="tight", dpi=150)
plt.close()
print("[fig2] cross-domain saved")


# ── FIG 3: INSTILLATION SWEEP ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Instillation Sweep: Suppressor Recovers, Crystallizer Is Fragile",
             fontsize=12, fontweight="bold")

# Left: Qwen14B amplify sweep (from R3 results)
qwen_alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
qwen_xfer   = [0.80, 0.80, 0.80, 0.90, 0.90]
ax1.plot(qwen_alphas, qwen_xfer, color=SUPP_COLOR, marker="o", markersize=8,
         linewidth=2.2, label="Qwen14B (amplify)")
ax1.axhline(y=0.80, color=SUPP_COLOR, linestyle=":", alpha=0.5, linewidth=1)
ax1.text(0.1, 0.81, "baseline=0.80", color=SUPP_COLOR, fontsize=9)
ax1.fill_between(qwen_alphas, 0.80, qwen_xfer, alpha=0.15, color=SUPP_COLOR)
ax1.set_xlabel("α (amplification scale)"); ax1.set_ylabel("Transfer accuracy @ L48")
ax1.set_title("Suppressor (Qwen2.5-14B)\nAmplify restores truth")
ax1.set_ylim(0.6, 1.05)
ax1.legend()

# Right: Mistral-NeMo subtract + amplify
mistral_sub_alphas  = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
mistral_sub_xfer    = [1.00, 1.00, 1.00, 1.00, 0.70, 0.60]
mistral_amp_alphas  = [0.0, 0.5, 1.0, 2.0, 5.0]
mistral_amp_xfer    = [1.00, 0.90, 0.60, 0.50, 0.50]

ax2.plot(mistral_sub_alphas, mistral_sub_xfer, color=CRYST_COLOR, marker="s",
         markersize=8, linewidth=2.2, label="Mistral-NeMo (subtract)")
ax2.plot(mistral_amp_alphas, mistral_amp_xfer, color=CRYST_COLOR, marker="^",
         markersize=8, linewidth=2.2, linestyle="--", label="Mistral-NeMo (amplify)")
ax2.axhline(y=1.00, color=CRYST_COLOR, linestyle=":", alpha=0.5, linewidth=1)
ax2.text(0.1, 1.01, "baseline=1.00", color=CRYST_COLOR, fontsize=9)
ax2.fill_between(mistral_sub_alphas, mistral_sub_xfer, 1.00, alpha=0.1, color=CRYST_COLOR)
ax2.fill_between(mistral_amp_alphas, mistral_amp_xfer, 1.00, alpha=0.1, color="red")
ax2.set_xlabel("α"); ax2.set_ylabel("Transfer accuracy @ L40")
ax2.set_title("Crystallizer (Mistral-NeMo-12B)\nAlready optimal — any perturbation hurts")
ax2.set_ylim(0.3, 1.1)
ax2.legend()

plt.tight_layout()
plt.savefig(FIGDIR / "fig3_instillation_sweep.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig3_instillation_sweep.png", bbox_inches="tight", dpi=150)
plt.close()
print("[fig3] instillation sweep saved")


# ── FIG 4: 2×2 ALIGNMENT MATRIX ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
fig.suptitle("The Suppressor–Crystallizer Dichotomy: 2×2 Cross-Model Comparison",
             fontsize=12, fontweight="bold")

# Grid lines
for x in [0, 1, 2]: ax.axvline(x, color="black", linewidth=1.5)
for y in [0, 1, 2]: ax.axhline(y, color="black", linewidth=1.5)

# Headers
ax.text(0.5, 2.12, "7B Scale",  ha="center", fontsize=11, fontweight="bold")
ax.text(1.5, 2.12, "12–14B Scale", ha="center", fontsize=11, fontweight="bold")
ax.text(-0.22, 1.5, "Chinese\nRLHF",  ha="center", va="center", fontsize=11,
        fontweight="bold", rotation=90)
ax.text(-0.22, 0.5, "Western\nRLHF", ha="center", va="center", fontsize=11,
        fontweight="bold", rotation=90)

# Cell backgrounds
from matplotlib.patches import FancyBboxPatch
for (x, y), color in [((0,1), SUPP_COLOR), ((1,1), SUPP_COLOR),
                       ((0,0), CRYST_COLOR), ((1,0), CRYST_COLOR)]:
    rect = plt.Rectangle((x+0.02, y+0.02), 0.96, 0.96,
                          facecolor=color, alpha=0.18, zorder=0)
    ax.add_patch(rect)

# Cell content
cells = {
    (0, 1): ("Qwen2.5-7B", "WEAK SUPPRESSOR", "1.28× compress", "xfer: 1.00→0.70", SUPP_COLOR),
    (1, 1): ("Qwen2.5-14B", "STRONG SUPPRESSOR", "3.35× compress", "xfer: 0.90→0.80", SUPP_COLOR),
    (0, 0): ("Mistral-7B-v0.3", "STRONG CRYSTALLIZER", "9.9× amplify", "xfer: 1.00→1.00", CRYST_COLOR),
    (1, 0): ("Mistral-NeMo-12B", "CRYSTALLIZER", "3.4× amplify", "xfer: 0.60→1.00", CRYST_COLOR),
}
for (x, y), (model, verdict, ratio, xfer_str, color) in cells.items():
    cx, cy = x + 0.5, y + 0.5
    ax.text(cx, cy+0.25, model, ha="center", va="center", fontsize=9,
            color="black", style="italic")
    ax.text(cx, cy+0.05, verdict, ha="center", va="center", fontsize=10,
            color=color, fontweight="bold")
    ax.text(cx, cy-0.12, ratio, ha="center", va="center", fontsize=9, color=color)
    ax.text(cx, cy-0.28, xfer_str, ha="center", va="center", fontsize=9,
            color="gray")

plt.tight_layout()
plt.savefig(FIGDIR / "fig4_2x2_table.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig4_2x2_table.png", bbox_inches="tight", dpi=150)
plt.close()
print("[fig4] 2×2 table saved")

print(f"\nAll figures saved to {FIGDIR}/")
print("Run 'pdflatex paper7.tex' twice to compile.")
