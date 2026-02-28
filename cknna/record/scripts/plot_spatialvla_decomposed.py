"""
Scatter: CKNNA_proprio vs WidowX SR, with SpatialVLA decomposed into
X (SigLIP), P' (ZoeDepth), X+P' (pre-LM), and post-LM variants.

Produces two figures:
  1. All models + decomposed SpatialVLA variants on same plot
  2. Focused view: SpatialVLA variants only, with horizontal reference lines

Uses the same visual style as plot_grouped_by_backbone.py.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

csv_path = os.path.join(os.path.dirname(__file__), "..", "cknna_action_proprio_simplerenv.csv")
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

SPATIALVLA_VARIANTS = {
    "spatialvla-sft-bridge",
    "spatialvla-X-only",
    "spatialvla-P-only",
    "spatialvla-XP-preLM",
}

EXCLUDE = {"RT-1-X", "Octo-base"}

GROUPS = [
    {
        "label": "Qwen2.5-VL-3B",
        "color": "#1f77b4",
        "marker": "o",
        "size": 70,
        "models": {
            "Qwen2.5-GR00T-Bridge",
            "Qwen2.5-GR00T-Bridge-RT-1",
            "Qwen2.5-FAST-Bridge-RT-1",
            "Qwen2.5-OFT-Bridge-RT-1",
        },
    },
    {
        "label": "Qwen3-VL-4B",
        "color": "#17becf",
        "marker": "^",
        "size": 80,
        "models": {
            "Qwen3-GR00T-Bridge-RT-1",
            "Qwen3-OFT-Bridge-RT-1",
        },
    },
    {
        "label": "Prismatic+Llama2-7B",
        "color": "#ff7f0e",
        "marker": "D",
        "size": 75,
        "models": {
            "openvla-7b-bridge",
            "openvla-7b-bridge-ft-200k",
            "CogACT-Small",
            "CogACT-Base",
            "CogACT-Large",
        },
    },
    {
        "label": "NVEagle",
        "color": "#9467bd",
        "marker": "*",
        "size": 120,
        "models": {
            "GR00T-N1.5-Bridge",
            "GR00T-N1.6-Bridge",
        },
    },
    {
        "label": "PaLiGemma (Pi0)",
        "color": "#2ca02c",
        "marker": "v",
        "size": 85,
        "models": {"pi0-lerobot-bridge"},
    },
    {
        "label": "SpatialVLA post-LM (X+P'->Gemma2)",
        "color": "#d62728",
        "marker": "s",
        "size": 110,
        "models": {"spatialvla-sft-bridge"},
    },
    {
        "label": "SpatialVLA X only (SigLIP)",
        "color": "#e377c2",
        "marker": "P",
        "size": 110,
        "models": {"spatialvla-X-only"},
    },
    {
        "label": "SpatialVLA P' only (ZoeDepth)",
        "color": "#8c564b",
        "marker": "X",
        "size": 110,
        "models": {"spatialvla-P-only"},
    },
    {
        "label": "SpatialVLA X+P' pre-LM",
        "color": "#bcbd22",
        "marker": "h",
        "size": 110,
        "models": {"spatialvla-XP-preLM"},
    },
]

DISPLAY_NAME = {
    "Qwen2.5-GR00T-Bridge": "GR00T-Bridge",
    "Qwen2.5-GR00T-Bridge-RT-1": "GR00T-Bridge-RT1",
    "Qwen2.5-FAST-Bridge-RT-1": "FAST-Bridge-RT1",
    "Qwen2.5-OFT-Bridge-RT-1": "OFT-Bridge-RT1",
    "Qwen3-GR00T-Bridge-RT-1": "Q3-GR00T-RT1",
    "Qwen3-OFT-Bridge-RT-1": "Q3-OFT-RT1",
    "spatialvla-sft-bridge": "SpatialVLA post-LM",
    "spatialvla-X-only": "SpatialVLA X (SigLIP)",
    "spatialvla-P-only": "SpatialVLA P' (ZoeDepth)",
    "spatialvla-XP-preLM": "SpatialVLA X+P' pre-LM",
    "pi0-lerobot-bridge": "Pi0",
    "openvla-7b-bridge": "OpenVLA-Base",
    "openvla-7b-bridge-ft-200k": "OpenVLA-FT-200k",
    "CogACT-Small": "CogACT-S",
    "CogACT-Base": "CogACT-B",
    "CogACT-Large": "CogACT-L",
    "GR00T-N1.5-Bridge": "GR00T-N1.5",
    "GR00T-N1.6-Bridge": "GR00T-N1.6",
}

outdir = os.path.join(os.path.dirname(__file__), "..", "runs", "spatialvla_decomposed")
os.makedirs(outdir, exist_ok=True)

k = 10
col = "CKNNA_proprio_k%d" % k
ylabel = "CKNNA (VLM, Proprio)  k=%d" % k


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# --- Figure 1: All models + decomposed variants ---
fig, ax = plt.subplots(figsize=(12, 8))
xs_all, ys_all = [], []
texts = []

for grp in GROUPS:
    xs, ys, labs = [], [], []
    for r in rows:
        name = r["Model"]
        if name not in grp["models"]:
            continue
        if name in EXCLUDE:
            continue
        if r[col] in ("N/A", "") or r["WidowX_avg_entire"] in ("N/A", ""):
            continue
        x = float(r["WidowX_avg_entire"])
        y = float(r[col])
        xs.append(x)
        ys.append(y)
        labs.append(DISPLAY_NAME.get(name, name))

    if not xs:
        continue

    ax.scatter(xs, ys, s=grp["size"], zorder=5,
               marker=grp["marker"], edgecolors="k", linewidths=0.6,
               color=grp["color"], label=grp["label"])

    for x, y, lab in zip(xs, ys, labs):
        t = ax.text(x + 0.8, y, lab, fontsize=9, fontweight="bold",
                    color="0.25", zorder=10, va="center")
        texts.append(t)

    if name not in SPATIALVLA_VARIANTS:
        xs_all.extend(xs)
        ys_all.extend(ys)

if xs_all:
    xs_arr = np.array(xs_all)
    ys_arr = np.array(ys_all)
    rho, pval = spearmanr(xs_arr, ys_arr)
    slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
    xline = np.linspace(0, 80, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray",
            linewidth=1, zorder=3, label="Trend (excl. SpatialVLA variants)")

ax.axvline(x=42.7, color="#d62728", linestyle=":", alpha=0.3, linewidth=1)

ax.set_title("SpatialVLA Decomposition: X (SigLIP) vs P' (ZoeDepth) vs Post-LM\n"
             "CKNNA (VLM, Proprio) k=%d  |  N=5000 Bridge samples" % k,
             fontsize=13, fontweight="bold")
ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=12, fontweight="bold")
ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
ax.tick_params(labelsize=11)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9,
          prop={"weight": "bold"}, ncol=2)
fig.tight_layout()

fname = "spatialvla_decomposed_CKNNA_proprio_k%d.png" % k
out = os.path.join(outdir, fname)
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)


# --- Figure 2: Bar chart of SpatialVLA variants ---
variant_data = []
for r in rows:
    name = r["Model"]
    if name not in SPATIALVLA_VARIANTS:
        continue
    if r[col] in ("N/A", ""):
        continue
    variant_data.append({
        "name": DISPLAY_NAME.get(name, name),
        "cknna_k5": float(r["CKNNA_proprio_k5"]),
        "cknna_k10": float(r["CKNNA_proprio_k10"]),
        "cknna_k20": float(r["CKNNA_proprio_k20"]),
    })

fig, ax = plt.subplots(figsize=(10, 6))
names = [d["name"] for d in variant_data]
x_pos = np.arange(len(names))
width = 0.25

bars_k5 = [d["cknna_k5"] for d in variant_data]
bars_k10 = [d["cknna_k10"] for d in variant_data]
bars_k20 = [d["cknna_k20"] for d in variant_data]

colors = ["#d62728", "#e377c2", "#8c564b", "#bcbd22"]

ax.bar(x_pos - width, bars_k5, width, label="k=5", color=colors, edgecolor="k", linewidth=0.5, alpha=0.9)
ax.bar(x_pos, bars_k10, width, label="k=10", color=colors, edgecolor="k", linewidth=0.5, alpha=0.7)
ax.bar(x_pos + width, bars_k20, width, label="k=20", color=colors, edgecolor="k", linewidth=0.5, alpha=0.5)

for i, (v5, v10, v20) in enumerate(zip(bars_k5, bars_k10, bars_k20)):
    ax.text(i - width, v5 + 0.002, "%.4f" % v5, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(i, v10 + 0.002, "%.4f" % v10, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(i + width, v20 + 0.002, "%.4f" % v20, ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10, fontweight="bold")
ax.set_ylabel("CKNNA (VLM, Proprio)", fontsize=12, fontweight="bold")
ax.set_title("SpatialVLA Feature Decomposition: CKNNA_proprio\n"
             "Post-LM (2304d) vs Pre-LM X/P'/X+P' (1152d)  |  N=5000",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right", prop={"weight": "bold"})
ax.tick_params(labelsize=10)
fig.tight_layout()

fname = "spatialvla_decomposed_bar_k_all.png"
out = os.path.join(outdir, fname)
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)


# --- Figure 3: Same scatter as plot_grouped_by_backbone but with decomposed variants ---
fig, ax = plt.subplots(figsize=(10, 7))
xs_all, ys_all = [], []

for grp in GROUPS:
    xs, ys, labs = [], [], []
    for r in rows:
        name = r["Model"]
        if name not in grp["models"]:
            continue
        if name in EXCLUDE:
            continue
        if r[col] in ("N/A", "") or r["WidowX_avg_entire"] in ("N/A", ""):
            continue
        x = float(r["WidowX_avg_entire"])
        y = float(r[col])
        xs.append(x)
        ys.append(y)
        labs.append(DISPLAY_NAME.get(name, name))

    if not xs:
        continue

    ax.scatter(xs, ys, s=grp["size"], zorder=5,
               marker=grp["marker"], edgecolors="k", linewidths=0.6,
               color=grp["color"], label=grp["label"])

    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="0.3", fontweight="bold")

    xs_all.extend(xs)
    ys_all.extend(ys)

xs_arr = np.array(xs_all)
ys_arr = np.array(ys_all)
rho, pval = spearmanr(xs_arr, ys_arr)
slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
xline = np.linspace(xs_arr.min() - 2, xs_arr.max() + 2, 100)
ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=1, zorder=3)

ax.set_title("CKNNA (VLM, Proprio) k=%d  -- with SpatialVLA decomposed\n"
             "Spearman rho=%.3f  p=%.4f %s  (N=%d)"
             % (k, rho, pval, _sig(pval), len(xs_all)),
             fontsize=13, fontweight="bold")
ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=13, fontweight="bold")
ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
ax.tick_params(labelsize=12)
ax.legend(fontsize=8, loc="best", framealpha=0.9, prop={"weight": "bold"}, ncol=2)
fig.tight_layout()

fname = "vla-only_CKNNA_proprio_k%d_decomposed.png" % k
out = os.path.join(outdir, fname)
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)

print("\nAll figures saved to:", outdir)
