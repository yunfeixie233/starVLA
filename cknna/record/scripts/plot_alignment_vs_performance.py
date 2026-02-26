import csv
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

csv_path = os.path.join(os.path.dirname(__file__), "..", "cknna_action_proprio_simplerenv.csv")
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

k = 10
drop_action = {"RT-1-X", "Octo-base"}

panels = [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio)  k=%d" % k, False),
    ("CKNNA_action_k%d" % k, "CKNNA (VLM, Action)  k=%d" % k, True),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d" % k, False),
    ("MutualKNN_action_k%d" % k, "Mutual k-NN (VLM, Action)  k=%d" % k, True),
]

SHORT = {
    "Qwen-GR00T-Bridge": "GR00T",
    "Qwen-GR00T-Bridge-RT-1": "GR00T-RT1",
    "Qwen3VL-GR00T-Bridge-RT-1": "Q3-GR00T",
    "Qwen-FAST-Bridge-RT-1": "FAST",
    "Qwen-OFT-Bridge-RT-1": "OFT",
    "Qwen3VL-OFT-Bridge-RT-1": "Q3-OFT",
    "spatialvla-sft-bridge": "SpatialVLA",
    "pi0-lerobot-bridge": "Pi0",
    "openvla-7b-bridge": "OpenVLA",
    "RT-1-X": "RT-1-X",
    "Octo-base": "Octo",
}

timestamp = time.strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(os.path.dirname(__file__), "..", "runs", timestamp)
os.makedirs(outdir, exist_ok=True)

for idx, (col, ylabel, do_drop) in enumerate(panels):
    fig, ax = plt.subplots(figsize=(6, 5))

    xs, ys, labels = [], [], []
    for r in rows:
        name = r["Model"]
        if do_drop and name in drop_action:
            continue
        x = float(r["WidowX_avg_entire"])
        y = float(r[col])
        xs.append(x)
        ys.append(y)
        labels.append(SHORT[name])

    xs = np.array(xs)
    ys = np.array(ys)

    ax.scatter(xs, ys, s=60, zorder=5, edgecolors="k", linewidths=0.5)

    rho, pval = spearmanr(xs, ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min() - 2, xs.max() + 2, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=1, zorder=3)

    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="0.3")

    n = len(xs)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
    ax.set_title("%s\nSpearman rho=%.3f  p=%.4f %s  (N=%d)" % (ylabel, rho, pval, sig, n),
                 fontsize=10)
    ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    fname = "%s_k%d.png" % (col, k)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

print("All figures saved to:", outdir)
