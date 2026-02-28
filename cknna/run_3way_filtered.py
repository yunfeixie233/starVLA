"""3-way CKNNA with empty-task-description filtering.

Filters out Bridge samples whose task_description is empty, then computes
CKNNA and MutualKNN for imgtext/img/txt feature variants.  Generates JSON
results, a combined CSV, and scatter + bar plots.

Usage:
    python run_3way_filtered.py [--device cuda]
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from compute_cknna_large import cknna_lowmem, mutual_knn_lowmem

DATA_DIR = os.path.join(SCRIPT_DIR, "cknna_data")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
RESULTS_DIR = os.path.join(DATA_DIR, "3way_cknna_results_filtered")
RECORD_DIR = os.path.join(SCRIPT_DIR, "record")
ORIG_CSV = os.path.join(RECORD_DIR, "cknna_action_proprio_simplerenv.csv")
OUT_CSV = os.path.join(RECORD_DIR, "cknna_3way_5k_filtered.csv")
PLOT_DIR = os.path.join(RECORD_DIR, "runs", "3way_5k_filtered")

TOPK_VALUES = [5, 10, 20]

MODELS_3WAY = [
    "Qwen-GR00T-Bridge",
    "Qwen-GR00T-Bridge-RT-1",
    "Qwen3VL-GR00T-Bridge-RT-1",
    "Qwen-FAST-Bridge-RT-1",
    "Qwen-OFT-Bridge-RT-1",
    "Qwen3VL-OFT-Bridge-RT-1",
    "spatialvla-sft-bridge",
    "pi0-lerobot-bridge",
    "openvla-7b-bridge",
    "openvla-7b-bridge-ft-200k",
    "cogact-small-bridge",
    "cogact-base-bridge",
    "cogact-large-bridge",
    "groot-n15-bridge",
    "groot-n16-bridge",
]

MODELS_IMGTEXT_ONLY = ["rt1x-bridge", "octo-base-bridge"]

DIR_TO_CSV = {
    "Qwen-GR00T-Bridge": "Qwen2.5-GR00T-Bridge",
    "Qwen-GR00T-Bridge-RT-1": "Qwen2.5-GR00T-Bridge-RT-1",
    "Qwen3VL-GR00T-Bridge-RT-1": "Qwen3-GR00T-Bridge-RT-1",
    "Qwen-FAST-Bridge-RT-1": "Qwen2.5-FAST-Bridge-RT-1",
    "Qwen-OFT-Bridge-RT-1": "Qwen2.5-OFT-Bridge-RT-1",
    "Qwen3VL-OFT-Bridge-RT-1": "Qwen3-OFT-Bridge-RT-1",
    "spatialvla-sft-bridge": "spatialvla-sft-bridge",
    "pi0-lerobot-bridge": "pi0-lerobot-bridge",
    "openvla-7b-bridge": "openvla-7b-bridge",
    "openvla-7b-bridge-ft-200k": "openvla-7b-bridge-ft-200k",
    "cogact-small-bridge": "CogACT-Small",
    "cogact-base-bridge": "CogACT-Base",
    "cogact-large-bridge": "CogACT-Large",
    "groot-n15-bridge": "GR00T-N1.5-Bridge",
    "groot-n16-bridge": "GR00T-N1.6-Bridge",
    "rt1x-bridge": "RT-1-X",
    "octo-base-bridge": "Octo-base",
}

SHORT = {
    "Qwen2.5-GR00T-Bridge": "GR00T",
    "Qwen2.5-GR00T-Bridge-RT-1": "GR00T-RT1",
    "Qwen3-GR00T-Bridge-RT-1": "Q3-GR00T",
    "Qwen2.5-FAST-Bridge-RT-1": "FAST",
    "Qwen2.5-OFT-Bridge-RT-1": "OFT",
    "Qwen3-OFT-Bridge-RT-1": "Q3-OFT",
    "spatialvla-sft-bridge": "SpatialVLA",
    "pi0-lerobot-bridge": "Pi0",
    "openvla-7b-bridge": "OpenVLA",
    "openvla-7b-bridge-ft-200k": "OpenVLA-FT",
    "CogACT-Small": "CogACT-S",
    "CogACT-Base": "CogACT-B",
    "CogACT-Large": "CogACT-L",
    "GR00T-N1.5-Bridge": "GR00T-1.5",
    "GR00T-N1.6-Bridge": "GR00T-1.6",
    "RT-1-X": "RT-1-X",
    "Octo-base": "Octo",
}


# ---------------------------------------------------------------------------
# Phase 1: Build filter
# ---------------------------------------------------------------------------

def build_non_empty_mask():
    with open(META_PATH) as f:
        meta = json.load(f)
    tasks = meta["task_descriptions"]
    idx = [i for i, t in enumerate(tasks) if t.strip()]
    print(f"Filter: {len(idx)}/{len(tasks)} samples with non-empty task descriptions")
    return torch.tensor(idx, dtype=torch.long)


# ---------------------------------------------------------------------------
# Phase 2: Compute CKNNA for all models
# ---------------------------------------------------------------------------

def compute_all(device):
    non_empty_idx = build_non_empty_mask()
    n_filtered = len(non_empty_idx)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feats_B_raw = torch.load(os.path.join(DATA_DIR, "feats_B.pt"), weights_only=True).float()
    feats_B_filtered = feats_B_raw[non_empty_idx]
    feats_B_norm = F.normalize(feats_B_filtered, p=2, dim=-1).to(device)
    del feats_B_raw, feats_B_filtered
    print(f"feats_B filtered shape: {tuple(feats_B_norm.shape)}")

    all_results = {}

    all_models = MODELS_3WAY + MODELS_IMGTEXT_ONLY
    for model_dir in all_models:
        csv_name = DIR_TO_CSV[model_dir]
        is_3way = model_dir in MODELS_3WAY
        variants = ["feats_A", "feats_A_img", "feats_A_txt"] if is_3way else ["feats_A"]

        model_path = os.path.join(DATA_DIR, model_dir)
        print(f"\n=== {csv_name} ({model_dir}) ===")

        entry = {"dir_name": model_dir}

        feats_action_path = os.path.join(model_path, "feats_action.pt")
        has_action = os.path.exists(feats_action_path)
        feats_action_norm = None
        if has_action:
            fa_raw = torch.load(feats_action_path, weights_only=True).float()
            fa_filtered = fa_raw[non_empty_idx]
            feats_action_norm = F.normalize(fa_filtered, p=2, dim=-1).to(device)
            del fa_raw, fa_filtered

        for variant in variants:
            suffix = {"feats_A": "imgtext", "feats_A_img": "img", "feats_A_txt": "txt"}[variant]
            fpath = os.path.join(model_path, f"{variant}.pt")
            feats_A_raw = torch.load(fpath, weights_only=True).float()
            feats_A_filtered = feats_A_raw[non_empty_idx]
            feats_A_norm = F.normalize(feats_A_filtered, p=2, dim=-1).to(device)
            del feats_A_raw, feats_A_filtered

            print(f"  {suffix}: shape {tuple(feats_A_norm.shape)}")

            for k in TOPK_VALUES:
                t0 = time.time()
                cknna_proprio = cknna_lowmem(feats_A_norm, feats_B_norm, topk=k)
                mknn_proprio = mutual_knn_lowmem(feats_A_norm, feats_B_norm, topk=k)
                elapsed = time.time() - t0
                entry[f"CKNNA_proprio_{suffix}_k{k}"] = cknna_proprio
                entry[f"MutualKNN_proprio_{suffix}_k{k}"] = mknn_proprio
                print(f"    proprio k={k}: CKNNA={cknna_proprio:.6f}  MutualKNN={mknn_proprio:.6f}  [{elapsed:.1f}s]")

                if has_action:
                    t0 = time.time()
                    cknna_action = cknna_lowmem(feats_A_norm, feats_action_norm, topk=k)
                    mknn_action = mutual_knn_lowmem(feats_A_norm, feats_action_norm, topk=k)
                    elapsed = time.time() - t0
                    entry[f"CKNNA_action_{suffix}_k{k}"] = cknna_action
                    entry[f"MutualKNN_action_{suffix}_k{k}"] = mknn_action
                    print(f"    action  k={k}: CKNNA={cknna_action:.6f}  MutualKNN={mknn_action:.6f}  [{elapsed:.1f}s]")

            del feats_A_norm
            torch.cuda.empty_cache()

        if feats_action_norm is not None:
            del feats_action_norm
            torch.cuda.empty_cache()

        all_results[csv_name] = entry

    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    out = {
        "_meta": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "N_original": 5000,
            "N_filtered": n_filtered,
            "filter": "non-empty task descriptions",
            "k_values": TOPK_VALUES,
        },
        "models": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON results saved to {results_path}")

    del feats_B_norm
    torch.cuda.empty_cache()
    return all_results


# ---------------------------------------------------------------------------
# Phase 3: Generate CSV
# ---------------------------------------------------------------------------

def generate_csv(all_results):
    orig_rows = {}
    with open(ORIG_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_rows[row["Model"]] = row

    fieldnames = [
        "Model", "dir_name", "WidowX_avg",
        "CKNNA_proprio_imgtext_k10", "CKNNA_proprio_img_k10", "CKNNA_proprio_txt_k10",
        "CKNNA_action_imgtext_k10", "CKNNA_action_img_k10", "CKNNA_action_txt_k10",
        "MutualKNN_proprio_imgtext_k10", "MutualKNN_proprio_img_k10", "MutualKNN_proprio_txt_k10",
        "MutualKNN_action_imgtext_k10", "MutualKNN_action_img_k10", "MutualKNN_action_txt_k10",
    ]

    rows = []
    all_dirs = MODELS_3WAY + MODELS_IMGTEXT_ONLY
    for model_dir in all_dirs:
        csv_name = DIR_TO_CSV[model_dir]
        orig = orig_rows.get(csv_name, {})
        success = orig.get("WidowX_avg_entire", "")
        entry = all_results.get(csv_name, {})

        row = {"Model": csv_name, "dir_name": model_dir, "WidowX_avg": success}
        for suffix in ["imgtext", "img", "txt"]:
            for metric in ["CKNNA_proprio", "CKNNA_action", "MutualKNN_proprio", "MutualKNN_action"]:
                key = f"{metric}_{suffix}_k10"
                row[key] = entry.get(key, "")
        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {OUT_CSV} ({len(rows)} models)")


# ---------------------------------------------------------------------------
# Phase 4: Generate plots
# ---------------------------------------------------------------------------

GROUPS = {
    "StarVLA-GR00T": {
        "models": [
            "Qwen2.5-GR00T-Bridge",
            "Qwen2.5-GR00T-Bridge-RT-1",
            "Qwen3-GR00T-Bridge-RT-1",
        ],
        "color": "#4477AA",
    },
    "StarVLA-Other": {
        "models": [
            "Qwen2.5-FAST-Bridge-RT-1",
            "Qwen2.5-OFT-Bridge-RT-1",
            "Qwen3-OFT-Bridge-RT-1",
        ],
        "color": "#EE6677",
    },
    "SpatialVLA/Pi0/OpenVLA": {
        "models": [
            "spatialvla-sft-bridge",
            "pi0-lerobot-bridge",
            "openvla-7b-bridge",
            "openvla-7b-bridge-ft-200k",
        ],
        "color": "#228833",
    },
    "CogACT": {
        "models": [
            "CogACT-Small",
            "CogACT-Base",
            "CogACT-Large",
        ],
        "color": "#AA3377",
    },
    "GR00T-NVIDIA": {
        "models": [
            "GR00T-N1.5-Bridge",
            "GR00T-N1.6-Bridge",
        ],
        "color": "#CCBB44",
    },
    "RT-1-X/Octo": {
        "models": [
            "RT-1-X",
            "Octo-base",
        ],
        "color": "#999999",
    },
}


def _load_csv_for_plot():
    rows = {}
    with open(OUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Model"]] = row
    return rows


def plot_scatter(data, metric_cols, title, ylabel, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    variant_labels = ["Image + Text", "Image Only", "Text Only"]

    for ax_idx, (col, vlabel) in enumerate(zip(metric_cols, variant_labels)):
        ax = axes[ax_idx]
        all_x, all_y = [], []

        for gname, ginfo in GROUPS.items():
            color = ginfo["color"]
            for model in ginfo["models"]:
                row = data.get(model)
                if row is None:
                    continue
                x_val = row.get(col, "")
                y_val = row.get("WidowX_avg", "")
                if x_val == "" or y_val == "":
                    continue
                x, y = float(x_val), float(y_val)
                label_txt = SHORT.get(model, model)
                ax.scatter(x, y, c=color, s=80, zorder=3,
                           edgecolors="black", linewidths=0.5, label=gname)
                ax.annotate(label_txt, (x, y), fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")
                all_x.append(x)
                all_y.append(y)

        if len(all_x) >= 3:
            rho, pval = stats.spearmanr(all_x, all_y)
            ax.set_title(f"{vlabel}\nSpearman rho={rho:.3f}, p={pval:.4f}", fontsize=11)
        else:
            ax.set_title(vlabel, fontsize=11)

        ax.set_xlabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("WidowX Success Rate (%)", fontsize=10)
    handles, labels = [], []
    for gname, ginfo in GROUPS.items():
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(color=ginfo["color"], label=gname))
    axes[2].legend(handles=handles, fontsize=7, loc="upper left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_bar(data, metric_prefix, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models_ordered = []
    for ginfo in GROUPS.values():
        models_ordered.extend(ginfo["models"])

    names, v_it, v_im, v_tx = [], [], [], []
    for m in models_ordered:
        row = data.get(m)
        if row is None:
            continue
        v1 = row.get(f"{metric_prefix}_imgtext_k10", "")
        v2 = row.get(f"{metric_prefix}_img_k10", "")
        v3 = row.get(f"{metric_prefix}_txt_k10", "")
        if v1 == "" and v2 == "" and v3 == "":
            continue
        names.append(SHORT.get(m, m))
        v_it.append(float(v1) if v1 else 0)
        v_im.append(float(v2) if v2 else 0)
        v_tx.append(float(v3) if v3 else 0)

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, v_it, w, label="Image+Text", color="#4477AA")
    ax.bar(x, v_im, w, label="Image Only", color="#EE6677")
    ax.bar(x + w, v_tx, w, label="Text Only", color="#228833")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score (k=10)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_plots():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = _load_csv_for_plot()

    plot_scatter(
        data,
        ["CKNNA_proprio_imgtext_k10", "CKNNA_proprio_img_k10", "CKNNA_proprio_txt_k10"],
        "CKNNA_proprio vs WidowX Success (k=10, N=3380 filtered)",
        "CKNNA_proprio (k=10)",
        os.path.join(PLOT_DIR, "scatter_CKNNA_proprio_3way.png"),
    )
    plot_scatter(
        data,
        ["CKNNA_action_imgtext_k10", "CKNNA_action_img_k10", "CKNNA_action_txt_k10"],
        "CKNNA_action vs WidowX Success (k=10, N=3380 filtered)",
        "CKNNA_action (k=10)",
        os.path.join(PLOT_DIR, "scatter_CKNNA_action_3way.png"),
    )
    plot_scatter(
        data,
        ["MutualKNN_proprio_imgtext_k10", "MutualKNN_proprio_img_k10", "MutualKNN_proprio_txt_k10"],
        "MutualKNN_proprio vs WidowX Success (k=10, N=3380 filtered)",
        "MutualKNN_proprio (k=10)",
        os.path.join(PLOT_DIR, "scatter_MutualKNN_proprio_3way.png"),
    )
    plot_scatter(
        data,
        ["MutualKNN_action_imgtext_k10", "MutualKNN_action_img_k10", "MutualKNN_action_txt_k10"],
        "MutualKNN_action vs WidowX Success (k=10, N=3380 filtered)",
        "MutualKNN_action (k=10)",
        os.path.join(PLOT_DIR, "scatter_MutualKNN_action_3way.png"),
    )
    plot_bar(
        data, "CKNNA_proprio",
        "CKNNA_proprio (k=10, filtered): Image+Text vs Image vs Text",
        os.path.join(PLOT_DIR, "bar_CKNNA_proprio_3way.png"),
    )
    plot_bar(
        data, "CKNNA_action",
        "CKNNA_action (k=10, filtered): Image+Text vs Image vs Text",
        os.path.join(PLOT_DIR, "bar_CKNNA_action_3way.png"),
    )
    print(f"\nAll plots saved to {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_compute", action="store_true",
                        help="Skip CKNNA computation, only regenerate CSV/plots from existing JSON")
    args = parser.parse_args()

    if args.skip_compute:
        results_path = os.path.join(RESULTS_DIR, "all_results.json")
        with open(results_path) as f:
            data = json.load(f)
        all_results = data["models"]
    else:
        all_results = compute_all(args.device)

    generate_csv(all_results)
    generate_plots()


if __name__ == "__main__":
    main()
