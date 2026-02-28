"""50K Full Sweep: CKNNA/MutualKNN for 3 feats_A variants x 5 feats_B variants.

feats_A variants: imgtext, img, txt
feats_B variants:
  (a) proprio_t          -- feats_B.pt at time t
  (b) proprio_seq_h{a}   -- feats_B_seq.pt[:, :a+1, :] flattened
  (c) feats_action_t     -- model_dir/feats_action.pt
  (d) real_action_t      -- actions.pt at time t
  (e) real_action_seq_h{a} -- actions_seq.pt[:, :a+1, :] flattened

Filters out Bridge samples with empty task descriptions before computation.
Generates JSON, long-format CSV, and plots.

Usage:
    python run_50k_cknna_computation.py [--device cuda] [--skip_compute]
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

DATA_DIR = os.path.join(SCRIPT_DIR, "cknna_data_50k")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
RESULTS_DIR = os.path.join(DATA_DIR, "50k_full_sweep_results")
RECORD_DIR = os.path.join(SCRIPT_DIR, "record")
ORIG_CSV = os.path.join(RECORD_DIR, "cknna_action_proprio_simplerenv.csv")
OUT_CSV = os.path.join(RECORD_DIR, "cknna_50k_full_sweep.csv")
PLOT_DIR = os.path.join(RECORD_DIR, "runs", "50k_full_sweep")

TOPK_VALUES = [5, 10, 20]
HORIZONS = [1, 3, 7, 15]

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


def build_non_empty_mask():
    with open(META_PATH) as f:
        meta = json.load(f)
    tasks = meta["task_descriptions"]
    idx = [i for i, t in enumerate(tasks) if t.strip()]
    print(f"Filter: {len(idx)}/{len(tasks)} samples with non-empty task descriptions")
    return torch.tensor(idx, dtype=torch.long)


def load_and_filter(path, non_empty_idx, device):
    raw = torch.load(path, weights_only=True).float()
    filtered = raw[non_empty_idx]
    normed = F.normalize(filtered, p=2, dim=-1).to(device)
    del raw, filtered
    return normed


def load_seq_and_filter(path, non_empty_idx, horizon, device):
    """Load sequential tensor (N, H+1, D), slice to horizon, flatten, filter, normalize."""
    raw = torch.load(path, weights_only=True).float()
    sliced = raw[:, :horizon + 1, :]
    flat = sliced.reshape(sliced.shape[0], -1)
    filtered = flat[non_empty_idx]
    normed = F.normalize(filtered, p=2, dim=-1).to(device)
    del raw, sliced, flat, filtered
    return normed


def compute_pair(feats_A_norm, feats_B_norm, k):
    t0 = time.time()
    cknna_val = cknna_lowmem(feats_A_norm, feats_B_norm, topk=k)
    mknn_val = mutual_knn_lowmem(feats_A_norm, feats_B_norm, topk=k)
    elapsed = time.time() - t0
    return cknna_val, mknn_val, elapsed


def compute_all(device):
    non_empty_idx = build_non_empty_mask()
    n_filtered = len(non_empty_idx)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feats_B_path = os.path.join(DATA_DIR, "feats_B.pt")
    actions_path = os.path.join(DATA_DIR, "actions.pt")
    feats_B_seq_path = os.path.join(DATA_DIR, "feats_B_seq.pt")
    actions_seq_path = os.path.join(DATA_DIR, "actions_seq.pt")

    has_seq = os.path.exists(feats_B_seq_path) and os.path.exists(actions_seq_path)
    if not has_seq:
        print("WARNING: feats_B_seq.pt / actions_seq.pt not found, skipping temporal variants")

    all_rows = []

    all_models = MODELS_3WAY + MODELS_IMGTEXT_ONLY
    for model_dir in all_models:
        csv_name = DIR_TO_CSV[model_dir]
        is_3way = model_dir in MODELS_3WAY
        a_variants = ["imgtext", "img", "txt"] if is_3way else ["imgtext"]

        model_path = os.path.join(DATA_DIR, model_dir)
        if not os.path.isdir(model_path):
            print(f"\n--- SKIP {csv_name}: directory {model_path} not found ---")
            continue
        print(f"\n{'='*60}\n  {csv_name} ({model_dir})\n{'='*60}")

        feats_action_path = os.path.join(model_path, "feats_action.pt")
        has_action = os.path.exists(feats_action_path)

        variant_map = {"imgtext": "feats_A", "img": "feats_A_img", "txt": "feats_A_txt"}

        for a_var in a_variants:
            fpath = os.path.join(model_path, f"{variant_map[a_var]}.pt")
            if not os.path.exists(fpath):
                print(f"  SKIP feats_A variant={a_var}: {fpath} not found")
                continue

            feats_A_norm = load_and_filter(fpath, non_empty_idx, device)
            print(f"  feats_A({a_var}): shape {tuple(feats_A_norm.shape)}")

            # (a) proprio at t
            print(f"  --- feats_B = proprio_t ---")
            feats_B_norm = load_and_filter(feats_B_path, non_empty_idx, device)
            for k in TOPK_VALUES:
                cknna_val, mknn_val, elapsed = compute_pair(feats_A_norm, feats_B_norm, k)
                print(f"    k={k}: CKNNA={cknna_val:.6f}  MutualKNN={mknn_val:.6f}  [{elapsed:.1f}s]")
                all_rows.append({
                    "model": csv_name, "dir_name": model_dir,
                    "feats_A": a_var, "feats_B": "proprio_t", "horizon": 0, "k": k,
                    "CKNNA": cknna_val, "MutualKNN": mknn_val,
                })
            del feats_B_norm
            torch.cuda.empty_cache()

            # (b) proprio from t to t+a
            if has_seq:
                for h in HORIZONS:
                    print(f"  --- feats_B = proprio_seq_h{h} ---")
                    feats_B_norm = load_seq_and_filter(feats_B_seq_path, non_empty_idx, h, device)
                    for k in TOPK_VALUES:
                        cknna_val, mknn_val, elapsed = compute_pair(feats_A_norm, feats_B_norm, k)
                        print(f"    k={k}: CKNNA={cknna_val:.6f}  MutualKNN={mknn_val:.6f}  [{elapsed:.1f}s]")
                        all_rows.append({
                            "model": csv_name, "dir_name": model_dir,
                            "feats_A": a_var, "feats_B": f"proprio_seq_h{h}", "horizon": h, "k": k,
                            "CKNNA": cknna_val, "MutualKNN": mknn_val,
                        })
                    del feats_B_norm
                    torch.cuda.empty_cache()

            # (c) feats_action at t
            if has_action:
                print(f"  --- feats_B = feats_action_t ---")
                feats_action_norm = load_and_filter(feats_action_path, non_empty_idx, device)
                for k in TOPK_VALUES:
                    cknna_val, mknn_val, elapsed = compute_pair(feats_A_norm, feats_action_norm, k)
                    print(f"    k={k}: CKNNA={cknna_val:.6f}  MutualKNN={mknn_val:.6f}  [{elapsed:.1f}s]")
                    all_rows.append({
                        "model": csv_name, "dir_name": model_dir,
                        "feats_A": a_var, "feats_B": "feats_action_t", "horizon": 0, "k": k,
                        "CKNNA": cknna_val, "MutualKNN": mknn_val,
                    })
                del feats_action_norm
                torch.cuda.empty_cache()

            # (d) real action at t
            print(f"  --- feats_B = real_action_t ---")
            feats_B_norm = load_and_filter(actions_path, non_empty_idx, device)
            for k in TOPK_VALUES:
                cknna_val, mknn_val, elapsed = compute_pair(feats_A_norm, feats_B_norm, k)
                print(f"    k={k}: CKNNA={cknna_val:.6f}  MutualKNN={mknn_val:.6f}  [{elapsed:.1f}s]")
                all_rows.append({
                    "model": csv_name, "dir_name": model_dir,
                    "feats_A": a_var, "feats_B": "real_action_t", "horizon": 0, "k": k,
                    "CKNNA": cknna_val, "MutualKNN": mknn_val,
                })
            del feats_B_norm
            torch.cuda.empty_cache()

            # (e) real action from t to t+a
            if has_seq:
                for h in HORIZONS:
                    print(f"  --- feats_B = real_action_seq_h{h} ---")
                    feats_B_norm = load_seq_and_filter(actions_seq_path, non_empty_idx, h, device)
                    for k in TOPK_VALUES:
                        cknna_val, mknn_val, elapsed = compute_pair(feats_A_norm, feats_B_norm, k)
                        print(f"    k={k}: CKNNA={cknna_val:.6f}  MutualKNN={mknn_val:.6f}  [{elapsed:.1f}s]")
                        all_rows.append({
                            "model": csv_name, "dir_name": model_dir,
                            "feats_A": a_var, "feats_B": f"real_action_seq_h{h}", "horizon": h, "k": k,
                            "CKNNA": cknna_val, "MutualKNN": mknn_val,
                        })
                    del feats_B_norm
                    torch.cuda.empty_cache()

            del feats_A_norm
            torch.cuda.empty_cache()

    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    out = {
        "_meta": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "N_original": 50000,
            "N_filtered": n_filtered,
            "filter": "non-empty task descriptions",
            "k_values": TOPK_VALUES,
            "horizons": HORIZONS,
            "feats_B_variants": ["proprio_t", "proprio_seq", "feats_action_t", "real_action_t", "real_action_seq"],
        },
        "rows": all_rows,
    }
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON results saved to {results_path}")

    return all_rows


def generate_csv(all_rows):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    orig_rows = {}
    if os.path.exists(ORIG_CSV):
        with open(ORIG_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_rows[row["Model"]] = row

    fieldnames = ["model", "dir_name", "WidowX_avg", "feats_A", "feats_B", "horizon", "k", "CKNNA", "MutualKNN"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            orig = orig_rows.get(row["model"], {})
            success = orig.get("WidowX_avg_entire", "")
            out_row = {
                "model": row["model"],
                "dir_name": row["dir_name"],
                "WidowX_avg": success,
                "feats_A": row["feats_A"],
                "feats_B": row["feats_B"],
                "horizon": row["horizon"],
                "k": row["k"],
                "CKNNA": row["CKNNA"],
                "MutualKNN": row["MutualKNN"],
            }
            writer.writerow(out_row)
    print(f"CSV saved to {OUT_CSV} ({len(all_rows)} rows)")


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
        "models": ["CogACT-Small", "CogACT-Base", "CogACT-Large"],
        "color": "#AA3377",
    },
    "GR00T-NVIDIA": {
        "models": ["GR00T-N1.5-Bridge", "GR00T-N1.6-Bridge"],
        "color": "#CCBB44",
    },
    "RT-1-X/Octo": {
        "models": ["RT-1-X", "Octo-base"],
        "color": "#999999",
    },
}


def _rows_to_pivot(all_rows):
    """Convert long-format rows into {model: {col_key: value}} for plotting."""
    pivot = {}
    for row in all_rows:
        model = row["model"]
        if model not in pivot:
            pivot[model] = {}
        k = row["k"]
        fA = row["feats_A"]
        fB = row["feats_B"]
        pivot[model][f"CKNNA_{fB}_{fA}_k{k}"] = row["CKNNA"]
        pivot[model][f"MutualKNN_{fB}_{fA}_k{k}"] = row["MutualKNN"]
    return pivot


def _get_success_rates():
    rates = {}
    if os.path.exists(ORIG_CSV):
        with open(ORIG_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("WidowX_avg_entire", "")
                if val:
                    rates[row["Model"]] = float(val)
    return rates


def plot_scatter(pivot, success_rates, metric_cols, title, ylabel, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    n_cols = len(metric_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax_idx, (col, vlabel) in enumerate(metric_cols):
        ax = axes[ax_idx]
        all_x, all_y = [], []

        for gname, ginfo in GROUPS.items():
            color = ginfo["color"]
            for model in ginfo["models"]:
                if model not in pivot or model not in success_rates:
                    continue
                x_val = pivot[model].get(col)
                if x_val is None:
                    continue
                y_val = success_rates[model]
                label_txt = SHORT.get(model, model)
                ax.scatter(x_val, y_val, c=color, s=80, zorder=3,
                           edgecolors="black", linewidths=0.5)
                ax.annotate(label_txt, (x_val, y_val), fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")
                all_x.append(x_val)
                all_y.append(y_val)

        if len(all_x) >= 3:
            rho, pval = stats.spearmanr(all_x, all_y)
            ax.set_title(f"{vlabel}\nrho={rho:.3f}, p={pval:.4f}", fontsize=11)
        else:
            ax.set_title(vlabel, fontsize=11)

        ax.set_xlabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("WidowX Success Rate (%)", fontsize=10)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=g["color"], label=n) for n, g in GROUPS.items()]
    axes[-1].legend(handles=handles, fontsize=7, loc="upper left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_temporal_curve(pivot, success_rates, feats_B_base, feats_A_var, metric_name, out_path):
    """Line plot: x=horizon, y=metric, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    hs = HORIZONS

    for gname, ginfo in GROUPS.items():
        color = ginfo["color"]
        for model in ginfo["models"]:
            if model not in pivot:
                continue
            vals = []
            for h in hs:
                key = f"{metric_name}_{feats_B_base}_h{h}_{feats_A_var}_k10"
                v = pivot[model].get(key)
                vals.append(v)

            if all(v is None for v in vals):
                continue

            label_txt = SHORT.get(model, model)
            xs = [h for h, v in zip(hs, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.plot(xs, ys, marker="o", markersize=5, label=label_txt, color=color, linewidth=1.5)

    ax.set_xlabel("Horizon (a)", fontsize=11)
    ax.set_ylabel(f"{metric_name} (k=10)", fontsize=11)
    ax.set_title(f"{metric_name} vs Horizon ({feats_B_base}, feats_A={feats_A_var})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_plots(all_rows):
    os.makedirs(PLOT_DIR, exist_ok=True)
    pivot = _rows_to_pivot(all_rows)
    success_rates = _get_success_rates()

    feats_B_variants_t = ["proprio_t", "feats_action_t", "real_action_t"]
    for fB in feats_B_variants_t:
        for metric in ["CKNNA", "MutualKNN"]:
            cols = [
                (f"{metric}_{fB}_imgtext_k10", "Image+Text"),
                (f"{metric}_{fB}_img_k10", "Image Only"),
                (f"{metric}_{fB}_txt_k10", "Text Only"),
            ]
            fB_label = fB.replace("_", " ")
            plot_scatter(
                pivot, success_rates, cols,
                f"{metric} vs Success ({fB_label}, k=10, 50K filtered)",
                f"{metric} (k=10)",
                os.path.join(PLOT_DIR, f"scatter_{metric}_{fB}_3way.png"),
            )

    for feats_B_base in ["proprio_seq", "real_action_seq"]:
        for feats_A_var in ["imgtext", "img", "txt"]:
            for metric in ["CKNNA", "MutualKNN"]:
                plot_temporal_curve(
                    pivot, success_rates, feats_B_base, feats_A_var, metric,
                    os.path.join(PLOT_DIR, f"temporal_{metric}_{feats_B_base}_{feats_A_var}.png"),
                )

    print(f"\nAll plots saved to {PLOT_DIR}/")


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
        all_rows = data["rows"]
    else:
        all_rows = compute_all(args.device)

    generate_csv(all_rows)
    generate_plots(all_rows)


if __name__ == "__main__":
    main()
