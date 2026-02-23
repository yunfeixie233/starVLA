"""
Phase 2: Extract VLM features (feats_A) from StarVLA models for CKNNA.

Loads a StarVLA checkpoint, runs a VLM-only forward pass on each image
from Phase 1 data, and saves the mean-pooled last hidden state as feats_A.

Works identically for all 4 frameworks (QwenFast, QwenOFT, QwenGR00T, QwenPI)
because they all share the same qwen_vl_interface.

The extraction is purely read-only: no hooks, no model mutation, no side effects.

Usage:
  python extract_features_starvla.py \
      --ckpt_path playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt \
      --data_dir ./cknna_data \
      --output_dir ./cknna_data/Qwen-GR00T-Bridge
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

STARVLA_ROOT = os.path.join(os.path.dirname(__file__), "..")
if STARVLA_ROOT not in sys.path:
    sys.path.insert(0, STARVLA_ROOT)


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens.

    Args:
        hidden_states: (B, seq_len, D) bfloat16 or float32
        attention_mask: (B, seq_len) int or bool

    Returns:
        pooled: (B, D) float32
    """
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def extract_feat_a(model, images_pil, instruction, device):
    """Run VLM prefill and extract mean-pooled last hidden state.

    This function is shared across all 4 StarVLA frameworks.
    It calls qwen_vl_interface directly, without going through
    the framework's forward() or predict_action() methods.

    Args:
        model: A loaded StarVLA framework instance.
        images_pil: List[PIL.Image] for a single sample (e.g., [img_0]).
        instruction: str task instruction.
        device: torch.device

    Returns:
        feat_a: (D,) float32 tensor on CPU.
    """
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil],
        instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]
    attention_mask = qwen_inputs["attention_mask"]
    pooled = masked_mean_pool(last_hidden, attention_mask)
    return pooled.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract StarVLA features.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 output directory (contains images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this sample index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = os.path.join(args.data_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading checkpoint: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()

    framework_class = type(model).__name__
    vlm_hidden_size = model.qwen_vl_interface.model.config.hidden_size
    print(f"  Framework: {framework_class}")
    print(f"  VLM hidden size: {vlm_hidden_size}")
    print(f"  Samples to process: {num_samples}")

    partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        feats_list = torch.load(partial_path, weights_only=True).tolist()
        feats_list = [torch.tensor(f) for f in feats_list[:args.resume_from]]
        print(f"  Resuming from sample {args.resume_from}")
    else:
        feats_list = []
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")

        instruction = task_descriptions[i]

        feat_a = extract_feat_a(model, [img], instruction, args.device)
        feats_list.append(feat_a)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - args.resume_from) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  feat_a shape=({vlm_hidden_size},)  "
                  f"rate={rate:.1f} samples/s  ETA={eta/60:.1f}min")

        if (i + 1) % 500 == 0:
            partial = torch.stack(feats_list)
            torch.save(partial, partial_path)

    feats_A = torch.stack(feats_list)
    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    torch.save(feats_A, feats_A_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "checkpoint": args.ckpt_path,
        "framework": framework_class,
        "vlm_hidden_size": vlm_hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "data_dir": args.data_dir,
    }
    meta_path = os.path.join(args.output_dir, "extraction_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
