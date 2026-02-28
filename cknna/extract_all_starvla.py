"""
Unified single-pass extraction: feats_A (imgtext/img/txt) + feats_action for StarVLA.

Loads the model ONCE, runs a single forward pass per sample, and captures both
VLM hidden states (for feats_A) and action representations (for feats_action).

For GR00T/PI: The VLM call in the action pipeline is identical to feats_A.
              We reuse hidden_states[-1] for feats_A masking, then run denoising.
For FAST:     generate() includes prefill. hidden_states[0][-1] = feats_A.
              Generated token hidden states = feats_action.
For OFT:      Two VLM calls needed (feats_A uses bare instruction, feats_action
              appends action tokens). Still saves 1 model load vs 2 scripts.

All hooks are read-only: .detach() on captured tensors, no tensor modification.

Usage:
  python extract_all_starvla.py \
      --ckpt_path playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt \
      --data_dir ./cknna_data_50k \
      --output_dir ./cknna_data_50k/Qwen-GR00T-Bridge
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

IMAGE_TOKEN_INDEX = 151655


def masked_mean_pool(hidden_states, attention_mask):
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return -1


def build_task_mask(input_ids_1d, tokenizer, task):
    mask = torch.zeros(len(input_ids_1d), dtype=torch.long, device=input_ids_1d.device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start = find_subsequence(ids_list, task_ids)
        if start >= 0:
            mask[start:start + len(task_ids)] = 1
            return mask
    return mask


def extract_vlm_features(last_hidden, attention_mask, input_ids, tokenizer, instruction):
    """From VLM hidden states, produce (feat_imgtext, feat_img, feat_txt)."""
    feat_imgtext = masked_mean_pool(last_hidden, attention_mask).squeeze(0).cpu()
    image_mask = (input_ids == IMAGE_TOKEN_INDEX).to(attention_mask.dtype)
    feat_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()
    task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
    feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()
    return feat_imgtext, feat_img, feat_txt


def get_state_for_model(model, feats_B_7d):
    if not hasattr(model, "action_model"):
        return None
    se = getattr(model.action_model, "state_encoder", None)
    if se is None:
        return None
    expected_dim = se.layer1.in_features
    n = feats_B_7d.shape[0]
    if expected_dim == feats_B_7d.shape[1]:
        return feats_B_7d
    result = np.zeros((n, expected_dim), dtype=np.float32)
    copy_len = min(expected_dim, feats_B_7d.shape[1])
    result[:, :copy_len] = feats_B_7d[:, :copy_len]
    return result


# ---------------------------------------------------------------------------
# Per-family single-pass extractors
# ---------------------------------------------------------------------------

def extract_single_pass_groot(model, images_pil, instruction, state_vec, tokenizer):
    """GR00T: 1 VLM call + 1 denoising loop. VLM hidden states reused for feats_A."""
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil], instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = model.qwen_vl_interface(
            **qwen_inputs, output_hidden_states=True, return_dict=True,
        )
        last_hidden = qwenvl_outputs.hidden_states[-1]

    feat_imgtext, feat_img, feat_txt = extract_vlm_features(
        last_hidden, qwen_inputs["attention_mask"], qwen_inputs["input_ids"],
        tokenizer, instruction,
    )

    action_reprs = []
    def pre_hook_fn(module, inputs):
        action_reprs.append(inputs[0].detach())

    handle = model.action_model.action_decoder.register_forward_pre_hook(pre_hook_fn)

    state_tensor = None
    if state_vec is not None and model.action_model.state_encoder is not None:
        state_tensor = torch.from_numpy(state_vec[np.newaxis, np.newaxis, :]).to(
            device=last_hidden.device, dtype=last_hidden.dtype
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        model.action_model.predict_action(last_hidden, state_tensor)

    handle.remove()

    final_repr = action_reprs[-1]
    action_horizon = model.action_model.action_horizon
    action_part = final_repr[:, -action_horizon:, :]
    feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt, feat_action


def extract_single_pass_pi(model, images_pil, instruction, state_vec, tokenizer):
    """PI: 1 VLM call + 1 denoising loop (layerwise). VLM hidden states reused."""
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil], instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = model.qwen_vl_interface(
            **qwen_inputs, output_hidden_states=True, return_dict=True,
        )
        all_hidden = qwenvl_outputs.hidden_states
        last_hidden = all_hidden[-1]
        expected_layers = len(model.action_model.model.transformer_blocks)
        vl_embs_list = list(all_hidden[-expected_layers:])

    feat_imgtext, feat_img, feat_txt = extract_vlm_features(
        last_hidden, qwen_inputs["attention_mask"], qwen_inputs["input_ids"],
        tokenizer, instruction,
    )

    action_reprs = []
    def pre_hook_fn(module, inputs):
        action_reprs.append(inputs[0].detach())

    handle = model.action_model.action_decoder.register_forward_pre_hook(pre_hook_fn)

    state_tensor = None
    if state_vec is not None and model.action_model.state_encoder is not None:
        state_tensor = torch.from_numpy(state_vec[np.newaxis, np.newaxis, :]).to(
            device=last_hidden.device, dtype=last_hidden.dtype
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        model.action_model.predict_action(vl_embs_list, state_tensor)

    handle.remove()

    final_repr = action_reprs[-1]
    action_horizon = model.action_model.action_horizon
    action_part = final_repr[:, -action_horizon:, :]
    feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt, feat_action


def extract_single_pass_fast(model, images_pil, instruction, tokenizer):
    """FAST: 1 generate() call. Prefill hidden states -> feats_A, generated -> feats_action."""
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil], instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface.model.generate(
            **qwen_inputs,
            max_length=2048,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    prefill_hidden = outputs.hidden_states[0][-1]
    feat_imgtext, feat_img, feat_txt = extract_vlm_features(
        prefill_hidden, qwen_inputs["attention_mask"], qwen_inputs["input_ids"],
        tokenizer, instruction,
    )

    num_gen_steps = len(outputs.hidden_states) - 1
    if num_gen_steps == 0:
        hidden_size = model.qwen_vl_interface.model.config.hidden_size
        feat_action = torch.zeros(hidden_size, dtype=torch.float32)
    else:
        action_hidden = [outputs.hidden_states[t][-1] for t in range(1, len(outputs.hidden_states))]
        action_repr = torch.cat(action_hidden, dim=1)
        feat_action = action_repr.float().mean(dim=1).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt, feat_action


def extract_single_pass_oft(model, images_pil, instruction, tokenizer):
    """OFT: 2 VLM calls (different inputs), but 1 model load. Not truly single-pass."""
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil], instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs, output_hidden_states=True, return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]
    feat_imgtext, feat_img, feat_txt = extract_vlm_features(
        last_hidden, qwen_inputs["attention_mask"], qwen_inputs["input_ids"],
        tokenizer, instruction,
    )

    action_tokens = model.action_token * model.chunk_len
    prompt_suffix = f" Please predict the next {model.chunk_len} robot actions: <action>{action_tokens}<action>."
    instruction_with_actions = instruction + prompt_suffix

    qwen_inputs_act = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil], instructions=[instruction_with_actions],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs_act = model.qwen_vl_interface(
            **qwen_inputs_act, output_hidden_states=True, return_dict=True,
        )

    last_hidden_act = outputs_act.hidden_states[-1]
    input_ids_act = qwen_inputs_act["input_ids"]
    action_queries = model._gather_action_token_embeddings(
        last_hidden_act, input_ids_act, action_token_id=model.action_token_id
    )
    feat_action = action_queries.float().mean(dim=1).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt, feat_action


FRAMEWORK_DISPATCH = {
    "Qwenvl_Fast": "fast",
    "Qwenvl_OFT": "oft",
    "Qwen_GR00T": "groot",
    "Qwen_PI": "pi",
}


def main():
    parser = argparse.ArgumentParser(description="Unified single-pass extraction for StarVLA.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    feats_B = torch.load(os.path.join(args.data_dir, "feats_B.pt"), weights_only=True)

    print(f"Loading checkpoint: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()

    framework_class = type(model).__name__
    family = FRAMEWORK_DISPATCH[framework_class]
    tokenizer = model.qwen_vl_interface.processor.tokenizer
    vlm_hidden_size = model.qwen_vl_interface.model.config.hidden_size
    print(f"  Framework: {framework_class} -> family: {family}")
    print(f"  VLM hidden size: {vlm_hidden_size}")
    print(f"  Mode: single-pass (feats_A + feats_action)")
    print(f"  Samples: {num_samples}")

    state_all = get_state_for_model(model, feats_B.numpy())

    feats_imgtext_list, feats_img_list, feats_txt_list, feats_action_list = [], [], [], []

    partial_path = os.path.join(args.output_dir, "extract_all_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial = torch.load(partial_path, weights_only=True)
        feats_imgtext_list = list(partial["feats_A"][:args.resume_from])
        feats_img_list = list(partial["feats_A_img"][:args.resume_from])
        feats_txt_list = list(partial["feats_A_txt"][:args.resume_from])
        feats_action_list = list(partial["feats_action"][:args.resume_from])
        print(f"  Resumed from sample {args.resume_from}")
    else:
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")
        instruction = task_descriptions[i]
        state_vec = state_all[i] if state_all is not None else None

        torch.manual_seed(args.seed + i)

        if family == "groot":
            f_it, f_im, f_tx, f_act = extract_single_pass_groot(
                model, [img], instruction, state_vec, tokenizer)
        elif family == "pi":
            f_it, f_im, f_tx, f_act = extract_single_pass_pi(
                model, [img], instruction, state_vec, tokenizer)
        elif family == "fast":
            f_it, f_im, f_tx, f_act = extract_single_pass_fast(
                model, [img], instruction, tokenizer)
        elif family == "oft":
            f_it, f_im, f_tx, f_act = extract_single_pass_oft(
                model, [img], instruction, tokenizer)

        feats_imgtext_list.append(f_it)
        feats_img_list.append(f_im)
        feats_txt_list.append(f_tx)
        feats_action_list.append(f_act)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  vlm=({vlm_hidden_size},) act=({f_act.shape[0]},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

        if (i + 1) % args.save_every == 0:
            partial_data = {
                "feats_A": torch.stack(feats_imgtext_list),
                "feats_A_img": torch.stack(feats_img_list),
                "feats_A_txt": torch.stack(feats_txt_list),
                "feats_action": torch.stack(feats_action_list),
            }
            torch.save(partial_data, partial_path)

    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list),
                          ("feats_action", feats_action_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "checkpoint": args.ckpt_path,
        "framework": framework_class,
        "family": family,
        "vlm_hidden_size": vlm_hidden_size,
        "num_samples": num_samples,
        "data_dir": args.data_dir,
        "seed": args.seed,
        "mode": "single_pass",
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt", "feats_action.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Single-Pass Extraction Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
