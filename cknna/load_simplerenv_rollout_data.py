"""
Phase 1 (SimplerEnv rollouts): Consolidate per-episode rollout data into CKNNA format.

Reads saved rollout data from maniskill2_evaluator's --save-rollout-dir output,
subsamples to N transitions, and outputs the standard CKNNA format compatible
with existing Phase 2 extraction scripts.

Input structure (per model):
  <rollout_dir>/<env_name>/episode_NNN/
    images/step_NNNN.png
    eef_poses.npy   (T, 8) float32
    env_actions.npy (T, 7) float32
    metadata.json

Output:
  <output_dir>/images/NNNNNN.png  -- individual PNG images
  <output_dir>/feats_B.pt        -- (N, 8) float32 eef_pos vectors
  <output_dir>/actions.pt        -- (N, 7) float32 action vectors
  <output_dir>/metadata.json     -- metadata

Usage:
  python load_simplerenv_rollout_data.py \
      --rollout_dir /data/rollouts/openvla_base \
      --output_dir /data/cknna/openvla_base \
      --num_samples 5000
"""

import argparse
import json
import os
import random
import shutil

import numpy as np
import torch


def collect_transitions(rollout_dir):
    """Walk all episode dirs and collect (image_path, eef_pose, env_action, task_desc) tuples."""
    transitions = []
    env_dirs = sorted([
        d for d in os.listdir(rollout_dir)
        if os.path.isdir(os.path.join(rollout_dir, d))
    ])
    for env_name in env_dirs:
        env_path = os.path.join(rollout_dir, env_name)
        ep_dirs = sorted([
            d for d in os.listdir(env_path)
            if os.path.isdir(os.path.join(env_path, d)) and d.startswith("episode_")
        ])
        for ep_dir_name in ep_dirs:
            ep_path = os.path.join(env_path, ep_dir_name)
            eef_path = os.path.join(ep_path, "eef_poses.npy")
            act_path = os.path.join(ep_path, "env_actions.npy")
            meta_path = os.path.join(ep_path, "metadata.json")
            img_dir = os.path.join(ep_path, "images")

            if not os.path.exists(eef_path) or not os.path.exists(act_path):
                print(f"  Skipping {ep_path}: missing npy files")
                continue

            eef_poses = np.load(eef_path)
            env_actions = np.load(act_path)

            with open(meta_path) as f:
                meta = json.load(f)
            task_desc = meta["task_description"]

            n_steps = min(len(eef_poses), len(env_actions))
            for step_idx in range(n_steps):
                img_path = os.path.join(img_dir, f"step_{step_idx:04d}.png")
                if not os.path.exists(img_path):
                    continue
                transitions.append({
                    "image_path": img_path,
                    "eef_pose": eef_poses[step_idx],
                    "env_action": env_actions[step_idx],
                    "task_description": task_desc,
                    "env_name": env_name,
                    "episode": ep_dir_name,
                })

    return transitions


def main():
    parser = argparse.ArgumentParser(description="Consolidate SimplerEnv rollout data for CKNNA.")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Root rollout dir for one model (contains env_name/episode_NNN/)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"=== Phase 1 (SimplerEnv rollouts): N={args.num_samples}, seed={args.seed} ===")
    print(f"Rollout dir: {args.rollout_dir}")
    print(f"Output dir:  {args.output_dir}")
    print()

    transitions = collect_transitions(args.rollout_dir)
    print(f"Total transitions found: {len(transitions)}")

    env_counts = {}
    for t in transitions:
        env_counts[t["env_name"]] = env_counts.get(t["env_name"], 0) + 1
    for env, cnt in sorted(env_counts.items()):
        print(f"  {env}: {cnt}")

    if len(transitions) <= args.num_samples:
        sampled = transitions
        print(f"\nUsing all {len(sampled)} transitions (fewer than {args.num_samples})")
    else:
        # Stratified sampling: proportional to task counts
        per_env = {}
        for t in transitions:
            per_env.setdefault(t["env_name"], []).append(t)

        sampled = []
        for env_name, env_trans in per_env.items():
            quota = int(round(args.num_samples * len(env_trans) / len(transitions)))
            quota = min(quota, len(env_trans))
            sampled.extend(random.sample(env_trans, quota))

        # Fill remaining slots if rounding left us short
        remaining = args.num_samples - len(sampled)
        if remaining > 0:
            used = set(id(s) for s in sampled)
            pool = [t for t in transitions if id(t) not in used]
            sampled.extend(random.sample(pool, min(remaining, len(pool))))

        random.shuffle(sampled)
        print(f"\nSubsampled to {len(sampled)} transitions")

    feats_B_list = []
    actions_list = []
    task_descriptions = []

    for i, t in enumerate(sampled):
        shutil.copy2(t["image_path"], os.path.join(images_dir, f"{i:06d}.png"))
        feats_B_list.append(t["eef_pose"])
        actions_list.append(t["env_action"])
        task_descriptions.append(t["task_description"])

        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  [{i+1}/{len(sampled)}]")

    feats_B = torch.tensor(np.stack(feats_B_list), dtype=torch.float32)
    torch.save(feats_B, os.path.join(args.output_dir, "feats_B.pt"))
    print(f"\nSaved feats_B: shape={tuple(feats_B.shape)}")

    actions = torch.tensor(np.stack(actions_list), dtype=torch.float32)
    torch.save(actions, os.path.join(args.output_dir, "actions.pt"))
    print(f"Saved actions: shape={tuple(actions.shape)}")

    metadata = {
        "num_samples": len(sampled),
        "state_dim": int(feats_B.shape[1]),
        "action_dim": int(actions.shape[1]),
        "state_keys": ["px", "py", "pz", "qw", "qx", "qy", "qz", "gripper_openness"],
        "source": "simplerenv_rollout",
        "rollout_dir": args.rollout_dir,
        "seed": args.seed,
        "sampling_strategy": "stratified_by_task",
        "task_descriptions": task_descriptions,
        "env_transition_counts": env_counts,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata.json")

    print(f"\n=== Phase 1 (SimplerEnv rollouts) complete ===")
    print(f"  feats_B:  {tuple(feats_B.shape)}")
    print(f"  actions:  {tuple(actions.shape)}")
    print(f"  images:   {images_dir}/  ({len(sampled)} files)")


if __name__ == "__main__":
    main()
