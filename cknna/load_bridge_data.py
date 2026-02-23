"""
Phase 1: Load Bridge dataset for CKNNA evaluation.

Downloads a subset of the IPEC-COMMUNITY/bridge_orig_lerobot dataset from
HuggingFace and extracts (image, proprioceptive_state, task_description) tuples.

State is 8D: [x, y, z, roll, pitch, yaw, pad, gripper].
We drop index 6 (pad, always 0) to get 7D feats_B.

Output:
  <output_dir>/feats_B.pt        -- (N, 7) float32 state vectors
  <output_dir>/images/NNNNNN.png -- individual PNG images
  <output_dir>/metadata.json     -- metadata (N, indices, tasks, ...)

Usage:
  python load_bridge_data.py --output_dir ./cknna_data --num_samples 5000
"""

import argparse
import json
import os
import random

import av
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_tree
from PIL import Image as PILImage


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
STATE_DIM_RAW = 8
STATE_DIM_EFFECTIVE = 7
PAD_INDEX = 6


def download_parquet_chunk(chunk_idx, cache_dir):
    """Download all parquet files in a data chunk."""
    prefix = f"data/chunk-{chunk_idx:03d}"
    items = list(list_repo_tree(DATASET_REPO, repo_type="dataset", path_in_repo=prefix))
    parquet_files = [it.path for it in items if it.path.endswith(".parquet")]
    local_paths = []
    for pf in parquet_files:
        lp = hf_hub_download(DATASET_REPO, pf, repo_type="dataset", cache_dir=cache_dir)
        local_paths.append(lp)
    return local_paths


def download_video_file(episode_index, chunk_idx, cache_dir):
    """Download a single episode video for image_0."""
    video_path = f"videos/chunk-{chunk_idx:03d}/observation.images.image_0/episode_{episode_index:06d}.mp4"
    return hf_hub_download(DATASET_REPO, video_path, repo_type="dataset", cache_dir=cache_dir)


def download_tasks(cache_dir):
    """Download tasks.jsonl to map task_index -> description."""
    path = hf_hub_download(DATASET_REPO, "meta/tasks.jsonl", repo_type="dataset", cache_dir=cache_dir)
    tasks = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


def load_all_frames_from_video(video_path):
    """Load all frames from an mp4 video file using PyAV.

    Returns a list of numpy arrays (H, W, 3) uint8 RGB.
    """
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(arr)
    container.close()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Load Bridge data for CKNNA.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_chunks", type=int, default=3,
                        help="Number of data chunks to download (1000 eps each)")
    parser.add_argument("--total_chunks", type=int, default=54,
                        help="Total number of chunks in the dataset (for even spacing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Downloading tasks metadata...")
    tasks = download_tasks(args.cache_dir)
    print(f"  Found {len(tasks)} tasks")

    step = max(1, args.total_chunks // args.num_chunks)
    chunk_indices = [i * step for i in range(args.num_chunks) if i * step < args.total_chunks]
    print(f"Selected chunks (evenly spaced): {chunk_indices}")

    all_frames = []
    for chunk_idx in chunk_indices:
        print(f"\nDownloading parquet chunk {chunk_idx}...")
        parquet_paths = download_parquet_chunk(chunk_idx, args.cache_dir)
        print(f"  Downloaded {len(parquet_paths)} parquet files")

        for pp in parquet_paths:
            df = pd.read_parquet(pp)
            ep_arr = df["episode_index"].values.astype(np.int64)
            fr_arr = df["frame_index"].values.astype(np.int64)
            ti_arr = df["task_index"].values.astype(np.int64)
            st_arr = np.stack(df["observation.state"].values).astype(np.float32)
            for j in range(len(df)):
                all_frames.append({
                    "episode_index": int(ep_arr[j]),
                    "frame_index": int(fr_arr[j]),
                    "state": st_arr[j],
                    "task_index": int(ti_arr[j]),
                    "chunk_idx": chunk_idx,
                })

    print(f"\nTotal frames available: {len(all_frames)}")

    num_to_sample = min(args.num_samples, len(all_frames))
    sampled_indices = sorted(random.sample(range(len(all_frames)), num_to_sample))
    sampled_frames = [all_frames[i] for i in sampled_indices]
    print(f"Sampled {num_to_sample} frames")

    episodes_needed = set()
    for sf in sampled_frames:
        episodes_needed.add((sf["episode_index"], sf["chunk_idx"]))
    print(f"Need videos from {len(episodes_needed)} unique episodes")

    ep_last_use = {}
    for i, sf in enumerate(sampled_frames):
        ep_last_use[(sf["episode_index"], sf["chunk_idx"])] = i

    print("\nDownloading video files and extracting frames...")
    frames_cache = {}
    states_list = []
    task_descriptions = []

    for i, sf in enumerate(sampled_frames):
        ep_key = (sf["episode_index"], sf["chunk_idx"])

        if ep_key not in frames_cache:
            video_path = download_video_file(sf["episode_index"], sf["chunk_idx"], args.cache_dir)
            frames_cache[ep_key] = load_all_frames_from_video(video_path)

        ep_frames = frames_cache[ep_key]
        fidx = sf["frame_index"]
        if fidx >= len(ep_frames):
            print(f"  WARNING: frame_index {fidx} >= num_frames {len(ep_frames)} for ep {sf['episode_index']}, using last frame")
            fidx = len(ep_frames) - 1
        frame = ep_frames[fidx]

        img = PILImage.fromarray(frame)
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img.save(img_path)

        raw_state = sf["state"]
        assert raw_state.shape == (STATE_DIM_RAW,), f"Expected {STATE_DIM_RAW}D state, got {raw_state.shape}"
        if i == 0:
            pad_val = raw_state[PAD_INDEX]
            assert pad_val == 0.0, f"state.pad (idx {PAD_INDEX}) should be 0.0, got {pad_val}"
        state_7d = np.concatenate([raw_state[:PAD_INDEX], raw_state[PAD_INDEX + 1:]])
        states_list.append(state_7d)

        task_desc = tasks[sf["task_index"]]
        task_descriptions.append(task_desc)

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1}/{num_to_sample}] ep={sf['episode_index']} frame={sf['frame_index']} state_7d={state_7d[:3].tolist()}...")

        if ep_last_use[ep_key] == i:
            del frames_cache[ep_key]

    feats_B = torch.tensor(np.stack(states_list), dtype=torch.float32)
    feats_B_path = os.path.join(args.output_dir, "feats_B.pt")
    torch.save(feats_B, feats_B_path)
    print(f"\nSaved feats_B: shape={tuple(feats_B.shape)} to {feats_B_path}")

    metadata = {
        "num_samples": num_to_sample,
        "state_dim_raw": STATE_DIM_RAW,
        "state_dim_effective": STATE_DIM_EFFECTIVE,
        "pad_index_dropped": PAD_INDEX,
        "state_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "dataset_repo": DATASET_REPO,
        "num_chunks_used": args.num_chunks,
        "chunk_indices": chunk_indices,
        "seed": args.seed,
        "task_descriptions": task_descriptions,
        "episode_indices": [sf["episode_index"] for sf in sampled_frames],
        "frame_indices": [sf["frame_index"] for sf in sampled_frames],
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n=== Phase 1 Complete ===")
    print(f"  feats_B: {feats_B_path}  shape={tuple(feats_B.shape)}")
    print(f"  images:  {images_dir}/  ({num_to_sample} files)")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
