"""
Phase 1 (full): Load Bridge dataset for CKNNA evaluation -- all 54 chunks.

Optimized for large N (50K+):
  - Parallel parquet/video downloads via ThreadPoolExecutor
  - 1 random frame per episode for maximum state diversity
  - Streaming video decode (no full-episode caching)

State is 8D: [x, y, z, roll, pitch, yaw, pad, gripper].
We drop index 6 (pad, always 0) to get 7D feats_B.
Action is 7D: [dx, dy, dz, droll, dpitch, dyaw, dgripper].

Output:
  <output_dir>/feats_B.pt            -- (N, 7) float32 state vectors (time t only)
  <output_dir>/actions.pt            -- (N, 7) float32 action vectors (time t only)
  <output_dir>/feats_B_seq.pt        -- (N, H+1, 7) float32 sequential states [t, t+1, ..., t+H]
  <output_dir>/actions_seq.pt        -- (N, H+1, 7) float32 sequential actions [t, t+1, ..., t+H]
  <output_dir>/images/NNNNNN.png     -- individual PNG images (time t only)
  <output_dir>/metadata.json         -- metadata

Usage:
  python load_bridge_data_full.py --output_dir ./cknna_data_50k --num_samples 50000
  python load_bridge_data_full.py --output_dir ./cknna_data_50k --num_samples 50000 --max_horizon 15
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import av
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_tree
from PIL import Image as PILImage


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
TOTAL_CHUNKS = 54
TOTAL_EPISODES = 53192
CHUNK_SIZE = 1000
STATE_DIM_RAW = 8
STATE_DIM_EFFECTIVE = 7
PAD_INDEX = 6


def download_tasks(cache_dir):
    path = hf_hub_download(DATASET_REPO, "meta/tasks.jsonl", repo_type="dataset", cache_dir=cache_dir)
    tasks = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


def download_one_parquet(episode_index, cache_dir):
    chunk_idx = episode_index // CHUNK_SIZE
    path = f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet"
    return hf_hub_download(DATASET_REPO, path, repo_type="dataset", cache_dir=cache_dir)


def download_one_video(episode_index, cache_dir):
    chunk_idx = episode_index // CHUNK_SIZE
    path = f"videos/chunk-{chunk_idx:03d}/observation.images.image_0/episode_{episode_index:06d}.mp4"
    return hf_hub_download(DATASET_REPO, path, repo_type="dataset", cache_dir=cache_dir)


def decode_frame_at_index(video_path, frame_index):
    container = av.open(video_path)
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_index:
            arr = frame.to_ndarray(format="rgb24")
            container.close()
            return arr
    container.close()
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 1 (full): Load Bridge data from all 54 chunks.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel download threads")
    parser.add_argument("--parquet_only", action="store_true",
                        help="Only download parquets + sample; skip video/image download")
    parser.add_argument("--max_horizon", type=int, default=15,
                        help="Max future horizon H: save states/actions from t to t+H")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    H = args.max_horizon
    print(f"=== Phase 1 (full): Bridge data, N={args.num_samples}, seed={args.seed}, max_horizon={H} ===")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print()

    # --- Step 1: Download tasks metadata ---
    print("Downloading tasks metadata...")
    tasks_map = download_tasks(args.cache_dir)
    print(f"  {len(tasks_map)} tasks")

    # --- Step 2: Sample episode indices ---
    num_eps = min(args.num_samples, TOTAL_EPISODES)
    sampled_episodes = sorted(random.sample(range(TOTAL_EPISODES), num_eps))
    print(f"\nSampled {num_eps} unique episodes (1 frame each) from {TOTAL_EPISODES} total")

    # --- Step 3: Download parquets in parallel ---
    print(f"\nDownloading {num_eps} parquets ({args.workers} threads)...")
    t0 = time.time()

    parquet_paths = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one_parquet, ep, args.cache_dir): ep
            for ep in sampled_episodes
        }
        done = 0
        for fut in as_completed(futures):
            ep = futures[fut]
            parquet_paths[ep] = fut.result()
            done += 1
            if done % 2000 == 0 or done == num_eps:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (num_eps - done) / rate if rate > 0 else 0
                print(f"  [{done}/{num_eps}] {rate:.0f} files/s, ETA {eta:.0f}s")

    t1 = time.time()
    print(f"  Parquets downloaded in {t1-t0:.0f}s ({num_eps/(t1-t0):.0f} files/s)")

    # --- Step 4: Read parquets, pick 1 random frame per episode with room for horizon ---
    print(f"\nReading parquets and sampling frames (horizon={H})...")
    sampled_frames = []
    skipped_short = 0
    for ep in sampled_episodes:
        df = pd.read_parquet(parquet_paths[ep])
        n_frames = len(df)
        if n_frames <= H:
            skipped_short += 1
            continue
        fidx = random.randint(0, n_frames - 1 - H)
        rows = df.iloc[fidx:fidx + H + 1]
        chunk_idx = ep // CHUNK_SIZE

        states_seq = np.array([np.array(r, dtype=np.float32) for r in rows["observation.state"]])
        actions_seq = np.array([np.array(r, dtype=np.float32) for r in rows["action"]])

        sampled_frames.append({
            "episode_index": int(rows.iloc[0]["episode_index"]),
            "frame_index": int(rows.iloc[0]["frame_index"]),
            "state": states_seq[0],
            "action": actions_seq[0],
            "states_seq": states_seq,
            "actions_seq": actions_seq,
            "task_index": int(rows.iloc[0]["task_index"]),
            "chunk_idx": chunk_idx,
        })

    if skipped_short:
        print(f"  Skipped {skipped_short} episodes shorter than horizon+1={H+1} frames")
    print(f"  {len(sampled_frames)} frames sampled")

    # --- Step 5: Save feats_B, actions, and sequential versions ---
    states_list = []
    actions_list = []
    states_seq_list = []
    actions_seq_list = []
    task_descriptions = []

    def _drop_pad(raw_state):
        return np.concatenate([raw_state[:PAD_INDEX], raw_state[PAD_INDEX + 1:]])

    for sf in sampled_frames:
        states_list.append(_drop_pad(sf["state"]))
        actions_list.append(sf["action"])
        states_seq_list.append(
            np.stack([_drop_pad(s) for s in sf["states_seq"]])
        )
        actions_seq_list.append(sf["actions_seq"])
        task_descriptions.append(tasks_map.get(sf["task_index"], ""))

    feats_B = torch.tensor(np.stack(states_list), dtype=torch.float32)
    torch.save(feats_B, os.path.join(args.output_dir, "feats_B.pt"))
    print(f"\nSaved feats_B: shape={tuple(feats_B.shape)}")

    actions = torch.tensor(np.stack(actions_list), dtype=torch.float32)
    torch.save(actions, os.path.join(args.output_dir, "actions.pt"))
    print(f"Saved actions: shape={tuple(actions.shape)}")

    feats_B_seq = torch.tensor(np.stack(states_seq_list), dtype=torch.float32)
    torch.save(feats_B_seq, os.path.join(args.output_dir, "feats_B_seq.pt"))
    print(f"Saved feats_B_seq: shape={tuple(feats_B_seq.shape)}")

    actions_seq = torch.tensor(np.stack(actions_seq_list), dtype=torch.float32)
    torch.save(actions_seq, os.path.join(args.output_dir, "actions_seq.pt"))
    print(f"Saved actions_seq: shape={tuple(actions_seq.shape)}")

    # --- Step 5b: Save metadata ---
    metadata = {
        "num_samples": len(sampled_frames),
        "max_horizon": H,
        "state_dim_raw": STATE_DIM_RAW,
        "state_dim_effective": STATE_DIM_EFFECTIVE,
        "pad_index_dropped": PAD_INDEX,
        "action_dim": 7,
        "state_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "dataset_repo": DATASET_REPO,
        "num_chunks_used": TOTAL_CHUNKS,
        "chunk_indices": list(range(TOTAL_CHUNKS)),
        "seed": args.seed,
        "sampling_strategy": "1_random_frame_per_episode_with_horizon",
        "task_descriptions": task_descriptions,
        "episode_indices": [sf["episode_index"] for sf in sampled_frames],
        "frame_indices": [sf["frame_index"] for sf in sampled_frames],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata.json")

    if args.parquet_only:
        print("\n--parquet_only: skipping video/image download.")
        print("=== Phase 1 (partial) complete ===")
        return

    # --- Step 6: Download videos in parallel ---
    episodes_needed = [sf["episode_index"] for sf in sampled_frames]
    print(f"\nDownloading {len(episodes_needed)} videos ({args.workers} threads)...")
    t2 = time.time()

    video_paths = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one_video, ep, args.cache_dir): ep
            for ep in episodes_needed
        }
        done = 0
        for fut in as_completed(futures):
            ep = futures[fut]
            video_paths[ep] = fut.result()
            done += 1
            if done % 2000 == 0 or done == len(episodes_needed):
                elapsed = time.time() - t2
                rate = done / elapsed
                eta = (len(episodes_needed) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(episodes_needed)}] {rate:.0f} files/s, ETA {eta:.0f}s")

    t3 = time.time()
    print(f"  Videos downloaded in {t3-t2:.0f}s")

    # --- Step 7: Extract frames and save images ---
    print(f"\nExtracting frames and saving images...")
    for i, sf in enumerate(sampled_frames):
        ep = sf["episode_index"]
        fidx = sf["frame_index"]
        vpath = video_paths[ep]
        arr = decode_frame_at_index(vpath, fidx)
        if arr is None:
            print(f"  WARNING: could not decode frame {fidx} from ep {ep}, skipping")
            continue
        img = PILImage.fromarray(arr)
        img.save(os.path.join(images_dir, f"{i:06d}.png"))

        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  [{i+1}/{len(sampled_frames)}]")

    t4 = time.time()
    print(f"  Images saved in {t4-t3:.0f}s")

    print(f"\n=== Phase 1 (full) complete ===")
    print(f"  feats_B:      {tuple(feats_B.shape)}")
    print(f"  feats_B_seq:  {tuple(feats_B_seq.shape)}")
    print(f"  actions:      {tuple(actions.shape)}")
    print(f"  actions_seq:  {tuple(actions_seq.shape)}")
    print(f"  images:       {images_dir}/  ({len(sampled_frames)} files)")
    print(f"  Total time: {t4-t0:.0f}s ({(t4-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
