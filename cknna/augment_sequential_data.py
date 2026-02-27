"""
Augment existing CKNNA Phase 1 data with sequential states/actions.

Reads the existing metadata.json to recover (episode_index, frame_index) pairs,
downloads parquets, and for each sample extracts [t, t+1, ..., t+H] states and
actions. Saves feats_B_seq.pt (N, H+1, 7) and actions_seq.pt (N, H+1, 7).

When a frame is too close to episode end (fidx + H >= n_frames), the sequence
is right-padded by repeating the last available frame.

Does NOT modify existing feats_B.pt, actions.pt, or images/.
Updates metadata.json with max_horizon and pad_count fields.

Usage:
  python augment_sequential_data.py --data_dir ./cknna_data --max_horizon 15
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
CHUNK_SIZE = 1000
PAD_INDEX = 6


def download_one_parquet(episode_index, cache_dir):
    chunk_idx = episode_index // CHUNK_SIZE
    path = f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet"
    return hf_hub_download(DATASET_REPO, path, repo_type="dataset", cache_dir=cache_dir)


def _drop_pad(raw_state):
    return np.concatenate([raw_state[:PAD_INDEX], raw_state[PAD_INDEX + 1:]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_horizon", type=int, default=15)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    H = args.max_horizon
    meta_path = os.path.join(args.data_dir, "metadata.json")
    meta = json.load(open(meta_path))
    N = meta["num_samples"]
    episode_indices = meta["episode_indices"]
    frame_indices = meta["frame_indices"]

    print(f"=== Augment sequential data: N={N}, max_horizon={H} ===")
    print(f"Data dir: {args.data_dir}")

    out_B = os.path.join(args.data_dir, "feats_B_seq.pt")
    out_A = os.path.join(args.data_dir, "actions_seq.pt")
    if os.path.exists(out_B) and os.path.exists(out_A):
        existing_B = torch.load(out_B, weights_only=True)
        if existing_B.shape[1] >= H + 1:
            print(f"feats_B_seq.pt already exists with shape {tuple(existing_B.shape)} (horizon>={H}). Nothing to do.")
            return

    unique_episodes = sorted(set(episode_indices))
    print(f"Unique episodes to download: {len(unique_episodes)}")

    t0 = time.time()
    parquet_paths = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one_parquet, ep, args.cache_dir): ep
            for ep in unique_episodes
        }
        done = 0
        for fut in as_completed(futures):
            ep = futures[fut]
            parquet_paths[ep] = fut.result()
            done += 1
            if done % 500 == 0 or done == len(unique_episodes):
                print(f"  [{done}/{len(unique_episodes)}] downloaded")

    t1 = time.time()
    print(f"Parquets downloaded in {t1-t0:.0f}s")

    ep_dfs = {}
    for ep in unique_episodes:
        ep_dfs[ep] = pd.read_parquet(parquet_paths[ep])

    states_seq_all = np.zeros((N, H + 1, 7), dtype=np.float32)
    actions_seq_all = np.zeros((N, H + 1, 7), dtype=np.float32)
    pad_count = 0

    for i in range(N):
        ep = episode_indices[i]
        fidx = frame_indices[i]
        df = ep_dfs[ep]
        n_frames = len(df)

        end = min(fidx + H + 1, n_frames)
        avail = end - fidx
        rows = df.iloc[fidx:end]

        for j, (_, row) in enumerate(rows.iterrows()):
            raw_state = np.array(row["observation.state"], dtype=np.float32)
            states_seq_all[i, j] = _drop_pad(raw_state)
            actions_seq_all[i, j] = np.array(row["action"], dtype=np.float32)

        if avail < H + 1:
            pad_count += 1
            for j in range(avail, H + 1):
                states_seq_all[i, j] = states_seq_all[i, avail - 1]
                actions_seq_all[i, j] = actions_seq_all[i, avail - 1]

    t2 = time.time()
    print(f"Sequential data extracted in {t2-t1:.0f}s")
    if pad_count:
        print(f"  {pad_count}/{N} samples required padding (frame near episode end)")

    feats_B_seq = torch.from_numpy(states_seq_all)
    torch.save(feats_B_seq, out_B)
    print(f"Saved feats_B_seq: {tuple(feats_B_seq.shape)}")

    actions_seq = torch.from_numpy(actions_seq_all)
    torch.save(actions_seq, out_A)
    print(f"Saved actions_seq: {tuple(actions_seq.shape)}")

    meta["max_horizon"] = H
    meta["pad_count"] = pad_count
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated metadata.json with max_horizon={H}")

    print(f"\n=== Done ({t2-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
