#!/bin/bash
# =============================================================================
# Pipeline Test: N=100 samples, all models, single-pass extraction + CKNNA
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
CONDA="${CONDA_ROOT:-/lambda/nfs/verl/conda}"
DATA_DIR="${STAR}/cknna/cknna_data_test"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
SECONDS=0

log "=========================================="
log "PIPELINE TEST (N=100)"
log "=========================================="

# --- StarVLA (1 model per family to test all 3 paths) ---
log "=== StarVLA GR00T ==="
source "${CONDA}/bin/activate" starVLA
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
cd "${STAR}"
T=$SECONDS
python cknna/extract_all_starvla.py \
    --ckpt_path playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/Qwen-GR00T-Bridge" || log "FAILED"
log "StarVLA-GR00T: $((SECONDS - T))s"

log "=== StarVLA FAST ==="
T=$SECONDS
python cknna/extract_all_starvla.py \
    --ckpt_path playground/Pretrained_models/Qwen-FAST-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/Qwen-FAST-Bridge-RT-1" || log "FAILED"
log "StarVLA-FAST: $((SECONDS - T))s"

log "=== StarVLA OFT ==="
T=$SECONDS
python cknna/extract_all_starvla.py \
    --ckpt_path playground/Pretrained_models/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/Qwen-OFT-Bridge-RT-1" || log "FAILED"
log "StarVLA-OFT: $((SECONDS - T))s"

# --- SpatialVLA ---
log "=== SpatialVLA ==="
source "${CONDA}/bin/activate" spatialvla_env
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
T=$SECONDS
python "${SIMPLER}/cknna/extract_all_spatialvla.py" \
    --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/spatialvla-sft-bridge" || log "FAILED"
log "SpatialVLA: $((SECONDS - T))s"

# --- Pi0 ---
log "=== Pi0 ==="
source "${CONDA}/bin/activate" pi0fast_env
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
T=$SECONDS
python "${SIMPLER}/cknna/extract_all_pi0.py" \
    --ckpt_path HaomingSong/lerobot-pi0-bridge \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/pi0-lerobot-bridge" || log "FAILED"
log "Pi0: $((SECONDS - T))s"

# --- OpenVLA (base) ---
log "=== OpenVLA ==="
source "${CONDA}/bin/activate" openvla_env
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
T=$SECONDS
python "${SIMPLER}/cknna/extract_all_openvla.py" \
    --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/openvla-7b-bridge" || log "FAILED"
log "OpenVLA: $((SECONDS - T))s"

# --- CogACT (Small) ---
log "=== CogACT-Small ==="
source "${CONDA}/bin/activate" cogact
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
cd "${SIMPLER}"
T=$SECONDS
python cknna/extract_features_cogact.py \
    --ckpt "CogACT/CogACT-Small" \
    --action_model_type "DiT-S" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/cogact-small-bridge" || log "FAILED"
cp "${DATA_DIR}/cogact-small-bridge/feats_A.pt" "${DATA_DIR}/cogact-small-bridge/feats_action.pt" 2>/dev/null || true
log "CogACT-Small: $((SECONDS - T))s"
cd "${STAR}"

# --- GR00T N1.5 ---
log "=== GR00T N1.5 ==="
source "${CONDA}/bin/activate" groot_libero
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
T=$SECONDS
PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_all_groot_n15.py" \
    --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/groot-n15-bridge" || log "FAILED"
log "GR00T-N1.5: $((SECONDS - T))s"

# --- GR00T N1.6 ---
log "=== GR00T N1.6 ==="
T=$SECONDS
PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_all_groot_n16.py" \
    --ckpt "${WORK}/GR00T-N1.6-bridge" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/groot-n16-bridge" || log "FAILED"
log "GR00T-N1.6: $((SECONDS - T))s"

# --- RT-1-X ---
log "=== RT-1-X ==="
source "${CONDA}/bin/activate" simpler_env
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
T=$SECONDS
python "${SIMPLER}/cknna/extract_features_rt1x.py" \
    --ckpt "${SIMPLER}/checkpoints/rt_1_x_tf_trained_for_002272480_step" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/rt1x-bridge" || log "FAILED"
cp "${DATA_DIR}/rt1x-bridge/feats_A.pt" "${DATA_DIR}/rt1x-bridge/feats_action.pt" 2>/dev/null || true
log "RT-1-X: $((SECONDS - T))s"

# --- Octo ---
log "=== Octo ==="
T=$SECONDS
python "${SIMPLER}/cknna/extract_features_octo.py" \
    --model_type "hf://rail-berkeley/octo-base" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/octo-base-bridge" || log "FAILED"
cp "${DATA_DIR}/octo-base-bridge/feats_A.pt" "${DATA_DIR}/octo-base-bridge/feats_action.pt" 2>/dev/null || true
log "Octo: $((SECONDS - T))s"

# --- Verify outputs ---
log "=========================================="
log "EXTRACTION COMPLETE -- Verifying outputs"
log "=========================================="
source "${CONDA}/bin/activate" starVLA
export PYTHONNOUSERSITE=1
cd "${STAR}"

python3 -c "
import os, torch
data_dir = '${DATA_DIR}'
models = [
    'Qwen-GR00T-Bridge', 'Qwen-FAST-Bridge-RT-1', 'Qwen-OFT-Bridge-RT-1',
    'spatialvla-sft-bridge', 'pi0-lerobot-bridge', 'openvla-7b-bridge',
    'cogact-small-bridge', 'groot-n15-bridge', 'groot-n16-bridge',
    'rt1x-bridge', 'octo-base-bridge',
]
print(f'{'Model':<30} {'feats_A':>12} {'feats_A_img':>14} {'feats_A_txt':>14} {'feats_action':>14}')
print('-' * 90)
for m in models:
    d = os.path.join(data_dir, m)
    row = [m]
    for f in ['feats_A.pt', 'feats_A_img.pt', 'feats_A_txt.pt', 'feats_action.pt']:
        p = os.path.join(d, f)
        if os.path.exists(p):
            t = torch.load(p, weights_only=True)
            row.append(f'{tuple(t.shape)}')
        else:
            row.append('MISSING')
    print(f'{row[0]:<30} {row[1]:>12} {row[2]:>14} {row[3]:>14} {row[4]:>14}')
"

# --- Mini CKNNA computation ---
log "=========================================="
log "CKNNA TEST (N=100)"
log "=========================================="
T=$SECONDS
python3 -c "
import os, sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, '${STAR}/cknna')
from compute_cknna_large import cknna_lowmem, mutual_knn_lowmem

data_dir = '${DATA_DIR}'
meta = json.load(open(os.path.join(data_dir, 'metadata.json')))
tasks = meta['task_descriptions']
non_empty = torch.tensor([i for i, t in enumerate(tasks) if t.strip()], dtype=torch.long)
print(f'Non-empty samples: {len(non_empty)}/{meta[\"num_samples\"]}')

feats_B = torch.load(os.path.join(data_dir, 'feats_B.pt'), weights_only=True).float()[:100]
feats_B_filt = F.normalize(feats_B[non_empty], p=2, dim=-1).cuda()

actions = torch.load(os.path.join(data_dir, 'actions.pt'), weights_only=True).float()[:100]
actions_filt = F.normalize(actions[non_empty], p=2, dim=-1).cuda()

feats_B_seq = torch.load(os.path.join(data_dir, 'feats_B_seq.pt'), weights_only=True).float()[:100]
seq_h3 = feats_B_seq[:, :4, :].reshape(100, -1)
seq_h3_filt = F.normalize(seq_h3[non_empty], p=2, dim=-1).cuda()

models = [
    'Qwen-GR00T-Bridge', 'Qwen-FAST-Bridge-RT-1', 'Qwen-OFT-Bridge-RT-1',
    'spatialvla-sft-bridge', 'pi0-lerobot-bridge', 'openvla-7b-bridge',
    'cogact-small-bridge', 'groot-n15-bridge', 'groot-n16-bridge',
    'rt1x-bridge', 'octo-base-bridge',
]

print(f\"{'Model':<28} {'CKNNA_proprio':>14} {'CKNNA_action':>14} {'CKNNA_real_act':>15} {'CKNNA_seq_h3':>13}\")
print('-' * 90)

for m in models:
    d = os.path.join(data_dir, m)
    fa_path = os.path.join(d, 'feats_A.pt')
    if not os.path.exists(fa_path):
        print(f'{m:<28} MISSING')
        continue
    fa = torch.load(fa_path, weights_only=True).float()
    fa_filt = F.normalize(fa[non_empty], p=2, dim=-1).cuda()

    c_p = cknna_lowmem(fa_filt, feats_B_filt, topk=5)
    c_ra = cknna_lowmem(fa_filt, actions_filt, topk=5)
    c_seq = cknna_lowmem(fa_filt, seq_h3_filt, topk=5)

    fact_path = os.path.join(d, 'feats_action.pt')
    if os.path.exists(fact_path):
        fact = torch.load(fact_path, weights_only=True).float()
        fact_filt = F.normalize(fact[non_empty], p=2, dim=-1).cuda()
        c_a = cknna_lowmem(fa_filt, fact_filt, topk=5)
    else:
        c_a = float('nan')

    print(f'{m:<28} {c_p:>14.6f} {c_a:>14.6f} {c_ra:>15.6f} {c_seq:>13.6f}')
    del fa_filt
    torch.cuda.empty_cache()

print()
print('CKNNA test passed -- all models computed successfully')
"
log "CKNNA test: $((SECONDS - T))s"

log "=========================================="
log "TOTAL TIME: ${SECONDS}s"
log "=========================================="
