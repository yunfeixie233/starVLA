#!/bin/bash
# =============================================================================
# CKNNA Pipeline for StarVLA Models on WidowX Bridge Data
# =============================================================================
#
# Runs all 3 phases:
#   Phase 1: Download Bridge data and extract (image, state) pairs
#   Phase 2: Extract VLM features (feats_A) for each checkpoint
#   Phase 3: Compute CKNNA scores
#
# Prerequisites:
#   - conda env "starVLA" is set up
#   - Checkpoints downloaded under playground/Pretrained_models/
#
# Usage:
#   bash cknna/run_cknna_starvla.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
DATA_DIR="${STAR}/cknna/cknna_data"
COMPUTE_CKNNA="${WORK}/lerobot/cknna/compute_cknna.py"

source "${CONDA}/bin/activate" starVLA
cd "${STAR}"

# Checkpoint name -> .pt file path
declare -A CHECKPOINTS
CHECKPOINTS=(
    ["Qwen-FAST-Bridge-RT-1"]="playground/Pretrained_models/Qwen-FAST-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    ["Qwen-OFT-Bridge-RT-1"]="playground/Pretrained_models/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    ["Qwen-GR00T-Bridge"]="playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
    ["Qwen-GR00T-Bridge-RT-1"]="playground/Pretrained_models/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
    ["Qwen3VL-GR00T-Bridge-RT-1"]="playground/Pretrained_models/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
    ["Qwen3VL-OFT-Bridge-RT-1"]="playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
)

NUM_SAMPLES=5000
NUM_CHUNKS=3


# ─── Phase 1: Data Loading ───────────────────────────────────────────────────

echo "============================================"
echo "Phase 1: Loading Bridge data"
echo "============================================"

if [ -f "${DATA_DIR}/feats_B.pt" ] && [ -f "${DATA_DIR}/metadata.json" ]; then
    echo "Phase 1 output already exists, skipping."
else
    python cknna/load_bridge_data.py \
        --output_dir "${DATA_DIR}" \
        --num_samples ${NUM_SAMPLES} \
        --num_chunks ${NUM_CHUNKS} \
        --seed 42
fi

echo ""


# ─── Phase 2: Feature Extraction ─────────────────────────────────────────────

echo "============================================"
echo "Phase 2: Extracting VLM features"
echo "============================================"

for name in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$name]}"
    out_dir="${DATA_DIR}/${name}"

    if [ -f "${out_dir}/feats_A.pt" ]; then
        echo "[${name}] feats_A.pt already exists, skipping."
        continue
    fi

    echo ""
    echo "--- ${name} ---"
    if python cknna/extract_features_starvla.py \
        --ckpt_path "${ckpt}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${out_dir}"; then
        echo "[${name}] SUCCESS"
    else
        echo "[${name}] FAILED -- skipping"
    fi
done

echo ""


# ─── Phase 3: CKNNA Computation ──────────────────────────────────────────────

echo "============================================"
echo "Phase 3: Computing CKNNA scores"
echo "============================================"

FEATS_A_PATHS=""
for name in "${!CHECKPOINTS[@]}"; do
    fa="${DATA_DIR}/${name}/feats_A.pt"
    if [ -f "${fa}" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${fa}"
    else
        echo "WARNING: ${fa} not found, skipping ${name}"
    fi
done

if [ -z "${FEATS_A_PATHS}" ]; then
    echo "ERROR: No feats_A files found. Cannot compute CKNNA."
    exit 1
fi

python "${COMPUTE_CKNNA}" \
    --feats_A ${FEATS_A_PATHS} \
    --feats_B "${DATA_DIR}/feats_B.pt" \
    --topk 5 10 20 \
    --also_mutual_knn \
    --output "${DATA_DIR}/cknna_results.json"

echo ""
echo "============================================"
echo "Pipeline complete. Results:"
echo "============================================"
cat "${DATA_DIR}/cknna_results.json"
