#!/bin/bash
# =============================================================================
# Action Representation CKNNA Pipeline for StarVLA Models on WidowX Bridge Data
# =============================================================================
#
# Phase 2b: Extract action representations (feats_action) for each checkpoint.
# Phase 3b: Compute CKNNA(feats_A, feats_action) for each model.
#
# Reuses existing Phase 1 data and Phase 2 feats_A from the main pipeline.
#
# Usage:
#   bash cknna/run_action_repr_starvla.sh
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

# Verify Phase 1 data
if [ ! -f "${DATA_DIR}/feats_B.pt" ] || [ ! -f "${DATA_DIR}/metadata.json" ]; then
    echo "ERROR: Phase 1 data not found. Run run_cknna_starvla.sh first."
    exit 1
fi

echo "============================================"
echo " Phase 2b: Extract Action Representations"
echo "============================================"

for name in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$name]}"
    out_dir="${DATA_DIR}/${name}"

    if [ -f "${out_dir}/feats_action.pt" ]; then
        echo "[${name}] feats_action.pt already exists, skipping."
        continue
    fi

    echo ""
    echo "--- ${name} ---"
    if python cknna/extract_action_repr_starvla.py \
        --ckpt_path "${ckpt}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${out_dir}" \
        --seed 42; then
        echo "[${name}] SUCCESS"
    else
        echo "[${name}] FAILED -- skipping"
    fi
done

echo ""
echo "============================================"
echo " Phase 3b: CKNNA(feats_A, feats_action)"
echo "============================================"

RESULTS_FILE="${DATA_DIR}/cknna_action_repr_results.json"

# Compute CKNNA(feats_A, feats_action) for each model individually
for name in "${!CHECKPOINTS[@]}"; do
    fa="${DATA_DIR}/${name}/feats_A.pt"
    fact="${DATA_DIR}/${name}/feats_action.pt"

    if [ ! -f "${fa}" ]; then
        echo "[${name}] feats_A.pt missing, skipping"
        continue
    fi
    if [ ! -f "${fact}" ]; then
        echo "[${name}] feats_action.pt missing, skipping"
        continue
    fi

    echo ""
    echo "--- ${name}: CKNNA(VLM, action_repr) ---"
    python "${COMPUTE_CKNNA}" \
        --feats_A "${fa}" \
        --feats_B "${fact}" \
        --topk 5 10 20 \
        --also_mutual_knn \
        --output "${DATA_DIR}/${name}/cknna_action_repr.json"
done

# Also compute cross-model comparison: all feats_A vs feats_action
echo ""
echo "--- Cross-model: all feats_A vs feats_action ---"
FEATS_A_PATHS=""
for name in "${!CHECKPOINTS[@]}"; do
    fa="${DATA_DIR}/${name}/feats_A.pt"
    if [ -f "${fa}" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${fa}"
    fi
done

# For each model's feats_action, compute CKNNA with all models' feats_A
for name in "${!CHECKPOINTS[@]}"; do
    fact="${DATA_DIR}/${name}/feats_action.pt"
    if [ -f "${fact}" ]; then
        echo ""
        echo "--- feats_B=${name}/feats_action.pt ---"
        python "${COMPUTE_CKNNA}" \
            --feats_A ${FEATS_A_PATHS} \
            --feats_B "${fact}" \
            --topk 10 \
            --output "${DATA_DIR}/${name}/cknna_action_repr_crossmodel.json"
    fi
done

echo ""
echo "============================================"
echo " Pipeline complete"
echo "============================================"
echo "Per-model results in: ${DATA_DIR}/<model>/cknna_action_repr.json"
