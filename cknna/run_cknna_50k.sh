#!/bin/bash
# =============================================================================
# CKNNA Pipeline (50K) -- Full Bridge dataset coverage
# =============================================================================
#
# Uses 50,000 samples from ALL 54 chunks of bridge_orig_lerobot (vs. 5,000
# from 3 chunks in the original pipeline). Data goes to cknna_data_50k/.
#
# Phase 1: load_bridge_data_full.py (parallel downloads, all 54 chunks)
# Phase 2: Same feature extraction scripts, pointed at cknna_data_50k/
# Phase 3: compute_cknna_large.py (memory-optimized for N=50K on H100 80GB)
#
# Covers 3 codebases:
#   A) StarVLA models (this script, starVLA conda env)
#   B) SimplerEnv-OpenVLA models (separate conda envs per model)
#   C) GR00T N1.5/N1.6, CogACT, RT-1-X, Octo (separate conda envs)
#
# Usage:
#   bash cknna/run_cknna_50k.sh          # StarVLA models only
#   bash cknna/run_cknna_50k.sh --all    # All models (all conda envs)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
DATA_DIR="${STAR}/cknna/cknna_data_50k"
COMPUTE_CKNNA="${STAR}/cknna/compute_cknna_large.py"

RUN_ALL=false
if [[ "${1:-}" == "--all" ]]; then
    RUN_ALL=true
fi

source "${CONDA}/bin/activate" starVLA
cd "${STAR}"


# =========================================================================
# Phase 1: Data Loading (50K from all 54 chunks)
# =========================================================================
echo "============================================"
echo "Phase 1: Loading Bridge data (50K, all chunks)"
echo "============================================"

if [ -f "${DATA_DIR}/feats_B.pt" ] && [ -f "${DATA_DIR}/metadata.json" ]; then
    N=$(python3 -c "import json; print(json.load(open('${DATA_DIR}/metadata.json'))['num_samples'])")
    echo "Phase 1 data already exists: N=${N} samples. Skipping."
else
    python cknna/load_bridge_data_full.py \
        --output_dir "${DATA_DIR}" \
        --num_samples 50000 \
        --workers 32 \
        --seed 42
fi
echo ""


# =========================================================================
# Phase 2A: StarVLA feature extraction
# =========================================================================
echo "============================================"
echo "Phase 2A: StarVLA models"
echo "============================================"

declare -A STARVLA_CKPTS
STARVLA_CKPTS=(
    ["Qwen-FAST-Bridge-RT-1"]="playground/Pretrained_models/Qwen-FAST-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    ["Qwen-OFT-Bridge-RT-1"]="playground/Pretrained_models/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    ["Qwen-GR00T-Bridge"]="playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
    ["Qwen-GR00T-Bridge-RT-1"]="playground/Pretrained_models/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
    ["Qwen3VL-GR00T-Bridge-RT-1"]="playground/Pretrained_models/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
    ["Qwen3VL-OFT-Bridge-RT-1"]="playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
)

for name in "${!STARVLA_CKPTS[@]}"; do
    ckpt="${STARVLA_CKPTS[$name]}"
    out_dir="${DATA_DIR}/${name}"

    if [ -f "${out_dir}/feats_A.pt" ]; then
        echo "[${name}] feats_A.pt exists, skipping."
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


# =========================================================================
# Phase 2B: SimplerEnv-OpenVLA models (requires env switching)
# =========================================================================
if $RUN_ALL; then

echo "============================================"
echo "Phase 2B: SimplerEnv-OpenVLA models"
echo "============================================"

# -- SpatialVLA --
SPATIALVLA_OUT="${DATA_DIR}/spatialvla-sft-bridge"
if [ -f "${SPATIALVLA_OUT}/feats_A.pt" ]; then
    echo "[spatialvla-sft-bridge] exists, skipping."
else
    echo "--- spatialvla-sft-bridge (spatialvla_env) ---"
    source "${CONDA}/bin/activate" spatialvla_env
    python "${SIMPLER}/cknna/extract_features_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUT}" \
        --unnorm_key "bridge_orig/1.0.0"
fi

# -- Pi0 lerobot --
PI0_OUT="${DATA_DIR}/pi0-lerobot-bridge"
if [ -f "${PI0_OUT}/feats_A.pt" ]; then
    echo "[pi0-lerobot-bridge] exists, skipping."
else
    echo "--- pi0-lerobot-bridge (pi0fast_env) ---"
    source "${CONDA}/bin/activate" pi0fast_env
    python "${SIMPLER}/cknna/extract_features_pi0_lerobot.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUT}"
fi

# -- OpenVLA-7B --
OPENVLA_OUT="${DATA_DIR}/openvla-7b-bridge"
if [ -f "${OPENVLA_OUT}/feats_A.pt" ]; then
    echo "[openvla-7b-bridge] exists, skipping."
else
    echo "--- openvla-7b-bridge (openvla_env) ---"
    source "${CONDA}/bin/activate" openvla_env
    python "${SIMPLER}/cknna/extract_features_openvla.py" \
        --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_OUT}"
fi

# -- RT-1-X --
RT1X_OUT="${DATA_DIR}/rt1x-bridge"
if [ -f "${RT1X_OUT}/feats_A.pt" ]; then
    echo "[rt1x-bridge] exists, skipping."
else
    echo "--- rt1x-bridge (simpler_env) ---"
    source "${CONDA}/bin/activate" simpler_env
    python "${SIMPLER}/cknna/extract_features_rt1x.py" \
        --ckpt "${SIMPLER}/checkpoints/rt_1_x_tf_trained_for_002272480_step" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RT1X_OUT}"
fi

# -- Octo-base --
OCTO_OUT="${DATA_DIR}/octo-base-bridge"
if [ -f "${OCTO_OUT}/feats_A.pt" ]; then
    echo "[octo-base-bridge] exists, skipping."
else
    echo "--- octo-base-bridge (simpler_env) ---"
    source "${CONDA}/bin/activate" simpler_env
    python "${SIMPLER}/cknna/extract_features_octo.py" \
        --ckpt hf://rail-berkeley/octo-base-1.5 \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OCTO_OUT}"
fi

# -- CogACT (Small, Base, Large) --
for SIZE in Small Base Large; do
    COGACT_OUT="${DATA_DIR}/cogact-$(echo ${SIZE} | tr '[:upper:]' '[:lower:]')-bridge"
    if [ -f "${COGACT_OUT}/feats_A.pt" ]; then
        echo "[cogact-${SIZE}] exists, skipping."
        continue
    fi
    echo "--- cogact-${SIZE} (cogact env) ---"
    source "${CONDA}/bin/activate" cogact
    python "${SIMPLER}/cknna/extract_features_cogact.py" \
        --ckpt "CogACT/CogACT-${SIZE}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${COGACT_OUT}"
done

# -- GR00T N1.5 --
GROOT15_OUT="${DATA_DIR}/groot-n15-bridge"
if [ -f "${GROOT15_OUT}/feats_A.pt" ]; then
    echo "[groot-n15-bridge] exists, skipping."
else
    echo "--- groot-n15-bridge (groot_libero env) ---"
    source "${CONDA}/bin/activate" groot_libero
    PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n15.py" \
        --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT15_OUT}"
fi

# -- GR00T N1.6 --
GROOT16_OUT="${DATA_DIR}/groot-n16-bridge"
if [ -f "${GROOT16_OUT}/feats_A.pt" ]; then
    echo "[groot-n16-bridge] exists, skipping."
else
    echo "--- groot-n16-bridge (groot16 env) ---"
    source "${CONDA}/bin/activate" groot16
    PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n16.py" \
        --ckpt "${WORK}/GR00T-N1.6-bridge" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT16_OUT}"
fi

echo ""

fi  # end RUN_ALL


# =========================================================================
# Phase 3: Compute CKNNA (memory-optimized for N=50K)
# =========================================================================
echo "============================================"
echo "Phase 3: Computing CKNNA scores (50K)"
echo "============================================"

source "${CONDA}/bin/activate" starVLA

FEATS_A_PATHS=""
for dir in "${DATA_DIR}"/*/; do
    if [ -f "${dir}feats_A.pt" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${dir}feats_A.pt"
        echo "  $(basename ${dir})"
    fi
done

if [ -z "${FEATS_A_PATHS}" ]; then
    echo "ERROR: No feats_A.pt files found."
    exit 1
fi

python "${COMPUTE_CKNNA}" \
    --feats_A ${FEATS_A_PATHS} \
    --feats_B "${DATA_DIR}/feats_B.pt" \
    --topk 5 10 20 \
    --also_mutual_knn \
    --output "${DATA_DIR}/cknna_results_50k.json"

echo ""
echo "============================================"
echo "Pipeline complete"
echo "============================================"
echo "Results: ${DATA_DIR}/cknna_results_50k.json"
