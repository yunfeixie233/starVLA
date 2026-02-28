#!/bin/bash
# =============================================================================
# 50K Full Sweep: Single-pass extraction (feats_A + feats_action)
# =============================================================================
# Uses unified extract_all_*.py scripts that extract BOTH VLM features and
# action representations in a single forward pass per sample.
#
# Step 1: Augment sequential data (feats_B_seq + actions_seq)
# Step 2: Single-pass extraction for all 17 models
# Step 3: CKNNA computation, CSV, plots
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
CONDA="${CONDA_ROOT:-/lambda/nfs/verl/conda}"
DATA_DIR="${STAR}/cknna/cknna_data_50k"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "50K Full Sweep -- Single-Pass Extraction"
log "Data dir: ${DATA_DIR}"
log "=========================================="

# =========================================================================
# STEP 1: Augment sequential data
# =========================================================================
log "STEP 1: Augment sequential data (feats_B_seq + actions_seq)"
source "${CONDA}/bin/activate" starVLA
export PYTHONNOUSERSITE=1
cd "${STAR}"

if [ -f "${DATA_DIR}/feats_B_seq.pt" ] && [ -f "${DATA_DIR}/actions_seq.pt" ]; then
    log "[step1] feats_B_seq.pt + actions_seq.pt already exist, skipping."
else
    python cknna/augment_sequential_data.py \
        --data_dir "${DATA_DIR}" \
        --max_horizon 15 || log "[step1] FAILED"
fi

# =========================================================================
# STEP 2A: StarVLA (6 models) -- single-pass
# =========================================================================
log "STEP 2A: StarVLA (6 models, single-pass)"

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

    if [ -f "${out_dir}/feats_A_img.pt" ] && [ -f "${out_dir}/feats_A_txt.pt" ] && [ -f "${out_dir}/feats_action.pt" ]; then
        log "[${name}] all outputs exist, skipping."
        continue
    fi

    log "--- ${name} (single-pass) ---"
    python cknna/extract_all_starvla.py \
        --ckpt_path "${ckpt}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${out_dir}" || log "[${name}] FAILED"
done

# =========================================================================
# STEP 2B: SpatialVLA -- single-pass
# =========================================================================
log "STEP 2B: SpatialVLA (single-pass)"
source "${CONDA}/bin/activate" spatialvla_env
export PYTHONNOUSERSITE=1
SPATIALVLA_OUT="${DATA_DIR}/spatialvla-sft-bridge"

if [ -f "${SPATIALVLA_OUT}/feats_A_img.pt" ] && [ -f "${SPATIALVLA_OUT}/feats_A_txt.pt" ] && [ -f "${SPATIALVLA_OUT}/feats_action.pt" ]; then
    log "[spatialvla] all outputs exist, skipping."
else
    log "--- spatialvla (single-pass) ---"
    python "${SIMPLER}/cknna/extract_all_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUT}" || log "[spatialvla] FAILED"
fi

# =========================================================================
# STEP 2C: Pi0 -- single-pass
# =========================================================================
log "STEP 2C: Pi0 (single-pass)"
source "${CONDA}/bin/activate" pi0fast_env
export PYTHONNOUSERSITE=1
PI0_OUT="${DATA_DIR}/pi0-lerobot-bridge"

if [ -f "${PI0_OUT}/feats_A_img.pt" ] && [ -f "${PI0_OUT}/feats_A_txt.pt" ] && [ -f "${PI0_OUT}/feats_action.pt" ]; then
    log "[pi0] all outputs exist, skipping."
else
    log "--- pi0 (single-pass) ---"
    PYTHONNOUSERSITE=1 python "${SIMPLER}/cknna/extract_all_pi0.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUT}" || log "[pi0] FAILED"
fi

# =========================================================================
# STEP 2D: OpenVLA (base + ft-200k) -- single-pass
# =========================================================================
log "STEP 2D: OpenVLA (single-pass)"
source "${CONDA}/bin/activate" openvla_env
export PYTHONNOUSERSITE=1

# openvla-7b (base)
OVLA_OUT="${DATA_DIR}/openvla-7b-bridge"
if [ -f "${OVLA_OUT}/feats_A_img.pt" ] && [ -f "${OVLA_OUT}/feats_A_txt.pt" ] && [ -f "${OVLA_OUT}/feats_action.pt" ]; then
    log "[openvla-7b-bridge] all outputs exist, skipping."
else
    log "--- openvla-7b-bridge (single-pass) ---"
    python "${SIMPLER}/cknna/extract_all_openvla.py" \
        --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OVLA_OUT}" || log "[openvla-7b-bridge] FAILED"
fi

# openvla-bridge-sft (ft-200k)
OVLA_OUT="${DATA_DIR}/openvla-7b-bridge-ft-200k"
if [ -f "${OVLA_OUT}/feats_A_img.pt" ] && [ -f "${OVLA_OUT}/feats_A_txt.pt" ] && [ -f "${OVLA_OUT}/feats_action.pt" ]; then
    log "[openvla-7b-bridge-ft-200k] all outputs exist, skipping."
else
    log "--- openvla-7b-bridge-ft-200k (single-pass) ---"
    python "${SIMPLER}/cknna/extract_all_openvla.py" \
        --ckpt "${SIMPLER}/checkpoints/openvla-bridge-sft" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OVLA_OUT}" || log "[openvla-7b-bridge-ft-200k] FAILED"
fi

# =========================================================================
# STEP 2E: CogACT (Small/Base/Large) -- feats_A only (feats_action = copy)
# =========================================================================
log "STEP 2E: CogACT"
source "${CONDA}/bin/activate" cogact
export PYTHONNOUSERSITE=1
cd "${SIMPLER}"

for SIZE in Small Base Large; do
    size_lower=$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')
    COGACT_OUT="${DATA_DIR}/cogact-${size_lower}-bridge"

    case "${SIZE}" in
        Small) DIT="DiT-S" ;;
        Base)  DIT="DiT-B" ;;
        Large) DIT="DiT-L" ;;
    esac

    if [ -f "${COGACT_OUT}/feats_A_img.pt" ] && [ -f "${COGACT_OUT}/feats_A_txt.pt" ]; then
        log "[cogact-${SIZE}] feats_A already done, skipping."
    else
        log "--- cogact-${SIZE} (feats_A) ---"
        python cknna/extract_features_cogact.py \
            --ckpt "CogACT/CogACT-${SIZE}" \
            --action_model_type "${DIT}" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${COGACT_OUT}" || log "[cogact-${SIZE}] FAILED"
    fi

    if [ ! -f "${COGACT_OUT}/feats_action.pt" ] && [ -f "${COGACT_OUT}/feats_A.pt" ]; then
        cp "${COGACT_OUT}/feats_A.pt" "${COGACT_OUT}/feats_action.pt"
        log "[cogact-${SIZE}] copied feats_A.pt -> feats_action.pt"
    fi
done
cd "${STAR}"

# =========================================================================
# STEP 2F: GR00T N1.5 + N1.6 -- single-pass
# =========================================================================
log "STEP 2F: GR00T N1.5 + N1.6 (single-pass)"
source "${CONDA}/bin/activate" groot_libero
export PYTHONNOUSERSITE=1

GROOT15_OUT="${DATA_DIR}/groot-n15-bridge"
if [ -f "${GROOT15_OUT}/feats_A_img.pt" ] && [ -f "${GROOT15_OUT}/feats_A_txt.pt" ] && [ -f "${GROOT15_OUT}/feats_action.pt" ]; then
    log "[groot-n15] all outputs exist, skipping."
else
    log "--- groot-n15 (single-pass) ---"
    PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_all_groot_n15.py" \
        --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT15_OUT}" || log "[groot-n15] FAILED"
fi

GROOT16_OUT="${DATA_DIR}/groot-n16-bridge"
if [ -f "${GROOT16_OUT}/feats_A_img.pt" ] && [ -f "${GROOT16_OUT}/feats_A_txt.pt" ] && [ -f "${GROOT16_OUT}/feats_action.pt" ]; then
    log "[groot-n16] all outputs exist, skipping."
else
    log "--- groot-n16 (single-pass) ---"
    PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_all_groot_n16.py" \
        --ckpt "${WORK}/GR00T-N1.6-bridge" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT16_OUT}" || log "[groot-n16] FAILED"
fi

# =========================================================================
# STEP 2G: RT-1-X -- feats_A only (feats_action = copy, TF model)
# =========================================================================
log "STEP 2G: RT-1-X"
source "${CONDA}/bin/activate" simpler_env
export PYTHONNOUSERSITE=1
RT1X_OUT="${DATA_DIR}/rt1x-bridge"

if [ -f "${RT1X_OUT}/feats_A.pt" ]; then
    log "[rt1x] feats_A already exists, skipping."
else
    log "--- rt1x (feats_A) ---"
    python "${SIMPLER}/cknna/extract_features_rt1x.py" \
        --ckpt "${SIMPLER}/checkpoints/rt_1_x_tf_trained_for_002272480_step" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RT1X_OUT}" || log "[rt1x] FAILED"
fi

if [ ! -f "${RT1X_OUT}/feats_action.pt" ] && [ -f "${RT1X_OUT}/feats_A.pt" ]; then
    cp "${RT1X_OUT}/feats_A.pt" "${RT1X_OUT}/feats_action.pt"
    log "[rt1x] copied feats_A.pt -> feats_action.pt"
fi

# =========================================================================
# STEP 2H: Octo -- feats_A only (feats_action = copy, JAX model)
# =========================================================================
log "STEP 2H: Octo"
OCTO_OUT="${DATA_DIR}/octo-base-bridge"

if [ -f "${OCTO_OUT}/feats_A.pt" ]; then
    log "[octo] feats_A already exists, skipping."
else
    log "--- octo (feats_A) ---"
    python "${SIMPLER}/cknna/extract_features_octo.py" \
        --model_type "hf://rail-berkeley/octo-base" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OCTO_OUT}" || log "[octo] FAILED"
fi

if [ ! -f "${OCTO_OUT}/feats_action.pt" ] && [ -f "${OCTO_OUT}/feats_A.pt" ]; then
    cp "${OCTO_OUT}/feats_A.pt" "${OCTO_OUT}/feats_action.pt"
    log "[octo] copied feats_A.pt -> feats_action.pt"
fi

# =========================================================================
# STEP 3: CKNNA Computation + CSV + Plots
# =========================================================================
log "STEP 3: CKNNA Computation (3 feats_A x 5 feats_B)"
source "${CONDA}/bin/activate" starVLA
export PYTHONNOUSERSITE=1
cd "${STAR}"

python cknna/run_50k_cknna_computation.py --device cuda || log "CKNNA COMPUTATION FAILED"

log "=========================================="
log "50K FULL SWEEP PIPELINE COMPLETE"
log "=========================================="
