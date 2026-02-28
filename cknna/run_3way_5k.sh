#!/bin/bash
# =============================================================================
# 3-Way Feature Extraction (imgtext/img/txt) + CKNNA at N=5K
# =============================================================================
# Re-extracts feats_A with 3 pooling variants for all models that support
# image/text decomposition (15 models). Skips RT-1-X and Octo (no text tokens).
# Then runs CKNNA_proprio and CKNNA_action for each variant.
#
# Output: feats_A.pt, feats_A_img.pt, feats_A_txt.pt per model dir
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
CONDA="${CONDA_ROOT:-/lambda/nfs/verl/conda}"
DATA_DIR="${STAR}/cknna/cknna_data"
COMPUTE_CKNNA="${STAR}/cknna/compute_cknna_large.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "3-Way Feature Extraction + CKNNA (N=5K)"
log "Data dir: ${DATA_DIR}"
log "=========================================="

# =========================================================================
# SECTION 1: StarVLA (6 models)
# =========================================================================
log "SECTION 1: StarVLA (6 models)"
source "${CONDA}/bin/activate" starVLA
export PYTHONNOUSERSITE=1
cd "${STAR}"

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

    if [ -f "${out_dir}/feats_A_img.pt" ] && [ -f "${out_dir}/feats_A_txt.pt" ]; then
        log "[${name}] feats_A_img/txt already exist, skipping."
        continue
    fi

    log "--- ${name} ---"
    python cknna/extract_features_starvla.py \
        --ckpt_path "${ckpt}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${out_dir}" || log "[${name}] FAILED"
done

# =========================================================================
# SECTION 2: SpatialVLA
# =========================================================================
log "SECTION 2: SpatialVLA"
source "${CONDA}/bin/activate" spatialvla_env
SPATIALVLA_OUT="${DATA_DIR}/spatialvla-sft-bridge"

if [ -f "${SPATIALVLA_OUT}/feats_A_img.pt" ] && [ -f "${SPATIALVLA_OUT}/feats_A_txt.pt" ]; then
    log "[spatialvla] already done, skipping."
else
    log "--- spatialvla-sft-bridge ---"
    python "${SIMPLER}/cknna/extract_features_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUT}" || log "[spatialvla] FAILED"
fi

# =========================================================================
# SECTION 3: Pi0
# =========================================================================
log "SECTION 3: Pi0"
source "${CONDA}/bin/activate" pi0fast_env
PI0_OUT="${DATA_DIR}/pi0-lerobot-bridge"

if [ -f "${PI0_OUT}/feats_A_img.pt" ] && [ -f "${PI0_OUT}/feats_A_txt.pt" ]; then
    log "[pi0] already done, skipping."
else
    log "--- pi0-lerobot-bridge ---"
    PYTHONNOUSERSITE=1 python "${SIMPLER}/cknna/extract_features_pi0_lerobot.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUT}" || log "[pi0] FAILED"
fi

# =========================================================================
# SECTION 4: OpenVLA (base + ft-200k)
# =========================================================================
log "SECTION 4: OpenVLA"
source "${CONDA}/bin/activate" openvla_env

for variant in openvla-7b openvla-bridge-sft; do
    if [ "${variant}" = "openvla-7b" ]; then
        out_name="openvla-7b-bridge"
    else
        out_name="openvla-7b-bridge-ft-200k"
    fi
    OVLA_OUT="${DATA_DIR}/${out_name}"

    if [ -f "${OVLA_OUT}/feats_A_img.pt" ] && [ -f "${OVLA_OUT}/feats_A_txt.pt" ]; then
        log "[${out_name}] already done, skipping."
        continue
    fi

    log "--- ${out_name} ---"
    python "${SIMPLER}/cknna/extract_features_openvla.py" \
        --ckpt "${SIMPLER}/checkpoints/${variant}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OVLA_OUT}" || log "[${out_name}] FAILED"
done

# =========================================================================
# SECTION 5: CogACT (Small/Base/Large)
# =========================================================================
log "SECTION 5: CogACT"
source "${CONDA}/bin/activate" cogact
cd "${SIMPLER}"

for SIZE in Small Base Large; do
    size_lower=$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')
    COGACT_OUT="${DATA_DIR}/cogact-${size_lower}-bridge"

    if [ -f "${COGACT_OUT}/feats_A_img.pt" ] && [ -f "${COGACT_OUT}/feats_A_txt.pt" ]; then
        log "[cogact-${SIZE}] already done, skipping."
        continue
    fi

    case "${SIZE}" in
        Small) DIT="DiT-S" ;;
        Base)  DIT="DiT-B" ;;
        Large) DIT="DiT-L" ;;
    esac

    log "--- cogact-${SIZE} ---"
    python cknna/extract_features_cogact.py \
        --ckpt "CogACT/CogACT-${SIZE}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${COGACT_OUT}" \
        --action_model_type "${DIT}" || log "[cogact-${SIZE}] FAILED"

    # CogACT: feats_action == feats_A
    if [ -f "${COGACT_OUT}/feats_A.pt" ] && [ ! -f "${COGACT_OUT}/feats_action.pt" ]; then
        cp "${COGACT_OUT}/feats_A.pt" "${COGACT_OUT}/feats_action.pt"
        log "[cogact-${SIZE}] copied feats_A.pt -> feats_action.pt"
    fi
done
cd "${STAR}"

# =========================================================================
# SECTION 6: GR00T N1.5
# =========================================================================
log "SECTION 6: GR00T N1.5"
source "${CONDA}/bin/activate" groot_libero
GROOT15_OUT="${DATA_DIR}/groot-n15-bridge"

if [ -f "${GROOT15_OUT}/feats_A_img.pt" ] && [ -f "${GROOT15_OUT}/feats_A_txt.pt" ]; then
    log "[groot-n15] already done, skipping."
else
    log "--- groot-n15-bridge ---"
    PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n15.py" \
        --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT15_OUT}" || log "[groot-n15] FAILED"
fi

# =========================================================================
# SECTION 7: GR00T N1.6
# =========================================================================
log "SECTION 7: GR00T N1.6"
GROOT16_OUT="${DATA_DIR}/groot-n16-bridge"

if [ -f "${GROOT16_OUT}/feats_A_img.pt" ] && [ -f "${GROOT16_OUT}/feats_A_txt.pt" ]; then
    log "[groot-n16] already done, skipping."
else
    log "--- groot-n16-bridge ---"
    PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n16.py" \
        --ckpt "${WORK}/GR00T-N1.6-bridge" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT16_OUT}" || log "[groot-n16] FAILED"
fi

# =========================================================================
# SECTION 8: CKNNA Computation (3 variants x 2 metrics)
# =========================================================================
log "SECTION 8: CKNNA Computation"
source "${CONDA}/bin/activate" starVLA
cd "${STAR}"

RESULTS_DIR="${DATA_DIR}/3way_cknna_results"
mkdir -p "${RESULTS_DIR}"

FEATS_B="${DATA_DIR}/feats_B.pt"

# Models with 3-way features (15 models, excluding RT-1-X and Octo)
MODELS_3WAY=(
    Qwen-FAST-Bridge-RT-1
    Qwen-OFT-Bridge-RT-1
    Qwen-GR00T-Bridge
    Qwen-GR00T-Bridge-RT-1
    Qwen3VL-GR00T-Bridge-RT-1
    Qwen3VL-OFT-Bridge-RT-1
    spatialvla-sft-bridge
    pi0-lerobot-bridge
    openvla-7b-bridge
    openvla-7b-bridge-ft-200k
    cogact-small-bridge
    cogact-base-bridge
    cogact-large-bridge
    groot-n15-bridge
    groot-n16-bridge
)

for VARIANT in feats_A feats_A_img feats_A_txt; do
    # CKNNA_proprio
    FEATS_A_PATHS=()
    for m in "${MODELS_3WAY[@]}"; do
        p="${DATA_DIR}/${m}/${VARIANT}.pt"
        if [ -f "$p" ]; then
            FEATS_A_PATHS+=("$p")
        else
            log "WARNING: ${p} not found, skipping ${m} for ${VARIANT}"
        fi
    done

    if [ ${#FEATS_A_PATHS[@]} -gt 0 ]; then
        log "--- CKNNA_proprio (${VARIANT}) ---"
        python "${COMPUTE_CKNNA}" \
            --feats_A "${FEATS_A_PATHS[@]}" \
            --feats_B "${FEATS_B}" \
            --topk 5 10 20 \
            --also_mutual_knn \
            --output "${RESULTS_DIR}/cknna_proprio_${VARIANT}.json" || log "CKNNA_proprio ${VARIANT} FAILED"
    fi

    # CKNNA_action (per-model, since feats_action differs)
    log "--- CKNNA_action (${VARIANT}) ---"
    ACTION_RESULTS="{\"models\": {"
    first=true
    for m in "${MODELS_3WAY[@]}"; do
        FA="${DATA_DIR}/${m}/${VARIANT}.pt"
        FACT="${DATA_DIR}/${m}/feats_action.pt"
        if [ ! -f "$FA" ] || [ ! -f "$FACT" ]; then
            continue
        fi

        TMPOUT=$(mktemp /tmp/cknna_action_XXXXXX.json)
        python "${COMPUTE_CKNNA}" \
            --feats_A "$FA" \
            --feats_B "$FACT" \
            --topk 5 10 20 \
            --also_mutual_knn \
            --output "$TMPOUT" 2>/dev/null || { log "[${m}] action FAILED"; rm -f "$TMPOUT"; continue; }

        ENTRY=$(python3 -c "
import json, sys
d = json.load(open('$TMPOUT'))
models = d['models']
key = list(models.keys())[0]
print(json.dumps(models[key]))
" 2>/dev/null)
        rm -f "$TMPOUT"

        if [ "$first" = true ]; then
            first=false
        else
            ACTION_RESULTS+=","
        fi
        ACTION_RESULTS+="\"${m}\": ${ENTRY}"
    done
    ACTION_RESULTS+="}}"
    echo "$ACTION_RESULTS" | python3 -m json.tool > "${RESULTS_DIR}/cknna_action_${VARIANT}.json" 2>/dev/null || \
        echo "$ACTION_RESULTS" > "${RESULTS_DIR}/cknna_action_${VARIANT}.json"
done

# Also run CKNNA_proprio for RT-1-X and Octo with their existing feats_A
log "--- CKNNA_proprio for RT-1-X and Octo (imgtext only) ---"
RT1X_OCTO_PATHS=()
for m in rt1x-bridge octo-base-bridge; do
    p="${DATA_DIR}/${m}/feats_A.pt"
    [ -f "$p" ] && RT1X_OCTO_PATHS+=("$p")
done
if [ ${#RT1X_OCTO_PATHS[@]} -gt 0 ]; then
    python "${COMPUTE_CKNNA}" \
        --feats_A "${RT1X_OCTO_PATHS[@]}" \
        --feats_B "${FEATS_B}" \
        --topk 5 10 20 \
        --also_mutual_knn \
        --output "${RESULTS_DIR}/cknna_proprio_rt1x_octo.json" || log "RT1X/Octo CKNNA FAILED"
fi

log "=========================================="
log "3-Way Pipeline Complete"
log "Results in: ${RESULTS_DIR}/"
log "=========================================="
