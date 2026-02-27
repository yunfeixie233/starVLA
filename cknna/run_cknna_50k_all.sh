#!/bin/bash
# =============================================================================
# CKNNA 50K Pipeline -- ALL models, feats_A + feats_action + CKNNA
# =============================================================================
# Sequential pipeline for all 17 models (6 StarVLA already done + 11 remaining).
# Runs feats_A extraction, feats_action extraction, then CKNNA computation.
#
# Models by conda env:
#   starVLA       : 6 StarVLA models (feats_A done, need feats_action)
#   spatialvla_env: spatialvla-sft-bridge
#   pi0fast_env   : pi0-lerobot-bridge
#   openvla_env   : openvla-7b-bridge, openvla-7b-bridge-ft-200k
#   simpler_env   : RT-1-X, Octo-base
#   cogact        : CogACT-Small, CogACT-Base, CogACT-Large
#   groot_libero  : GR00T-N1.5-Bridge
#   groot_libero  : GR00T-N1.6-Bridge (same env, different PYTHONPATH)
#
# Usage:
#   bash cknna/run_cknna_50k_all.sh 2>&1 | tee /tmp/cknna_50k_all.log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
DATA_DIR="${STAR}/cknna/cknna_data_50k"
COMPUTE_CKNNA="${STAR}/cknna/compute_cknna_large.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "CKNNA 50K Full Pipeline -- ALL models"
log "Data dir: ${DATA_DIR}"
log "=========================================="

# =========================================================================
# SECTION 1: StarVLA feats_action (6 models, starVLA env)
# =========================================================================
log "SECTION 1: StarVLA feats_action extraction"
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

    if [ -f "${out_dir}/feats_action.pt" ]; then
        log "[${name}] feats_action.pt exists, skipping."
        continue
    fi

    log "--- ${name} feats_action ---"
    python cknna/extract_action_repr_starvla.py \
        --ckpt_path "${ckpt}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${out_dir}" \
        --seed 42 || log "[${name}] FAILED"
done


# =========================================================================
# SECTION 2: SpatialVLA (feats_A + feats_action)
# =========================================================================
log "SECTION 2: SpatialVLA"
SPATIALVLA_OUT="${DATA_DIR}/spatialvla-sft-bridge"
source "${CONDA}/bin/activate" spatialvla_env

if [ -f "${SPATIALVLA_OUT}/feats_A.pt" ]; then
    log "[spatialvla] feats_A.pt exists, skipping."
else
    log "--- spatialvla feats_A ---"
    python "${SIMPLER}/cknna/extract_features_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUT}" \
        --unnorm_key "bridge_orig/1.0.0"
fi

if [ -f "${SPATIALVLA_OUT}/feats_action.pt" ]; then
    log "[spatialvla] feats_action.pt exists, skipping."
else
    log "--- spatialvla feats_action ---"
    python "${SIMPLER}/cknna/extract_action_repr_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUT}" \
        --seed 42
fi


# =========================================================================
# SECTION 3: Pi0 lerobot (feats_A + feats_action)
# =========================================================================
log "SECTION 3: Pi0 lerobot"
PI0_OUT="${DATA_DIR}/pi0-lerobot-bridge"
source "${CONDA}/bin/activate" pi0fast_env

if [ -f "${PI0_OUT}/feats_A.pt" ]; then
    log "[pi0] feats_A.pt exists, skipping."
else
    log "--- pi0 feats_A ---"
    python "${SIMPLER}/cknna/extract_features_pi0_lerobot.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUT}"
fi

if [ -f "${PI0_OUT}/feats_action.pt" ]; then
    log "[pi0] feats_action.pt exists, skipping."
else
    log "--- pi0 feats_action ---"
    python "${SIMPLER}/cknna/extract_action_repr_pi0_lerobot.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUT}" \
        --seed 42
fi


# =========================================================================
# SECTION 4: OpenVLA-7B base (feats_A + feats_action)
# =========================================================================
log "SECTION 4: OpenVLA-7B base"
OPENVLA_OUT="${DATA_DIR}/openvla-7b-bridge"
source "${CONDA}/bin/activate" openvla_env

if [ -f "${OPENVLA_OUT}/feats_A.pt" ]; then
    log "[openvla-base] feats_A.pt exists, skipping."
else
    log "--- openvla-base feats_A ---"
    python "${SIMPLER}/cknna/extract_features_openvla.py" \
        --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_OUT}"
fi

if [ -f "${OPENVLA_OUT}/feats_action.pt" ]; then
    log "[openvla-base] feats_action.pt exists, skipping."
else
    log "--- openvla-base feats_action ---"
    python "${SIMPLER}/cknna/extract_action_repr_openvla.py" \
        --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_OUT}" \
        --seed 42
fi


# =========================================================================
# SECTION 5: OpenVLA-7B FT-200k (feats_A + feats_action)
# =========================================================================
log "SECTION 5: OpenVLA-7B FT-200k"
OPENVLA_FT_OUT="${DATA_DIR}/openvla-7b-bridge-ft-200k"

if [ -f "${OPENVLA_FT_OUT}/feats_A.pt" ]; then
    log "[openvla-ft-200k] feats_A.pt exists, skipping."
else
    log "--- openvla-ft-200k feats_A ---"
    python "${SIMPLER}/cknna/extract_features_openvla.py" \
        --ckpt "${WORK}/openvla/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--200000_chkpt" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_FT_OUT}"
fi

if [ -f "${OPENVLA_FT_OUT}/feats_action.pt" ]; then
    log "[openvla-ft-200k] feats_action.pt exists, skipping."
else
    log "--- openvla-ft-200k feats_action ---"
    python "${SIMPLER}/cknna/extract_action_repr_openvla.py" \
        --ckpt "${WORK}/openvla/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--200000_chkpt" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_FT_OUT}" \
        --seed 42
fi


# =========================================================================
# SECTION 6: CogACT Small/Base/Large (feats_A; feats_action == feats_A)
# =========================================================================
log "SECTION 6: CogACT Small/Base/Large"
source "${CONDA}/bin/activate" cogact

for SIZE in Small Base Large; do
    size_lower=$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')
    COGACT_OUT="${DATA_DIR}/cogact-${size_lower}-bridge"

    if [ -f "${COGACT_OUT}/feats_A.pt" ]; then
        log "[cogact-${SIZE}] feats_A.pt exists, skipping."
    else
        log "--- cogact-${SIZE} feats_A ---"
        # DiT variant names: DiT-Small, DiT-B, DiT-L
        case "${SIZE}" in
            Small) DIT="DiT-Small" ;;
            Base)  DIT="DiT-B" ;;
            Large) DIT="DiT-L" ;;
        esac
        python "${SIMPLER}/cknna/extract_features_cogact.py" \
            --ckpt "CogACT/CogACT-${SIZE}" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${COGACT_OUT}" \
            --action_model_type "${DIT}"
    fi

    # CogACT: feats_action == feats_A (same VLM hidden_states, different pooling mask subset)
    # At N=5K, there was no separate action_repr script -- CSV notes feats_action_source is
    # "VLM hidden_states[-1] masked mean-pool (cognition feature is last-token subset)"
    # which is the SAME extraction as feats_A. Copy feats_A as feats_action.
    if [ -f "${COGACT_OUT}/feats_A.pt" ] && [ ! -f "${COGACT_OUT}/feats_action.pt" ]; then
        log "[cogact-${SIZE}] copying feats_A.pt -> feats_action.pt (feats_action==feats_A)"
        cp "${COGACT_OUT}/feats_A.pt" "${COGACT_OUT}/feats_action.pt"
    fi
done


# =========================================================================
# SECTION 7: RT-1-X (feats_A; feats_action == feats_A)
# =========================================================================
log "SECTION 7: RT-1-X"
RT1X_OUT="${DATA_DIR}/rt1x-bridge"
source "${CONDA}/bin/activate" simpler_env

if [ -f "${RT1X_OUT}/feats_A.pt" ]; then
    log "[rt1x] feats_A.pt exists, skipping."
else
    log "--- rt1x feats_A ---"
    python "${SIMPLER}/cknna/extract_features_rt1x.py" \
        --ckpt "${SIMPLER}/checkpoints/rt_1_x_tf_trained_for_002272480_step" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RT1X_OUT}"
fi

# RT-1-X: feats_action == feats_A (post-EfficientNet+FiLM, pre-Transformer; TF SavedModel locked)
if [ -f "${RT1X_OUT}/feats_A.pt" ] && [ ! -f "${RT1X_OUT}/feats_action.pt" ]; then
    log "[rt1x] copying feats_A.pt -> feats_action.pt (feats_action==feats_A)"
    cp "${RT1X_OUT}/feats_A.pt" "${RT1X_OUT}/feats_action.pt"
fi


# =========================================================================
# SECTION 8: Octo-base (feats_A; feats_action == feats_A)
# =========================================================================
log "SECTION 8: Octo-base"
OCTO_OUT="${DATA_DIR}/octo-base-bridge"

if [ -f "${OCTO_OUT}/feats_A.pt" ]; then
    log "[octo] feats_A.pt exists, skipping."
else
    log "--- octo feats_A ---"
    python "${SIMPLER}/cknna/extract_features_octo.py" \
        --ckpt hf://rail-berkeley/octo-base-1.5 \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OCTO_OUT}"
fi

# Octo: feats_action == feats_A (readout_action tokens are both condition and action-head input)
if [ -f "${OCTO_OUT}/feats_A.pt" ] && [ ! -f "${OCTO_OUT}/feats_action.pt" ]; then
    log "[octo] copying feats_A.pt -> feats_action.pt (feats_action==feats_A)"
    cp "${OCTO_OUT}/feats_A.pt" "${OCTO_OUT}/feats_action.pt"
fi


# =========================================================================
# SECTION 9: GR00T N1.5 (feats_A + feats_action)
# =========================================================================
log "SECTION 9: GR00T N1.5"
GROOT15_OUT="${DATA_DIR}/groot-n15-bridge"
source "${CONDA}/bin/activate" groot_libero

if [ -f "${GROOT15_OUT}/feats_A.pt" ]; then
    log "[groot-n15] feats_A.pt exists, skipping."
else
    log "--- groot-n15 feats_A ---"
    PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n15.py" \
        --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT15_OUT}"
fi

if [ -f "${GROOT15_OUT}/feats_action.pt" ]; then
    log "[groot-n15] feats_action.pt exists, skipping."
else
    log "--- groot-n15 feats_action ---"
    PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_action_repr_groot_n15.py" \
        --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT15_OUT}" \
        --seed 42
fi


# =========================================================================
# SECTION 10: GR00T N1.6 (feats_A + feats_action)
# =========================================================================
log "SECTION 10: GR00T N1.6"
GROOT16_OUT="${DATA_DIR}/groot-n16-bridge"

if [ -f "${GROOT16_OUT}/feats_A.pt" ]; then
    log "[groot-n16] feats_A.pt exists, skipping."
else
    log "--- groot-n16 feats_A ---"
    PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n16.py" \
        --ckpt "${WORK}/GR00T-N1.6-bridge" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT16_OUT}"
fi

if [ -f "${GROOT16_OUT}/feats_action.pt" ]; then
    log "[groot-n16] feats_action.pt exists, skipping."
else
    log "--- groot-n16 feats_action ---"
    PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_action_repr_groot_n16.py" \
        --ckpt "${WORK}/GR00T-N1.6-bridge" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${GROOT16_OUT}" \
        --seed 42
fi


# =========================================================================
# SECTION 11: Compute CKNNA_proprio for ALL models (memory-optimized)
# =========================================================================
log "SECTION 11: CKNNA_proprio for ALL models"
source "${CONDA}/bin/activate" starVLA

FEATS_A_PATHS=""
for dir in "${DATA_DIR}"/*/; do
    if [ -f "${dir}feats_A.pt" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${dir}feats_A.pt"
        log "  Found: $(basename ${dir})"
    fi
done

log "Computing CKNNA_proprio (feats_A vs feats_B) ..."
python "${COMPUTE_CKNNA}" \
    --feats_A ${FEATS_A_PATHS} \
    --feats_B "${DATA_DIR}/feats_B.pt" \
    --topk 5 10 20 \
    --also_mutual_knn \
    --output "${DATA_DIR}/cknna_results_proprio_50k.json"


# =========================================================================
# SECTION 12: Compute CKNNA_action for each model (feats_A vs feats_action)
# =========================================================================
log "SECTION 12: CKNNA_action per model"

for dir in "${DATA_DIR}"/*/; do
    name=$(basename "${dir}")
    fa="${dir}feats_A.pt"
    fact="${dir}feats_action.pt"

    if [ ! -f "${fa}" ] || [ ! -f "${fact}" ]; then
        log "[${name}] missing feats_A or feats_action, skipping"
        continue
    fi

    result="${dir}cknna_action_50k.json"
    if [ -f "${result}" ]; then
        log "[${name}] cknna_action_50k.json exists, skipping"
        continue
    fi

    log "--- ${name}: CKNNA(VLM, action) ---"
    python "${COMPUTE_CKNNA}" \
        --feats_A "${fa}" \
        --feats_B "${fact}" \
        --topk 5 10 20 \
        --also_mutual_knn \
        --output "${result}" || log "[${name}] FAILED"
done


# =========================================================================
# SECTION 13: Summary
# =========================================================================
log "=========================================="
log "PIPELINE COMPLETE"
log "=========================================="

log "feats_A extracted:"
find "${DATA_DIR}" -name "feats_A.pt" | wc -l

log "feats_action extracted:"
find "${DATA_DIR}" -name "feats_action.pt" | wc -l

log "CKNNA_proprio results:"
cat "${DATA_DIR}/cknna_results_proprio_50k.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  N={d[\"_meta\"][\"N\"]}, models={len(d[\"models\"])}')
for name, m in sorted(d['models'].items(), key=lambda x: x[1]['cknna_k10'], reverse=True):
    print(f'  {name:<35s} k10={m[\"cknna_k10\"]:.4f}')
"

log "CKNNA_action results:"
for f in "${DATA_DIR}"/*/cknna_action_50k.json; do
    name=$(basename $(dirname "${f}"))
    python3 -c "
import json
d = json.load(open('${f}'))
m = list(d['models'].values())[0]
print(f'  ${name}:<35s} k10={m[\"cknna_k10\"]:.4f}')
" 2>/dev/null || true
done

log "Done."
