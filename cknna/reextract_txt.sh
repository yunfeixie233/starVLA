#!/bin/bash
set -euo pipefail

CONDA="/lambda/nfs/verl/conda"
WORK="/home/ubuntu/verl"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
DATA_DIR="${WORK}/starVLA/cknna/cknna_data"
RESULTS_DIR="${DATA_DIR}/3way_cknna_results"
COMPUTE_CKNNA="${WORK}/starVLA/cknna/compute_cknna_large.py"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=========================================="
log "Re-extract feats_A_txt + CKNNA computation"
log "=========================================="

# =========================================================================
# PHASE 1: Re-extract feats for 9 non-StarVLA models
# =========================================================================

# --- SpatialVLA ---
log "--- spatialvla-sft-bridge ---"
source "${CONDA}/bin/activate" spatialvla_env
cd "${SIMPLER}"
python cknna/extract_features_spatialvla.py \
    --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/spatialvla-sft-bridge"
log "spatialvla DONE"

# --- Pi0 ---
log "--- pi0-lerobot-bridge ---"
source "${CONDA}/bin/activate" pi0fast_env
cd "${SIMPLER}"
python cknna/extract_features_pi0_lerobot.py \
    --ckpt HaomingSong/lerobot-pi0-bridge \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/pi0-lerobot-bridge"
log "pi0 DONE"

# --- OpenVLA ---
log "--- openvla-7b-bridge ---"
source "${CONDA}/bin/activate" openvla
cd "${SIMPLER}"
python cknna/extract_features_openvla.py \
    --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/openvla-7b-bridge"
log "openvla-7b DONE"

log "--- openvla-7b-bridge-ft-200k ---"
python cknna/extract_features_openvla.py \
    --ckpt "${SIMPLER}/checkpoints/openvla-bridge-sft" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/openvla-7b-bridge-ft-200k"
log "openvla-ft DONE"

# --- CogACT ---
source "${CONDA}/bin/activate" cogact
cd "${SIMPLER}"
for SIZE in Small Base Large; do
    size_lower=$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')
    case "${SIZE}" in
        Small) DIT="DiT-S" ;;
        Base)  DIT="DiT-B" ;;
        Large) DIT="DiT-L" ;;
    esac
    log "--- cogact-${size_lower}-bridge ---"
    python cknna/extract_features_cogact.py \
        --ckpt "CogACT/CogACT-${SIZE}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${DATA_DIR}/cogact-${size_lower}-bridge" \
        --action_model_type "${DIT}"
    log "cogact-${SIZE} DONE"
done

# --- GR00T N1.5 ---
log "--- groot-n15-bridge ---"
source "${CONDA}/bin/activate" groot_libero
PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n15.py" \
    --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/groot-n15-bridge"
log "groot-n15 DONE"

# --- GR00T N1.6 ---
log "--- groot-n16-bridge ---"
PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n16.py" \
    --ckpt "${WORK}/GR00T-N1.6-bridge" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/groot-n16-bridge"
log "groot-n16 DONE"

log "=========================================="
log "PHASE 1 COMPLETE - all extractions done"
log "=========================================="

# =========================================================================
# PHASE 2: CKNNA computation
# =========================================================================
source "${CONDA}/bin/activate" starVLA
cd "${WORK}/starVLA"
mkdir -p "${RESULTS_DIR}"

MODELS="Qwen-FAST-Bridge-RT-1 Qwen-OFT-Bridge-RT-1 Qwen-GR00T-Bridge Qwen-GR00T-Bridge-RT-1 Qwen3VL-GR00T-Bridge-RT-1 Qwen3VL-OFT-Bridge-RT-1 spatialvla-sft-bridge pi0-lerobot-bridge openvla-7b-bridge openvla-7b-bridge-ft-200k cogact-small-bridge cogact-base-bridge cogact-large-bridge groot-n15-bridge groot-n16-bridge"

# --- CKNNA_proprio for all 3 variants ---
for VARIANT in feats_A feats_A_img feats_A_txt; do
    log "--- CKNNA_proprio (${VARIANT}) ---"
    PATHS=""
    for m in $MODELS; do PATHS="${PATHS} ${DATA_DIR}/${m}/${VARIANT}.pt"; done
    python "${COMPUTE_CKNNA}" --feats_A $PATHS --feats_B "${DATA_DIR}/feats_B.pt" \
        --topk 5 10 20 --also_mutual_knn \
        --output "${RESULTS_DIR}/cknna_proprio_${VARIANT}.json"
    log "CKNNA_proprio ${VARIANT} DONE"
done

# --- CKNNA_action for all 3 variants (per-model) ---
for VARIANT in feats_A feats_A_img feats_A_txt; do
    log "--- CKNNA_action (${VARIANT}) ---"
    TMPDIR=$(mktemp -d /tmp/cknna_action_XXXXXX)
    for m in $MODELS; do
        FA="${DATA_DIR}/${m}/${VARIANT}.pt"
        FACT="${DATA_DIR}/${m}/feats_action.pt"
        [ ! -f "$FA" ] || [ ! -f "$FACT" ] && continue
        python "${COMPUTE_CKNNA}" \
            --feats_A "$FA" --feats_B "$FACT" \
            --topk 5 10 20 --also_mutual_knn \
            --output "${TMPDIR}/${m}.json" 2>&1 | grep -E "CKNNA|mutual" || true
    done

    # Merge per-model JSONs
    python3 -c "
import json, os, glob
merged = {'models': {}}
for f in sorted(glob.glob('${TMPDIR}/*.json')):
    d = json.load(open(f))
    merged['models'].update(d['models'])
with open('${RESULTS_DIR}/cknna_action_${VARIANT}.json', 'w') as out:
    json.dump(merged, out, indent=2)
print(f'Merged {len(merged[\"models\"])} models -> ${RESULTS_DIR}/cknna_action_${VARIANT}.json')
"
    rm -rf "${TMPDIR}"
    log "CKNNA_action ${VARIANT} DONE"
done

# --- RT-1-X and Octo (imgtext only) ---
log "--- CKNNA for RT-1-X and Octo ---"
RT1X_OCTO_PATHS=""
for m in rt1x-bridge octo-base-bridge; do
    p="${DATA_DIR}/${m}/feats_A.pt"
    [ -f "$p" ] && RT1X_OCTO_PATHS="${RT1X_OCTO_PATHS} $p"
done
if [ -n "$RT1X_OCTO_PATHS" ]; then
    python "${COMPUTE_CKNNA}" --feats_A $RT1X_OCTO_PATHS \
        --feats_B "${DATA_DIR}/feats_B.pt" \
        --topk 5 10 20 --also_mutual_knn \
        --output "${RESULTS_DIR}/cknna_proprio_rt1x_octo.json"
    for m in rt1x-bridge octo-base-bridge; do
        FA="${DATA_DIR}/${m}/feats_A.pt"
        FACT="${DATA_DIR}/${m}/feats_action.pt"
        [ ! -f "$FA" ] || [ ! -f "$FACT" ] && continue
        python "${COMPUTE_CKNNA}" \
            --feats_A "$FA" --feats_B "$FACT" \
            --topk 5 10 20 --also_mutual_knn \
            --output "${RESULTS_DIR}/cknna_action_rt1x_octo_${m}.json"
    done
fi
log "RT-1-X/Octo DONE"

log "=========================================="
log "ALL COMPLETE"
log "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
log "=========================================="
