#!/bin/bash
# =============================================================================
# Temporal CKNNA Sweep -- VLM features at t vs proprio/action from t to t+a
# =============================================================================
#
# Sweeps horizon a in {0, 1, 2, 3, 4, 5, 10, 15} and computes
# CKNNA(feats_A, concat(proprio_t, ..., proprio_{t+a})) for all models.
#
# Also runs the same sweep for actions:
# CKNNA(feats_A, concat(action_t, ..., action_{t+a})).
#
# Uses existing N=5000 data in cknna_data/. Augments with sequential
# states/actions if not already present.
#
# Prerequisites:
#   - Phase 1 data (feats_B.pt, images/, metadata.json) in cknna_data/
#   - Phase 2 feats_A already extracted for all models
#
# Usage:
#   bash cknna/run_temporal_cknna.sh
#   bash cknna/run_temporal_cknna.sh --data_dir ./cknna_data_50k
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
COMPUTE="${SCRIPT_DIR}/compute_cknna_large.py"

DATA_DIR="${SCRIPT_DIR}/cknna_data"
if [[ "${1:-}" == "--data_dir" ]]; then
    DATA_DIR="${2}"
    shift 2
fi

MAX_HORIZON=15
HORIZONS=(0 1 2 3 4 5 10 15)
TOPK="5 10 20"
RESULTS_DIR="${DATA_DIR}/temporal_cknna_results"
mkdir -p "${RESULTS_DIR}"

source "${CONDA}/bin/activate" starVLA

# --- Augment with sequential data if needed ---
if [ ! -f "${DATA_DIR}/feats_B_seq.pt" ] || [ ! -f "${DATA_DIR}/actions_seq.pt" ]; then
    echo "Sequential data not found. Running augmentation..."
    python "${SCRIPT_DIR}/augment_sequential_data.py" \
        --data_dir "${DATA_DIR}" \
        --max_horizon "${MAX_HORIZON}" \
        --workers 32
    echo ""
fi

# --- Collect all available feats_A ---
FEATS_A_PATHS=""
for dir in "${DATA_DIR}"/*/; do
    if [ -f "${dir}feats_A.pt" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${dir}feats_A.pt"
    fi
done

if [ -z "${FEATS_A_PATHS}" ]; then
    echo "ERROR: No feats_A.pt files found in ${DATA_DIR}/"
    exit 1
fi

N_MODELS=$(echo ${FEATS_A_PATHS} | wc -w)
echo "============================================="
echo "  Temporal CKNNA Sweep (N=5000)"
echo "============================================="
echo "Data dir  : ${DATA_DIR}"
echo "Models    : ${N_MODELS}"
echo "Horizons  : ${HORIZONS[*]}"
echo "k values  : ${TOPK}"
echo "Results   : ${RESULTS_DIR}/"
echo ""

echo "Models found:"
for f in ${FEATS_A_PATHS}; do
    echo "  - $(basename $(dirname ${f}))"
done
echo ""


# =========================================================================
# Sweep: CKNNA(VLM_t, proprio_{t:t+a})
# =========================================================================
echo "============================================="
echo "  Proprio sweep: CKNNA(VLM_t, proprio_{t:t+a})"
echo "============================================="

for a in "${HORIZONS[@]}"; do
    OUT="${RESULTS_DIR}/cknna_proprio_horizon_${a}.json"
    if [ -f "${OUT}" ]; then
        echo "[proprio a=${a}] already exists, skipping."
        continue
    fi

    echo ""
    echo "--- proprio horizon a=${a} ---"
    python "${COMPUTE}" \
        --feats_A ${FEATS_A_PATHS} \
        --feats_B_seq "${DATA_DIR}/feats_B_seq.pt" \
        --horizon "${a}" \
        --topk ${TOPK} \
        --also_mutual_knn \
        --output "${OUT}"
    echo "[proprio a=${a}] saved to ${OUT}"
done
echo ""


# =========================================================================
# Sweep: CKNNA(VLM_t, action_{t:t+a})
# =========================================================================
echo "============================================="
echo "  Action sweep: CKNNA(VLM_t, action_{t:t+a})"
echo "============================================="

for a in "${HORIZONS[@]}"; do
    OUT="${RESULTS_DIR}/cknna_action_horizon_${a}.json"
    if [ -f "${OUT}" ]; then
        echo "[action a=${a}] already exists, skipping."
        continue
    fi

    echo ""
    echo "--- action horizon a=${a} ---"
    python "${COMPUTE}" \
        --feats_A ${FEATS_A_PATHS} \
        --feats_B_seq "${DATA_DIR}/actions_seq.pt" \
        --horizon "${a}" \
        --topk ${TOPK} \
        --also_mutual_knn \
        --output "${OUT}"
    echo "[action a=${a}] saved to ${OUT}"
done
echo ""


# =========================================================================
# Summary
# =========================================================================
echo "============================================="
echo "  Temporal CKNNA Sweep Complete"
echo "============================================="
echo ""
echo "Results directory: ${RESULTS_DIR}/"
echo ""
echo "Files:"
ls -1 "${RESULTS_DIR}/"
