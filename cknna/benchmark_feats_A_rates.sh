#!/bin/bash
# Quick benchmark: run feats_A extraction for 300 samples per model to get real rates.
# Then extrapolate to N=50K.
#
# Usage: bash cknna/benchmark_feats_A_rates.sh 2>&1 | tee /tmp/benchmark_feats_A.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${STAR}/.." && pwd)"
SIMPLER="${WORK}/SimplerEnv-OpenVLA"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
DATA_DIR="${STAR}/cknna/cknna_data_50k"
BENCH_SAMPLES=300

log() { echo "[$(date '+%H:%M:%S')] $*"; }
rate_to_eta() {
    # $1 = rate (samples/s), print ETA for 50K
    python3 -c "r=$1; print(f'{50000/r/3600:.2f}h ({50000/r/60:.0f}min)')"
}

log "=== feats_A extraction rate benchmark (N=${BENCH_SAMPLES} per model) ==="
log "Data dir: ${DATA_DIR}"
log ""

results=""

# Helper: run extraction script on N samples and report rate
benchmark_model() {
    local name=$1
    local script=$2
    local extra_args=$3
    local bench_out="${DATA_DIR}/_bench_${name}"
    mkdir -p "${bench_out}"

    local t0=$(date +%s%N)
    python "${script}" \
        ${extra_args} \
        --data_dir "${DATA_DIR}" \
        --output_dir "${bench_out}" \
        --num_samples "${BENCH_SAMPLES}" \
        2>&1 | grep -E "rate=|samples/s|Phase 2|Time:" | tail -5
    local t1=$(date +%s%N)
    local elapsed_s=$(( (t1 - t0) / 1000000000 ))
    local rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed_s}:.2f}')" 2>/dev/null || echo "?")

    log "  ${name}: ${rate} samples/s -> 50K ETA = $(rate_to_eta ${rate} 2>/dev/null || echo '?')"
    results="${results}\n  ${name}: ${rate} s/s"

    # Clean up bench output
    rm -rf "${bench_out}"
}


# =====================================================================
# NOTE: extract_features scripts read exactly num_samples from metadata.
# We temporarily patch metadata to N=300 by passing --num_samples if
# the script supports it, OR we rely on early-exit from N=300 images.
# Most scripts use metadata["num_samples"] to iterate -- so we pass
# a temporary data dir with a patched metadata.json pointing to same images.
# =====================================================================

make_bench_data_dir() {
    local bench_dir="${DATA_DIR}/_benchdata"
    mkdir -p "${bench_dir}"
    # Symlink images dir
    if [ ! -L "${bench_dir}/images" ]; then
        ln -s "${DATA_DIR}/images" "${bench_dir}/images"
    fi
    # Write patched metadata with only 300 samples
    python3 -c "
import json, random
m = json.load(open('${DATA_DIR}/metadata.json'))
m['num_samples'] = ${BENCH_SAMPLES}
m['task_descriptions'] = m['task_descriptions'][:${BENCH_SAMPLES}]
json.dump(m, open('${bench_dir}/metadata.json', 'w'))
"
    # Symlink feats_B (scripts may need it for state)
    if [ ! -L "${bench_dir}/feats_B.pt" ]; then
        ln -s "${DATA_DIR}/feats_B.pt" "${bench_dir}/feats_B.pt"
    fi
    echo "${bench_dir}"
}

BENCH_DATA=$(make_bench_data_dir)
log "Bench data dir: ${BENCH_DATA} (${BENCH_SAMPLES} samples)"
log ""


# =====================================================================
# 1. StarVLA (feats_A already done at N=50K, but measure for reference)
# =====================================================================
log "--- 1. StarVLA (Qwen-GR00T-Bridge, reference) ---"
source "${CONDA}/bin/activate" starVLA
export PYTHONNOUSERSITE=1
cd "${STAR}"

t0=$(date +%s%N)
python cknna/extract_features_starvla.py \
    --ckpt_path "playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt" \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_starvla" 2>&1 | grep -E "rate=|samples/s|Phase 2|=== Phase" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA_starvla=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  StarVLA-GR00T feats_A: ${rate} s/s -> 50K = ${ETA_starvla} (already done)"
rm -rf "${BENCH_DATA}/_out_starvla"


# =====================================================================
# 2. SpatialVLA
# =====================================================================
log ""
log "--- 2. SpatialVLA ---"
source "${CONDA}/bin/activate" spatialvla_env

t0=$(date +%s%N)
python "${SIMPLER}/cknna/extract_features_spatialvla.py" \
    --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_spatialvla" \
    --unnorm_key "bridge_orig/1.0.0" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  spatialvla feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_spatialvla"


# =====================================================================
# 3. Pi0 lerobot
# =====================================================================
log ""
log "--- 3. Pi0 lerobot ---"
source "${CONDA}/bin/activate" pi0fast_env

t0=$(date +%s%N)
python "${SIMPLER}/cknna/extract_features_pi0_lerobot.py" \
    --ckpt_path HaomingSong/lerobot-pi0-bridge \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_pi0" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  pi0 feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_pi0"


# =====================================================================
# 4. OpenVLA-7B base
# =====================================================================
log ""
log "--- 4. OpenVLA-7B base ---"
source "${CONDA}/bin/activate" openvla_env

t0=$(date +%s%N)
python "${SIMPLER}/cknna/extract_features_openvla.py" \
    --ckpt "${SIMPLER}/checkpoints/openvla-7b" \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_openvla" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  openvla-base feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_openvla"


# =====================================================================
# 5. OpenVLA-7B FT-200k (same script, different ckpt -- same rate)
# =====================================================================
log ""
log "  openvla-ft-200k feats_A: same rate as base (same architecture)"


# =====================================================================
# 6. CogACT-Base (one size is enough -- same backbone for all 3)
# =====================================================================
log ""
log "--- 6. CogACT-Base ---"
source "${CONDA}/bin/activate" cogact

t0=$(date +%s%N)
python "${SIMPLER}/cknna/extract_features_cogact.py" \
    --ckpt "CogACT/CogACT-Base" \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_cogact" \
    --action_model_type "DiT-B" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  cogact feats_A: ${rate} s/s -> 50K = ${ETA} (x3 for Small/Base/Large)"
rm -rf "${BENCH_DATA}/_out_cogact"


# =====================================================================
# 7. RT-1-X
# =====================================================================
log ""
log "--- 7. RT-1-X ---"
source "${CONDA}/bin/activate" simpler_env

t0=$(date +%s%N)
python "${SIMPLER}/cknna/extract_features_rt1x.py" \
    --ckpt "${SIMPLER}/checkpoints/rt_1_x_tf_trained_for_002272480_step" \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_rt1x" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  rt1x feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_rt1x"


# =====================================================================
# 8. Octo-base (same env)
# =====================================================================
log ""
log "--- 8. Octo-base ---"

t0=$(date +%s%N)
python "${SIMPLER}/cknna/extract_features_octo.py" \
    --ckpt hf://rail-berkeley/octo-base-1.5 \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_octo" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  octo feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_octo"


# =====================================================================
# 9. GR00T N1.5
# =====================================================================
log ""
log "--- 9. GR00T N1.5 ---"
source "${CONDA}/bin/activate" groot_libero

t0=$(date +%s%N)
PYTHONPATH="${WORK}/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n15.py" \
    --ckpt "${WORK}/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_groot15" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  groot-n15 feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_groot15"


# =====================================================================
# 10. GR00T N1.6
# =====================================================================
log ""
log "--- 10. GR00T N1.6 ---"

t0=$(date +%s%N)
PYTHONPATH="${WORK}/gr00t_1p6/Isaac-GR00T" python "${SIMPLER}/cknna/extract_features_groot_n16.py" \
    --ckpt "${WORK}/GR00T-N1.6-bridge" \
    --data_dir "${BENCH_DATA}" \
    --output_dir "${BENCH_DATA}/_out_groot16" 2>&1 | grep -E "rate=|samples/s|\[300" | tail -3
t1=$(date +%s%N)
elapsed=$(python3 -c "print(($t1-$t0)//1000000000)")
rate=$(python3 -c "print(f'{${BENCH_SAMPLES}/${elapsed}:.1f}')")
ETA=$(python3 -c "print(f'{50000/${BENCH_SAMPLES}*${elapsed}/3600:.2f}h')")
log "  groot-n16 feats_A: ${rate} s/s -> 50K = ${ETA}"
rm -rf "${BENCH_DATA}/_out_groot16"


# =====================================================================
# Cleanup
# =====================================================================
rm -rf "${BENCH_DATA}"
log ""
log "=== Benchmark complete ==="
