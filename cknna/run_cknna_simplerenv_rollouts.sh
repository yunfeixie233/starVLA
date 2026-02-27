#!/bin/bash
set -e

CONDA_BASE=/home/ubuntu/verl/conda
SIMPLER_ENV_DIR=/home/ubuntu/verl/SimplerEnv-OpenVLA
STARVLA_DIR=/home/ubuntu/verl/starVLA
ROLLOUT_BASE=/home/ubuntu/verl/cknna_rollout_data
GPU_ID=${GPU_ID:-0}

export DISPLAY=:1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Tasks and their configs
TASKS_V1="PutCarrotOnPlateInScene-v0 StackGreenCubeOnYellowCubeBakedTexInScene-v0 PutSpoonOnTableClothInScene-v0"
TASK_V2="PutEggplantInBasketScene-v0"

# V1 params
V1_ROBOT=widowx
V1_SCENE=bridge_table_1_v1
V1_OVERLAY=${SIMPLER_ENV_DIR}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
V1_INIT_X=0.147
V1_INIT_Y=0.028
V1_MAX_STEPS=60

# V2 params (PutEggplant)
V2_ROBOT=widowx_sink_camera_setup
V2_SCENE=bridge_table_1_v2
V2_OVERLAY=${SIMPLER_ENV_DIR}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
V2_INIT_X=0.127
V2_INIT_Y=0.06
V2_MAX_STEPS=120

run_eval_v1() {
    local policy_model=$1
    local ckpt_path=$2
    local save_dir=$3
    local env_name=$4
    local extra_args="${5:-}"

    echo "  [V1] ${env_name} ..."
    python ${SIMPLER_ENV_DIR}/simpler_env/main_inference.py \
        --policy-model ${policy_model} \
        --ckpt-path "${ckpt_path}" \
        --robot ${V1_ROBOT} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps ${V1_MAX_STEPS} \
        --env-name "${env_name}" --scene-name ${V1_SCENE} \
        --rgb-overlay-path ${V1_OVERLAY} \
        --robot-init-x ${V1_INIT_X} ${V1_INIT_X} 1 \
        --robot-init-y ${V1_INIT_Y} ${V1_INIT_Y} 1 \
        --obj-variation-mode episode --obj-episode-range 0 24 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --save-rollout-dir "${save_dir}" \
        --logging-dir "${save_dir}/eval_videos" \
        ${extra_args}
}

run_eval_v2() {
    local policy_model=$1
    local ckpt_path=$2
    local save_dir=$3
    local extra_args="${4:-}"

    echo "  [V2] PutEggplantInBasketScene-v0 ..."
    python ${SIMPLER_ENV_DIR}/simpler_env/main_inference.py \
        --policy-model ${policy_model} \
        --ckpt-path "${ckpt_path}" \
        --robot ${V2_ROBOT} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps ${V2_MAX_STEPS} \
        --env-name PutEggplantInBasketScene-v0 --scene-name ${V2_SCENE} \
        --rgb-overlay-path ${V2_OVERLAY} \
        --robot-init-x ${V2_INIT_X} ${V2_INIT_X} 1 \
        --robot-init-y ${V2_INIT_Y} ${V2_INIT_Y} 1 \
        --obj-variation-mode episode --obj-episode-range 0 24 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --save-rollout-dir "${save_dir}" \
        --logging-dir "${save_dir}/eval_videos" \
        ${extra_args}
}

run_all_tasks() {
    local policy_model=$1
    local ckpt_path=$2
    local save_dir=$3
    local extra_args="${4:-}"

    for task in ${TASKS_V1}; do
        run_eval_v1 "${policy_model}" "${ckpt_path}" "${save_dir}" "${task}" "${extra_args}"
    done
    run_eval_v2 "${policy_model}" "${ckpt_path}" "${save_dir}" "${extra_args}"
}

run_starvla_eval_v1() {
    local ckpt_path=$1
    local save_dir=$2
    local env_name=$3
    local port=${4:-6678}

    echo "  [V1-StarVLA] ${env_name} ..."
    python ${STARVLA_DIR}/examples/SimplerEnv/eval_files/start_simpler_env.py \
        --ckpt-path "${ckpt_path}" \
        --port ${port} \
        --robot ${V1_ROBOT} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps ${V1_MAX_STEPS} \
        --env-name "${env_name}" --scene-name ${V1_SCENE} \
        --rgb-overlay-path ${V1_OVERLAY} \
        --robot-init-x ${V1_INIT_X} ${V1_INIT_X} 1 \
        --robot-init-y ${V1_INIT_Y} ${V1_INIT_Y} 1 \
        --obj-variation-mode episode --obj-episode-range 0 24 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --save-rollout-dir "${save_dir}" \
        --logging-dir "${save_dir}/eval_videos"
}

run_starvla_eval_v2() {
    local ckpt_path=$1
    local save_dir=$2
    local port=${3:-6678}

    echo "  [V2-StarVLA] PutEggplantInBasketScene-v0 ..."
    python ${STARVLA_DIR}/examples/SimplerEnv/eval_files/start_simpler_env.py \
        --ckpt-path "${ckpt_path}" \
        --port ${port} \
        --robot ${V2_ROBOT} --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps ${V2_MAX_STEPS} \
        --env-name PutEggplantInBasketScene-v0 --scene-name ${V2_SCENE} \
        --rgb-overlay-path ${V2_OVERLAY} \
        --robot-init-x ${V2_INIT_X} ${V2_INIT_X} 1 \
        --robot-init-y ${V2_INIT_Y} ${V2_INIT_Y} 1 \
        --obj-variation-mode episode --obj-episode-range 0 24 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --save-rollout-dir "${save_dir}" \
        --logging-dir "${save_dir}/eval_videos"
}

run_starvla_all_tasks() {
    local ckpt_path=$1
    local save_dir=$2
    local port=${3:-6678}

    for task in ${TASKS_V1}; do
        run_starvla_eval_v1 "${ckpt_path}" "${save_dir}" "${task}" "${port}"
    done
    run_starvla_eval_v2 "${ckpt_path}" "${save_dir}" "${port}"
}

# ============================================================
# If a model name is given as $1, run only that model.
# Otherwise, run all models.
# ============================================================
TARGET_MODEL=${1:-all}

# ========== GROUP A: simpler_env conda env ==========
run_group_a_simpler() {
    source ${CONDA_BASE}/bin/activate simpler_env
    export LD_LIBRARY_PATH=${CONDA_BASE}/envs/simpler_env/lib:$LD_LIBRARY_PATH

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "openvla_base" ]]; then
        echo "=== OpenVLA-7B base ==="
        run_all_tasks openvla openvla/openvla-7b ${ROLLOUT_BASE}/openvla_base
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "openvla_ft200k" ]]; then
        echo "=== OpenVLA-7B FT-200k ==="
        local FT200K=/home/ubuntu/verl/openvla/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--200000_chkpt
        run_all_tasks openvla "${FT200K}" ${ROLLOUT_BASE}/openvla_ft200k
    fi

    # NOTE: OpenVLA FT-5k/10k/15k checkpoints need local paths.
    # Set OPENVLA_FT_5K, OPENVLA_FT_10K, OPENVLA_FT_15K env vars if available.
    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "openvla_ft5k" ]]; then
        if [ -n "${OPENVLA_FT_5K:-}" ]; then
            echo "=== OpenVLA-7B FT-5k ==="
            run_all_tasks openvla "${OPENVLA_FT_5K}" ${ROLLOUT_BASE}/openvla_ft5k
        else
            echo "SKIP openvla_ft5k: set OPENVLA_FT_5K env var"
        fi
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "openvla_ft10k" ]]; then
        if [ -n "${OPENVLA_FT_10K:-}" ]; then
            echo "=== OpenVLA-7B FT-10k ==="
            run_all_tasks openvla "${OPENVLA_FT_10K}" ${ROLLOUT_BASE}/openvla_ft10k
        else
            echo "SKIP openvla_ft10k: set OPENVLA_FT_10K env var"
        fi
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "openvla_ft15k" ]]; then
        if [ -n "${OPENVLA_FT_15K:-}" ]; then
            echo "=== OpenVLA-7B FT-15k ==="
            run_all_tasks openvla "${OPENVLA_FT_15K}" ${ROLLOUT_BASE}/openvla_ft15k
        else
            echo "SKIP openvla_ft15k: set OPENVLA_FT_15K env var"
        fi
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "rt1x" ]]; then
        echo "=== RT-1-X ==="
        run_all_tasks rt1 rt_1_x_tf_trained_for_002272480_step ${ROLLOUT_BASE}/rt1x
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "octo_base" ]]; then
        echo "=== Octo-base ==="
        run_all_tasks octo-base octo-base ${ROLLOUT_BASE}/octo_base
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "pi0_lerobot" ]]; then
        echo "=== Pi0 (LeRobot) ==="
        # NOTE: Pi0 needs pi0fast_env with a lerobot version that has lerobot.common.optim
        # Switch env for this model:
        source ${CONDA_BASE}/bin/activate pi0fast_env
        export LD_LIBRARY_PATH=${CONDA_BASE}/envs/pi0fast_env/lib:$LD_LIBRARY_PATH
        # ckpt_path must be the resolved HF snapshot local path
        local PI0_CKPT=${PI0_LEROBOT_CKPT:-$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('HaomingSong/lerobot-pi0-bridge'))" 2>/dev/null)}
        run_all_tasks lerobotpifast "${PI0_CKPT}" ${ROLLOUT_BASE}/pi0_lerobot
        # Switch back
        source ${CONDA_BASE}/bin/activate simpler_env
        export LD_LIBRARY_PATH=${CONDA_BASE}/envs/simpler_env/lib:$LD_LIBRARY_PATH
    fi
}

# ========== GROUP A: cogact conda env ==========
run_group_a_cogact() {
    source ${CONDA_BASE}/bin/activate cogact
    export LD_LIBRARY_PATH=${CONDA_BASE}/envs/cogact/lib:$LD_LIBRARY_PATH

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "cogact_small" ]]; then
        echo "=== CogACT-Small ==="
        run_all_tasks cogact CogACT/CogACT-Small ${ROLLOUT_BASE}/cogact_small "--action-model-type DiT-S"
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "cogact_base" ]]; then
        echo "=== CogACT-Base ==="
        run_all_tasks cogact CogACT/CogACT-Base ${ROLLOUT_BASE}/cogact_base "--action-model-type DiT-B"
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "cogact_large" ]]; then
        echo "=== CogACT-Large ==="
        run_all_tasks cogact CogACT/CogACT-Large ${ROLLOUT_BASE}/cogact_large "--action-model-type DiT-L"
    fi

    # SpatialVLA needs spatialvla_env (transformers>=4.47 for Unpack import)
    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "spatialvla" ]]; then
        echo "=== SpatialVLA ==="
        source ${CONDA_BASE}/bin/activate spatialvla_env
        export LD_LIBRARY_PATH=${CONDA_BASE}/envs/spatialvla_env/lib:$LD_LIBRARY_PATH
        run_all_tasks spatialvla IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge ${ROLLOUT_BASE}/spatialvla
        source ${CONDA_BASE}/bin/activate cogact
        export LD_LIBRARY_PATH=${CONDA_BASE}/envs/cogact/lib:$LD_LIBRARY_PATH
    fi
}

# ========== GROUP A: GR00T (groot_libero conda env) ==========
run_group_a_groot() {
    source ${CONDA_BASE}/bin/activate groot_libero
    export LD_LIBRARY_PATH=${CONDA_BASE}/envs/groot_libero/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/ubuntu/verl/Isaac-GR00T:${PYTHONPATH:-}

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "groot_n15" ]]; then
        echo "=== GR00T-N1.5-Bridge ==="
        # NOTE: ckpt_path needs local path or HF repo ID for GR00T N1.5
        if [ -n "${GROOT_N15_CKPT:-}" ]; then
            run_all_tasks gr00t "${GROOT_N15_CKPT}" ${ROLLOUT_BASE}/groot_n15
        else
            echo "SKIP groot_n15: set GROOT_N15_CKPT env var"
        fi
    fi

    if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" == "groot_n16" ]]; then
        echo "=== GR00T-N1.6-Bridge ==="
        if [ -n "${GROOT_N16_CKPT:-}" ]; then
            run_all_tasks gr00t "${GROOT_N16_CKPT}" ${ROLLOUT_BASE}/groot_n16
        else
            echo "SKIP groot_n16: set GROOT_N16_CKPT env var"
        fi
    fi
}

# ========== GROUP B: StarVLA (server + client) ==========
run_group_b_starvla() {
    local CKPT_BASE=${STARVLA_DIR}/playground/Pretrained_models
    local STARVLA_PORT=6678

    declare -A STARVLA_MODELS
    STARVLA_MODELS=(
        ["qwen_groot_bridge"]="${CKPT_BASE}/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
        ["qwen_groot_bridge_rt1"]="${CKPT_BASE}/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
        ["qwen3vl_groot_bridge_rt1"]="${CKPT_BASE}/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
        ["qwen_fast_bridge_rt1"]="${CKPT_BASE}/Qwen-FAST-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
        ["qwen_oft_bridge_rt1"]="${CKPT_BASE}/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
        ["qwen3vl_oft_bridge_rt1"]="${CKPT_BASE}/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
    )

    for model_name in "${!STARVLA_MODELS[@]}"; do
        if [[ "$TARGET_MODEL" != "all" && "$TARGET_MODEL" != "${model_name}" ]]; then
            continue
        fi

        local ckpt=${STARVLA_MODELS[$model_name]}
        if [ ! -f "${ckpt}" ]; then
            echo "SKIP ${model_name}: checkpoint not found at ${ckpt}"
            continue
        fi

        echo "=== StarVLA: ${model_name} ==="

        # Start server in starVLA env
        echo "  Starting policy server..."
        source ${CONDA_BASE}/bin/activate starVLA
        export LD_LIBRARY_PATH=${CONDA_BASE}/envs/starVLA/lib:$LD_LIBRARY_PATH
        cd ${STARVLA_DIR}
        PYTHONPATH=$(pwd):${PYTHONPATH:-} python deployment/model_server/server_policy.py \
            --ckpt_path "${ckpt}" --port ${STARVLA_PORT} &
        local SERVER_PID=$!
        echo "  Server PID: ${SERVER_PID}"
        sleep 30

        # Run eval client in simpler_env
        source ${CONDA_BASE}/bin/activate simpler_env
        export LD_LIBRARY_PATH=${CONDA_BASE}/envs/simpler_env/lib:$LD_LIBRARY_PATH
        cd ${STARVLA_DIR}
        export PYTHONPATH=$(pwd):${PYTHONPATH:-}
        run_starvla_all_tasks "${ckpt}" ${ROLLOUT_BASE}/${model_name} ${STARVLA_PORT}

        # Stop server
        echo "  Stopping server PID ${SERVER_PID}..."
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        echo "  Server stopped."
    done
}

# ============================================================
# MAIN
# ============================================================
echo "CKNNA SimplerEnv Rollout Data Collection"
echo "Target: ${TARGET_MODEL}"
echo "Output: ${ROLLOUT_BASE}"
echo "GPU: ${GPU_ID}"
echo ""

mkdir -p ${ROLLOUT_BASE}

cd ${SIMPLER_ENV_DIR}

if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" =~ ^(openvla|rt1x|octo|pi0) ]]; then
    run_group_a_simpler
fi

if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" =~ ^(cogact|spatialvla) ]]; then
    run_group_a_cogact
fi

if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" =~ ^groot ]]; then
    run_group_a_groot
fi

if [[ "$TARGET_MODEL" == "all" || "$TARGET_MODEL" =~ ^qwen ]]; then
    run_group_b_starvla
fi

echo ""
echo "=== Rollout collection complete ==="
echo "Data saved to: ${ROLLOUT_BASE}"
echo ""
echo "Next steps:"
echo "  1. Run consolidation: python starVLA/cknna/load_simplerenv_rollout_data.py --rollout_dir <dir> --output_dir <dir>"
echo "  2. Run Phase 2 feature extraction"
echo "  3. Run Phase 3 CKNNA computation"
