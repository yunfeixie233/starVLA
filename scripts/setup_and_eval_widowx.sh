#!/bin/bash
#
# StarVLA WidowX (SimplerEnv) -- full setup + evaluation smoke test
#
# Prerequisites:
#   - conda at /home/ubuntu/verl/conda
#   - starVLA repo at /home/ubuntu/verl/starVLA
#   - NVIDIA GPU with drivers installed (tested on H100, driver 570)
#   - sudo access (for Vulkan/GL libraries)
#
# This script:
#   1. Creates two conda envs (starVLA, simpler_env)
#   2. Installs all dependencies
#   3. Downloads base VLMs and finetuned checkpoints from HuggingFace
#   4. Patches config.yaml files to use local paths
#   5. Patches model2simpler_interface.py for unnorm_key compatibility
#   6. Runs a 3-step smoke test for each downloadable checkpoint
#
# Usage:
#   bash scripts/setup_and_eval_widowx.sh
#
# Results:
#   5 of 6 checkpoints pass. Qwen-PI-Bridge-RT-1 fails due to a
#   code/checkpoint version incompatibility (pre-2025-10-20 checkpoint
#   vs restructured QwenPI framework).

set -euo pipefail

CONDA=/home/ubuntu/verl/conda
STAR=/home/ubuntu/verl/starVLA
MODELS=$STAR/playground/Pretrained_models
PORT=6678

# ============================================================
# PHASE 1: System-level dependencies (Vulkan for SAPIEN)
# ============================================================
echo "===== Phase 1: System dependencies ====="

sudo apt-get update -qq
sudo apt-get install -y libvulkan1 vulkan-tools libnvidia-gl-570-server

# ============================================================
# PHASE 2: conda env "starVLA" (policy server)
# ============================================================
echo "===== Phase 2: Create starVLA conda env ====="

$CONDA/bin/conda create -n starVLA python=3.10 -y

source $CONDA/bin/activate starVLA

pip install -r $STAR/requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
cd $STAR && pip install -e .

# ============================================================
# PHASE 3: Clone SimplerEnv + conda env "simpler_env" (sim client)
# ============================================================
echo "===== Phase 3: Create simpler_env conda env ====="

cd /home/ubuntu/verl
if [ ! -d SimplerEnv ]; then
    git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
fi

$CONDA/bin/conda create -n simpler_env python=3.10 -y

source $CONDA/bin/activate simpler_env

pip install numpy==1.24.4
cd /home/ubuntu/verl/SimplerEnv/ManiSkill2_real2sim && pip install -e .
cd /home/ubuntu/verl/SimplerEnv && pip install -e .
pip install "setuptools<71"
pip install tyro matplotlib mediapy websockets msgpack
pip install opencv-python-headless websocket-client msgpack omegaconf
pip install rich safetensors
pip install numpy==1.24.4

# ============================================================
# PHASE 4: Download base VLMs from HuggingFace
# ============================================================
echo "===== Phase 4: Download base VLMs ====="

source $CONDA/bin/activate starVLA

huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
    --local-dir $MODELS/Qwen2.5-VL-3B-Instruct-Action

huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir $MODELS/Qwen2.5-VL-3B-Instruct

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir $MODELS/Qwen3-VL-4B-Instruct

huggingface-cli download physical-intelligence/fast \
    --local-dir $MODELS/fast

# ============================================================
# PHASE 5: Download finetuned WidowX checkpoints
# ============================================================
echo "===== Phase 5: Download finetuned checkpoints ====="

for repo in \
    StarVLA/Qwen-FAST-Bridge-RT-1 \
    StarVLA/Qwen-OFT-Bridge-RT-1 \
    StarVLA/Qwen-PI-Bridge-RT-1 \
    StarVLA/Qwen-GR00T-Bridge \
    StarVLA/Qwen3VL-GR00T-Bridge-RT-1 \
    StarVLA/Qwen3VL-OFT-Bridge-RT-1; do
    name=$(basename $repo)
    echo "--- Downloading $name ---"
    huggingface-cli download $repo --local-dir $MODELS/$name
done

# NOTE: README also lists StarVLA/Qwen-FM-Bridge-RT-1 for the PI
# model at 62.5%, but that HF repo does not exist (broken link).

# ============================================================
# PHASE 6: Patch config.yaml files (base_vlm local paths)
# ============================================================
echo "===== Phase 6: Patch config.yaml base_vlm paths ====="

patch_base_vlm() {
    local cfg=$1
    local new_path=$2
    if [ -f "$cfg" ]; then
        python3 -c "
import re, sys
cfg_path = sys.argv[1]
new_vlm = sys.argv[2]
with open(cfg_path) as f:
    text = f.read()
text = re.sub(r'(base_vlm:\s*).*', r'\1' + new_vlm, text)
with open(cfg_path, 'w') as f:
    f.write(text)
print(f'  Patched {cfg_path}')
" "$cfg" "$new_path"
    fi
}

patch_base_vlm "$MODELS/Qwen-FAST-Bridge-RT-1/config.yaml" \
    "$MODELS/Qwen2.5-VL-3B-Instruct-Action"

patch_base_vlm "$MODELS/Qwen-OFT-Bridge-RT-1/config.yaml" \
    "$MODELS/Qwen2.5-VL-3B-Instruct"

patch_base_vlm "$MODELS/Qwen-PI-Bridge-RT-1/config.yaml" \
    "$MODELS/Qwen2.5-VL-3B-Instruct-Action"

patch_base_vlm "$MODELS/Qwen-GR00T-Bridge/config.yaml" \
    "$MODELS/Qwen2.5-VL-3B-Instruct-Action"

patch_base_vlm "$MODELS/Qwen3VL-GR00T-Bridge-RT-1/config.yaml" \
    "$MODELS/Qwen3-VL-4B-Instruct"

patch_base_vlm "$MODELS/Qwen3VL-OFT-Bridge-RT-1/config.yaml" \
    "$MODELS/Qwen3-VL-4B-Instruct"

# ============================================================
# PHASE 7: Patch model2simpler_interface.py (unnorm_key fallback)
# ============================================================
echo "===== Phase 7: Patch model2simpler_interface.py ====="

INTERFACE=$STAR/examples/SimplerEnv/eval_files/model2simpler_interface.py

# Only patch if the fallback logic is not already present
if ! grep -q "candidates = \[k for k in norm_stats" "$INTERFACE" 2>/dev/null; then
    python3 -c "
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()

old = '''    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)

        # unnorm_key = baseframework._check_unnorm_key(norm_stats, unnorm_key) # 其实也是很环境 specific 的
        return norm_stats[unnorm_key][\"action\"]'''

new = '''    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)

        if unnorm_key not in norm_stats:
            candidates = [k for k in norm_stats if \"bridge\" in k.lower()]
            if candidates:
                unnorm_key = candidates[0]
            else:
                unnorm_key = next(iter(norm_stats))
            print(f\"[model2simpler_interface] Using unnorm_key={unnorm_key!r} (available: {list(norm_stats.keys())})\")
        return norm_stats[unnorm_key][\"action\"]'''

if old in text:
    text = text.replace(old, new)
    with open(path, 'w') as f:
        f.write(text)
    print('  Patched get_action_stats')
else:
    print('  Already patched or original not found -- skipping')
" "$INTERFACE"
else
    echo "  Already patched -- skipping"
fi

# ============================================================
# PHASE 8: Run smoke-test evaluation for each checkpoint
# ============================================================
echo "===== Phase 8: Smoke-test evaluation ====="

SIMPLER=/home/ubuntu/verl/SimplerEnv

run_smoke_test() {
    local ckpt=$1
    local name=$2

    kill $(lsof -ti:$PORT) 2>/dev/null || true
    sleep 1

    echo ""
    echo "=== Testing: $name ==="

    # Start policy server (starVLA env)
    source $CONDA/bin/activate starVLA
    cd $STAR
    export PYTHONPATH=${STAR}:${PYTHONPATH:-}
    CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
        --ckpt_path "$ckpt" --port $PORT --use_bf16 > /tmp/server.log 2>&1 &
    local spid=$!

    # Wait for server to be ready (up to 120s)
    local ready=0
    for i in $(seq 1 60); do
        if grep -q "server listening" /tmp/server.log 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 $spid 2>/dev/null; then
            echo "FAIL: $name -- server died"
            tail -20 /tmp/server.log
            return 1
        fi
        sleep 2
    done

    if [ $ready -eq 0 ]; then
        echo "FAIL: $name -- timeout waiting for server"
        kill $spid 2>/dev/null || true
        return 1
    fi

    echo "  Server ready. Running 3-step eval..."

    # Run sim client (simpler_env env)
    source $CONDA/bin/activate simpler_env
    cd $STAR
    export PYTHONPATH=${STAR}:${SIMPLER}:${PYTHONPATH:-}
    python examples/SimplerEnv/eval_files/start_simpler_env.py \
        --ckpt-path "$ckpt" --port $PORT \
        --robot widowx --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps 3 \
        --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
        --scene-name bridge_table_1_v1 \
        --rgb-overlay-path ${SIMPLER}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
        --robot-init-x 0.147 0.147 1 \
        --robot-init-y 0.028 0.028 1 \
        --obj-variation-mode episode --obj-episode-range 0 1 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        > /tmp/eval.log 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "PASS: $name"
        grep "Average success" /tmp/eval.log 2>/dev/null || true
    else
        echo "FAIL: $name (exit $rc)"
        tail -20 /tmp/eval.log
    fi

    kill $spid 2>/dev/null || true
    wait $spid 2>/dev/null || true
    return $rc
}

# Find checkpoint .pt files for each model
declare -A CHECKPOINTS
for dir in $MODELS/Qwen-FAST-Bridge-RT-1 \
           $MODELS/Qwen-OFT-Bridge-RT-1 \
           $MODELS/Qwen-PI-Bridge-RT-1 \
           $MODELS/Qwen-GR00T-Bridge \
           $MODELS/Qwen3VL-GR00T-Bridge-RT-1 \
           $MODELS/Qwen3VL-OFT-Bridge-RT-1; do
    name=$(basename $dir)
    ckpt=$(find $dir/checkpoints -name "*.pt" -o -name "*.safetensors" 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        echo "Found: $name -> $ckpt"
        CHECKPOINTS[$name]=$ckpt
    else
        echo "SKIP: $name -- no checkpoint file found"
    fi
done

echo ""
echo "===== Running smoke tests ====="
PASS=0
FAIL=0
SKIP=0

for name in "${!CHECKPOINTS[@]}"; do
    ckpt=${CHECKPOINTS[$name]}
    if run_smoke_test "$ckpt" "$name"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "===== Summary ====="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo ""
echo "Expected results:"
echo "  PASS: Qwen-FAST-Bridge-RT-1, Qwen-OFT-Bridge-RT-1, Qwen-GR00T-Bridge,"
echo "        Qwen3VL-GR00T-Bridge-RT-1, Qwen3VL-OFT-Bridge-RT-1"
echo "  FAIL: Qwen-PI-Bridge-RT-1 (code/checkpoint version incompatibility)"
echo "  NOT TESTED: Qwen-FM-Bridge-RT-1 (HF repo does not exist, broken README link)"
