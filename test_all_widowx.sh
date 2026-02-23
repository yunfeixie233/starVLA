#!/bin/bash
set -e

STAR_VLA_DIR=/home/ubuntu/verl/starVLA
SIMPLER_ENV_DIR=/home/ubuntu/verl/SimplerEnv
CONDA=/home/ubuntu/verl/conda/bin/activate
PORT=6678

CKPTS=(
  "Qwen-FAST-Bridge-RT-1:steps_10000_pytorch_model.pt"
  "Qwen-OFT-Bridge-RT-1:steps_10000_pytorch_model.pt"
  "Qwen-PI-Bridge-RT-1:steps_30000_pytorch_model.pt"
  "Qwen-GR00T-Bridge:steps_45000_pytorch_model.pt"
  "Qwen3VL-GR00T-Bridge-RT-1:steps_20000_pytorch_model.pt"
  "Qwen3VL-OFT-Bridge-RT-1:steps_5000_pytorch_model.pt"
)

RESULTS_FILE="${STAR_VLA_DIR}/widowx_smoke_test_results.txt"
> "$RESULTS_FILE"

for entry in "${CKPTS[@]}"; do
  IFS=":" read -r model_name pt_file <<< "$entry"
  ckpt_path="${STAR_VLA_DIR}/playground/Pretrained_models/${model_name}/checkpoints/${pt_file}"

  echo "========================================" | tee -a "$RESULTS_FILE"
  echo "Testing: ${model_name}" | tee -a "$RESULTS_FILE"
  echo "Checkpoint: ${ckpt_path}" | tee -a "$RESULTS_FILE"
  echo "========================================" | tee -a "$RESULTS_FILE"

  # Start policy server in background
  source $CONDA starVLA
  cd $STAR_VLA_DIR
  export PYTHONPATH=${STAR_VLA_DIR}:${PYTHONPATH}
  CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
    --ckpt_path "$ckpt_path" \
    --port $PORT \
    --use_bf16 \
    > /tmp/server_${model_name}.log 2>&1 &
  SERVER_PID=$!
  echo "Server PID: $SERVER_PID"

  # Wait for server to be ready
  echo "Waiting for server to load model..."
  for i in $(seq 1 60); do
    if grep -q "server listening" /tmp/server_${model_name}.log 2>/dev/null; then
      echo "Server ready after ${i}s"
      break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "FAIL: Server process died" | tee -a "$RESULTS_FILE"
      cat /tmp/server_${model_name}.log >> "$RESULTS_FILE"
      break 2
    fi
    sleep 2
  done

  if ! grep -q "server listening" /tmp/server_${model_name}.log 2>/dev/null; then
    echo "FAIL: Server did not start within timeout" | tee -a "$RESULTS_FILE"
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    continue
  fi

  # Run sim client (3 steps only)
  source $CONDA simpler_env
  cd $STAR_VLA_DIR
  export PYTHONPATH=${STAR_VLA_DIR}:${SIMPLER_ENV_DIR}:${PYTHONPATH}
  python examples/SimplerEnv/eval_files/start_simpler_env.py \
    --ckpt-path "$ckpt_path" \
    --port $PORT \
    --robot widowx \
    --policy-setup widowx_bridge \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps 3 \
    --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
    --scene-name bridge_table_1_v1 \
    --rgb-overlay-path ${SIMPLER_ENV_DIR}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    --robot-init-x 0.147 0.147 1 \
    --robot-init-y 0.028 0.028 1 \
    --obj-variation-mode episode \
    --obj-episode-range 0 1 \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    > /tmp/eval_${model_name}.log 2>&1
  EVAL_EXIT=$?

  if [ $EVAL_EXIT -eq 0 ]; then
    echo "PASS: ${model_name}" | tee -a "$RESULTS_FILE"
  else
    echo "FAIL: ${model_name} (exit code: $EVAL_EXIT)" | tee -a "$RESULTS_FILE"
    tail -20 /tmp/eval_${model_name}.log >> "$RESULTS_FILE"
  fi

  # Kill server
  kill $SERVER_PID 2>/dev/null
  wait $SERVER_PID 2>/dev/null
  sleep 2
  echo "" | tee -a "$RESULTS_FILE"
done

echo ""
echo "====== SUMMARY ======"
cat "$RESULTS_FILE"
