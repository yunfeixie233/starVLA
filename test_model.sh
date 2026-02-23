#!/bin/bash
CKPT=$1
NAME=$2
PORT=6678
STAR=/home/ubuntu/verl/starVLA
SIMPLER=/home/ubuntu/verl/SimplerEnv
CONDA=/home/ubuntu/verl/conda/bin/activate

kill $(lsof -ti:$PORT) 2>/dev/null
sleep 1

echo "=== Starting server for: $NAME ==="
source $CONDA starVLA
cd $STAR
export PYTHONPATH=${STAR}:${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port $PORT --use_bf16 > /tmp/server.log 2>&1 &
SPID=$!

for i in $(seq 1 60); do
  if grep -q "server listening" /tmp/server.log 2>/dev/null; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "FAIL: $NAME -- server died"; tail -20 /tmp/server.log; exit 1; fi
  sleep 2
done

if ! grep -q "server listening" /tmp/server.log 2>/dev/null; then
  echo "FAIL: $NAME -- timeout"
  kill $SPID 2>/dev/null
  exit 1
fi

echo "Server ready, running eval..."
source $CONDA simpler_env
cd $STAR
export PYTHONPATH=${STAR}:${SIMPLER}:${PYTHONPATH}
python examples/SimplerEnv/eval_files/start_simpler_env.py \
  --ckpt-path "$CKPT" --port $PORT --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 3 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
  --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ${SIMPLER}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 1 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 > /tmp/eval.log 2>&1
RC=$?

if [ $RC -eq 0 ]; then
  echo "PASS: $NAME"
  grep "Average success" /tmp/eval.log
else
  echo "FAIL: $NAME (exit $RC)"
  tail -20 /tmp/eval.log
fi

kill $SPID 2>/dev/null
wait $SPID 2>/dev/null
