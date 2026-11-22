#!/usr/bin/env bash
# Start `vllm serve` on a hybrid export (BF16, then FP8 dynamic), run the client benchmark + accuracy subset, stop.
#   usage: flock locks/gpu1-slotB.lock scripts/run_hybrid_serving_bench.sh <export_dir> [port] [precisions]
set -u -o pipefail
cd "$(dirname "$0")/.."
EXPORT=$1; PORT=${2:-8011}; PRECS=${3:-"bf16 fp8"}
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
VLLM=~/.conda/envs/vlm-eval/bin/vllm
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} VLLM_USE_FLASHINFER_SAMPLER=0
NAME=$(basename $EXPORT); OUT=reports/serving; mkdir -p $OUT logs/serving-bench
for P in $PRECS; do
  Q=""; [ "$P" = fp8 ] && Q="--quantization fp8"
  LOG=logs/serving-bench/serve-$NAME-$P.log
  for attempt in 1 2 3 4 5 6; do
    setsid $VLLM serve $EXPORT --served-model-name $NAME --port $PORT --dtype bfloat16 $Q \
      --gpu-memory-utilization ${GPU_MEM:-0.45} --max-model-len 16384 --limit-mm-per-prompt '{"image":1}' > $LOG 2>&1 &
    SPID=$!
    until curl -sf localhost:$PORT/health >/dev/null || ! kill -0 $SPID 2>/dev/null; do sleep 5; done
    kill -0 $SPID 2>/dev/null && break
    echo "server start failed (attempt $attempt), see $LOG"; tail -3 $LOG; sleep 120
  done
  kill -0 $SPID 2>/dev/null || { echo "giving up on $P"; continue; }
  echo "=== $P server up $(date -u +%T)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  $EVAL_PY scripts/bench_hybrid_serving.py --url http://localhost:$PORT --model $NAME --tag $P ${BENCH_ARGS:-} --out $OUT/bench-$NAME-$P${SUFFIX:-}.json
  echo "=== $P bench exit $? $(date -u +%T)"
  kill -INT -- -$SPID 2>/dev/null; kill -INT $SPID 2>/dev/null
  for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  kill -9 -- -$SPID 2>/dev/null
  sleep 10
done
echo "=== done $(date -u +%T)"
