#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
while ! grep -q "=== done" experiments/gpu0_queue.log; do sleep 60; done
for attempt in 1 2 3; do
  echo "=== bench attempt $attempt $(date -u +%T)"
  CUDA_VISIBLE_DEVICES=0 GPU_MEM=0.35 VLLM_USE_FLASHINFER_SAMPLER=0 TASKS=ocrbench,pope,chartqa \
    scripts/run_benchmarks.sh exports/s1-q3-1b7-384 quick > reports/bench_s1_smoke.log 2>&1 && { echo "bench ok"; break; } || echo "bench failed"
  sleep 60
done
echo "=== done $(date -u +%T)"
