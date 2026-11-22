#!/usr/bin/env bash
# After the S2-prep GPU queue: vLLM plugin parity on the S1 export, then a quick benchmark smoke run.
set -u
cd "$(dirname "$0")/.."
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
while ! grep -q "=== done" experiments/gpu_queue_s2prep.log; do sleep 60; done
echo "=== parity $(date -u +%T)"
$EVAL_PY scripts/test_vllm_parity.py --export exports/s1-q3-1b7-384 --run experiments/EXP-S1-Q3-1B7-S2P16-384-MLP-50K-LR1e-3 > reports/serving_parity_s1.log 2>&1 && echo "parity PASS" || echo "parity FAIL (see reports/serving_parity_s1.log)"
echo "=== benchmark smoke $(date -u +%T)"
TASKS=ocrbench,pope,chartqa scripts/run_benchmarks.sh exports/s1-q3-1b7-384 quick > reports/bench_s1_smoke.log 2>&1 || echo "bench failed"
echo "=== done $(date -u +%T)"
