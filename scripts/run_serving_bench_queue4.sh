#!/usr/bin/env bash
# serving-bench queue 4 (GPU 1 slot B): same-pixel-budget ceiling = Qwen3-VL-2B-Instruct with max_pixels 1.6M.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 GPU_MEM=0.45 VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== ceiling-1.6MP quick start $(date -u +%T)"
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa CHAT_TEMPLATE=none \
  scripts/run_benchmarks.sh exports/serve-bench-qwen3vl2b-1p6mp quick; echo "exit $?"
echo "=== done $(date -u +%T)"
