#!/usr/bin/env bash
# serving-bench queue (GPU 1 slot B): quick lmms-eval (single-image tasks, image limit 1) for ours + Qwen3-VL-2B,
# then mmmu_val (multi-image, image limit 5) for both.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 GPU_MEM=0.45 VLLM_USE_FLASHINFER_SAMPLER=0
for M in exports/serve-hyb-1b7-s2000 Qwen/Qwen3-VL-2B-Instruct; do
  echo "=== $M quick start $(date -u +%T)"
  TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa scripts/run_benchmarks.sh $M quick; echo "exit $?"
done
for M in exports/serve-hyb-1b7-s2000 Qwen/Qwen3-VL-2B-Instruct; do
  echo "=== $M mmmu start $(date -u +%T)"
  TASKS=mmmu_val MM_LIMIT=5 OUT_TAG=mmmu scripts/run_benchmarks.sh $M quick; echo "exit $?"
done
echo "=== done $(date -u +%T)"
