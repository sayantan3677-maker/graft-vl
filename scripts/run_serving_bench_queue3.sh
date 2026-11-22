#!/usr/bin/env bash
# serving-bench queue 3 (GPU 1 slot B): Qwen3-VL-2B ceiling quick run at max_model_len 32768 (its default pixel cap
# yields >16K-token prompts on long infographics), then a repeat of the BF16/FP8 serving perf benchmark.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 GPU_MEM=0.45 VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== ceiling quick start $(date -u +%T)"
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa MAX_LEN=32768 \
  scripts/run_benchmarks.sh Qwen/Qwen3-VL-2B-Instruct quick; echo "exit $?"
echo "=== perf repeat start $(date -u +%T)"
BENCH_ARGS="--acc-per-source 0 --min-requests 24 --waves 6" SUFFIX=-rep2 scripts/run_hybrid_serving_bench.sh exports/serve-hyb-1b7-s2000 8011
echo "=== done $(date -u +%T)"
