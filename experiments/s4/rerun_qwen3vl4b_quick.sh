#!/usr/bin/env bash
# Rerun of the Qwen3-VL-4B quick benchmarks with MAX_LEN 32768 (first run hit a 16,413-token DocVQA prompt). GPU1 slot B.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
exec 9>locks/gpu1-slotB.lock; flock 9
echo "=== bench Qwen/Qwen3-VL-4B-Instruct (MAX_LEN 32768) $(date +%T)"
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa MAX_LEN=32768 \
  scripts/run_benchmarks.sh Qwen/Qwen3-VL-4B-Instruct quick || echo "!!! bench failed"
echo "=== all done $(date +%T)"
