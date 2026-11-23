#!/usr/bin/env bash
# Upper-bound reference for the model card: Qwen3-VL-4B-Instruct (same vision encoder, Qwen's full VL training),
# same lmms-eval quick settings as our runs. MAX_LEN 32768 as for the Qwen3-VL-2B reference: at Qwen's
# default 16.7 MP pixel cap some DocVQA pages exceed 16K tokens. Starts after the S4 evaluation (GPU1).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
until grep -q "eval done" experiments/s4/eval.log 2>/dev/null; do sleep 60; done
exec 9>locks/gpu1-slotB.lock; flock 9
M=Qwen/Qwen3-VL-4B-Instruct
echo "=== bench $M $(date +%T)"
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa MAX_LEN=32768 scripts/run_benchmarks.sh $M quick || echo "!!! bench failed"
echo "=== mmmu $M $(date +%T)"
TASKS=mmmu_val MM_LIMIT=5 OUT_TAG=mmmu scripts/run_benchmarks.sh $M quick || echo "!!! mmmu failed"
echo "=== all done $(date +%T)"
