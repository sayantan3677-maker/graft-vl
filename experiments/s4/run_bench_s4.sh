#!/usr/bin/env bash
# lmms-eval quick + MMMU for the best S4 export (public-benchmark check of the S4 vs S3 trend). GPU1 slot B, after the Qwen3-VL-4B reference.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "all done" experiments/s4/qwen3vl4b.log 2>/dev/null; do sleep 60; done
exec 9>locks/gpu1-slotB.lock; flock 9
E=hyb-4b-s4-s2000-l05
echo "=== bench $E $(date +%T)"
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa scripts/run_benchmarks.sh exports/$E quick || echo "!!! bench $E failed"
TASKS=mmmu_val MM_LIMIT=5 OUT_TAG=mmmu scripts/run_benchmarks.sh exports/$E quick || echo "!!! mmmu $E failed"
echo "=== all done $(date +%T)"
