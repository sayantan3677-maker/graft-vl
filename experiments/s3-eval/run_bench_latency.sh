#!/usr/bin/env bash
# S3 release candidate: lmms-eval quick benchmarks + mmmu, then BF16 serving latency (GPU1 slot B, run alone).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>locks/gpu1-slotB.lock; flock 9
E=${1:-hyb-4b-s3-s4000-l05}
echo "=== bench $E $(date +%T)"
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa scripts/run_benchmarks.sh exports/$E quick || echo "!!! bench $E failed"
echo "=== mmmu $E $(date +%T)"
TASKS=mmmu_val MM_LIMIT=5 OUT_TAG=mmmu scripts/run_benchmarks.sh exports/$E quick || echo "!!! mmmu $E failed"
echo "=== latency $E $(date +%T)"
SUFFIX=-final scripts/run_hybrid_serving_bench.sh exports/$E 8011 bf16 || echo "!!! latency $E failed"
echo "=== all done $(date +%T)"
