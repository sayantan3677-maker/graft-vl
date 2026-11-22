#!/usr/bin/env bash
# lmms-eval quick benchmarks (limit 500) for the 4B exports on GPU1 (slot B), vLLM at 0.45 memory.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
exec 9>locks/gpu1-slotB.lock; flock 9
for E in hyb-4b-s2000-l05 hyb-4b-s2000-l10; do
  echo "=== bench $E $(date +%T)"
  TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa scripts/run_benchmarks.sh exports/$E quick || echo "!!! bench $E failed"
  echo "=== mmmu $E $(date +%T)"
  TASKS=mmmu_val MM_LIMIT=5 OUT_TAG=mmmu scripts/run_benchmarks.sh exports/$E quick || echo "!!! mmmu $E failed"
done
echo "=== bench done $(date +%T)"
