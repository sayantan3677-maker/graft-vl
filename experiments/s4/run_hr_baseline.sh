#!/usr/bin/env bash
# Resolution-only baseline: S3 release served at a 2.56 MP cap (vs 1.6 MP), GPU1 beside S4 training.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>locks/gpu1-slotB.lock; flock 9
E=hyb-4b-s3-s4000-l05-hr; D=experiments/s4; mkdir -p $D/vision
echo "=== qa $E $(date +%T)"
~/.conda/envs/vlm-train/bin/python scripts/eval_hybrid_export.py qa --export exports/$E --n-qa 200 --n-probe 100 --max-pixels 2560000 \
  --out $D/vision/$E.qa.json > $D/vision/$E.qa.log 2>&1 || echo "!!! qa failed"
echo "=== bench $E $(date +%T)"
GPU_MEM=0.3 TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench scripts/run_benchmarks.sh exports/$E quick || echo "!!! bench failed"
echo "=== all done $(date +%T)"
