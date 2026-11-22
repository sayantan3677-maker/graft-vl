#!/usr/bin/env bash
# S4: ViT top-8 unfreeze + 2.56 MP doc-heavy stage on top of the S3 release (GPU1).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>locks/gpu1.lock; flock 9
~/.conda/envs/vlm-train/bin/python scripts/train_hybrid.py --config configs/train/hybrid_s4_vit_hr_4b.json \
  > experiments/HYB-S4-Q3-4B-VIT8-HR-2000.train.log 2>&1
echo "exit $? $(date +%T)" >> experiments/s4/train.done
