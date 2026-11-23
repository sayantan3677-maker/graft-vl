#!/usr/bin/env bash
# S4F release run: S4 recipe + rationale-style answer normalization (GPU1).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>locks/gpu1.lock; flock 9
~/.conda/envs/vlm-train/bin/python scripts/train_hybrid.py --config configs/train/hybrid_s4f_release_4b.json \
  > experiments/HYB-S4F-Q3-4B-VIT8-HR-2000.train.log 2>&1
echo "exit $? $(date +%T)" >> experiments/s4f/train.done
