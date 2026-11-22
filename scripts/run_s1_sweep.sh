#!/usr/bin/env bash
# S1 projector-LR sweep on s1-5k, each run followed by the image-reliance eval.
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for LR in 1e-4 3e-4 1e-3; do
  RUN=EXP-S1-Q3-1B7-S2P16-384-MLP-5K-LR${LR}
  echo "=== train $RUN $(date -u +%T)"
  $PY scripts/train.py --config configs/train/s1_5k_lr${LR}.json > experiments/${RUN}.train.log 2>&1 || { echo "train failed: $RUN"; continue; }
  echo "=== eval $RUN $(date -u +%T)"
  $PY scripts/eval_s1.py --run experiments/$RUN --eval-data manifests/s1-eval300 --train-data manifests/s1-5k --n 300 --n-generate 12 > experiments/${RUN}.eval.log 2>&1 || echo "eval failed: $RUN"
done
echo "=== sweep done $(date -u +%T)"
