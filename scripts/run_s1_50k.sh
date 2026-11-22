#!/usr/bin/env bash
# S1 on 50K; evaluates every saved checkpoint for image reliance.
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=EXP-S1-Q3-1B7-S2P16-384-MLP-50K-LR1e-3
echo "=== train $(date -u +%T)"
$PY scripts/train.py --config configs/train/s1_50k_lr1e-3.json > experiments/${RUN}.train.log 2>&1 || { echo "train failed"; exit 1; }
for S in 425 850 1275 1700; do
  echo "=== eval step $S $(date -u +%T)"
  $PY scripts/eval_s1.py --run experiments/$RUN --step $S --eval-data manifests/s1-eval300 --train-data manifests/s1-50k --n 300 --n-generate 12 > experiments/${RUN}.eval-$S.log 2>&1 || echo "eval failed: $S"
done
echo "=== done $(date -u +%T)"
