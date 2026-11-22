#!/usr/bin/env bash
# M1.5: S1 alignment for SigLIP2-512 (needed before the 384-vs-512 comparison). Waits for the S2-prep GPU queue.
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q "=== done" experiments/serving_check.log 2>/dev/null; do sleep 60; done
RUN=EXP-S1-Q3-1B7-S2P16-512-MLP-50K-LR1e-3
echo "=== train $RUN $(date -u +%T)"
$PY scripts/train.py --config configs/train/s1_50k_512_lr1e-3.json > experiments/${RUN}.train.log 2>&1 || { echo "train failed"; exit 1; }
echo "=== eval $(date -u +%T)"
$PY scripts/eval_s1.py --run experiments/$RUN --eval-data manifests/s1-eval300 --train-data manifests/s1-50k --n 300 > experiments/${RUN}.eval.log 2>&1 || echo "eval failed"
echo "=== done $(date -u +%T)"
