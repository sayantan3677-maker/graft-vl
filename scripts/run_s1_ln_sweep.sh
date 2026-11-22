#!/usr/bin/env bash
# Waits for the first S1 sweep, then runs the LayerNorm-projector variants (+ eval).
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q "sweep done" experiments/s1_sweep.log; do sleep 30; done
for LR in 3e-4 1e-3; do
  RUN=EXP-S1-Q3-1B7-S2P16-384-MLPLN-5K-LR${LR}
  echo "=== train $RUN $(date -u +%T)"
  $PY scripts/train.py --config configs/train/s1_5k_ln_lr${LR}.json > experiments/${RUN}.train.log 2>&1 || { echo "train failed: $RUN"; continue; }
  echo "=== eval $RUN $(date -u +%T)"
  $PY scripts/eval_s1.py --run experiments/$RUN --eval-data manifests/s1-eval300 --train-data manifests/s1-5k --n 300 --n-generate 12 > experiments/${RUN}.eval.log 2>&1 || echo "eval failed: $RUN"
done
echo "=== ln sweep done $(date -u +%T)"
