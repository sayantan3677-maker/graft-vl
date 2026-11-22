#!/usr/bin/env bash
# M1.5 decision: equal short S2 pilots at 384 vs 512 (doc/screen/chart-heavy), compared on held-out QA per image type.
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q "=== done" experiments/m15.log 2>/dev/null; do sleep 60; done
for RES in 384 512; do
  RUN=PILOT-M15-Q3-1B7-${RES}-FULLFT-800
  echo "=== train $RUN $(date -u +%T)"
  $PY scripts/train.py --config configs/train/pilot_m15_${RES}.json > experiments/${RUN}.train.log 2>&1 || { echo "train failed $RES"; continue; }
  echo "=== eval $RUN $(date -u +%T)"
  $PY scripts/eval_vqa.py --run experiments/$RUN --eval-data manifests/qa-eval-1500 --train-data manifests/s1-50k $(ls -d manifests/s2/*/ | grep -v "/_") > experiments/${RUN}.vqa.log 2>&1 || echo "eval failed $RES"
done
echo "=== done $(date -u +%T)"
