#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=PILOT-M15-Q3-1B7-384-FULLFT-VIS-800
echo "=== train $(date -u +%T)"
$PY scripts/train.py --config configs/train/pilot_m15_384_vis.json > experiments/${RUN}.train.log 2>&1 || { echo "train failed"; exit 1; }
echo "=== eval $(date -u +%T)"
$PY scripts/eval_vqa.py --run experiments/$RUN --eval-data manifests/qa-eval-1500 --train-data manifests/s1-50k $(ls -d manifests/s2/*/ | grep -v "/_") > experiments/${RUN}.vqa.log 2>&1 || echo "eval failed"
echo "=== done $(date -u +%T)"
