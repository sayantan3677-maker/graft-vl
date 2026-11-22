#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=PILOT-LR-Q3-1B7-384-VIS9-LLM2e-5-1000
TRAIN=$(ls -d manifests/s1-50k manifests/s2/*/ | grep -v "/_")
echo "=== train $(date -u +%T)"
$PY scripts/train.py --config configs/train/pilot_lr_2e-5.json > experiments/$RUN.train.log 2>&1 || { echo "train failed"; exit 1; }
for S in 500 1000; do
  echo "=== eval $S $(date -u +%T)"
  $PY scripts/eval_s1.py --run experiments/$RUN --step $S --eval-data manifests/diag-train-docvqa --n 60 --n-generate 3 > experiments/$RUN.diag-docvqa-$S.log 2>&1
  $PY scripts/eval_vqa.py --run experiments/$RUN --step $S --eval-data manifests/eval-official --train-data $TRAIN --n 1400 > experiments/$RUN.vqa-official-$S.log 2>&1 && mv experiments/$RUN/eval/vqa-short-step-$(printf %07d $S).json experiments/$RUN/eval/vqa-official-step-$(printf %07d $S).json
done
echo "=== done $(date -u +%T)"
