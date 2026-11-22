#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 3994947 2>/dev/null; do sleep 60; done  # old-S2 text regression (thinking QA dropped)
RUN=PILOT-S2R-Q3-1B7-384-VIS9-READING-LLM1e-5-1000
TRAIN=$(ls -d manifests/s1-50k manifests/s2/*/ | grep -v "/_")
echo "=== train $(date -u +%T)"
$PY scripts/train.py --config configs/train/pilot_s2r.json > experiments/$RUN.train.log 2>&1 || { echo "train failed"; exit 1; }
for S in 500 1000; do
  echo "=== probe+eval $S $(date -u +%T)"
  $PY scripts/ocr_probe.py --run experiments/$RUN --step $S --n 30 > experiments/$RUN.probe-$S.log 2>&1; grep "^font" experiments/$RUN.probe-$S.log
  $PY scripts/eval_vqa.py --run experiments/$RUN --step $S --eval-data manifests/eval-official --train-data $TRAIN --n 1400 > experiments/$RUN.vqa-official-$S.log 2>&1 && mv experiments/$RUN/eval/vqa-short-step-$(printf %07d $S).json experiments/$RUN/eval/vqa-official-step-$(printf %07d $S).json
done
echo "=== done $(date -u +%T)"
