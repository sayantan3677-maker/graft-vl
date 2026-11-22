#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
TRAIN=$(ls -d manifests/s1-50k manifests/s2/*/ | grep -v "/_")
for RUN in PILOT-M15-Q3-1B7-384-FULLFT-800 PILOT-M15-Q3-1B7-512-FULLFT-800 PILOT-M15-Q3-1B7-384-FULLFT-VIS-800; do
  echo "=== $RUN $(date -u +%T)"
  CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_vqa.py --run experiments/$RUN --eval-data manifests/eval-official --train-data $TRAIN --n 1400 > experiments/$RUN.vqa-official.log 2>&1 || echo "failed $RUN"
  mv experiments/$RUN/eval/vqa-short-step-0000800.json experiments/$RUN/eval/vqa-official-step-0000800.json 2>/dev/null
done
echo "=== done $(date -u +%T)"
