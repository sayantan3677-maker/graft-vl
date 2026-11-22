#!/usr/bin/env bash
# Reading alignment (projector-only) → OCR probe + official-split QA.
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f manifests/synth-reading-200k/manifest.json ]; do sleep 30; done
RUN=EXP-S1R-Q3-1B7-384-MLP-READING-2K
echo "=== train $(date -u +%T)"
$PY scripts/train.py --config configs/train/s1r_reading_projector.json > experiments/$RUN.train.log 2>&1 || { echo "train failed"; exit 1; }
for S in 1000 2000; do
  echo "=== probe $S $(date -u +%T)"
  $PY scripts/ocr_probe.py --run experiments/$RUN --step $S --n 30 > experiments/$RUN.probe-$S.log 2>&1
  grep "^font" experiments/$RUN.probe-$S.log
done
echo "=== done $(date -u +%T)"
