#!/usr/bin/env bash
# Injection + thinking checks for S3 candidates (GPU1 slot A).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/.conda/envs/vlm-train/bin/python; D=experiments/s3-eval; mkdir -p $D/injection $D/thinking
exec 9>locks/gpu1-slotA.lock; flock 9
for E in hyb-4b-s3-s4000-l075 hyb-4b-s3-s4000-l05 hyb-4b-s3-s4000-l10; do
  echo "=== injection $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py injection --export exports/$E --out $D/injection/$E.json > $D/injection/$E.log 2>&1 || echo "!!! injection $E failed"
  echo "=== thinking $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py thinking --export exports/$E --n-qa 40 --nothink-json $D/vision/$E.qa.json --out $D/thinking/$E.json > $D/thinking/$E.log 2>&1 || echo "!!! thinking $E failed"
done
echo "=== aux done $(date +%T)"
