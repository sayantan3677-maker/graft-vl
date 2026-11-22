#!/usr/bin/env bash
# HF-side evals for the 4B exports on GPU1 (slot A): full official QA, injection, thinking.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
PY=~/.conda/envs/vlm-train/bin/python; D=experiments/final-tests; mkdir -p $D/vision $D/injection $D/thinking
exec 9>locks/gpu1-slotA.lock; flock 9
for E in hyb-4b-s2000-l05 hyb-4b-s2000-l10; do
  echo "=== qa $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py qa --export exports/$E --n-qa 200 --n-probe 100 --out $D/vision/$E.qa.json > $D/vision/$E.qa.log 2>&1 || echo "!!! qa $E failed"
done
for E in hyb-4b-s2000-l05 hyb-4b-s2000-l10; do
  echo "=== injection $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py injection --export exports/$E --out $D/injection/$E.json > $D/injection/$E.log 2>&1 || echo "!!! injection $E failed"
  echo "=== thinking $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py thinking --export exports/$E --n-qa 40 --nothink-json $D/vision/$E.qa.json --out $D/thinking/$E.json > $D/thinking/$E.log 2>&1 || echo "!!! thinking $E failed"
done
echo "=== hf done $(date +%T)"
