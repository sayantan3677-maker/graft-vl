#!/usr/bin/env bash
# S3 (4B) checkpoint sweep: export (merged LoRA + serving prep) → full official QA (1,400) on GPU1.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/.conda/envs/vlm-train/bin/python; D=experiments/s3-eval; mkdir -p $D/vision
exec 9>locks/gpu1-slotA.lock; flock 9
for spec in 4000:1.0 3000:1.0 2000:1.0 4000:0.5 4000:0.75; do
  S=${spec%:*}; L=${spec#*:}; E=hyb-4b-s3-s${S}-l${L/./}
  echo "=== export $E $(date +%T)"
  [ -f exports/$E/model.safetensors ] || [ -f exports/$E/model.safetensors.index.json ] || \
    $PY scripts/export_hybrid.py --run experiments/HYB-S3-Q3-4B-LORA64-4000 --step $S --lora-scale $L --out exports/$E > $D/export-$E.log 2>&1 || { echo "!!! export $E failed"; continue; }
  echo "=== qa $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py qa --export exports/$E --n-qa 200 --n-probe 100 --out $D/vision/$E.qa.json > $D/vision/$E.qa.log 2>&1 || echo "!!! qa $E failed"
done
echo "=== qa done $(date +%T)"
