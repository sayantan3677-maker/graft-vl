#!/bin/bash
# quality-retention aux queue: runs inside slot A while run_main.sh (which holds locks/gpu1-slotA.lock) is active;
# waits for the vision sub-queue to finish so only one HF job shares the slot with vLLM (0.36).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
PY=~/.conda/envs/vlm-train/bin/python; D=experiments/quality-retention; mkdir -p $D/injection $D/thinking
until [ ! -f $D/vision.running ]; do sleep 30; done
pgrep -f run_main.sh > /dev/null || { echo "main queue (lock holder) not running; abort"; exit 1; }
touch $D/vision.running
for E in hybrid-q3-1b7-aligned hyb-1b7-s2000-l10 hyb-1b7-s3000-l10 hyb-1b7-s2000-l05 hyb-1b7-s3000-l05 hyb-1b7-s2000-l075 hyb-1b7-s3000-l075; do
  [ -f $D/injection/$E.json ] && continue
  echo "=== injection $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py injection --export exports/$E --out $D/injection/$E.json > $D/injection/$E.log 2>&1 || echo "!!! injection $E failed"
  python3 -c "import json;d=json.load(open('$D/injection/$E.json'));print(d['canary_rate'],d['by_style'],d['minutes'])"
done
for E in $THINK_EXPORTS; do
  [ -f $D/thinking/$E.json ] && continue
  echo "=== thinking $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py thinking --export exports/$E --n-qa 40 --nothink-json $D/vision/$E.qa.json --out $D/thinking/$E.json > $D/thinking/$E.log 2>&1 || echo "!!! thinking $E failed"
done
rm -f $D/vision.running; echo "=== aux done $(date +%T)"
