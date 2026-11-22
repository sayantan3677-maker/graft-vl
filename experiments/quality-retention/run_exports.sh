#!/bin/bash
# quality-retention: 6 exports (steps 2000/3000 x LoRA scale 1.0/0.75/0.5). Run under locks/gpu1-slotA.lock.
cd /home/sayantan_dev/vlm_training
PY=~/.conda/envs/vlm-train/bin/python
for STEP in 2000 3000; do for S in 1.0 0.75 0.5; do
  TAG=$(echo $S | sed 's/\.//; s/^10$/10/'); case $S in 1.0) TAG=10;; 0.75) TAG=075;; 0.5) TAG=05;; esac
  OUT=exports/hyb-1b7-s${STEP}-l${TAG}
  if [ -f $OUT/export_info.json ] && [ -f $OUT-llm/config.json ]; then echo "skip $OUT"; continue; fi
  echo "=== export $OUT $(date +%T)"; df -h / | tail -1
  CUDA_VISIBLE_DEVICES=1 $PY scripts/export_hybrid.py --run experiments/HYB-S2-Q3-1B7-LORA32-3000 --step $STEP --lora-scale $S --out $OUT || exit 1
done; done
echo "=== exports done $(date +%T)"; du -sh exports/hyb-1b7-*
