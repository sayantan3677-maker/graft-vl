#!/bin/bash
# quality-retention text queue v2 (replaces the per-task loop of run_main.sh after the baseline finished):
# same settings, but eval_text_batched.py (one combined vLLM call) to avoid long-tail GPU idling.
# Run under locks/gpu1-slotA.lock. vLLM 0.36 while an HF job shares the slot (vision.running), else 0.45.
cd /home/sayantan_dev/vlm_training
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
EVAL_PY=~/.conda/envs/vlm-eval/bin/python; D=experiments/quality-retention; T=reports/text/quality-retention
for N in hyb-1b7-s2000-l10 hyb-1b7-s2000-l05 hyb-1b7-s3000-l10 hyb-1b7-s3000-l05 hyb-1b7-s2000-l075 hyb-1b7-s3000-l075; do
  OUT=$T/$N.json
  if [ -f $OUT ] && python3 -c "import json,sys;sys.exit(0 if len(json.load(open('$OUT'))['results'])==8 else 1)"; then continue; fi
  GM=0.45; [ -f $D/vision.running ] && GM=0.36
  echo "=== text $N gpu-mem $GM $(date +%T)"
  $EVAL_PY scripts/eval_text_batched.py --model exports/$N-llm --tasks mmlu_pro gsm8k ifeval math500 --modes think nothink --seeds 2 \
    --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem $GM --out $OUT --items-out $T/$N.items.json > $D/text-$N.log 2>&1 || echo "!!! text $N failed"
  grep -E "acc .* ±|total" $D/text-$N.log
done
echo "=== text queue done $(date +%T)"
