#!/bin/bash
# quality-retention main queue (run under locks/gpu1-slotA.lock, GPU 1 only).
# Two parallel sub-queues inside the slot: (a) vLLM text retention, (b) HF vision eval of exports.
# Slot memory cap ~45 GB: while (b) runs (~<10 GB), vLLM uses 0.36; afterwards 0.45.
cd /home/sayantan_dev/vlm_training
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
PY=~/.conda/envs/vlm-train/bin/python; EVAL_PY=~/.conda/envs/vlm-eval/bin/python
D=experiments/quality-retention; mkdir -p $D/vision reports/text/quality-retention
EXPORTS="hyb-1b7-s2000-l10 hyb-1b7-s3000-l10 hyb-1b7-s2000-l05 hyb-1b7-s3000-l05 hyb-1b7-s2000-l075 hyb-1b7-s3000-l075"
rm -f $D/vision.running; touch $D/vision.running
(
  for E in hybrid-q3-1b7-aligned $EXPORTS; do
    [ -f $D/vision/$E.qa.json ] && continue
    echo "=== vision $E $(date +%T)"
    $PY scripts/eval_hybrid_export.py qa --export exports/$E --n-qa 200 --n-probe 100 --out $D/vision/$E.qa.json > $D/vision/$E.qa.log 2>&1 \
      || echo "!!! vision $E failed"
    tail -n 30 $D/vision/$E.qa.log | grep -A12 '^{' | head -14
  done
  rm -f $D/vision.running; echo "=== vision queue done $(date +%T)"
) > $D/vision_queue.log 2>&1 &
VPID=$!
for M in Qwen/Qwen3-1.7B $EXPORTS; do
  N=$(basename $M); [ $N = Qwen3-1.7B ] && N=qwen3-1.7b-original-subset && MP=$M || MP=exports/$M-llm
  OUT=reports/text/quality-retention/$N.json
  if [ -f $OUT ] && python3 -c "import json,sys;sys.exit(0 if len(json.load(open('$OUT'))['results'])==8 else 1)"; then continue; fi
  GM=0.45; [ -f $D/vision.running ] && GM=0.36
  echo "=== text $N gpu-mem $GM $(date +%T)"
  $EVAL_PY scripts/eval_text.py --model $MP --tasks mmlu_pro gsm8k ifeval math500 --modes think nothink --seeds 2 \
    --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem $GM --out $OUT --items-out reports/text/quality-retention/$N.items.json \
    > $D/text-$N.log 2>&1 || echo "!!! text $N failed"
  grep -E "acc .* ±" $D/text-$N.log
done
echo "=== text queue done $(date +%T)"
wait $VPID
echo "=== all done $(date +%T)"
