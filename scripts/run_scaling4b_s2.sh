#!/usr/bin/env bash
# scaling-4b stage 3: 4B S2 LoRA (4B self-distilled replay), checkpoint evals, full eval of best, exports (λ=1.0, 0.5),
# text baseline (original Qwen3-4B) + retention evals. Run under locks/gpu0.lock (GPU 0 only).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0
RUN=HYB-S2-Q3-4B-LORA32-2000
CFG=configs/train/hybrid_s2_lora_4b.json
R=reports/agents/scaling-4b
TEXT_ARGS="--tasks mmlu_pro gsm8k ifeval math500 --seeds 2 --limit mmlu_pro=500 gsm8k=500 math500=200"
for f in exports/hybrid-q3-4b-aligned/export_info.json manifests/replay-q3-4b-16k/manifest.json manifests/replay-q3-4b-ifsafety-4k/manifest.json; do
  [ -f $f ] || { echo "missing prerequisite $f"; exit 1; }
done
evalck() {  # $1 step
  f=experiments/$RUN/eval/hybrid-eval-step-$(printf %07d $1).json
  [ -f $f ] || $PY scripts/eval_hybrid.py --run experiments/$RUN --step $1 --n-qa 60 2>&1 | grep '^{"checkpoint'
}
text_eval() {  # $1 model, $2 out, $3 gpu mem
  [ -f $2.done ] && return
  $EVAL_PY scripts/eval_text.py --model $1 $TEXT_ARGS --gpu-mem $3 --out $2.json --items-out $2.items.json > $2.log 2>&1 && touch $2.done
  echo "text eval $1 exit $? $(date -u +%T)"
}
echo "=== S2 train start $(date -u +%T)"
if [ ! -f experiments/$RUN/checkpoints/step-0002000/trainable.pt ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_hybrid.py --config $CFG > experiments/$RUN.train.log 2>&1 &
  TP=$!
  until ! kill -0 $TP 2>/dev/null || [ "$(wc -l < experiments/$RUN/metrics.jsonl 2>/dev/null || echo 0)" -ge 2 ]; do sleep 30; done
  if ! kill -0 $TP 2>/dev/null && grep -q -i "out of memory" experiments/$RUN.train.log; then
    echo "OOM without gradient checkpointing → retry with it $(date -u +%T)"
    CFG=configs/train/hybrid_s2_lora_4b_gc.json
    $PY -c "import json; c=json.load(open('configs/train/hybrid_s2_lora_4b.json')); c['gradient_checkpointing']=True; json.dump(c,open('$CFG','w'),indent=2)"
    rm -f experiments/$RUN/metrics.jsonl
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_hybrid.py --config $CFG > experiments/$RUN.train.log 2>&1 &
    TP=$!
    until ! kill -0 $TP 2>/dev/null || [ "$(wc -l < experiments/$RUN/metrics.jsonl 2>/dev/null || echo 0)" -ge 2 ]; do sleep 30; done
  fi
  PEAK=$(tail -1 experiments/$RUN/metrics.jsonl 2>/dev/null | $PY -c "import json,sys; print(int(json.loads(sys.stdin.read())['peak_alloc_gb']))" 2>/dev/null || echo 999)
  echo "S2 peak after 50 steps: ${PEAK} GB $(date -u +%T)"
  if [ "$PEAK" -lt 55 ]; then
    echo "co-scheduling text baseline at gpu-mem 0.30 $(date -u +%T)"
    text_eval Qwen/Qwen3-4B $R/text-qwen3-4b-original 0.30
  fi
  if [ "$PEAK" -lt 75 ]; then  # evaluate step 1000 while training continues
    until ! kill -0 $TP 2>/dev/null || [ -f experiments/$RUN/checkpoints/step-0001000/info.json ]; do sleep 60; done
    sleep 30; kill -0 $TP 2>/dev/null && { echo "=== eval step 1000 (co-scheduled) $(date -u +%T)"; evalck 1000; }
  fi
  wait $TP; echo "S2 train exit $? $(date -u +%T)"
fi
for s in 1000 2000; do echo "=== eval step $s $(date -u +%T)"; evalck $s; done
BEST=$($PY -c "
import json,glob
r=[(json.load(open(f))['qa_overall'],f) for f in glob.glob('experiments/$RUN/eval/hybrid-eval-step-*.json')]
print(int(max(r)[1].split('step-')[-1][:7]))")
echo "=== best step $BEST; full 1,400 eval $(date -u +%T)"
[ -f $R/s2-4b-step$BEST-full1400.json ] || $PY scripts/eval_hybrid.py --run experiments/$RUN --step $BEST --n-qa 200 --out $R/s2-4b-step$BEST-full1400.json 2>&1 | grep '^{"checkpoint'
[ -f $R/s2-1b7-step2000-full1400.json ] || $PY scripts/eval_hybrid.py --run experiments/HYB-S2-Q3-1B7-LORA32-3000 --step 2000 --n-qa 200 --out $R/s2-1b7-step2000-full1400.json 2>&1 | grep '^{"checkpoint'
for sc in 10 05; do
  out=exports/hyb-4b-s$BEST-l$sc
  echo "=== export $out $(date -u +%T)"
  [ -f $out-llm/config.json ] || $PY scripts/export_hybrid.py --run experiments/$RUN --step $BEST --lora-scale ${sc:0:1}.${sc:1:1} --out $out 2>&1 | grep -E "merged|base|→|Error"
done
echo "=== text evals $(date -u +%T)"
text_eval Qwen/Qwen3-4B $R/text-qwen3-4b-original 0.85
text_eval exports/hyb-4b-s$BEST-l10-llm $R/text-hyb-4b-s$BEST-l10 0.85
text_eval exports/hyb-4b-s$BEST-l05-llm $R/text-hyb-4b-s$BEST-l05 0.85
echo "=== done $(date -u +%T)"
