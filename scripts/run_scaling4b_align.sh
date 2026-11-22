#!/usr/bin/env bash
# scaling-4b stage 2: 4B alignment training (+ 4B replay generation co-scheduled on the same GPU if memory allows),
# alignment checkpoint evals, aligned export. Run under locks/gpu0.lock (GPU 0 only).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0
RUN=HYB-ALIGN-Q3-4B-VIT4B-MERGER-1500
L=experiments/scaling-4b
replay() {  # $1 = gpu mem fraction
  [ -f manifests/replay-q3-4b-16k/manifest.json ] || $EVAL_PY scripts/gen_replay.py --model Qwen/Qwen3-4B --n 16000 --name replay-q3-4b-16k --gpu-mem $1 > $L/gen_replay-4b-16k.log 2>&1
  [ -f manifests/replay-q3-4b-ifsafety-4k/manifest.json ] || $EVAL_PY scripts/gen_replay.py --model Qwen/Qwen3-4B --categories instruction_following safety --n 4000 --name replay-q3-4b-ifsafety-4k --gpu-mem $1 > $L/gen_replay-4b-ifsafety.log 2>&1
  echo "replay done $(date -u +%T): $(cat manifests/replay-q3-4b-16k/manifest.json manifests/replay-q3-4b-ifsafety-4k/manifest.json 2>/dev/null | tr -d '\n ' )"
}
echo "=== align train start $(date -u +%T)"
if [ ! -f experiments/$RUN/checkpoints/step-0001500/trainable.pt ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_hybrid.py --config configs/train/hybrid_align_4b.json > experiments/$RUN.train.log 2>&1 &
  TP=$!
  until ! kill -0 $TP 2>/dev/null || [ "$(wc -l < experiments/$RUN/metrics.jsonl 2>/dev/null || echo 0)" -ge 2 ]; do sleep 30; done
  PEAK=$(tail -1 experiments/$RUN/metrics.jsonl 2>/dev/null | $PY -c "import json,sys; print(int(json.loads(sys.stdin.read())['peak_alloc_gb']))" 2>/dev/null || echo 999)
  echo "train peak after 50 steps: ${PEAK} GB $(date -u +%T)"
  if [ "$PEAK" -lt 40 ]; then echo "co-scheduling replay at gpu-mem 0.40"; replay 0.40; fi
  wait $TP; echo "train exit $? $(date -u +%T)"
fi
replay 0.85
for s in 500 1000 1500; do
  f=experiments/$RUN/eval/hybrid-eval-step-$(printf %07d $s).json
  [ -f $f ] && continue
  echo "=== eval step $s $(date -u +%T)"
  $PY scripts/eval_hybrid.py --run experiments/$RUN --step $s --n-qa 60 2>&1 | grep '^{"checkpoint'
done
BEST=$($PY -c "
import json,glob
r=[(json.load(open(f))['qa_overall'],f) for f in glob.glob('experiments/$RUN/eval/hybrid-eval-step-*.json')]
print(int(max(r)[1].split('step-')[-1][:7]))")
echo "=== export aligned (best step $BEST) $(date -u +%T)"
[ -f exports/hybrid-q3-4b-aligned/export_info.json ] || $PY scripts/export_hybrid.py --run experiments/$RUN --step $BEST --out exports/hybrid-q3-4b-aligned --no-llm-only
echo "=== done $(date -u +%T)"
