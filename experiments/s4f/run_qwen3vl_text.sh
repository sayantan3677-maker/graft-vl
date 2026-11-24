#!/usr/bin/env bash
# Text ability of Qwen's own VL models vs Qwen3-4B, same tasks/subsets/seeds as our retention gate:
# Qwen3-VL-4B-Instruct (non-thinking) and Qwen3-VL-4B-Thinking (thinking). GPU1, in parallel when memory allows.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
free_gb() { nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
EVAL_PY=~/.conda/envs/vlm-eval/bin/python; T=reports/text/qwen3-vl
for SPEC in Qwen3-VL-4B-Instruct:nothink Qwen3-VL-4B-Thinking:think; do
  IFS=: read M MODE <<< "$SPEC"
  until [ "$(free_gb)" -ge 22 ]; do sleep 60; done
  echo "=== text $M ($MODE) $(date +%T)"
  $EVAL_PY scripts/eval_text_batched.py --model Qwen/$M --tasks mmlu_pro gsm8k ifeval math500 --modes $MODE --seeds 2 \
    --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem 0.2 --out $T/$M.json --items-out $T/$M.items.json > $T/$M.log 2>&1 &
  sleep 300
done
wait
echo "=== done $(date +%T)"
