#!/usr/bin/env bash
# Text retention for S3 exports (same settings as the 4B S2 gate: subsets, 2 seeds, think+nothink), GPU1 slot B.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
EVAL_PY=~/.conda/envs/vlm-eval/bin/python; D=experiments/s3-eval; T=reports/text/s3; mkdir -p $T
for N in "$@"; do
  until [ -f exports/$N-llm/config.json ] && ls exports/$N-llm/*.safetensors >/dev/null 2>&1; do sleep 60; done
  sleep 60
  exec 9>locks/gpu1-slotB.lock; flock 9
  echo "=== text $N $(date +%T)"
  $EVAL_PY scripts/eval_text_batched.py --model exports/$N-llm --tasks mmlu_pro gsm8k ifeval math500 --modes think nothink --seeds 2 \
    --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem 0.45 --out $T/$N.json --items-out $T/$N.items.json > $D/text-$N.log 2>&1 || echo "!!! text $N failed"
  flock -u 9
done
echo "=== text done $(date +%T)"
