#!/usr/bin/env bash
# Text retention on GPU0's free memory (root vLLM holds ~80 GB there): small KV budget, same eval settings.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0
EVAL_PY=~/.conda/envs/vlm-eval/bin/python; D=experiments/s3-eval; T=reports/text/s3
exec 9>locks/gpu0.lock; flock 9
for N in "$@"; do
  echo "=== text $N $(date +%T)"
  $EVAL_PY scripts/eval_text_batched.py --model exports/$N-llm --tasks mmlu_pro gsm8k ifeval math500 --modes think nothink --seeds 2 \
    --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem 0.17 --out $T/$N.json --items-out $T/$N.items.json > $D/text-$N.log 2>&1 || echo "!!! text $N failed"
done
echo "=== text0 done $(date +%T)"
