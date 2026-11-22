#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
while ! grep -q "=== done\|train failed" experiments/pilot_lr.log; do sleep 60; done
echo "=== gpqa $(date -u +%T)"
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 ~/.conda/envs/vlm-eval/bin/python scripts/eval_text.py --model Qwen/Qwen3-1.7B \
  --tasks gpqa_diamond --seeds 4 --gpu-mem 0.35 --out reports/text/qwen3-1.7b-original.json >> reports/text/qwen3-1.7b-original.log 2>&1 && echo ok || echo failed
echo "=== done $(date -u +%T)"
