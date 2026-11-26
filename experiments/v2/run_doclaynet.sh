#!/usr/bin/env bash
# v2 full run: teacher QA for every DocLayNet train page (72 shards, ~69K pages). Resumable: rerun to continue.
# Shares GPU1 with other services: 8B teacher at 0.28 of memory, started only when that much is free.
cd "$(dirname "$0")/../.."
free_gb() { nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
until grep -q "download done" experiments/v2/download.log 2>/dev/null; do sleep 60; done
until [ "$(free_gb)" -ge 28 ]; do sleep 120; done
echo "=== start $(date '+%F %T') (free $(free_gb) GB)"
exec 9>locks/gpu1-slotA.lock; flock 9
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 ~/.conda/envs/vlm-eval/bin/python experiments/v2/gen_doc_qa.py \
  --parquet data/v2/raw/doclaynet/data/train-*.parquet --gpu-mem 0.28 --out-dir data/v2/doclaynet-qa
echo "=== exit $? $(date '+%F %T')"
