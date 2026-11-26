#!/usr/bin/env bash
# v2 teacher-QA pilot (300 DocLayNet pages). Waits until GPU1 has room for the 8B teacher (shared GPU), then runs.
cd "$(dirname "$0")/../.."
free_gb() { nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
until [ "$(free_gb)" -ge 36 ]; do sleep 120; done
echo "=== start $(date '+%F %T') (free $(free_gb) GB)"
exec 9>locks/gpu1-slotA.lock; flock 9
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 ~/.conda/envs/vlm-eval/bin/python experiments/v2/gen_doc_qa.py \
  --parquet data/v2/raw/doclaynet/data/*.parquet --n-pages 300 --out data/v2/doclaynet-qa-pilot.jsonl
echo "=== exit $? $(date '+%F %T')"
