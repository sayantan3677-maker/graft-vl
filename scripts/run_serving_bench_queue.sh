#!/usr/bin/env bash
# serving-bench agent queue (GPU 1 slot B): lmms-eval quick runs for our export and the Qwen3-VL-2B ceiling.
#   usage: flock locks/gpu1-slotB.lock scripts/run_serving_bench_queue.sh <mode: smoke|quick> [models...]
set -u
cd "$(dirname "$0")/.."
MODE=$1; shift
export CUDA_VISIBLE_DEVICES=1 GPU_MEM=0.45 VLLM_USE_FLASHINFER_SAMPLER=0
export TASKS=${TASKS:-docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa,mmmu_val}
for M in "$@"; do
  echo "=== $M $MODE start $(date -u +%T)"
  if [ "$MODE" = smoke ]; then
    # tiny limit to catch task/metric/template issues before the long run
    OUT=reports/benchmarks/smoke-$(basename $M); mkdir -p $OUT
    TPL=""; [ -d "$M" ] && TPL=",chat_template=$M/chat_template_nothink.jinja"
    ~/.conda/envs/vlm-eval/bin/python -m lmms_eval --model vllm \
      --model_args "model=$M,gpu_memory_utilization=$GPU_MEM,max_new_tokens=256$TPL,max_model_len=16384,limit_mm_per_prompt={\"image\":1}" \
      --tasks $TASKS --batch_size 64 --limit 8 --output_path $OUT --log_samples > $OUT/run.log 2>&1
    echo "exit $?"; grep -E "^\|" $OUT/run.log
  else
    scripts/run_benchmarks.sh $M quick; echo "exit $?"
  fi
  echo "=== $M $MODE end $(date -u +%T)"
done
