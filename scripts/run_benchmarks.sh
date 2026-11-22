#!/usr/bin/env bash
# Public multimodal benchmarks through the vLLM serving plugin + lmms-eval (roadmap §7.3).
#   usage: scripts/run_benchmarks.sh <export_dir> [quick|full]
#   quick: --limit 500 per task (intermediate checkpoints); full: whole splits (gates / final)
#   <export_dir> may also be a HF model id (e.g. Qwen/Qwen3-VL-2B-Instruct for the ceiling).
#   CHAT_TEMPLATE: jinja file to use (default: <export>/chat_template_nothink.jinja if it exists, else the model's own
#   template); set CHAT_TEMPLATE=none to force the model's own template.
#   MM_LIMIT: max images per prompt (default 1; multi-image tasks such as mmmu_val need ~5). MAX_LEN: max_model_len (16384).
set -u -o pipefail
cd "$(dirname "$0")/.."
EXPORT=$1; MODE=${2:-quick}
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
TASKS=${TASKS:-docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,mmmu_val,pope,realworldqa}
LIMIT=""; [ "$MODE" = quick ] && LIMIT="--limit 500"
OUT=reports/benchmarks/$(basename $EXPORT)-$MODE${OUT_TAG:+-$OUT_TAG}   # OUT_TAG: optional suffix (separate runs)
TPL_ARG=""
if [ "${CHAT_TEMPLATE:-}" = none ] || { [ -z "${CHAT_TEMPLATE:-}" ] && [ ! -d "$EXPORT" ]; }; then
  echo "using the model's own chat template"            # HF model id (e.g. the Qwen3-VL ceiling) or explicit none
else
  CHAT_TEMPLATE=${CHAT_TEMPLATE:-$EXPORT/chat_template_nothink.jinja}
  [ -f "$CHAT_TEMPLATE" ] || { echo "missing $CHAT_TEMPLATE (set CHAT_TEMPLATE=<file> or none)"; exit 1; }
  TPL_ARG=",chat_template=$CHAT_TEMPLATE"
fi
mkdir -p $OUT
$EVAL_PY -m lmms_eval --model vllm \
  --model_args "model=$EXPORT,gpu_memory_utilization=${GPU_MEM:-0.45},max_new_tokens=256$TPL_ARG,max_model_len=${MAX_LEN:-16384},limit_mm_per_prompt={\"image\":${MM_LIMIT:-1}}" \
  --tasks $TASKS --batch_size 64 $LIMIT --output_path $OUT --log_samples > $OUT/run.log 2>&1
STATUS=$?
grep -E "^\|" $OUT/run.log
if [ $STATUS -ne 0 ] || ! find $OUT -name "*results*.json" | grep -q .; then
  echo "benchmark FAILED (exit $STATUS, see $OUT/run.log)"; exit 1
fi
