#!/usr/bin/env bash
# Start vLLM on a hybrid export and the Gradio demo in front of it.
#   demo/launch.sh exports/<name> [vllm_port=8011] [app_port=7860]
# Env: CUDA_VISIBLE_DEVICES, GPU_MEM (vLLM memory fraction, default 0.35), VLLM (path to vllm), DEMO_PY (python with gradio).
set -euo pipefail
cd "$(dirname "$0")/.."
EXPORT=${1:?usage: demo/launch.sh exports/<name> [vllm_port] [app_port]}
VPORT=${2:-8011}; APORT=${3:-7860}
VLLM=${VLLM:-~/.conda/envs/vlm-eval/bin/vllm}
DEMO_PY=${DEMO_PY:-~/.venvs/vlm-demo/bin/python}
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -f demo/examples/invoice.png ] || $DEMO_PY demo/make_examples.py

$VLLM serve "$EXPORT" --served-model-name vlm --port "$VPORT" --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEM:-0.35}" --max-model-len 16384 --limit-mm-per-prompt '{"image":1}' \
  > demo/vllm.log 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
until curl -sf "localhost:$VPORT/v1/models" >/dev/null; do
  kill -0 $VPID 2>/dev/null || { echo "vLLM exited; see demo/vllm.log"; exit 1; }
  sleep 3
done
echo "vLLM ready on :$VPORT — starting demo on :$APORT"
VLM_API_BASE="http://localhost:$VPORT/v1" VLM_MODEL=vlm PORT=$APORT $DEMO_PY demo/app.py
