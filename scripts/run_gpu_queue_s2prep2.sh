#!/usr/bin/env bash
# Continuation of the S2-prep queue: waits for the in-flight replay (pid 1634818), then traces.
set -u
cd "$(dirname "$0")/.."
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
while kill -0 1634818 2>/dev/null; do sleep 30; done
step() { echo "=== $1 $(date -u +%T)"; }
step describe; $EVAL_PY scripts/gen_traces.py describe --n 6000 --name traces-src-6k --gpu-mem 0.45 > manifests/gen_traces_describe.log 2>&1 || echo "describe failed"
step reason;   $EVAL_PY scripts/gen_traces.py reason --src traces-src-6k --model Qwen/Qwen3-1.7B --name traces-q3-1b7 --gpu-mem 0.45 > manifests/gen_traces_reason-q3-1b7.log 2>&1 || echo "reason failed"
step done
