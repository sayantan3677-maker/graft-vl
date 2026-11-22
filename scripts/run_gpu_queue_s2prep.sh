#!/usr/bin/env bash
# GPU queue after S1-50K: replay → traces → text baselines (all vLLM, `vlm-eval` env).
set -u
cd "$(dirname "$0")/.."
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
while ! grep -q "=== done\|train failed" experiments/s1_50k.log; do sleep 30; done
mkdir -p reports/text
step() { echo "=== $1 $(date -u +%T)"; }
step replay;   $EVAL_PY scripts/gen_replay.py --model Qwen/Qwen3-1.7B --n 20000 --name replay-q3-1b7-20k --gpu-mem 0.45 > manifests/gen_replay-q3-1b7.log 2>&1 || echo "replay failed"
step describe; $EVAL_PY scripts/gen_traces.py describe --n 6000 --name traces-src-6k --gpu-mem 0.45 > manifests/gen_traces_describe.log 2>&1 || echo "describe failed"
step reason;   $EVAL_PY scripts/gen_traces.py reason --src traces-src-6k --model Qwen/Qwen3-1.7B --name traces-q3-1b7 --gpu-mem 0.45 > manifests/gen_traces_reason-q3-1b7.log 2>&1 || echo "reason failed"
step baseline; $EVAL_PY scripts/eval_text.py --model Qwen/Qwen3-1.7B --tasks mmlu_pro gsm8k ifeval math500 gpqa_diamond --seeds 4 --limit mmlu_pro=1000 --gpu-mem 0.45 --out reports/text/qwen3-1.7b-original.json > reports/text/qwen3-1.7b-original.log 2>&1 || echo "baseline failed"
step aime;     $EVAL_PY scripts/eval_text.py --model Qwen/Qwen3-1.7B --tasks aime24 aime25 --modes think --seeds 16 --gpu-mem 0.45 --out reports/text/qwen3-1.7b-original.json >> reports/text/qwen3-1.7b-original.log 2>&1 || echo "aime failed"
step done
