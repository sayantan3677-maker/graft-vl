#!/usr/bin/env bash
# GPU 0 side queue (shared GPU, 35% cap): benchmark smoke → original Qwen3-1.7B text baselines → more traces.
set -u
cd "$(dirname "$0")/.."
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 GPU_MEM=0.35
step() { echo "=== $1 $(date -u +%T)"; }
step bench;    TASKS=ocrbench,pope,chartqa scripts/run_benchmarks.sh exports/s1-q3-1b7-384 quick > reports/bench_s1_smoke.log 2>&1 && echo "bench ok" || echo "bench FAILED"
step baseline; $EVAL_PY scripts/eval_text.py --model Qwen/Qwen3-1.7B --tasks mmlu_pro gsm8k ifeval math500 gpqa_diamond --seeds 4 --limit mmlu_pro=1000 --gpu-mem 0.35 --out reports/text/qwen3-1.7b-original.json > reports/text/qwen3-1.7b-original.log 2>&1 || echo "baseline failed"
step aime;     $EVAL_PY scripts/eval_text.py --model Qwen/Qwen3-1.7B --tasks aime24 aime25 --modes think --seeds 16 --gpu-mem 0.35 --out reports/text/qwen3-1.7b-original.json >> reports/text/qwen3-1.7b-original.log 2>&1 || echo "aime failed"
step describe; $EVAL_PY scripts/gen_traces.py describe --n 9000 --seed 2 --name traces-src-9k-s2 --gpu-mem 0.35 > manifests/gen_traces_describe-s2.log 2>&1 || echo "describe failed"
step reason;   $EVAL_PY scripts/gen_traces.py reason --src traces-src-9k-s2 --model Qwen/Qwen3-1.7B --name traces-q3-1b7-s2 --gpu-mem 0.35 > manifests/gen_traces_reason-s2.log 2>&1 || echo "reason failed"
step done
