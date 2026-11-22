#!/usr/bin/env bash
# After the vision pilot: parity (near-tie analysis) → traces v2 (fixed prompt/filter) → replay IF+safety top-up.
set -u
cd "$(dirname "$0")/.."
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
while ! grep -q "=== done" experiments/pilot_vis.log 2>/dev/null; do sleep 60; done
step() { echo "=== $1 $(date -u +%T)"; }
step parity;  $EVAL_PY scripts/test_vllm_parity.py --export exports/s1-q3-1b7-384 --run experiments/EXP-S1-Q3-1B7-S2P16-384-MLP-50K-LR1e-3 > reports/serving_parity_s1_v2.log 2>&1 && echo "parity PASS" || echo "parity FAIL"
step traces;  $EVAL_PY scripts/gen_traces.py reason --src traces-src-6k --model Qwen/Qwen3-1.7B --name traces-q3-1b7-v2 --gpu-mem 0.45 > manifests/gen_traces_reason-q3-1b7-v2.log 2>&1 || echo "traces failed"
step replay;  $EVAL_PY scripts/gen_replay.py --model Qwen/Qwen3-1.7B --n 5000 --categories instruction_following safety --name replay-q3-1b7-ifsafety-5k --gpu-mem 0.45 --seed 1 > manifests/gen_replay-ifsafety.log 2>&1 || echo "replay failed"
step done
