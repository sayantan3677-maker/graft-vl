#!/usr/bin/env bash
# scaling-4b: build 4B hybrid, FP32 equivalence check, zero-shot eval (GPU 0, run under locks/gpu0.lock)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=0
echo "=== build $(date -u +%T)"
[ -f exports/hybrid-q3-4b-vit4b/hybrid_info.json ] || $PY scripts/build_hybrid.py --vl Qwen/Qwen3-VL-4B-Instruct --llm Qwen/Qwen3-4B --out exports/hybrid-q3-4b-vit4b
echo "=== verify fp32 $(date -u +%T)"
$PY scripts/verify_hybrid_fp32.py --hybrid exports/hybrid-q3-4b-vit4b --out reports/agents/scaling-4b/text-equivalence-fp32.json
echo "=== zero-shot eval $(date -u +%T)"
$PY scripts/eval_hybrid.py --init exports/hybrid-q3-4b-vit4b --n-qa 60 --out reports/agents/scaling-4b/hybrid-eval-4b-init.json 2>&1 | grep -E '^\{"checkpoint|Error|error' || true
echo "=== done $(date -u +%T)"
