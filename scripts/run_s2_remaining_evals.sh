#!/usr/bin/env bash
# Remaining evals of the S2 final checkpoint: injection (now), then text regression + thinking QA (after GPQA).
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python; EVAL_PY=~/.conda/envs/vlm-eval/bin/python
RUN=EXP-S2-Q3-1B7-384-FULLFT-VIS9-0.5B; OUT=experiments/$RUN/eval
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== injection $(date -u +%T)"
$PY scripts/injection_eval.py eval --run experiments/$RUN > $OUT/injection.log 2>&1 && echo ok || echo failed
while ! grep -q "=== done" experiments/gpqa_baseline.log; do sleep 60; done
echo "=== export LLM $(date -u +%T)"
$PY scripts/export_llm.py --run experiments/$RUN --out exports/$RUN-llm > $OUT/export-llm.log 2>&1 || { echo "export failed"; exit 1; }
echo "=== text regression $(date -u +%T)"
$EVAL_PY scripts/eval_text.py --model exports/$RUN-llm --tasks mmlu_pro gsm8k ifeval math500 --seeds 2 --limit mmlu_pro=500 gsm8k=500 math500=200 \
  --gpu-mem 0.35 --out $OUT/text-step-0015600.json > $OUT/text.log 2>&1 && echo ok || echo failed
echo "=== thinking QA $(date -u +%T)"
TRAIN=$(ls -d manifests/s1-50k manifests/s2/*/ | grep -v "/_")
$PY scripts/eval_vqa.py --run experiments/$RUN --eval-data manifests/eval-official --train-data $TRAIN --thinking --n 280 > $OUT/vqa-think.log 2>&1 && echo ok || echo failed
echo "=== done $(date -u +%T)"
