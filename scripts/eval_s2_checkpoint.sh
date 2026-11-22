#!/usr/bin/env bash
# Full gate evaluation of one S2 checkpoint (roadmap §10 S2 gate).
#   usage: scripts/eval_s2_checkpoint.sh <run_id> <step> [quick|full]
set -u
cd "$(dirname "$0")/.."
RUN=$1; STEP=$2; MODE=${3:-quick}
PY=~/.conda/envs/vlm-train/bin/python
EVAL_PY=~/.conda/envs/vlm-eval/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TAG=$(printf "step-%07d" $STEP)
OUT=experiments/$RUN/eval; mkdir -p $OUT exports
echo "=== [$RUN $TAG] export LLM $(date -u +%T)"
$PY scripts/export_llm.py --run experiments/$RUN --step $STEP --out exports/$RUN-$TAG-llm > $OUT/export-$TAG.log 2>&1 || { echo "export failed"; exit 1; }
echo "=== text regression ($MODE) $(date -u +%T)"
if [ "$MODE" = full ]; then
  $EVAL_PY scripts/eval_text.py --model exports/$RUN-$TAG-llm --tasks mmlu_pro gsm8k ifeval math500 gpqa_diamond --seeds 4 --limit mmlu_pro=1000 --gpu-mem 0.45 --out $OUT/text-$TAG.json > $OUT/text-$TAG.log 2>&1
  $EVAL_PY scripts/eval_text.py --model exports/$RUN-$TAG-llm --tasks aime24 aime25 --modes think --seeds 16 --gpu-mem 0.45 --out $OUT/text-$TAG.json >> $OUT/text-$TAG.log 2>&1
else
  $EVAL_PY scripts/eval_text.py --model exports/$RUN-$TAG-llm --tasks mmlu_pro gsm8k ifeval math500 --seeds 2 --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem 0.45 --out $OUT/text-$TAG.json > $OUT/text-$TAG.log 2>&1
fi
echo "=== image reliance $(date -u +%T)"
$PY scripts/eval_s1.py --run experiments/$RUN --step $STEP --eval-data manifests/s1-eval300 --train-data manifests/s1-50k --n 300 --n-generate 12 > $OUT/reliance-$TAG.log 2>&1
echo "=== QA accuracy (short) $(date -u +%T)"
$PY scripts/eval_vqa.py --run experiments/$RUN --step $STEP --eval-data manifests/qa-eval-1500 --train-data manifests/s1-50k $(ls -d manifests/s2/*/ | grep -v "/_") > $OUT/vqa-short-$TAG.log 2>&1
echo "=== QA accuracy (thinking, 300) $(date -u +%T)"
$PY scripts/eval_vqa.py --run experiments/$RUN --step $STEP --eval-data manifests/qa-eval-1500 --train-data manifests/s1-50k $(ls -d manifests/s2/*/ | grep -v "/_") --thinking --n 300 > $OUT/vqa-think-$TAG.log 2>&1
echo "=== done $(date -u +%T)"
