#!/usr/bin/env bash
# S4F evaluation, started when training finishes: export s2000 at LoRA 0.5 and 1.0 → official QA (2.56 MP)
# + "Answer:" quirk count → text retention (both, in parallel) → lmms-eval quick + MMMU for l0.5 → serving latency.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU1 is shared with root-owned vLLM servers: start each vLLM job only when enough memory is free.
free_gb() { nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
wait_free() { until [ "$(free_gb)" -ge "$1" ]; do sleep 60; done; }
PY=~/.conda/envs/vlm-train/bin/python; EVAL_PY=~/.conda/envs/vlm-eval/bin/python
R=experiments/HYB-S4F-Q3-4B-VIT8-HR-2000; D=experiments/s4f; T=reports/text/s4f; mkdir -p $D/vision $T
until [ -f $D/train.done ]; do sleep 60; done
grep -q "exit 0" $D/train.done || { echo "!!! training failed: $(cat $D/train.done)"; exit 1; }
exec 9>locks/gpu1.lock; flock 9
for SPEC in 2000:0.5:05 2000:1.0:10; do
  IFS=: read S L TAG <<< "$SPEC"; E=hyb-4b-s4f-s$S-l$TAG
  echo "=== export $E $(date +%T)"
  $PY scripts/export_hybrid.py --run $R --step $S --lora-scale $L --out exports/$E > $D/export-$E.log 2>&1 || { echo "!!! export $E failed"; continue; }
  echo "=== qa $E $(date +%T)"
  $PY scripts/eval_hybrid_export.py qa --export exports/$E --n-qa 200 --n-probe 100 --max-pixels 2560000 \
    --out $D/vision/$E.qa.json > $D/vision/$E.qa.log 2>&1 || echo "!!! qa $E failed"
done
$PY scripts/score_clean.py s3-l05=experiments/s3-eval/vision/hyb-4b-s3-s4000-l05.qa.json \
  s4-l05=experiments/s4/vision/hyb-4b-s4-s2000-l05.qa.json \
  $(for f in $D/vision/hyb-4b-s4f-*.qa.json; do n=$(basename $f .qa.json); echo "${n#hyb-4b-}=$f"; done)
for E in hyb-4b-s4f-s2000-l05 hyb-4b-s4f-s2000-l10; do
  wait_free 22; echo "=== text $E $(date +%T) (free $(free_gb) GB)"
  $EVAL_PY scripts/eval_text_batched.py --model exports/$E-llm --tasks mmlu_pro gsm8k ifeval math500 --modes think nothink --seeds 2 \
    --limit mmlu_pro=500 gsm8k=500 math500=200 --gpu-mem 0.2 --out $T/$E.json --items-out $T/$E.items.json > $D/text-$E.log 2>&1 &
  sleep 240
done
wait
echo "=== text done $(date +%T)"
E=hyb-4b-s4f-s2000-l05
wait_free 31; echo "=== bench $E $(date +%T)"
export GPU_MEM=0.3
TASKS=docvqa_val,chartqa,textvqa_val,infovqa_val,ocrbench,ai2d,mmbench_en_dev,pope,realworldqa scripts/run_benchmarks.sh exports/$E quick || echo "!!! bench $E failed"
TASKS=mmmu_val MM_LIMIT=5 OUT_TAG=mmmu scripts/run_benchmarks.sh exports/$E quick || echo "!!! mmmu $E failed"
wait_free 31; echo "=== latency $E $(date +%T)"
SUFFIX=-s4f scripts/run_hybrid_serving_bench.sh exports/$E 8011 bf16 || echo "!!! latency $E failed"
echo "=== eval done $(date +%T)"
