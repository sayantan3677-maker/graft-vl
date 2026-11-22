#!/usr/bin/env bash
# Full S2 (1.7B, full FT + top-9 vision layers), with official-split QA eval of each checkpoint as it lands.
set -u
cd "$(dirname "$0")/.."
PY=~/.conda/envs/vlm-train/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/train/s2_q3-1b7_full.json
RUN=$(python3 -c "import json;print(json.load(open('$CFG'))['run_id'])")
while ! grep -q "=== done" experiments/queue_fixups.log 2>/dev/null; do sleep 30; done
$PY scripts/make_s2_config.py --base $CFG --mixture configs/data/s2_textrich.json --mm 0.77 \
    --traces manifests/traces-q3-1b7-v3 --trace-share 0.03 \
    --replay manifests/replay-q3-1b7-20k:0.15 manifests/replay-q3-1b7-ifsafety-5k:0.05 --replay-share 0.20
echo "=== train $RUN $(date -u +%T)"
$PY scripts/train.py --config $CFG > experiments/$RUN.train.log 2>&1 &
TRAIN_PID=$!
TRAIN=$(ls -d manifests/s1-50k manifests/s2/*/ | grep -v "/_")
done_steps=""
while kill -0 $TRAIN_PID 2>/dev/null || [ -n "$(comm -23 <(ls experiments/$RUN/checkpoints 2>/dev/null | sort) <(echo "$done_steps" | tr ' ' '\n' | sort))" ]; do
  for ck in $(ls experiments/$RUN/checkpoints 2>/dev/null); do
    [[ " $done_steps " == *" $ck "* ]] && continue
    [ -f experiments/$RUN/checkpoints/$ck/info.json ] || continue
    step=$((10#${ck#step-}))
    echo "=== eval $ck $(date -u +%T)"
    $PY scripts/eval_vqa.py --run experiments/$RUN --step $step --eval-data manifests/eval-official --train-data $TRAIN --n 1400 > experiments/$RUN.vqa-official-$ck.log 2>&1 \
      && mv experiments/$RUN/eval/vqa-short-$ck.json experiments/$RUN/eval/vqa-official-$ck.json
    done_steps="$done_steps $ck"
  done
  sleep 120
done
wait $TRAIN_PID; echo "=== train exit $? $(date -u +%T)"
echo "=== done $(date -u +%T)"
