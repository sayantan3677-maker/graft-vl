#!/usr/bin/env bash
# Evaluate each new checkpoint of a hybrid run on GPU 1 as it lands.  usage: watch_hybrid_evals.sh <run_id> <n_expected>
set -u
cd "$(dirname "$0")/.."
RUN=$1; N=$2; done_list=""
while [ $(echo $done_list | wc -w) -lt $N ]; do
  for ck in $(ls experiments/$RUN/checkpoints 2>/dev/null); do
    [[ " $done_list " == *" $ck "* ]] && continue
    [ -f experiments/$RUN/checkpoints/$ck/info.json ] || continue
    sleep 20  # let the checkpoint file finish writing
    step=$((10#${ck#step-}))
    echo "=== eval $ck $(date -u +%T)"
    CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vlm-train/bin/python scripts/eval_hybrid.py --run experiments/$RUN --step $step --n-qa 60 2>&1 | grep '^{"checkpoint'
    done_list="$done_list $ck"
  done
  sleep 60
done
echo "=== done $(date -u +%T)"
