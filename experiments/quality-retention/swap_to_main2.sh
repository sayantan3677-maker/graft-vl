#!/bin/bash
# wait until run_main.sh finished the baseline, then stop its session (my own processes only) and start run_main2.sh
cd "$(dirname "$0")/../.."; D=experiments/quality-retention
until grep -q '=== text hyb-1b7-s2000-l10' $D/main.log; do sleep 10; done
PG=$(ps -eo pgid,args | grep "flock locks/gpu1-slotA.lock experiments/quality-retention/run_main.sh" | grep -v grep | awk '{print $1}' | head -1)
echo "killing process group $PG $(date +%T)"; ps -eo pid,pgid,args | awk -v g=$PG '$2==g'
kill -TERM -$PG; sleep 20; kill -KILL -$PG 2>/dev/null
until ! pgrep -f "eval_text.py --model exports/hyb-1b7-s2000-l10-llm" > /dev/null; do sleep 5; done
rm -f reports/text/quality-retention/hyb-1b7-s2000-l10.json reports/text/quality-retention/hyb-1b7-s2000-l10.items.json
sleep 10; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
setsid nohup flock locks/gpu1-slotA.lock $D/run_main2.sh > $D/main2.log 2>&1 < /dev/null &
echo "started main2 $(date +%T)"
