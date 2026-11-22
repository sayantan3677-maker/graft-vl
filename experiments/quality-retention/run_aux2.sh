#!/bin/bash
# wait for aux queue #1 (injection) to finish, then run thinking evals via run_aux.sh
cd "$(dirname "$0")/../.."; D=experiments/quality-retention
until grep -q "aux done" $D/aux.log; do sleep 30; done
exec $D/run_aux.sh
