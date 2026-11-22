#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
while ! grep -q "=== done" experiments/queue_fixups.log 2>/dev/null; do sleep 60; done
echo "=== bench smoke $(date -u +%T)"
CUDA_VISIBLE_DEVICES=1 TASKS=ocrbench,pope,chartqa scripts/run_benchmarks.sh exports/s1-q3-1b7-384 quick > reports/bench_s1_smoke.log 2>&1 && echo ok || echo "bench failed"
echo "=== done $(date -u +%T)"
