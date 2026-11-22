#!/usr/bin/env bash
# BF16 serving latency, same conditions: 4B release candidate vs 1.7B step-3000 candidate (GPU1 slot B).
cd "$(dirname "$0")/../.."
exec 9>locks/gpu1-slotB.lock; flock 9
for E in hyb-4b-s2000-l05 hyb-1b7-s3000-l10; do
  echo "=== latency $E $(date +%T)"; SUFFIX=-final scripts/run_hybrid_serving_bench.sh exports/$E 8011 bf16
done
echo "=== latency done $(date +%T)"
