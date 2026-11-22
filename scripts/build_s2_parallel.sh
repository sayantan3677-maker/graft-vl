#!/usr/bin/env bash
# Build one manifest per S2 source in parallel: manifests/s2/<source>. Quotas = weight × TOTAL.
#   usage: scripts/build_s2_parallel.sh [TOTAL=300000] [PARALLEL=6]
set -u
cd "$(dirname "$0")/.."
TOTAL=${1:-300000}; PAR=${2:-6}
PY=~/.conda/envs/vlm-train/bin/python
mkdir -p manifests/s2
$PY - "$TOTAL" <<'PY' > manifests/s2/_jobs.txt
import json, sys
w = json.load(open("configs/data/s2.json"))["weights"]
for name, wt in sorted(w.items(), key=lambda x: -x[1]):
    print(name, max(500, round(wt * int(sys.argv[1]))))
PY
build_one() {
  name=$1; n=$2
  if [ -f manifests/s2/$name/manifest.json ]; then echo "skip $name (exists)"; return; fi
  CUDA_VISIBLE_DEVICES="" $PY -u scripts/build_shards.py --source $name --n $n --name s2/$name --seed 3 \
      --shard-size 1000 --max-target-tokens 2048 > manifests/s2/_build_$name.log 2>&1 \
    && echo "done $name $(date -u +%T)" || echo "FAILED $name"
}
export -f build_one; export PY
xargs -P $PAR -L 1 bash -c 'build_one "$@"' _ < manifests/s2/_jobs.txt
echo "=== all done $(date -u +%T)"
