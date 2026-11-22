#!/usr/bin/env bash
# Build S3 doc/chart/OCR manifests (one per source) in parallel: manifests/s3/<source>.
set -u
cd "$(dirname "$0")/../.."
PAR=${1:-8}; PY=~/.conda/envs/vlm-train/bin/python
mkdir -p manifests/s3
build_one() {
  name=$1; n=$2
  if [ -f manifests/s3/$name/manifest.json ]; then echo "skip $name (exists)"; return; fi
  CUDA_VISIBLE_DEVICES="" $PY -u scripts/build_shards.py --source $name --n $n --name s3/$name --seed 11 \
      --shard-size 1000 --max-target-tokens 2048 --tokenizer Qwen/Qwen3-4B $(grep -qx $name experiments/s3/all_turns.txt && echo --all-turns) > manifests/s3/_build_$name.log 2>&1 \
    && echo "done $name $(date -u +%T)" || echo "FAILED $name $(date -u +%T)"
}
export -f build_one; export PY
xargs -P $PAR -L 1 bash -c 'build_one "$@"' _ < experiments/s3/jobs.txt
echo "=== all done $(date -u +%T)"
