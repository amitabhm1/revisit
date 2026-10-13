#!/usr/bin/env bash
# Per-stage runtime: the room benchmark three times, then median [min-max] per stage and comparison.
# The reported table used one H100 80GB per run with 3 runs per machine at a time.
#
#   bash reproduce/runtime.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for r in 1 2 3; do
  rm -rf "outputs/runtime/r$r"
  OUT="outputs/runtime/r$r" bash reproduce/room.sh
done
for v in qwen sensenova; do
  python scripts/summarize_timing.py --verifier "$v" outputs/runtime/r1 outputs/runtime/r2 outputs/runtime/r3 \
    --out "outputs/runtime/$v.json"
done
