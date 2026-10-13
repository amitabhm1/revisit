#!/usr/bin/env bash
# SceneDiff table: proposals only, two zero-shot VLMs and the three RL adapters, on the test split.
#
#   bash reproduce/scenediff.sh metrics   # CPU, minutes: rescore our released verifier outputs
#   bash reproduce/scenediff.sh verify    # GPUs: rerun every verifier on our released Stage-1 proposals (default)
#   bash reproduce/scenediff.sh full      # GPUs: also recompute the Stage-1 proposals from the raw benchmark
#
# Needs: python scripts/download_data.py --scenediff --artifacts
#        python scripts/download_checkpoints.py --set scenediff   (verify and full)
# Environment: SPLIT (default test), NGPU (default: all visible GPUs).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-verify}
SPLIT=${SPLIT:-test}
NGPU=${NGPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
DATA=${REVISIT_DATA_DIR:-data}
OUT=outputs/scenediff
PROPOSALS=$DATA/artifacts/scenediff/proposals
ROWS="none qwen-zeroshot sensenova-zeroshot qwen-grpo qwen-gdpo qwen-gdpo-rlcr"
mkdir -p "$OUT/logs"

sharded() {  # sharded <log-name> <command...>: one process per GPU with --shard/--num-shards, then wait
  local name=$1; shift
  for i in $(seq 0 $(( NGPU - 1 ))); do
    CUDA_VISIBLE_DEVICES=$i "$@" --shard "$i" --num-shards "$NGPU" > "$OUT/logs/${name}_${i}.log" 2>&1 &
  done
  wait
}

if [ "$MODE" = metrics ]; then
  for row in $ROWS; do
    rm -rf "$OUT/$row" && mkdir -p "$OUT/$row"
    cp -r "$DATA/artifacts/scenediff/verifier_outputs/$row/." "$OUT/$row/"
    python scripts/scenediff_verify.py --verifier "$row" --split "$SPLIT" --proposals "$PROPOSALS" --score-only \
      > "$OUT/logs/score_${row}.log" 2>&1
  done
else
  if [ "$MODE" = full ]; then
    PROPOSALS=$OUT/proposals
    sharded proposals python scripts/scenediff_proposals.py --split "$SPLIT" --output-dir "$PROPOSALS"
  fi
  for row in $ROWS; do
    base=""
    case $row in qwen-grpo|qwen-gdpo|qwen-gdpo-rlcr) base="--base-outputs $OUT/qwen-zeroshot" ;; esac
    if [ "$row" = none ]; then
      python scripts/scenediff_verify.py --verifier none --split "$SPLIT" --proposals "$PROPOSALS" > "$OUT/logs/none.log" 2>&1
    else
      sharded "$row" python scripts/scenediff_verify.py --verifier "$row" --split "$SPLIT" --proposals "$PROPOSALS" $base
      python scripts/scenediff_verify.py --verifier "$row" --split "$SPLIT" --proposals "$PROPOSALS" --score-only \
        > "$OUT/logs/score_${row}.log" 2>&1
    fi
  done
fi
python scripts/make_tables.py scenediff --outputs "$OUT" --split "$SPLIT" --compare results/scenediff.json
