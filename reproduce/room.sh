#!/usr/bin/env bash
# Room benchmark with the final configuration: 3 comparisons x 2 verifiers, then the scoreboard.
#
#   bash reproduce/room.sh
#
# Each run uses one GPU. PARALLEL runs go at once (default: 3, or fewer GPUs if you have fewer).
# Extra arguments are passed to scripts/run_room.py, e.g. --sam-backend sam1 --keyframes 36.
# Environment: OUT (default outputs/room), PARALLEL, VERIFIERS (default "qwen sensenova").
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
OUT=${OUT:-outputs/room}
PARALLEL=${PARALLEL:-$(( NGPU < 3 ? NGPU : 3 ))}
VERIFIERS=${VERIFIERS:-"qwen sensenova"}
ROOM=${REVISIT_DATA_DIR:-data}/room
mkdir -p "$OUT/logs"
: > "$OUT/logs/wall.tsv"

declare -A ARGS=(
  [Take_1_vs_Take_2]="--before $ROOM/Take_1 --after $ROOM/Take_2"
  [Take_1_vs_Take_3]="--before $ROOM/Take_1 --after $ROOM/Take_3"
  [Take_1_vs_Take_3_relook_Take_4]="--before $ROOM/Take_1 --after $ROOM/Take_3 --relook $ROOM/Take_4"
)
jobs=()
for v in $VERIFIERS; do
  for pair in Take_1_vs_Take_2 Take_1_vs_Take_3 Take_1_vs_Take_3_relook_Take_4; do jobs+=("$v $pair"); done
done

slot=0
for job in "${jobs[@]}"; do
  read -r v pair <<< "$job"
  gpu=$(( slot % PARALLEL ))
  (
    t0=$(date +%s%N)
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_room.py ${ARGS[$pair]} --verifier "$v" \
      --output-dir "$OUT/$v" "$@" > "$OUT/logs/${v}_${pair}.log" 2>&1
    awk -v v="$v" -v p="$pair" -v a="$t0" -v b="$(date +%s%N)" 'BEGIN { printf "%s\t%s\t%.1f\n", v, p, (b - a) / 1e9 }' \
      >> "$OUT/logs/wall.tsv"
  ) &
  slot=$(( slot + 1 ))
  if (( slot % PARALLEL == 0 )); then wait; fi
done
wait

names=$(for v in $VERIFIERS; do printf "%s=%s " "$v" "$OUT/$v"; done)
if [ $# -eq 0 ]; then
  python scripts/score_room.py $names --out "$OUT/scoreboard.json" \
    --compare results/room/scoreboard.json --reference-dir results/room
else
  python scripts/score_room.py $names --out "$OUT/scoreboard.json"
fi
