#!/usr/bin/env bash
# Mask backend x keyframe ablation on the room benchmark. The final cell (SAM 2.1, 28 keyframes) is
# reproduce/room.sh. Needs the extra checkpoints: python scripts/download_checkpoints.py --set ablation
#
#   bash reproduce/mask_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/.."
cells="sam1:28 sam2.1:36 sam1:36 sam3.1:36"
for cell in $cells; do
  sam=${cell%%:*}; kf=${cell##*:}
  OUT=outputs/ablation/${sam}_kf${kf} bash reproduce/room.sh --sam-backend "$sam" --keyframes "$kf"
done
runs=""
for cell in $cells; do
  sam=${cell%%:*}; kf=${cell##*:}
  for v in qwen sensenova; do runs+="${sam}_kf${kf}-${v}=outputs/ablation/${sam}_kf${kf}/${v} "; done
done
python scripts/score_room.py $runs --out outputs/ablation/scoreboard.json --compare results/room/ablation.json
