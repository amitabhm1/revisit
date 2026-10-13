#!/usr/bin/env bash
# Metric scale: ARKit trajectory scale per take vs the LiDAR-median estimate. CPU, under a minute.
# Needs the Stage 1-2 outputs of reproduce/room.sh.
#
#   bash reproduce/metric_scale.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/check_metric_scale.py "${1:-outputs/room/qwen}" --out outputs/metric_scale.json \
  --compare results/metric_scale.json
