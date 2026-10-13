#!/usr/bin/env bash
# Train a verifier adapter on one node, then score it on the SceneDiff test split.
#
#   bash reproduce/train_verifier.sh                                          # GDPO+RLCR, 45 steps (~8 min)
#   bash reproduce/train_verifier.sh configs/train/qwen3vl8b_grpo_step45.yaml
#   bash reproduce/train_verifier.sh configs/train/qwen3vl8b_gdpo_rlcr_room.yaml  # room recipe (~3 days)
#
# Assumes one machine with 8 GPUs (80 GB each). Set NGPU for fewer; the per-step batch shrinks with it.
# Needs: python scripts/download_data.py --scenediff --artifacts
#        python scripts/download_checkpoints.py --set train
# The test-split scoring reuses outputs/scenediff/qwen-zeroshot from reproduce/scenediff.sh if present.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/train/qwen3vl8b_gdpo_rlcr_step45.yaml}
NGPU=${NGPU:-8}
torchrun --standalone --nproc_per_node "$NGPU" scripts/train_verifier.py --config "$CONFIG"

OUTDIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")
BASE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['base_model'])" "$CONFIG")
MODE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['mode'])" "$CONFIG")
MODEL=$(python -c "from revisit import paths; print(paths.model_dir('$BASE'))")
reuse=""
[ "$BASE" = qwen3vl8b ] && [ -d outputs/scenediff/qwen-zeroshot ] && reuse="--base-outputs outputs/scenediff/qwen-zeroshot"
for i in $(seq 0 $(( NGPU - 1 ))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/scenediff_verify.py --verifier custom --model "$MODEL" --adapter "$OUTDIR/adapter" \
    --rl-mode "$MODE" --split test --output-dir "$OUTDIR/scenediff_test" $reuse --shard "$i" --num-shards "$NGPU" &
done
wait
python scripts/scenediff_verify.py --verifier custom --model "$MODEL" --adapter "$OUTDIR/adapter" --rl-mode "$MODE" \
  --split test --output-dir "$OUTDIR/scenediff_test" --score-only
