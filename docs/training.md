# Training a verifier

The verifier is a LoRA adapter (on every attention and MLP projection) for Qwen3-VL-8B-Instruct or
SenseNova-SI-1.3-Qwen3-VL-8B, trained with online RL on evidence cards built from SceneDiff
**validation** scenes. Test scenes are never used for training.

```bash
python scripts/download_data.py --scenediff --artifacts
python scripts/download_checkpoints.py --set train
bash reproduce/train_verifier.sh                       # GDPO+RLCR, 45 steps on 8 GPUs, then scores it on test
torchrun --standalone --nproc_per_node 8 scripts/train_verifier.py \
    --config configs/train/qwen3vl8b_gdpo_rlcr_step45.yaml --max_steps 100   # any config value can be overridden
```

The scripts assume one machine with 8 GPUs of 80 GB. With fewer GPUs, set `--nproc_per_node` (or `NGPU` for
the shell script); each step then sees fewer cards, so raise `max_steps` to keep the total. Started with plain
`python` instead of `torchrun`, the trainer runs on one GPU without `torch.distributed`, which is handy for
debugging.

## What one step does

1. Every GPU takes one card and samples a group of answers (temperature 0.95, top-p 0.95, at most 192 tokens).
2. Each answer gets three rewards. Labels come from matching the proposal to SceneDiff's ground truth
   (stacked-mask IoU above 0.5): moved, added, removed, or none.

   | Reward | Definition |
   |:--|:--|
   | format | parseable `<answer>{...}</answer>`, a short `<think>` that cites geometry, appearance and visibility |
   | verdict | +1 exact; +0.35 right that something changed but wrong type; -1 for a hallucinated change when the 3D cost is low, -0.65 otherwise; -0.6 for a missed change |
   | calibration | 1 - (0.6 Brier(p_change, changed) + 0.4 Brier(confidence, verdict correct)) |

3. Advantages within the group. GRPO adds the weighted rewards (format 0.25, verdict 1.0) and normalises once.
   GDPO normalises each reward on its own first. GDPO+RLCR adds the calibration reward with weight 0.80.
4. Policy-gradient loss with a per-token KL penalty to the base model (beta 0.02, adapter disabled for the
   reference). LoRA gradients are averaged over all GPUs; AdamW with weight decay 0.01 and gradient clipping at 1.0.

The room recipes balance the card pool: they take the hardest false positives (highest Stage-1 score), twice as
many as there are cards that match a real change, and alternate one real-change card with one false positive, so
each real-change card appears twice (lists shuffled with seed 42). The step-45 recipes use every validation card
in split order: 938 cards, 170 of them real changes.

## Recipes

| Config | Reproduces | Card pool | Steps x GPUs, group | LoRA rank | Time on 8 H100 |
|:--|:--|:--|:--|--:|--:|
| `qwen3vl8b_{grpo,gdpo,gdpo_rlcr}_step45.yaml` | the three SceneDiff RL rows | SceneDiff proposals, val scenes (938 cards) | 45 x 8, 6 | 16 | ~8 min |
| `qwen3vl8b_gdpo_rlcr_room.yaml` | `qwen3vl8b-gdpo-rlcr-step2460` | 5 Stage-1 variants, val scenes (2,748 cards) | 12,740 x 8, 8 | 32 | ~3 days |
| `sensenova8b_gdpo_rlcr_room.yaml` | `sensenova8b-gdpo-rlcr-step2290` | same | 11,900 x 8, 8 | 32 | ~3 days |

The step-45 recipes match how those adapters were trained: one node with 8 GPUs, constant learning rate 1.5e-5,
no augmentation. RL sampling is not deterministic, so a retrained adapter will be close to ours, not
bit-identical. The room recipes use learning rate 3e-5 with the cosine schedule described below.

## How the room adapters were trained

Each room adapter was trained on six machines with 8 GPUs each: gradients averaged within a machine every step,
LoRA weights averaged across machines every 5 steps, a cosine learning-rate decay to 25% over the first 1,500
steps, and light photometric and score jitter on the cards. One machine per model stopped near step 450. A
staging error left one of the remaining Qwen machines and two of the SenseNova machines with three of the five
card sources (1,436 cards instead of 2,748); their weights were averaged with the others as usual. The released
checkpoints, steps 2,460 (Qwen) and 2,290 (SenseNova), sit at whole epochs of the trainer's counter (43 and 40,
counted for 48 GPUs and the full pool). About 102k and 95k cards were sampled up to those steps. The
single-machine configs above use the full pool and see the same number of cards with the same schedule per
card, not the same optimisation path. The multi-machine launcher behind the released adapters is not included;
retrain with the single-machine configs above.

The five card sources are the SceneDiff proposals plus four Stage-1 variants run on the validation scenes:
Pi3X with DINOv3-H+/16, DINOv3-L/16 or DINOv2-B/14 (all with SAM 1), and Pi3X + DINOv3-H+/16 + SAM 3.
The SAM 3 variant was made before we fixed a bug that placed its point prompts outside the image, so its masks
are few and coarse and most of its cards are labelled "none". It stays in the pool because the released
adapters were trained on it. All five sources are in `data/artifacts/` after `download_data.py --artifacts`.

## Outputs

`outputs/train/<name>/` holds `adapter/` (with `revisit.json`: base model, RL variant, steps), `checkpoints/latest`
(resume state), milestone adapters `checkpoints/step_*`, `training_log.json` and the card list. Rerunning the
same command resumes from `checkpoints/latest`. To score an adapter on SceneDiff:

```bash
python scripts/scenediff_verify.py --verifier custom --model checkpoints/Qwen/Qwen3-VL-8B-Instruct \
    --adapter outputs/train/qwen3vl8b-gdpo-rlcr-step45/adapter --rl-mode gdpo_rlcr --split test
```

Reward curves of our runs are in `results/training_curves.csv`.
