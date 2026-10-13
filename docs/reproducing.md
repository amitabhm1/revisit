# Reproducing the results

| Result | Command | Needs | Time (H100 80GB) |
|:--|:--|:--|:--|
| Room benchmark | `bash reproduce/room.sh` | `--set room`, `--room` | ~25 min, 3 GPUs |
| SceneDiff table, from our verifier outputs | `bash reproduce/scenediff.sh metrics` | `--scenediff --artifacts` | minutes, CPU |
| SceneDiff table, rerun the verifiers | `bash reproduce/scenediff.sh verify` | `--set scenediff`, `--scenediff --artifacts` | ~2 h on 8 GPUs |
| SceneDiff table, from raw videos | `bash reproduce/scenediff.sh full` | same, plus `ffmpeg` | ~3 h on 8 GPUs |
| Mask backend and keyframes | `bash reproduce/mask_ablation.sh` | `--set room` and `--set ablation`, `[ablation]` extra | ~2.5 h, 3 GPUs |
| Metric scale | `bash reproduce/metric_scale.sh` | room outputs | < 1 min, CPU |
| Runtime | `bash reproduce/runtime.sh` | as room | ~75 min, 3 GPUs |
| Retrain a verifier | `bash reproduce/train_verifier.sh` | `--set train`, `--scenediff --artifacts` | see [training.md](training.md) |

"Needs" refers to `scripts/download_checkpoints.py --set ...` and `scripts/download_data.py ...`. The mask
ablation also needs the SAM 3 package: `SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e ".[ablation]"`.
Every script compares its output with the files in `results/` and says whether they match.

## Room benchmark

Three comparisons per verifier (no-change pair, before/after, before/after with re-look) with the final
configuration in `configs/room.yaml`. The script runs three comparisons at a time, one per GPU, then scores them
and checks every per-proposal verdict against `results/room/<verifier>/`. Rerunning this code from the raw
captures gave the same proposals, verdicts and probabilities, field for field, as the runs behind the tables
(on H100 GPUs; on other GPUs a few low-margin verdicts may differ).
Each comparison peaks at 39 GB of GPU memory and 85 GB of host RAM (53 GB and 126 GB with the re-look); three
at a time fit a machine with 283 GB of RAM. Set `PARALLEL=1` if memory is short.

## SceneDiff

All rows start from the same Stage-1 proposals (Pi3 + DINOv3 ViT-H+/16 + SAM 1, `configs/scenediff.yaml`).
`none` scores the proposals with their own Stage-1 scores. The zero-shot rows ask the base VLMs. The RL rows use
the step-45 adapters; with an adapter, the base model's answer and the Stage-1 scores are combined with the
adapter's answer by a fixed rule (`fuse_with_evidence` in `revisit/verifier.py`). The verifier then prunes and
re-scores proposals (`revisit/eval/scenediff_verifier.py`) and the official SceneDiff evaluator scores the result.

- The RL adapters were trained on cards from validation scenes, and every pruning and fusion constant was set on
  the validation split before the test split was run. Validation numbers in `results/scenediff.json` are
  therefore in-sample for the RL rows; the README reports test only.
- The test split lists 250 scenes. Our Stage-1 run wrote no output for `living_room_19_living_room_20` (that
  process failed and was not retried; rerunning it succeeds). The official evaluator skips scenes without
  predictions, so every reported row covers the same 249 scenes. `metrics` and `verify` use our released
  proposals and reproduce this setting. `full` recomputes the proposals, includes that scene, and gets proposals
  that match ours closely but not exactly (in a check of three scenes, one differed in a few masks), so its
  numbers differ slightly.
- As in our runs, the RL rows reuse the base model's answers from the Qwen zero-shot run (`--base-outputs`).
- The GRPO and GDPO test rows are clean reruns of the released adapters with this code. In our first
  evaluation, the test runs of those two adapters resumed from an earlier pass and sent its already fused
  answers through the fusion rule a second time, which moved their AP by up to 0.8 points. Every other row,
  including the validation rows of these two adapters, matches this code exactly.
- AUROC, Brier and ECE are computed per proposal against the matched ground truth (2,231 test proposals), with
  the verifier's p(change) or, for `none`, the Stage-1 score.

## Mask backend and keyframes

Room benchmark, keyword score ([room_benchmark.md](room_benchmark.md#keyword-score); it over-credits, so read
it as a relative measure). The first two rows are the final configuration.

| Masks | Keyframes | Verifier | No-change pair: false / proposals | Before/after: found / static | With re-look: found / static | Plant reported with re-look |
|:--|--:|:--|--:|--:|--:|:--|
| SAM 2.1 | 28 | Qwen | 11 / 17 | 7 / 1 | 6 / 2 | yes |
| SAM 2.1 | 28 | SenseNova | 10 / 17 | 8 / 1 | 7 / 3 | yes |
| SAM 1 | 28 | Qwen | 9 / 20 | 7 / 2 | 6 / 4 | yes |
| SAM 1 | 28 | SenseNova | 10 / 20 | 7 / 2 | 6 / 6 | yes |
| SAM 2.1 | 36 | Qwen | 13 / 25 | 7 / 1 | 4 / 3 | no |
| SAM 2.1 | 36 | SenseNova | 14 / 25 | 8 / 4 | 4 / 6 | no |
| SAM 1 | 36 | Qwen | 25 / 44 | 7 / 0 | 6 / 6 | no |
| SAM 1 | 36 | SenseNova | 24 / 44 | 7 / 4 | 5 / 9 | no |
| SAM 3.1 | 36 | Qwen | 29 / 51 | 6 / 1 | 6 / 7 | no |
| SAM 3.1 | 36 | SenseNova | 23 / 51 | 7 / 3 | 6 / 12 | no |

The keyframe count matters more than the mask model: with 36 keyframes per scan, every backend proposes more
false changes and none reports the plant after the re-look. Both settings were chosen on this room. Every cell
uses the ARKit scale. Our first runs of the 36-keyframe cells used the older LiDAR estimate; rerunning them with
this code changed only the re-look false flags of the SAM 3.1 cell (8 to 7 with Qwen, 13 to 12 with SenseNova).

## Metric scale

Metres per Pi3 unit, from the ARKit camera path (median over keyframe pairs within each take) and from the older
LiDAR estimate (median LiDAR depth over median Pi3 object distance).

| Comparison | ARKit, per take | ARKit, pooled (used) | LiDAR estimate | LiDAR estimate, capped at 12 |
|:--|:--|--:|--:|--:|
| Take 1 vs Take 2 | 8.219, 8.064 | 8.145 | 15.60 | 12.00 |
| Take 1 vs Take 3 | 8.478, 8.359 | 8.427 | 14.11 | 12.00 |
| Take 1 vs Take 3 + Take 4 | 8.507, 8.379, 8.835 | 8.445 | 15.30 | 12.00 |

The per-take ARKit values agree within 2-5%. The LiDAR estimate compares depth dominated by walls with object
distances and reads about 1.8 times too high; it is only logged.

## Runtime

Seconds per comparison with the Qwen verifier, median [min-max] over 3 repeats on 3 machines, one H100 80GB per
comparison and three comparisons per machine at a time (PyTorch 2.5.1). Every run started from the raw captures.

| Stage | No-change pair | Before/after | With re-look |
|:--|--:|--:|--:|
| Keyframes per scan (second scan + re-look) | 28 / 28 | 28 / 28 | 28 / 42 |
| Proposals verified | 17 | 19 | 24 |
| Stage 0: keyframes and LiDAR | 25.9 | 27.7 | 36.0 |
| Stage 1-2: Pi3 + SAM 2.1 + DINOv3 | 214.8 | 213.1 | 270.7 |
| of which Pi3 | 15.0 | 14.2 | 14.0 |
| of which SAM 2.1 masks | 72.0 | 71.5 | 88.7 |
| of which cost maps | 53.8 | 54.8 | 76.3 |
| of which voxelization and features | 62.5 | 61.0 | 76.4 |
| Stage 4: load VLM | 16.2 | 16.6 | 13.7 |
| Stage 4: verify proposals | 234.8 | 269.5 | 334.7 |
| Total in process | 493.9 [487.7-496.2] | 528.0 [507.0-532.7] | 657.8 [655.4-661.5] |
| Wall clock | 518.3 | 550.7 | 682.3 |

SenseNova takes 563, 598 and 754 s (wall clock), at about 16.7 s per proposal instead of 13.9 s. Each proposal
costs two unbatched generations (adapter and base model). Full tables: `results/runtime_{qwen,sensenova}.json`.
