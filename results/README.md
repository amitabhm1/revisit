# Reference results

Files written by the scripts in `reproduce/` and used by them to check a rerun.

| File | Contents |
|:--|:--|
| `scenediff.json` | every row of the SceneDiff table, test and validation splits (official metrics plus verifier AUROC, Brier, ECE, 4-way accuracy) |
| `room/scoreboard.json` | room benchmark scores of the final configuration, per verifier and per change |
| `room/<verifier>/<comparison>/change_report.{md,json}` | the full reports behind those scores; reruns must reproduce every verdict |
| `room/ablation.json` | the same scores for the mask backend and keyframe ablation |
| `metric_scale.json` | ARKit and LiDAR scale estimates per comparison |
| `runtime_{qwen,sensenova}.json` | seconds per stage and comparison, three repeats |
| `training_curves.csv` | mean rewards of our training runs, per step (step-45 runs) or per counter epoch (room runs) |
