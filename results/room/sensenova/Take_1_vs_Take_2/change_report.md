# Revisit change report: `Take_1_vs_Take_2`

> Pipeline: `Pi3` + `SAM 2.1 Hiera-L` + `DINOv3-ViT-H+/16` + 3D evidence ledger + `SenseNova-SI-1.3-Qwen3-VL-8B` (adapter `sensenova8b-gdpo-rlcr-step2290`)

> Metric scale from the ARKit trajectory: `s_hat=8.1451` m per Pi3 unit (`per_take={'Take_1': 8.2189, 'Take_2': 8.0641}`, `rel_spread=0.031`)

### Changes

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |
|---:|:---|:---|---:|---:|:---|
| `1` | **MOVED** | `couch` | `0.718` | `1.015` | The couch has visibly shifted position between frames, confirmed by 3D displacement and consistent object appearance. |
| `2` | **REMOVED** | `picture frame` | `0.835` | `0.000` | The picture frame is absent in Video 2, confirmed by zero cross-view DINOv3 similarity and zero 3D displacement. |
| `3` | **REMOVED** | `poster` | `0.844` | `0.000` | The poster is absent in Video 2, confirmed by zero DINOv3 similarity and zero 3D displacement, despite partial occlusion in Video 1. |
| `5` | **MOVED** | `chair` | `0.836` | `6.810` | The chair was physically relocated, confirmed by 3D displacement and consistent DINOv3 object similarity across views. |
| `6` | **UNVERIFIED** | `chair` | `0.057` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.07`). Look from `pos=[1.822, -0.594, 2.353]m` toward `target=[1.618, 0.01, 2.916]m`, `yaw=-19.9°`. |
| `8` | **MOVED** | `couch pillow` | `0.553` | `1.073` | A black couch pillow was physically relocated from a different position on the sofa. |
| `9` | **MOVED** | `chair` | `0.543` | `0.339` | Chair visibly shifted position between frames, confirmed by low geometric cost and high cross-view object similarity. |
| `11` | **MOVED** | `chair` | `0.465` | `1.595` | The chair's position shifted slightly between frames, confirmed by low geometric cost and high cross-view visibility. |
| `13` | **MOVED** | `coffee table` | `0.352` | `0.337` | The coffee table's position shifted slightly between frames, confirmed by low geometric cost and high cross-view object similarity. |
| `14` | **ADDED** | `book` | `0.486` | `0.000` | A book appears in Video 2 where none existed in Video 1, confirmed by zero geometric displacement and high cross-view similarity. |
| `15` | **ADDED** | `can` | `0.596` | `0.000` | A can appears in Video 2 where none existed in Video 1, confirmed by high geometric and semantic change scores and zero cross-view similarity. |

### Rejected proposals

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |
|---:|:---|:---|---:|---:|:---|
| `0` | none (proposal: `removed`) | `chair` | `0.270` | `0.000` | The chair appears unchanged despite viewpoint shift, confirmed by near-zero 3D displacement and low cross-view DINOv3 similarity. |
| `4` | none (proposal: `removed`) | `poster` | `0.327` | `0.000` | The poster appears unchanged despite different camera angles and lighting, confirmed by near-zero 3D displacement and DINOv3 similarity. |
| `7` | none (proposal: `removed`) | `chair` | `0.236` | `0.000` | The chair appears in both frames with identical position and orientation, confirmed by zero 3D displacement and high cross-view camera overlap. |
| `10` | none (proposal: `removed`) | `chair` | `0.559` | `0.000` | The chair appears in both frames with identical position and orientation, confirmed by near-zero 3D displacement and high cross-view similarity. |
| `12` | none (proposal: `removed`) | `floor` | `0.078` | `0.000` | The striped floor area remains unchanged between frames, confirmed by zero 3D displacement and near-zero DINOv3 similarity. |
| `16` | none (proposal: `added`) | `floor` | `0.114` | `0.000` | The floor region shows no physical change; the low DINOv3 similarity and zero displacement confirm it's a false positive due to viewpoint shift. |
