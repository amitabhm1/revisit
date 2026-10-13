# Revisit change report: `Take_1_vs_Take_3_relook_Take_4`

> Pipeline: `Pi3` + `SAM 2.1 Hiera-L` + `DINOv3-ViT-H+/16` + 3D evidence ledger + `SenseNova-SI-1.3-Qwen3-VL-8B` (adapter `sensenova8b-gdpo-rlcr-step2290`)

> Metric scale from the ARKit trajectory: `s_hat=8.4449` m per Pi3 unit (`per_take={'Take_1': 8.5074, 'Take_3': 8.3787, 'Take_4': 8.8354}`, `rel_spread=0.029`, `relook=Take_4`)

### Changes

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |
|---:|:---|:---|---:|---:|:---|
| `1` | **MOVED** | `chair` | `0.642` | `2.003` | The chair's position shifted between frames, confirmed by 3D displacement and consistent DINOv3 object matching. |
| `2` | **MOVED** | `couch` | `0.696` | `1.215` | The couch has visibly shifted position between frames, confirmed by consistent 3D displacement and high cross-view object similarity. |
| `3` | **REMOVED** | `chair` | `0.873` | `0.000` | The chair is removed, confirmed by zero 3D displacement and near-zero DINOv3 similarity despite high visibility coverage. |
| `5` | **MOVED** | `poster` | `0.639` | `6.596` | The poster is physically relocated from its original wall position to a new location on the floor. |
| `6` | **REMOVED** | `poster` | `0.718` | `0.000` | The poster is absent in Video 2 compared to Video 1, confirmed by zero DINOv3 similarity and zero 3D displacement. |
| `7` | **MOVED** | `chair` | `0.628` | `0.646` | The chair's position shifted slightly between frames, confirmed by low geometric cost and high cross-view similarity. |
| `9` | **MOVED** | `chair` | `0.930` | `0.803` | The chair is physically relocated within the room, confirmed by consistent 3D geometry and high DINOv3 similarity despite camera shift. |
| `10` | **MOVED** | `chair` | `0.933` | `0.319` | The chair's position shifted slightly between frames, confirmed by low geometric cost and high cross-view similarity. |
| `11` | **REMOVED** | `object` | `0.959` | `0.000` | Object is removed as it appears in Video 1 but not Video 2, confirmed by zero 3D displacement and near-zero DINOv3 similarity. |
| `12` | **REMOVED** | `potted plant` | `0.727` | `0.000` | The potted plant is absent in Video 2, confirmed by zero DINOv3 similarity and zero 3D displacement. |
| `13` | **MOVED** | `chair` | `0.895` | `6.641` | The chair was physically relocated, confirmed by consistent DINOv3 similarity and significant 3D displacement. |
| `14` | **REMOVED** | `book` | `0.631` | `0.000` | The stack of books is absent in Video 2, confirmed by zero DINOv3 similarity and zero 3D displacement. |
| `15` | **REMOVED** | `stack of books` | `0.426` | `0.000` | The stack of books is removed from the table, confirmed by zero 3D displacement and near-zero DINOv3 similarity. |
| `17` | **REMOVED** | `book` | `0.531` | `0.000` | The blue book is absent in Video 2, confirmed by zero 3D displacement and near-zero DINOv3 similarity. |
| `19` | **ADDED** | `stack of books` | `0.804` | `0.000` | A stack of books appears in Video 2 where only floor was visible in Video 1, confirmed by low semantic similarity and high visibility coverage. |
| `20` | **ADDED** | `cup` | `0.418` | `0.000` | A white cup appears in the second frame where none existed before, confirmed by low semantic similarity and high visibility coverage. |

### Rejected proposals

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |
|---:|:---|:---|---:|---:|:---|
| `0` | none (proposal: `removed`) | `chair` | `0.413` | `0.000` | The chair appears unchanged in position and appearance across both frames, confirmed by near-zero 3D displacement and high cross-view similarity. |
| `4` | none (proposal: `replaced`) | `poster` | `0.229` | `0.093` | The poster appears visually consistent across both frames, with low geometric and semantic change scores indicating no physical alteration. |
| `8` | none (proposal: `removed`) | `chair` | `0.148` | `0.000` | Chair appears unchanged despite viewpoint shift, confirmed by near-zero 3D displacement and high cross-view similarity. |
| `16` | none (proposal: `removed`) | `poster` | `0.089` | `0.000` | The poster appears unchanged in position and appearance across both frames, with near-zero 3D displacement and near-zero DINOv3 similarity, indicating no real change. |
| `18` | none (proposal: `removed`) | `book` | `0.120` | `0.000` | The book appears in both frames with identical position and orientation, confirmed by zero 3D displacement and near-zero DINOv3 similarity, indicating no change. |
| `21` | none (proposal: `added`) | `chair` | `0.192` | `0.000` | The chair appears identical in both frames, with no displacement or occlusion changes indicating a real object change. |
| `22` | none (proposal: `added`) | `poster` | `0.083` | `0.000` | The poster's position and appearance are consistent across both frames, with no detectable change. |
| `23` | none (proposal: `added`) | `poster` | `0.081` | `0.000` | The poster's appearance is consistent across frames, with no geometric or semantic change detected despite camera shift. |
