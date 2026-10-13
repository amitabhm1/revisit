# Revisit change report: `Take_1_vs_Take_3`

> Pipeline: `Pi3` + `SAM 2.1 Hiera-L` + `DINOv3-ViT-H+/16` + 3D evidence ledger + `SenseNova-SI-1.3-Qwen3-VL-8B` (adapter `sensenova8b-gdpo-rlcr-step2290`)

> Metric scale from the ARKit trajectory: `s_hat=8.4266` m per Pi3 unit (`per_take={'Take_1': 8.4783, 'Take_3': 8.3592}`, `rel_spread=0.022`)

### Changes

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |
|---:|:---|:---|---:|---:|:---|
| `0` | **MOVED** | `chair` | `0.663` | `1.402` | The chair's position changed between frames, confirmed by 3D displacement and consistent DINOv3 appearance matching. |
| `1` | **REMOVED** | `chair` | `0.509` | `0.000` | Chair appears unchanged despite viewpoint shift, confirmed by near-zero 3D displacement and high cross-view similarity. |
| `2` | **UNVERIFIED** | `poster` | `0.146` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.11`). Look from `pos=[-1.416, -1.694, 3.527]m` toward `target=[-0.827, -1.084, 3.472]m`, `yaw=95.4°`. |
| `3` | **MOVED** | `poster` | `0.691` | `6.633` | The poster is physically relocated from its original wall position to a new location on the floor. |
| `4` | **UNVERIFIED** | `chair` | `0.209` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.11`). Look from `pos=[-1.447, -1.417, 3.178]m` toward `target=[-0.766, -1.145, 3.608]m`, `yaw=57.7°`. |
| `6` | **MOVED** | `chair` | `0.931` | `0.827` | The chair is confirmed to have moved slightly, evidenced by consistent DINOv3 similarity and low geometric displacement. |
| `8` | **MOVED** | `chair` | `0.911` | `6.811` | The chair's position changed between frames, confirmed by 3D displacement and consistent object appearance. |
| `9` | **MOVED** | `chair` | `0.498` | `7.328` | The chair was physically relocated, confirmed by 3D displacement and consistent DINOv3 object similarity across views. |
| `10` | **REMOVED** | `book` | `0.540` | `0.000` | The book stack is removed, confirmed by zero 3D displacement and near-zero DINOv3 similarity. |
| `11` | **REMOVED** | `book` | `0.454` | `0.000` | The stack of books is absent in Video 2, confirmed by zero DINOv3 similarity and zero 3D displacement. |
| `12` | **UNVERIFIED** | `book` | `0.548` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.04`). Look from `pos=[-2.371, 0.771, -0.385]m` toward `target=[-1.884, 1.298, 0.07]m`, `yaw=47.0°`. |
| `14` | **ADDED** | `cup` | `0.433` | `0.000` | A white cup appears in Video 2 where none existed in Video 1, confirmed by low DINOv3 similarity and zero 3D displacement. |
| `16` | **ADDED** | `pillow` | `0.508` | `0.000` | A white pillow is added to the sofa in Video 2, confirmed by high geometric and semantic costs with zero cross-view similarity. |
| `17` | **ADDED** | `books` | `0.786` | `0.000` | A stack of books appears in Video 2 where only floor texture was visible in Video 1, confirmed by high geometric and semantic change scores. |
| `18` | **ADDED** | `books` | `0.660` | `0.000` | A stack of books appears in Video 2 where nothing was present in Video 1, confirmed by high geometric and semantic change scores. |

### Rejected proposals

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |
|---:|:---|:---|---:|---:|:---|
| `5` | none (proposal: `removed`) | `chair` | `0.198` | `0.000` | The chair appears unchanged in position and appearance across both frames, confirmed by near-zero 3D displacement and DINOv3 similarity. |
| `7` | none (proposal: `removed`) | `object` | `0.084` | `0.000` | Object appears unchanged despite low DINOv3 similarity due to consistent camera position and occlusion. |
| `13` | none (proposal: `added`) | `couch` | `0.118` | `0.000` | The couch appears identical in both frames, with no detectable movement or addition/removal, confirmed by zero 3D displacement and near-zero DINOv3 similarity. |
| `15` | none (proposal: `added`) | `chair` | `0.225` | `0.000` | The chair appears visually identical in both frames, with zero 3D displacement and near-zero semantic mismatch, indicating no real change. |
