# Revisit change report: `Take_1_vs_Take_3`

> Pipeline: `Pi3` + `SAM 2.1 Hiera-L` + `DINOv3-ViT-H+/16` + 3D evidence ledger + `Qwen3-VL-8B-Instruct` (adapter `qwen3vl8b-gdpo-rlcr-step2460`)

> Metric scale from the ARKit trajectory: `s_hat=8.4266` m per Pi3 unit (`per_take={'Take_1': 8.4783, 'Take_3': 8.3592}`, `rel_spread=0.022`)

### Changes

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |
|---:|:---|:---|---:|---:|:---|
| `0` | **MOVED** | `chair` | `0.633` | `1.402` | The black chair appears to have been repositioned slightly between the two video frames, confirmed by consistent 3D geometry and semantic matching despite camera angle shift. |
| `2` | **UNVERIFIED** | `chair armrest` | `0.143` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.11`). Look from `pos=[-1.416, -1.694, 3.527]m` toward `target=[-0.827, -1.084, 3.472]m`, `yaw=95.4°`. |
| `3` | **MOVED** | `sign` | `0.671` | `6.633` | The sign has been relocated from its original position on the wall to a new spot near the floor, confirmed by 3D displacement and consistent DINOv3 appearance matching. |
| `4` | **UNVERIFIED** | `wall texture` | `0.205` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.11`). Look from `pos=[-1.447, -1.417, 3.178]m` toward `target=[-0.766, -1.145, 3.608]m`, `yaw=57.7°`. |
| `6` | **MOVED** | `chair` | `0.966` | `0.827` | Verified moved chair across multi-view 3D registration (mean_cost=1.07, max_cost=1.36, render_diff=0.66, delta_3d=0.827m, vis_coverage=0.86). |
| `8` | **MOVED** | `chair` | `0.913` | `6.811` | Verified moved chair across multi-view 3D registration (mean_cost=0.89, max_cost=1.25, render_diff=0.74, delta_3d=6.811m, vis_coverage=0.86). |
| `9` | **MOVED** | `chair` | `0.465` | `7.328` | The chair was physically relocated from its original position to a new spot near the wall, confirmed by 3D displacement and consistent DINOv3 similarity. |
| `10` | **REMOVED** | `book` | `0.712` | `0.000` | Verified removed book across multi-view 3D registration (mean_cost=0.74, max_cost=0.74, render_diff=0.64, delta_3d=0.000m, vis_coverage=0.84). |
| `11` | **REMOVED** | `stack of books` | `0.626` | `0.000` | Verified removed stack of books across multi-view 3D registration (mean_cost=0.68, max_cost=0.68, render_diff=0.72, delta_3d=0.000m, vis_coverage=0.94). |
| `12` | **UNVERIFIED** | `book` | `0.714` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.04`). Look from `pos=[-2.371, 0.771, -0.385]m` toward `target=[-1.884, 1.298, 0.07]m`, `yaw=47.0°`. |
| `14` | **ADDED** | `cup` | `0.616` | `0.000` | Verified added cup across multi-view 3D registration (mean_cost=0.67, max_cost=0.67, render_diff=0.59, delta_3d=0.000m, vis_coverage=0.88). |
| `16` | **ADDED** | `pillow` | `0.670` | `0.000` | Verified added pillow across multi-view 3D registration (mean_cost=0.72, max_cost=0.72, render_diff=0.49, delta_3d=0.000m, vis_coverage=0.88). |
| `17` | **ADDED** | `stack of books` | `0.786` | `0.000` | Verified added stack of books across multi-view 3D registration (mean_cost=0.80, max_cost=0.80, render_diff=0.75, delta_3d=0.000m, vis_coverage=0.89). |
| `18` | **ADDED** | `stack of books` | `0.655` | `0.000` | Verified added stack of books across multi-view 3D registration (mean_cost=0.70, max_cost=0.70, render_diff=0.87, delta_3d=0.000m, vis_coverage=0.98). |

### Rejected proposals

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |
|---:|:---|:---|---:|---:|:---|
| `1` | none (proposal: `removed`) | `chair` | `0.345` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `5` | none (proposal: `removed`) | `table` | `0.198` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `7` | none (proposal: `removed`) | `object shadow` | `0.082` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `13` | none (proposal: `added`) | `cushion` | `0.126` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `15` | none (proposal: `added`) | `chair` | `0.225` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero geometric displacement, zero semantic similarity, and full visibility coverage with no cross-view object mat |
