# Revisit change report: `Take_1_vs_Take_3_relook_Take_4`

> Pipeline: `Pi3` + `SAM 2.1 Hiera-L` + `DINOv3-ViT-H+/16` + 3D evidence ledger + `Qwen3-VL-8B-Instruct` (adapter `qwen3vl8b-gdpo-rlcr-step2460`)

> Metric scale from the ARKit trajectory: `s_hat=8.4449` m per Pi3 unit (`per_take={'Take_1': 8.5074, 'Take_3': 8.3787, 'Take_4': 8.8354}`, `rel_spread=0.029`, `relook=Take_4`)

### Changes

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |
|---:|:---|:---|---:|---:|:---|
| `1` | **MOVED** | `chair armrest` | `0.647` | `2.003` | The black textured armrest of a chair has visibly shifted position between frames, confirmed by consistent 3D displacement and high cross-view object similarity. |
| `2` | **MOVED** | `sofa` | `0.701` | `1.215` | The sofa has visibly shifted position between frames, confirmed by consistent 3D geometry and high cross-view object similarity despite camera angle change. |
| `5` | **MOVED** | `sign` | `0.644` | `6.596` | The sign has been relocated from a wall-mounted position to a lower surface, confirmed by 3D displacement and consistent DINOv3 similarity. |
| `6` | **REMOVED** | `object` | `0.709` | `0.000` | Object is removed as it appears in Video 1 but absent in Video 2, confirmed by zero DINOv3 similarity and zero 3D displacement. |
| `7` | **MOVED** | `chair` | `0.633` | `0.646` | The chair's position shifted slightly between frames, confirmed by low geometric cost and high DINOv3 similarity, indicating a real movement. |
| `9` | **MOVED** | `chair` | `0.966` | `0.803` | Verified moved chair across multi-view 3D registration (mean_cost=1.05, max_cost=1.40, render_diff=0.66, delta_3d=0.803m, vis_coverage=0.86). |
| `10` | **MOVED** | `chair` | `0.934` | `0.319` | The chair was physically repositioned, confirmed by consistent 3D geometry and semantic matching despite minor viewpoint shift. |
| `12` | **REMOVED** | `potted plant` | `0.732` | `0.000` | Verified removed potted plant across multi-view 3D registration (mean_cost=0.74, max_cost=0.80, render_diff=0.44, delta_3d=0.000m, vis_coverage=0.56). |
| `13` | **MOVED** | `chair` | `0.895` | `6.641` | Verified moved chair across multi-view 3D registration (mean_cost=0.88, max_cost=1.15, render_diff=0.60, delta_3d=6.641m, vis_coverage=0.81). |
| `14` | **REMOVED** | `stack of books` | `0.636` | `0.000` | Verified removed stack of books across multi-view 3D registration (mean_cost=0.68, max_cost=0.71, render_diff=0.66, delta_3d=0.000m, vis_coverage=0.89). |
| `15` | **REMOVED** | `stack of books` | `0.431` | `0.000` | The stack of books is absent in the second view, confirmed by zero DINOv3 similarity and zero 3D displacement, indicating removal. |
| `17` | **REMOVED** | `book` | `0.695` | `0.000` | Verified removed book across multi-view 3D registration (mean_cost=0.73, max_cost=0.73, render_diff=0.77, delta_3d=0.000m, vis_coverage=1.00). |
| `19` | **ADDED** | `stack of books` | `0.807` | `0.000` | Verified added stack of books across multi-view 3D registration (mean_cost=0.81, max_cost=0.86, render_diff=0.79, delta_3d=0.000m, vis_coverage=0.92). |
| `20` | **ADDED** | `cup` | `0.602` | `0.000` | Verified added cup across multi-view 3D registration (mean_cost=0.66, max_cost=0.66, render_diff=0.60, delta_3d=0.000m, vis_coverage=0.89). |

### Rejected proposals

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |
|---:|:---|:---|---:|---:|:---|
| `0` | none (proposal: `removed`) | `table` | `0.408` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `3` | none (proposal: `removed`) | `chair` | `0.305` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `4` | none (proposal: `replaced`) | `ceiling light fixture` | `0.225` | `0.093` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by low geometric displacement, high semantic similarity, and full visibility coverage with no cross-view object matc |
| `8` | none (proposal: `removed`) | `chair` | `0.153` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `11` | none (proposal: `removed`) | `object shadow` | `0.740` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `16` | none (proposal: `removed`) | `wall poster` | `0.087` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `18` | none (proposal: `removed`) | `object` | `0.117` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `21` | none (proposal: `added`) | `chair` | `0.192` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `22` | none (proposal: `added`) | `pillow` | `0.081` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `23` | none (proposal: `added`) | `wall poster` | `0.081` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
