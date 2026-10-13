# Revisit change report: `Take_1_vs_Take_2`

> Pipeline: `Pi3` + `SAM 2.1 Hiera-L` + `DINOv3-ViT-H+/16` + 3D evidence ledger + `Qwen3-VL-8B-Instruct` (adapter `qwen3vl8b-gdpo-rlcr-step2460`)

> Metric scale from the ARKit trajectory: `s_hat=8.1451` m per Pi3 unit (`per_take={'Take_1': 8.2189, 'Take_2': 8.0641}`, `rel_spread=0.031`)

### Changes

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |
|---:|:---|:---|---:|---:|:---|
| `1` | **MOVED** | `circular sofa` | `0.843` | `1.015` | Verified moved circular sofa across multi-view 3D registration (mean_cost=0.81, max_cost=1.06, render_diff=0.56, delta_3d=1.015m, vis_coverage=0.87). |
| `2` | **REMOVED** | `wall art` | `0.838` | `0.000` | The wall art piece is absent in Video 2, confirmed by zero frames, high geometric/semantic mismatch, and zero cross-view similarity. |
| `3` | **REMOVED** | `plaque` | `0.844` | `0.000` | The golden plaque is absent in Video 2, confirmed by zero frames, zero DINOv3 similarity, and zero 3D displacement, indicating removal. |
| `4` | **REMOVED** | `object` | `0.788` | `0.000` | The object is absent in Video 2, confirmed by zero frames and zero DINOv3 similarity, despite low geometric cost suggesting it was never there. |
| `5` | **MOVED** | `chair` | `0.839` | `6.810` | The chair has visibly shifted position between frames, confirmed by 3D displacement and consistent object identity across views. |
| `6` | **UNVERIFIED** | `chair` | `0.056` | `0.000` | Not seen in scan 2 (`frustum_ratio=0.07`). Look from `pos=[1.822, -0.594, 2.353]m` toward `target=[1.618, 0.01, 2.916]m`, `yaw=-19.9°`. |
| `8` | **MOVED** | `cushion` | `0.520` | `1.073` | A black cushion was repositioned on the curved sofa between the two video frames. |
| `9` | **MOVED** | `chair` | `0.709` | `0.339` | Verified moved chair across multi-view 3D registration (mean_cost=0.74, max_cost=0.81, render_diff=0.60, delta_3d=0.339m, vis_coverage=0.84). |
| `11` | **MOVED** | `chair` | `0.470` | `1.595` | The wooden chair with black cushion appears to have been repositioned slightly between frames, confirmed by consistent 3D geometry and high cross-view similarity. |
| `13` | **MOVED** | `coffee table` | `0.357` | `0.337` | The coffee table appears to have been slightly repositioned between the two video frames, evidenced by consistent geometry, high cross-view similarity, and minimal displacement. |
| `14` | **ADDED** | `box` | `0.486` | `0.000` | Verified added box across multi-view 3D registration (mean_cost=0.59, max_cost=0.59, render_diff=0.75, delta_3d=0.000m, vis_coverage=1.00). |
| `15` | **ADDED** | `can` | `0.570` | `0.000` | Verified added can across multi-view 3D registration (mean_cost=0.65, max_cost=0.69, render_diff=0.46, delta_3d=0.000m, vis_coverage=0.78). |

### Rejected proposals

| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |
|---:|:---|:---|---:|---:|:---|
| `0` | none (proposal: `removed`) | `chair armrest` | `0.278` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `7` | none (proposal: `removed`) | `chair` | `0.244` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `10` | none (proposal: `removed`) | `object shadow` | `0.553` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `12` | none (proposal: `removed`) | `carpet` | `0.076` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
| `16` | none (proposal: `added`) | `object` | `0.114` | `0.000` | The candidate region shows no real object change — it's a false positive caused by camera viewpoint shift and lighting variation, confirmed by zero 3D displacement, zero semantic similarity, and full visibility coverage with no cross-view object match. |
