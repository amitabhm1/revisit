# Changes to SceneDiff

This folder is adapted from [SceneDiff](https://github.com/yuqunw/scene_diff) at commit
`62dcbd285fcfe984bb3db55dc01bdcd1a0055714` (MIT, see `LICENSE`). The change-detection math is SceneDiff's.
Changes, by file:

| Here | Upstream | What changed |
|:--|:--|:--|
| `scenediff.py` | `modules/scenediff.py` | Writes `scene_candidates.pkl` (merged 3D objects, poses, intrinsics) and `verifier_cards.json` with crops and side-by-side images for every proposal, used by the verifier and the evidence ledger. Keeps the cost components and records per-stage wall time in `stage_times`. Mask cache file named per mask backend, with an optional separate cache folder. |
| `geometry.py` | `modules/geometry_model.py` | Pi3 is imported as an installed package. Optional Pi3X support (not used for reported results). Loads local checkpoint folders. |
| `masks.py` | `modules/mask_model.py` | SAM 2.1, SAM 3 and SAM 3.1 can replace SAM 1 inside the same point-grid automatic mask generator. SAM 3/3.1 point prompts are normalized correctly. The SAM 3.1 checkpoint is remapped for single-image use, and loading fails if any point-prompt weight would stay randomly initialized. |
| `semantics.py` | `modules/semantic_model.py` | DINOv3 loads through Hugging Face `transformers` from a local folder. The `torch.hub` path is removed. |
| `utils.py` | `utils.py` | Thread limits for faiss and ffmpeg. Feature extraction reads the patch size from the model. Guards for empty mask levels and empty regions. Mask NMS computes the same pairwise IoUs with one matrix product instead of a double loop. |
| `../eval/scenediff_metrics.py` | `scripts/evaluate_multiview.py` | Reads the split files next to the data folder. Metric code unchanged. |
| `../../scripts/scenediff_proposals.py`, `../pipeline.py` | `scripts/predict_multiview.py` | Same per-scene processing, split into a sharded command line tool and shared helpers. |
| `segment_anything_ls/` | [segment-anything-langsplat-modified](https://github.com/yuqunw/segment-anything-langsplat-modified) `100f1ca` | Vendored (Apache-2.0, derived from Meta's Segment Anything). Guards for crops with no masks, tensors kept on one device, and a relative import in `predictor.py`. |

Not included: `modules/config_manager.py` (unused), `scripts/demo.py`, and the git submodules
(Pi3 is installed with pip; the DINOv3 code is not needed with `transformers`).
