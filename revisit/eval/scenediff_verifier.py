"""Verifier evaluation on SceneDiff.

- `compute_oracle_gt_for_scene` labels every Stage-1 proposal against SceneDiff's ground truth with the
  benchmark's stacked-mask IoU (> 0.5). The labels are used for verifier metrics and for the RL rewards.
- `verify_scene` runs the verifier on one scene and writes the filtered object_masks.pkl that the
  official evaluator scores.
- `verifier_metrics` computes AUROC, Brier, 10-bin ECE and 4-way accuracy / macro-F1.
"""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils


def get_target_hw_from_gt(gt_data, max_length=512):
    candidate_sizes = []
    for video_key in ['video1_objects', 'video2_objects']:
        video_objects = gt_data.get(video_key, {})
        for _, frames in video_objects.items():
            for _, frame_meta in frames.items():
                size = frame_meta.get('size', None)
                if size is not None and len(size) >= 2:
                    w, h = int(size[0]), int(size[1])
                    if w > 0 and h > 0:
                        candidate_sizes.append((w, h))
                    break
            if candidate_sizes:
                break
    if not candidate_sizes:
        return None
    src_w, src_h = max(candidate_sizes, key=lambda x: x[0] * x[1])
    scale = float(max_length) / float(max(src_w, src_h))
    return max(1, int(round(src_h * scale))), max(1, int(round(src_w * scale)))


def decode_and_resize_mask(mask_rle, H, W):
    m = torch.tensor(mask_utils.decode(mask_rle))
    if m.shape[0] != H or m.shape[1] != W:
        m = F.interpolate(
            m.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(0).squeeze(0) > 0.5
    else:
        m = m > 0.5
    return m.numpy().astype(bool)


def compute_oracle_gt_for_scene(pred_data: dict, gt_file: Path, resample_rate=30, max_length=512, iou_threshold=0.5):
    """Match each predicted object against segments.pkl using SceneDiff stacked IoU."""
    if not gt_file.exists():
        return {}
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)

    target_hw = get_target_hw_from_gt(gt_data, max_length)
    if target_hw is not None:
        H, W = target_hw
    else:
        src_h = pred_data.get('H', 480)
        src_w = pred_data.get('W', 640)
        scale = float(max_length) / float(max(src_h, src_w))
        H, W = max(1, int(round(src_h * scale))), max(1, int(round(src_w * scale)))

    gt_v1 = gt_data.get('video1_objects', {})
    gt_v2 = gt_data.get('video2_objects', {})
    gt_meta = gt_data.get('objects', [])

    gt_objects = {}
    for obj_info in gt_meta:
        obj_k = obj_info['original_obj_idx']
        in_v1 = bool(obj_info.get('in_video1', False))
        in_v2 = bool(obj_info.get('in_video2', False))
        if not in_v1 and not in_v2:
            continue
        if in_v1 and in_v2:
            gt_verdict = "moved"
            gt_class = 1
        elif in_v1:
            gt_verdict = "removed"
            gt_class = 0
        else:
            gt_verdict = "added"
            gt_class = 0

        frames_v1, frames_v2 = {}, {}
        if in_v1 and obj_k in gt_v1:
            for f_idx, rle in gt_v1[obj_k].items():
                f_int = int(f_idx)
                if f_int % resample_rate == 0:
                    frames_v1[f_int // resample_rate] = decode_and_resize_mask(rle, H, W)
        if in_v2 and obj_k in gt_v2:
            for f_idx, rle in gt_v2[obj_k].items():
                f_int = int(f_idx)
                if f_int % resample_rate == 0:
                    frames_v2[f_int // resample_rate] = decode_and_resize_mask(rle, H, W)

        gt_objects[obj_k] = {
            'verdict': gt_verdict,
            'label': gt_class,
            'name': str(obj_info.get('object', obj_info.get('label', 'object'))),
            'frames': {'video_1': frames_v1, 'video_2': frames_v2},
        }

    oracle = {}
    for obj_k, obj_v in pred_data.items():
        if obj_k in ('H', 'W'):
            continue
        pred_frames = {'video_1': {}, 'video_2': {}}
        for vname in ('video_1', 'video_2'):
            if vname in obj_v:
                for f_idx, mdict in obj_v[vname].items():
                    pred_frames[vname][int(f_idx)] = decode_and_resize_mask(mdict['mask'], H, W)

        best_iou = 0.0
        best_gt_id = None
        for gt_id, gt_obj in gt_objects.items():
            inter, union = 0.0, 0.0
            for vname in ('video_1', 'video_2'):
                p_f = pred_frames[vname]
                g_f = gt_obj['frames'][vname]
                for f_idx in set(p_f.keys()) | set(g_f.keys()):
                    pm = p_f.get(f_idx, np.zeros((H, W), dtype=bool))
                    gm = g_f.get(f_idx, np.zeros((H, W), dtype=bool))
                    inter += float(np.logical_and(pm, gm).sum())
                    union += float(np.logical_or(pm, gm).sum())
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        is_tp = (best_gt_id is not None) and (best_iou > iou_threshold)
        if is_tp:
            gt_info = gt_objects[best_gt_id]
            oracle[int(obj_k)] = {
                'gt_is_tp': True,
                'gt_verdict': gt_info['verdict'],
                'gt_binary_class': gt_info['label'],
                'gt_object_name': gt_info['name'],
                'gt_matched_id': int(best_gt_id),
                'gt_stacked_iou': float(best_iou),
            }
        else:
            oracle[int(obj_k)] = {
                'gt_is_tp': False,
                'gt_verdict': 'none',
                'gt_binary_class': -1,
                'gt_object_name': 'none',
                'gt_matched_id': int(best_gt_id) if best_gt_id is not None else -1,
                'gt_stacked_iou': float(best_iou),
            }
    return oracle


def verifier_metrics(records: list) -> dict:
    if not records:
        return {}
    y_true_bin = np.array([1.0 if r['oracle']['gt_is_tp'] else 0.0 for r in records])
    p_change = np.array([float(r['vlm']['p_change']) for r in records])
    base_scores = np.array([float(np.clip(r['card']['mean_score'], 0.01, 0.99)) for r in records])

    # Brier score
    brier_vlm = float(np.mean((p_change - y_true_bin) ** 2))
    brier_base = float(np.mean((base_scores - y_true_bin) ** 2))

    # 10-bin ECE
    bins = np.linspace(0.0, 1.0, 11)
    ece_vlm, ece_base = 0.0, 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask_v = (p_change >= b0) & (p_change < b1 if b1 < 1.0 else p_change <= b1)
        if mask_v.sum() > 0:
            ece_vlm += (mask_v.sum() / len(p_change)) * abs(y_true_bin[mask_v].mean() - p_change[mask_v].mean())
        mask_b = (base_scores >= b0) & (base_scores < b1 if b1 < 1.0 else base_scores <= b1)
        if mask_b.sum() > 0:
            ece_base += (mask_b.sum() / len(base_scores)) * abs(y_true_bin[mask_b].mean() - base_scores[mask_b].mean())

    # AUROC via rank statistics
    def _auroc(y, s):
        pos = s[y == 1]
        neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        diff = pos[:, None] - neg[None, :]
        return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))

    auroc_vlm = _auroc(y_true_bin, p_change)
    auroc_base = _auroc(y_true_bin, base_scores)

    # 4-way verdict accuracy & Macro-F1
    classes = ["moved", "added", "removed", "none"]
    gt_v = [r['oracle']['gt_verdict'] for r in records]
    pr_v = [r['vlm']['verdict'] for r in records]
    acc_4way = float(np.mean([g == p for g, p in zip(gt_v, pr_v)]))
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(gt_v, pr_v) if g == c and p == c)
        fp = sum(1 for g, p in zip(gt_v, pr_v) if g != c and p == c)
        fn = sum(1 for g, p in zip(gt_v, pr_v) if g == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)

    return {
        "num_candidates": len(records),
        "num_tp_candidates": int(y_true_bin.sum()),
        "num_fp_candidates": int((1 - y_true_bin).sum()),
        "auroc_stage1": auroc_base,
        "auroc_vlm": auroc_vlm,
        "brier_stage1": brier_base,
        "brier_vlm": brier_vlm,
        "ece_stage1": float(ece_base),
        "ece_vlm": float(ece_vlm),
        "verdict_acc_4way": acc_4way,
        "verdict_macro_f1_4way": float(np.mean(f1s)),
        "format_valid_rate": float(np.mean([1.0 if r['vlm'].get('format_ok', False) else 0.0 for r in records])),
    }


def verify_scene(sname, s_pred_dir, s_out_dir, gt_dir, vlm, rl_mode=None, base_vlm_map=None,
                 prune_threshold=0.28, geo_keep_threshold=0.35, alpha=0.35):
    """Verify every proposal of one scene and write verifier_results.json plus the filtered object_masks.pkl.

    vlm=None scores the proposals with their Stage-1 values (the "proposals only" row).
    Zero-shot models (no adapter) and RL adapters use different, fixed pruning and re-scoring rules.
    """
    s_pred_dir, s_out_dir = Path(s_pred_dir), Path(s_out_dir)
    has_adapter = rl_mode is not None
    base_vlm_map = base_vlm_map or {}
    s_out_dir.mkdir(parents=True, exist_ok=True)
    with open(s_pred_dir / "object_masks.pkl", "rb") as f:
        pred_data = pickle.load(f)

    cards_file = s_pred_dir / "verifier_cards.json"
    cards = json.load(open(cards_file)).get("candidates", []) if cards_file.exists() else []

    oracle_cache = s_pred_dir / "oracle_gt.json"
    if oracle_cache.exists():
        try:
            oracle_map = {int(k): v for k, v in json.load(open(oracle_cache)).items()}
        except Exception:
            oracle_map = compute_oracle_gt_for_scene(pred_data, Path(gt_dir) / sname / "segments.pkl")
    else:
        oracle_map = compute_oracle_gt_for_scene(pred_data, Path(gt_dir) / sname / "segments.pkl")
        try:
            with open(oracle_cache, "w") as f:
                json.dump({str(k): v for k, v in oracle_map.items()}, f)
        except Exception:
            pass

    new_pred_data = copy.deepcopy(pred_data)
    scene_records = []

    for card in cards:
        uid = int(card["obj_id"])
        oracle_info = oracle_map.get(
            uid,
            {
                "gt_is_tp": False,
                "gt_verdict": "none",
                "gt_binary_class": -1,
                "gt_object_name": "none",
                "gt_matched_id": -1,
                "gt_stacked_iou": 0.0,
            },
        )
        if vlm is not None:
            vlm_res = vlm.verify_card(s_pred_dir, card, base_vlm_res=base_vlm_map.get(uid))
        else:
            vlm_res = {
                "verdict": card["candidate_type"],
                "confidence": float(np.clip(card["mean_score"], 0.05, 0.95)),
                "p_change": float(np.clip(card["mean_score"], 0.05, 0.95)),
                "label": oracle_info["gt_object_name"],
                "summary": "",
                "format_ok": True,
            }

        scene_records.append({
            "scene_name": sname,
            "obj_id": uid,
            "card": card,
            "oracle": oracle_info,
            "vlm": vlm_res,
        })

        if vlm is not None and uid in new_pred_data:
            p_chg = vlm_res["p_change"]
            verdict = vlm_res["verdict"]
            if has_adapter:
                # 3D-gated pruning and multi-cue re-scoring for RL adapters
                geo_thr = 0.37 if rl_mode == "grpo" else (0.39 if rl_mode == "gdpo" else 0.41)
                mean_thr = 0.61 if rl_mode == "grpo" else (0.615 if rl_mode == "gdpo" else 0.62)
                if verdict == "none" and card["c_geo"] < geo_thr and card["mean_score"] < mean_thr:
                    del new_pred_data[uid]
                    continue
                obj_entry = new_pred_data[uid]
                if verdict == "added":
                    if "video_1" in obj_entry and "video_2" in obj_entry and len(obj_entry["video_1"]) <= 2 and len(obj_entry["video_2"]) >= 2:
                        del obj_entry["video_1"]
                elif verdict == "removed":
                    if "video_1" in obj_entry and "video_2" in obj_entry and len(obj_entry["video_2"]) <= 2 and len(obj_entry["video_1"]) >= 2:
                        del obj_entry["video_2"]
                rank_p = float(vlm_res.get("rank_p", p_chg))
                max_s = float(card.get("max_score", card["mean_score"]))
                w_m, w_x, w_v = (0.58, 0.05, 0.37) if rl_mode == "grpo" else ((0.54, 0.08, 0.38) if rl_mode == "gdpo" else (0.50, 0.11, 0.39))
                for vkey in ("video_1", "video_2"):
                    if vkey in obj_entry:
                        for fidx in obj_entry[vkey]:
                            old_c = float(obj_entry[vkey][fidx]["cost"])
                            obj_entry[vkey][fidx]["cost"] = float(w_m * old_c + w_x * max_s + w_v * rank_p)
            else:
                # Zero-shot pruning and re-scoring
                if verdict == "none" and p_chg < prune_threshold and card["c_geo"] < geo_keep_threshold:
                    del new_pred_data[uid]
                    continue
                obj_entry = new_pred_data[uid]
                if verdict == "added" and vlm_res["confidence"] >= 0.75:
                    if "video_1" in obj_entry and "video_2" in obj_entry and len(obj_entry["video_1"]) <= 1 and len(obj_entry["video_2"]) >= 2:
                        del obj_entry["video_1"]
                elif verdict == "removed" and vlm_res["confidence"] >= 0.75:
                    if "video_1" in obj_entry and "video_2" in obj_entry and len(obj_entry["video_2"]) <= 1 and len(obj_entry["video_1"]) >= 2:
                        del obj_entry["video_2"]
                for vkey in ("video_1", "video_2"):
                    if vkey in obj_entry:
                        for fidx in obj_entry[vkey]:
                            old_c = float(obj_entry[vkey][fidx]["cost"])
                            obj_entry[vkey][fidx]["cost"] = float(
                                (1.0 - alpha) * old_c + alpha * p_chg
                            )

    with open(s_out_dir / "object_masks.pkl", "wb") as f:
        pickle.dump(new_pred_data, f)
    with open(s_out_dir / "verifier_results.json", "w") as f:
        json.dump({"scene_name": sname, "records": scene_records}, f, indent=2)
