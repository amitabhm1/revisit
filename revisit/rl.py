"""GDPO + RLCR building blocks: rewards, group advantages, the card pool and its augmentation.

Rewards per sampled answer:
  R_fmt  format: parseable <answer>{...}</answer>, a short <think> that cites geometry, appearance and visibility.
  R_vrd  verdict: +1 exact, +0.35 right "changed" but wrong type, -1 / -0.65 hallucinated change
         (harsher when the 3D cost is low), -0.6 missed change.
  R_cal  calibration (RLCR): 1 - (0.6 * Brier(p_change, changed) + 0.4 * Brier(confidence, verdict correct)).

GRPO sums the weighted rewards and normalises once per group. GDPO normalises each reward within the
group first, then sums with weights fmt 0.25, verdict 1.0 and (GDPO+RLCR only) calibration 0.80.
"""
import copy
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

from .eval.scenediff_verifier import compute_oracle_gt_for_scene


def compute_reward_components(parsed: dict, oracle: dict, card: dict) -> tuple[float, float, float]:
    """
    Returns (R_format, R_verdict, R_cal) for a single sampled rollout.
    """
    raw = parsed.get("raw_response", "")
    raw_low = raw.lower()
    think_m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    has_think = think_m is not None
    has_ans = bool(re.search(r"<answer>\s*\{.*?\}\s*</answer>", raw, flags=re.DOTALL))
    if parsed.get("format_ok", False):
        r_fmt = 0.45
        if has_think:
            r_fmt += 0.20
        if has_ans:
            r_fmt += 0.15
        cites_geo = any(k in raw_low for k in ("c_geo", "geometr", "depth", "3d", "surface"))
        cites_sem = any(k in raw_low for k in ("c_render", "c_match", "dino", "semantic", "appearance"))
        cites_vis = any(k in raw_low for k in ("vis_coverage", "occlusion", "viewpoint", "overlap", "coverage"))
        r_fmt += 0.05 * (int(cites_geo) + int(cites_sem) + int(cites_vis))
        n_words = len((think_m.group(1) if think_m else raw).split())
        r_fmt += 0.05 * math.exp(-((n_words - 45.0) / 35.0) ** 2)
    else:
        r_fmt = -1.0 + 0.001 * min(len(raw.split()), 100)

    gt_v = oracle["gt_verdict"]  # moved | added | removed | none
    pr_v = parsed["verdict"]
    gt_tp = bool(oracle["gt_is_tp"])
    pr_tp = (pr_v != "none")

    if pr_v == gt_v:
        r_vrd = 1.0
    elif gt_tp and pr_tp:
        r_vrd = 0.35
    elif (not gt_tp) and pr_tp:
        r_vrd = -1.0 if card.get("c_geo", 0.0) < 0.30 else -0.65
    else:
        r_vrd = -0.60

    y_bin = 1.0 if gt_tp else 0.0
    p_chg = float(np.clip(parsed["p_change"], 0.01, 0.99))
    brier = (p_chg - y_bin) ** 2
    y_4way = 1.0 if (pr_v == gt_v) else 0.0
    conf = float(np.clip(parsed["confidence"], 0.01, 0.99))
    brier_4way = (conf - y_4way) ** 2
    r_cal = 1.0 - (0.6 * brier + 0.4 * brier_4way)

    return float(r_fmt), float(r_vrd), float(r_cal)


def compute_group_advantages(
    r_fmt: torch.Tensor,
    r_vrd: torch.Tensor,
    r_cal: torch.Tensor,
    mode: str = "gdpo_rlcr",
    w_fmt: float = 0.25,
    w_vrd: float = 1.0,
    w_cal: float = 0.80,
    eps: float = 1e-5,
) -> torch.Tensor:
    if mode == "grpo":
        r_tot = w_fmt * r_fmt + w_vrd * r_vrd
        std = r_tot.std(unbiased=False)
        if std < eps:
            return torch.zeros_like(r_tot)
        return (r_tot - r_tot.mean()) / (std + eps)

    def _norm(x: torch.Tensor) -> torch.Tensor:
        s = x.std(unbiased=False)
        if s < eps:
            return torch.zeros_like(x)
        return (x - x.mean()) / (s + eps)

    a_fmt = _norm(r_fmt)
    a_vrd = _norm(r_vrd)
    if mode == "gdpo":
        a_comb = w_fmt * a_fmt + w_vrd * a_vrd
    else:
        a_cal = _norm(r_cal)
        a_comb = w_fmt * a_fmt + w_vrd * a_vrd + w_cal * a_cal

    s_comb = a_comb.std(unbiased=False)
    if s_comb > eps:
        a_comb = (a_comb - a_comb.mean()) / (s_comb + eps)
    return a_comb


def load_training_cards(pred_dirs_str: str, gt_dir: Path, balance: bool = True) -> list[dict]:
    import pickle
    splits_dir = gt_dir.parent / "splits"
    val_s = json.load(open(splits_dir / "val_split.json"))
    val_scenes = val_s["varied"] + val_s["kitchen"]

    pred_dirs = [Path(p.strip()) for p in pred_dirs_str.split(",") if p.strip()]
    items = []
    for pred_dir in pred_dirs:
        if not pred_dir.exists():
            continue
        for sname in val_scenes:
            s_pred = pred_dir / sname
            if not (s_pred / "object_masks.pkl").exists() or not (s_pred / "verifier_cards.json").exists():
                continue
            with open(s_pred / "object_masks.pkl", "rb") as f:
                pred_data = pickle.load(f)
            cards = json.load(open(s_pred / "verifier_cards.json")).get("candidates", [])
            oracle_cache = s_pred / "oracle_gt.json"
            if oracle_cache.exists():
                try:
                    oracle_map = {int(k): v for k, v in json.load(open(oracle_cache)).items()}
                except Exception:
                    oracle_map = compute_oracle_gt_for_scene(pred_data, gt_dir / sname / "segments.pkl")
            else:
                oracle_map = compute_oracle_gt_for_scene(pred_data, gt_dir / sname / "segments.pkl")
                try:
                    with open(oracle_cache, "w") as f:
                        json.dump({str(k): v for k, v in oracle_map.items()}, f)
                except Exception:
                    pass
            for c in cards:
                uid = int(c["obj_id"])
                if uid not in oracle_map:
                    continue
                v1_p = s_pred / c["v1_crop_path"]
                v2_p = s_pred / c["v2_crop_path"]
                pair_p = s_pred / c["pair_image_path"]
                if v1_p.exists() and v2_p.exists() and pair_p.exists():
                    items.append({
                        "scene_dir": str(s_pred),
                        "backbone_source": pred_dir.name,
                        "card": c,
                        "oracle": oracle_map[uid],
                    })

    if not balance:  # the step-45 adapters were trained on every card, in split order
        return items
    rng = random.Random(42)
    tp_items = [x for x in items if x["oracle"].get("gt_is_tp", False)]
    fp_items = [x for x in items if not x["oracle"].get("gt_is_tp", False)]
    fp_items.sort(key=lambda x: float(x["card"].get("mean_score", 0.0)), reverse=True)
    hard_fps = fp_items[: max(len(tp_items) * 2, 1)]
    rng.shuffle(tp_items)
    rng.shuffle(hard_fps)
    balanced = []
    n_pairs = max(len(tp_items), len(hard_fps))
    for i in range(n_pairs):
        if tp_items:
            balanced.append(tp_items[i % len(tp_items)])
        if hard_fps:
            balanced.append(hard_fps[i % len(hard_fps)])
    return balanced if balanced else items


def augment_multiview_sample(
    v1_crop: Image.Image,
    v2_crop: Image.Image,
    pair_img: Image.Image,
    card: dict,
    enable_aug: bool = True,
) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
    if not enable_aug:
        return (
            v1_crop.resize((384, 384)),
            v2_crop.resize((384, 384)),
            pair_img.resize((768, 384)),
            card,
        )

    def _jitter_img(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
        w, h = im.size
        scale = random.uniform(0.93, 1.0)
        nw, nh = max(int(w * scale), 16), max(int(h * scale), 16)
        x0 = random.randint(0, max(w - nw, 0))
        y0 = random.randint(0, max(h - nh, 0))
        im = im.crop((x0, y0, x0 + nw, y0 + nh)).resize((target_w, target_h))
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.90, 1.10))
        im = ImageEnhance.Contrast(im).enhance(random.uniform(0.92, 1.08))
        return im

    aug_card = copy.deepcopy(card)
    for k in ("c_geo", "c_render", "c_match", "mean_score", "max_score", "vis_coverage"):
        if k in aug_card and isinstance(aug_card[k], (int, float)):
            aug_card[k] = float(np.clip(float(aug_card[k]) + random.gauss(0.0, 0.012), 0.0, 1.0))

    return (
        _jitter_img(v1_crop, 384, 384),
        _jitter_img(v2_crop, 384, 384),
        _jitter_img(pair_img, 768, 384),
        aug_card,
    )
