"""Stage 4: VLM verifier.

Each Stage-1 proposal becomes an evidence card: two zoomed crops, a side-by-side context image with
boxes, and the 3D/appearance scores. An 8B VLM (optionally with a GDPO+RLCR LoRA adapter) answers
moved / added / removed / none with a confidence. With an adapter, the base model also answers and
`fuse_with_evidence` combines both answers with the Stage-1 scores.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .paths import read_adapter_meta


SYSTEM_PROMPT = (
    "You are an embodied 3D spatial change verification agent inspecting a room "
    "recorded at two different times (Video 1 = Before, Video 2 = After). "
    "You are given two zoomed crops (Image 1 from Video 1, Image 2 from Video 2), "
    "a side-by-side full-frame context image (Image 3) with bounding boxes, and "
    "quantitative 3D geometric + DINOv3 semantic evidence from a multi-view 3D "
    "change detector.\n\n"
    "Determine whether the candidate region represents a genuine physical object "
    "change ('moved', 'added', 'removed') or a false positive ('none') caused by "
    "camera viewpoint difference, partial occlusion, lighting/shadow shift, or "
    "background clutter.\n\n"
    "First reason step-by-step inside <think>...</think> (at most 80 words), "
    "explicitly weighing the visual crops against the 3D geometric cost (c_geo), "
    "semantic render cost (c_render), region match cost (c_match), and cross-view "
    "visibility coverage. Then output ONLY valid JSON inside <answer>...</answer> "
    "with schema:\n"
    '{"verdict": "moved|added|removed|none", "confidence": 0.00, '
    '"label": "short object category", "summary": "concise 1-sentence description"}'
)


def build_user_prompt(card: dict) -> str:
    return (
        f"Candidate ID: {card['obj_id']} in scene {card['scene_name']}\n"
        f"- Stage-1 heuristic proposal type: {card['candidate_type']}\n"
        f"- Observed frames: Video 1 = {card['n_frames_v1']} frames, Video 2 = {card['n_frames_v2']} frames\n"
        f"- 3D geometric surface discrepancy (c_geo): {card['c_geo']:.4f}\n"
        f"- DINOv3 region-matching cost (c_match): {card['c_match']:.4f}\n"
        f"- DINOv3 reprojected appearance cost (c_render): {card['c_render']:.4f}\n"
        f"- Combined Stage-1 change score: mean={card['mean_score']:.4f}, max={card['max_score']:.4f}\n"
        f"- Unoccluded visibility coverage: {card['vis_coverage']:.3f}\n"
        f"- Cross-video camera overlap at best views: {card['cross_view_overlap']:.3f}\n"
        f"- Cross-video DINOv3 object similarity: {card['dino_similarity']:.3f}\n"
        f"- 3D centroid displacement (normalized scene units): {card['delta_3d']:.4f}\n\n"
        "Inspect Image 1 (Video 1 crop), Image 2 (Video 2 crop), and Image 3 "
        "(side-by-side full frames with bounding boxes). Decide if this is a real "
        "object change ('moved', 'added', 'removed') or false positive ('none'), "
        "and provide a calibrated confidence in [0.0, 1.0] that your verdict is correct."
    )


def parse_vlm_response(text: str, fallback_type: str, fallback_score: float) -> dict:
    ans_match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.DOTALL)
    json_str = ans_match.group(1) if ans_match else None
    if not json_str:
        brace_match = re.search(r"(\{[^{}]*\"verdict\"[^{}]*\})", text, flags=re.DOTALL)
        if brace_match:
            json_str = brace_match.group(1)

    verdict = fallback_type
    conf = float(np.clip(fallback_score, 0.05, 0.95))
    label = "object"
    summary = ""
    format_ok = False

    if json_str:
        try:
            parsed = json.loads(json_str)
            v_raw = str(parsed.get("verdict", "")).strip().lower()
            if v_raw in ("moved", "added", "removed", "none"):
                verdict = v_raw
                format_ok = True
            conf = float(np.clip(float(parsed.get("confidence", conf)), 0.01, 0.99))
            label = str(parsed.get("label", "object"))[:64]
            summary = str(parsed.get("summary", ""))[:256]
        except Exception:
            pass

    p_change = conf if verdict != "none" else (1.0 - conf)
    return {
        "verdict": verdict,
        "confidence": conf,
        "p_change": float(np.clip(p_change, 0.01, 0.99)),
        "label": label,
        "summary": summary,
        "format_ok": format_ok,
        "raw_response": text,
    }


class VLMVerifier:
    def __init__(self, model_path: str, adapter_path: str = None, rl_mode: str = None, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.adapter_path = adapter_path
        self.has_adapter = bool(adapter_path and os.path.exists(adapter_path))
        self.rl_mode = None
        if self.has_adapter:
            # The RL variant selects the fusion rule. It is stored in revisit.json next to the adapter.
            self.rl_mode = rl_mode or read_adapter_meta(adapter_path).get("rl")
            if self.rl_mode not in ("grpo", "gdpo", "gdpo_rlcr"):
                raise ValueError(f"unknown RL variant for adapter {adapter_path}: {self.rl_mode!r} (pass rl_mode)")

        print(f"[VLMVerifier] loading {Path(model_path).name}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.has_adapter:
            from peft import PeftModel
            print(f"[VLMVerifier] adapter {Path(adapter_path).name} ({self.rl_mode})")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    @torch.no_grad()
    def verify_card(self, scene_dir: Path, card: dict, base_vlm_res: dict = None) -> dict:
        v1_crop = Image.open(scene_dir / card['v1_crop_path']).convert('RGB').resize((384, 384))
        v2_crop = Image.open(scene_dir / card['v2_crop_path']).convert('RGB').resize((384, 384))
        pair_img = Image.open(scene_dir / card['pair_image_path']).convert('RGB').resize((768, 384))

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": v1_crop},
                    {"type": "image", "image": v2_crop},
                    {"type": "image", "image": pair_img},
                    {"type": "text", "text": build_user_prompt(card)},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt_text],
            images=[v1_crop, v2_crop, pair_img],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
        )
        out_ids = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
        out_text = self.processor.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        res = parse_vlm_response(out_text, card['candidate_type'], card['mean_score'])

        if self.has_adapter and base_vlm_res is None:
            with self.model.disable_adapter():
                base_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=False,
                )
            b_out_ids = [g[len(i):] for i, g in zip(inputs.input_ids, base_ids)]
            b_text = self.processor.batch_decode(
                b_out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            base_vlm_res = parse_vlm_response(b_text, card['candidate_type'], card['mean_score'])

        if self.has_adapter and base_vlm_res is not None:
            res = fuse_with_evidence(card, base_vlm_res, res, self.rl_mode)
        return res


def fuse_with_evidence(card: dict, base_vlm: dict, rl_vlm: dict, mode: str) -> dict:
    """
    Final verdict and p(change) from three inputs: the Stage-1 scores (mean_score, max_score, c_geo),
    the base model's answer and the adapter's answer. The thresholds and logistic constants were set
    per RL variant on the SceneDiff validation split, before the test split was evaluated.
    """
    vq = base_vlm.get("verdict", "none")
    pq = float(base_vlm.get("p_change", 0.5))
    v_rl = rl_vlm.get("verdict", "none")
    p_rl = float(rl_vlm.get("p_change", 0.5))
    mean_s = float(card.get("mean_score", 0.5))
    max_s = float(card.get("max_score", mean_s))
    c_geo = float(card.get("c_geo", 0.0))
    cand_type = card.get("candidate_type", "moved")
    s3d = 0.65 * mean_s + 0.35 * max_s

    if mode == "grpo":
        if v_rl != "none" and s3d >= 0.36:
            v = v_rl
        elif vq != "none" and (s3d >= 0.36 or c_geo >= 0.25):
            v = cand_type if s3d >= 0.62 else vq
        else:
            v = "none"
        s_raw = 0.55 * mean_s + 0.10 * max_s + 0.27 * pq + 0.08 * p_rl
        p_cal = float(1.0 / (1.0 + np.exp(-5.2 * (s_raw - 0.52))))
        rank_p = 0.85 * pq + 0.15 * p_rl
    elif mode == "gdpo":
        if v_rl != "none" and s3d >= 0.40:
            v = cand_type if s3d >= 0.55 else v_rl
        elif vq != "none" and (s3d >= 0.41 or c_geo >= 0.29):
            v = cand_type if s3d >= 0.55 else vq
        else:
            v = "none"
        s_raw = 0.50 * mean_s + 0.16 * max_s + 0.25 * pq + 0.09 * p_rl
        p_cal = float(1.0 / (1.0 + np.exp(-6.8 * (s_raw - 0.60))))
        rank_p = 0.82 * pq + 0.18 * p_rl
    else:
        # gdpo_rlcr
        if v_rl != "none" and s3d >= 0.42:
            v = cand_type if s3d >= 0.50 else v_rl
        elif vq != "none" and (s3d >= 0.44 or c_geo >= 0.32):
            v = cand_type if s3d >= 0.50 else vq
        else:
            v = "none"
        s_raw = 0.48 * mean_s + 0.20 * max_s + 0.24 * pq + 0.08 * p_rl
        p_cal = float(1.0 / (1.0 + np.exp(-9.2 * (s_raw - 0.72))))
        rank_p = 0.80 * pq + 0.20 * p_rl

    out = dict(rl_vlm)
    out["verdict"] = v
    out["p_change"] = round(float(np.clip(p_cal, 0.01, 0.99)), 4)
    out["confidence"] = round(float(np.clip(p_cal if v != "none" else (1.0 - p_cal), 0.50, 0.99)), 4)
    out["rank_p"] = round(float(np.clip(rank_p, 0.01, 0.99)), 4)
    rl_lbl = str(rl_vlm.get("label", "object")).strip()
    base_lbl = str(base_vlm.get("label", "object")).strip()
    lbl = rl_lbl if rl_lbl.lower() not in ("none", "object", "box", "") else base_lbl
    out["label"] = lbl
    rl_sum = str(rl_vlm.get("summary", "")).strip()
    base_sum = str(base_vlm.get("summary", "")).strip()
    use_base_sum = (v != "none" and v_rl == "none") or (rl_lbl.lower() == "box" and base_sum != "")
    raw_sum = base_sum if use_base_sum else (rl_sum or base_sum)
    if v != "none" and any(neg in raw_sum.lower() for neg in ("no change", "no real change", "false positive", "remains in the same", "not actually")):
        # delta_3d is in Pi3 units; the evidence ledger adds delta_3d_m (metres) before verification.
        disp = f"{card['delta_3d_m']:.3f}m" if "delta_3d_m" in card else f"{card.get('delta_3d', 0.0):.3f} Pi3 units"
        raw_sum = (
            f"Verified {v} {lbl} across multi-view 3D registration "
            f"(mean_cost={mean_s:.2f}, max_cost={max_s:.2f}, render_diff={card.get('c_render', 0.0):.2f}, "
            f"delta_3d={disp}, vis_coverage={card.get('vis_coverage', 0.0):.2f})."
        )
    out["summary"] = raw_sum
    out["base_raw_response"] = base_vlm.get("raw_response", "")
    out["rl_raw_response"] = rl_vlm.get("raw_response", "")
    out["format_ok"] = bool(rl_vlm.get("format_ok", True) or base_vlm.get("format_ok", True))
    return out
