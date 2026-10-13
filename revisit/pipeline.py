"""End-to-end comparison of two Stray Scanner captures, with an optional re-look capture.

Stages: 0 keyframes, 1-2 Pi3 + SAM + DINOv3 proposals, 1B metric scale, 3 evidence ledger,
4 VLM verifier, 5 report. Writes change_report.{md,json}, scene_diff.rrd and timing.json.
"""
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np  # noqa: F401  (imported before the thread limits, matching the reported runs)
import yaml

from .capture import load_stray_scanner_frames
from .ledger import apply_evidence_ledger
from .metric_scale import arkit_trajectory_scale, lidar_median_scale
from .paths import model_dir, relative_to_roots
from .report import write_report

SAM_CHECKPOINTS = {"sam1": "sam1", "sam2.1": "sam2.1", "sam3.1": "sam3.1"}
SAM_LABELS = {"sam1": "SAM 1 ViT-H", "sam2.1": "SAM 2.1 Hiera-L", "sam3.1": "SAM 3.1"}


def load_config(config_path):
    """Load a YAML config and expand ${VARS} in the models and dataset sections (SceneDiff convention)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for section in ["models", "dataset"]:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, str):
                    config[section][key] = os.path.expandvars(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            config[section][key][subkey] = os.path.expandvars(subvalue)
    return config


def prepare_scene_data(scene_dir, resample_rate):
    """SceneDiff-style folder with two videos: extract frames once, keep every `resample_rate`-th frame."""
    from .proposals.utils import process_video_to_frames

    video_1_dir = scene_dir / "video1_frames"
    video_2_dir = scene_dir / "video2_frames"
    video_files = sorted([x for x in os.listdir(scene_dir) if x.lower().endswith((".mp4", ".mov", ".avi"))])
    if len(video_files) >= 2:
        if (not video_1_dir.exists() or not video_2_dir.exists()
                or len(list(video_1_dir.glob("*.jpg"))) == 0 or len(list(video_2_dir.glob("*.jpg"))) == 0):
            print("  Extracting frames from videos...")
            process_video_to_frames(str(scene_dir / video_files[0]), str(video_1_dir))
            process_video_to_frames(str(scene_dir / video_files[1]), str(video_2_dir))
    img_1_list = sorted(video_1_dir.glob("*.jpg"))[::resample_rate]
    img_2_list = sorted(video_2_dir.glob("*.jpg"))[::resample_rate]
    if len(img_1_list) == 0 or len(img_2_list) == 0:
        return None
    return img_1_list, img_2_list


def compare(
    output_dir,
    config_path,
    scan1=None,
    scan2=None,
    relook=None,
    scene_dir=None,
    keyframes=None,
    sam_backend=None,
    sam_checkpoint=None,
    vlm_model=None,
    vlm_adapter=None,
    rl_mode=None,
    lidar_mode="auto",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from .runtime import limit_threads, seed_everything

    limit_threads()
    from .proposals.scenediff import SceneDiff
    from .verifier import VLMVerifier

    seed_everything(42)
    timing = {}

    def _lap(key, t0):
        # Synchronize so GPU work launched in the stage is attributed to that stage.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[key] = round(time.perf_counter() - t0, 3)
        return time.perf_counter()

    t_main = time.perf_counter()
    config = load_config(config_path)
    room_cfg = config.get("room", {})
    keyframes = int(keyframes or room_cfg.get("keyframes", 28))
    sam_cfg = config["models"]["sam"]
    if sam_backend:
        sam_cfg["backend"] = sam_backend
        sam_cfg["checkpoint"] = str(sam_checkpoint or model_dir(SAM_CHECKPOINTS[sam_backend]))
    elif sam_checkpoint:
        sam_cfg["checkpoint"] = str(sam_checkpoint)
    sam_backend = str(sam_cfg.get("backend", "sam1")).lower()
    sam_checkpoint = os.path.expandvars(str(sam_cfg["checkpoint"]))
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint for backend '{sam_backend}' not found: {sam_checkpoint}")
    sam_label = SAM_LABELS.get(sam_backend, sam_backend)
    print(f"[Stage 2] mask backend {sam_backend} ({sam_label}), checkpoint {relative_to_roots(sam_checkpoint)}")
    m1, m2 = [], []
    scale_segments = []
    has_relook = False

    t = time.perf_counter()
    if scan1 and scan2:
        s1, s2 = Path(scan1), Path(scan2)
        sr = Path(relook) if relook else None
        has_relook = sr is not None and sr.exists()
        scene_name = f"{s1.name}_vs_{s2.name}" + (f"_relook_{sr.name}" if has_relook else "")
        # SceneDiff names the scene after Path(file_list[0]).parents[1], so stage frames under the scene name.
        work_scene_dir = output_dir / "_staged_inputs" / scene_name
        f1, m1 = load_stray_scanner_frames(s1, work_scene_dir / "video1_frames", target_keyframes=keyframes)
        f2, m2 = load_stray_scanner_frames(s2, work_scene_dir / "video2_frames", target_keyframes=keyframes)
        scale_segments = [(s1, list(f1)), (s2, list(f2))]  # Pi3 file order, one entry per take
        if has_relook:
            f_r, m_r = load_stray_scanner_frames(sr, work_scene_dir / "relook_frames", target_keyframes=max(12, keyframes // 2))
            scale_segments.append((sr, list(f_r)))
            f2 = f2 + f_r
            m2 = m2 + m_r
        img_1_list = [str(p) for p in f1]
        img_2_list = [str(p) for p in f2]
        input_ids = [s1.name, s2.name, sr.name if has_relook else None]
    elif scene_dir:
        scene_dir = Path(scene_dir)
        scene_name = scene_dir.name
        work_scene_dir = scene_dir
        res = prepare_scene_data(scene_dir, config["dataset"]["resample_rate"])
        if res is None:
            raise RuntimeError(f"No frames found in {scene_dir}")
        img_1_list, img_2_list = res
        input_ids = [scene_dir.name]
    else:
        raise ValueError("Provide scan1 and scan2 (Stray Scanner folders) or scene_dir")
    t = _lap("stage0_keyframes_s", t)

    file_list = img_1_list + img_2_list
    scene_out_dir = output_dir / scene_name
    cand_pkl = scene_out_dir / "scene_candidates.pkl"
    cards_file = scene_out_dir / "verifier_cards.json"
    meta_file = scene_out_dir / "stage12_meta.json"
    stage12_meta = {
        "pi3": os.path.basename(str(config["models"]["pi3"]["name"]).rstrip("/")),
        "sam_backend": sam_backend,
        "sam_checkpoint": os.path.basename(sam_checkpoint),
        "dinov3": str(config["models"]["dinov3"].get("model_name")),
        "keyframes": int(keyframes),
        "inputs": input_ids,
        "n_frames": [len(img_1_list), len(img_2_list)],
    }
    cached_meta = json.load(open(meta_file)) if meta_file.exists() else None

    if cand_pkl.exists() and cards_file.exists() and cached_meta == stage12_meta:
        print(f"[Stage 1-2] reusing cached Pi3 + {sam_label} + DINOv3 outputs in {scene_out_dir}")
        timing["stage12_cached"] = True
    else:
        if cand_pkl.exists() and cached_meta != stage12_meta:
            print(f"[Stage 1-2] cached outputs in {scene_out_dir} were made with other settings; recomputing")
            # SAM masks are cached next to the staged frames; drop them so a new setting never reuses them.
            for stale in Path(work_scene_dir).glob("*_masks.pkl"):
                stale.unlink()
            shutil.rmtree(scene_out_dir / "verifier_crops", ignore_errors=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scenediff = SceneDiff(config, device)
        scenediff.process_scene(file_list, len(img_1_list), len(img_2_list), output_dir)
        timing["scenediff_stages_s"] = dict(scenediff.stage_times)
        scenediff.cleanup()
        with open(meta_file, "w") as f:
            json.dump(stage12_meta, f, indent=2)
        timing["stage12_cached"] = False
    t = _lap("stage12_pi3_sam_dinov3_s", t)

    scale_factor, lidar_info = lidar_median_scale(cand_pkl, m1, m2, lidar_mode=lidar_mode)
    lidar_info["scale_source"] = "lidar_median_fallback" if lidar_info.get("status") == "applied_1dof_metric_scale" else "none"
    if lidar_mode != "off" and scale_segments:
        s_ark, ark_info = arkit_trajectory_scale(cand_pkl, scale_segments)
        if s_ark is not None:
            # The LiDAR-median estimate is logged as a cross-check only; it reads ~1.8x high on these captures.
            ark_info["lidar_median_estimate_capped"] = lidar_info.get("scale_factor")
            ark_info["lidar_valid_ratio"] = lidar_info.get("valid_ratio")
            scale_factor, lidar_info = s_ark, {"lidar_mode": lidar_mode, **ark_info}
    if has_relook and relook:
        lidar_info["relook_scan"] = Path(relook).name
    print(f"[Stage 1B] {scene_name}: {lidar_info}")
    t = _lap("stage1b_metric_scale_s", t)

    cards = json.load(open(cards_file)).get("candidates", []) if cards_file.exists() else []

    cards = apply_evidence_ledger(cand_pkl, cards, scale_factor=scale_factor, has_relook=has_relook)
    t = _lap("stage3_ledger_s", t)

    vlm_records = {}
    if vlm_model and os.path.exists(vlm_model):
        vlm = VLMVerifier(str(vlm_model), adapter_path=str(vlm_adapter) if vlm_adapter else None, rl_mode=rl_mode)
        t = _lap("stage4_vlm_load_s", t)
        for c in cards:
            uid = int(c["obj_id"])
            vrec = vlm.verify_card(scene_out_dir, c)
            if c.get("candidate_type") == "unverified":
                vrec["verdict"] = "unverified"
                vrec["look_from_pose"] = c.get("look_from_pose")
            elif vrec.get("verdict") != "none" and c.get("candidate_type") in ("replaced", "swapped"):
                vrec["verdict"] = c["candidate_type"]
            vlm_records[uid] = vrec
        t = _lap("stage4_vlm_verify_s", t)
        timing["vlm_cards"] = len(cards)
    elif vlm_model:
        raise FileNotFoundError(f"VLM not found: {vlm_model} (run scripts/download_checkpoints.py)")

    vlm_name = os.path.basename(str(vlm_model).rstrip("/")) if vlm_model else None
    adapter_label = Path(vlm_adapter).name if vlm_adapter else None
    stack_info = f"`Pi3` + `{sam_label}` + `DINOv3-ViT-H+/16` + 3D evidence ledger" + (
        f" + `{vlm_name}`" + (f" (adapter `{adapter_label}`)" if adapter_label else " (no adapter)") if vlm_name else ""
    )

    with open(scene_out_dir / "change_report.json", "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "geometry_backbone": f"Pi3 ({stage12_meta['pi3']}) + {sam_label} + DINOv3 ({stage12_meta['dinov3']})",
                "sam_backend": sam_backend,
                "sam_checkpoint": relative_to_roots(sam_checkpoint),
                "vlm_model": vlm_name,
                "vlm_adapter": relative_to_roots(vlm_adapter) if vlm_adapter else None,
                "lidar_calibration": lidar_info,
                "candidates": cards,
                "vlm_verdicts": vlm_records,
            },
            f,
            indent=2,
        )

    write_report(
        scene_name=scene_name,
        out_dir=scene_out_dir,
        candidates_pkl=cand_pkl,
        cards=cards,
        vlm_records=vlm_records,
        rrd_path=scene_out_dir / "scene_diff.rrd",
        scale_factor=scale_factor,
        lidar_info=lidar_info,
        stack_info=stack_info,
    )
    t = _lap("stage5_report_export_s", t)
    timing["total_in_process_s"] = round(time.perf_counter() - t_main, 3)
    timing.update({
        "scene_name": scene_name,
        "n_frames": [len(img_1_list), len(img_2_list)],
        "sam_backend": sam_backend,
        "vlm_model": vlm_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "peak_host_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else 0.0,
    })
    with open(scene_out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"[Timing] {json.dumps(timing)}")
    return scene_out_dir
