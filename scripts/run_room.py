#!/usr/bin/env python3
"""Compare two Stray Scanner captures of a room, optionally with a short re-look capture.

  python scripts/run_room.py --before data/room/Take_1 --after data/room/Take_3 --relook data/room/Take_4

Writes <output-dir>/<before>_vs_<after>[_relook_<relook>]/ with change_report.md, change_report.json,
scene_diff.rrd (open with `rerun`) and timing.json.
"""
import argparse
from pathlib import Path

from revisit import paths
from revisit.pipeline import compare


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", type=Path, help="Stray Scanner folder of the first scan")
    p.add_argument("--after", type=Path, help="Stray Scanner folder of the second scan")
    p.add_argument("--relook", type=Path, help="optional extra scan of places the second scan missed")
    p.add_argument("--scene-dir", type=Path, help="instead of scans: a folder with two plain videos (no metric scale)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/room"))
    p.add_argument("--config", type=Path, default=paths.CONFIG_DIR / "room.yaml")
    p.add_argument("--verifier", default="qwen", help="qwen | sensenova | qwen-zeroshot | sensenova-zeroshot | none")
    p.add_argument("--vlm-model", type=Path, help="override: base VLM folder")
    p.add_argument("--vlm-adapter", type=Path, help="override: LoRA adapter folder")
    p.add_argument("--rl-mode", choices=["grpo", "gdpo", "gdpo_rlcr"], help="override: RL variant of --vlm-adapter")
    p.add_argument("--sam-backend", choices=["sam1", "sam2.1", "sam3.1"], help="default: value in --config (sam2.1)")
    p.add_argument("--keyframes", type=int, help="keyframes per scan (default: value in --config, 28)")
    p.add_argument("--no-metric-scale", action="store_true", help="skip LiDAR/ARKit scale (distances in Pi3 units)")
    args = p.parse_args()

    if not ((args.before and args.after) or args.scene_dir):
        p.error("pass --before and --after, or --scene-dir")
    model, adapter, rl_mode = None, None, None
    if args.verifier != "none":
        v = paths.verifier(args.verifier)
        model, adapter, rl_mode = v["model"], v["adapter"], v["rl_mode"]
    model = args.vlm_model or model
    if args.vlm_adapter:
        adapter, rl_mode = args.vlm_adapter, args.rl_mode
    if adapter is not None and not Path(adapter).exists():
        raise SystemExit(f"adapter not found: {adapter}\nrun: python scripts/download_checkpoints.py --set room")

    compare(
        output_dir=args.output_dir,
        config_path=args.config,
        scan1=args.before,
        scan2=args.after,
        relook=args.relook,
        scene_dir=args.scene_dir,
        keyframes=args.keyframes,
        sam_backend=args.sam_backend,
        vlm_model=model,
        vlm_adapter=adapter,
        rl_mode=rl_mode,
        lidar_mode="off" if args.no_metric_scale else "auto",
    )


if __name__ == "__main__":
    main()
