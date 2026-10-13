#!/usr/bin/env python3
"""Stage 1-2 on SceneDiff: Pi3 + SAM 1 + DINOv3 proposals for every scene of a split.

  python scripts/scenediff_proposals.py --split test

One process per GPU:
  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python scripts/scenediff_proposals.py --split all --shard $i --num-shards 8 &
  done; wait

Scenes that already have object_masks.pkl in the output folder are skipped, so the command resumes.
"""
from revisit.runtime import limit_threads

limit_threads()  # before numpy loads, matching the reported runs

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from revisit import paths  # noqa: E402
from revisit.pipeline import load_config, prepare_scene_data  # noqa: E402
from revisit.runtime import seed_everything  # noqa: E402


def get_scenes_to_process(config, args):
    val_split_path = Path(config["dataset"]["splits"]["val"])
    test_split_path = Path(config["dataset"]["splits"]["test"])
    gt_dir_parent = Path(config["dataset"]["gt_dir"]).parent
    if not val_split_path.exists():
        val_split_path = gt_dir_parent / "splits" / "val_split.json"
    if not test_split_path.exists():
        test_split_path = gt_dir_parent / "splits" / "test_split.json"
    val_split = json.load(open(val_split_path, "r"))
    val_split["all"] = val_split.get("varied", []) + val_split.get("kitchen", [])
    test_split = json.load(open(test_split_path, "r"))
    test_split["all"] = test_split.get("varied", []) + test_split.get("kitchen", [])
    if args.split == "val":
        valid_scenes = val_split["all"]
    elif args.split == "test":
        valid_scenes = test_split["all"]
    else:
        valid_scenes = val_split["all"] + test_split["all"]
    if args.num_shards > 1:
        valid_scenes = valid_scenes[args.shard::args.num_shards]
    return valid_scenes


def filter_already_processed_scenes(valid_scenes, output_dir):
    finished_scenes = []
    for scene_name in list(valid_scenes):
        for output_subdir in Path(output_dir).glob(f"{scene_name}*"):
            if output_subdir.is_dir() and (output_subdir / "object_masks.pkl").exists():
                print(f"{output_subdir} already processed")
                valid_scenes.remove(scene_name)
                finished_scenes.append(scene_name)
                break
    return valid_scenes, finished_scenes


def process_single_scene(scene_name, scene_dir, scene_diff_model, config):
    print(f"\n{'=' * 80}\nProcessing scene: {scene_name}\n{'=' * 80}\n")
    result = prepare_scene_data(scene_dir, config["dataset"]["resample_rate"])
    if result is None:
        print(f"Skipping {scene_name}: no frames found")
        return False
    img_1_list, img_2_list = result
    print(f"  Video 1: {len(img_1_list)} frames\n  Video 2: {len(img_2_list)} frames")
    try:
        scene_diff_model.process_scene(img_1_list + img_2_list, len(img_1_list), len(img_2_list),
                                       Path(config["output"]["output_dir"]))
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"Error processing {scene_name}: {e}")
        import traceback

        traceback.print_exc()
        torch.cuda.empty_cache()
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(paths.CONFIG_DIR / "scenediff.yaml"))
    p.add_argument("--split", choices=["val", "test", "all"], default="all")
    p.add_argument("--output-dir", default="outputs/scenediff/proposals")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--scenes", help="comma-separated scene names (default: every scene of --split)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    config = load_config(args.config)
    config["output"]["output_dir"] = args.output_dir
    valid_scenes = args.scenes.split(",") if args.scenes else get_scenes_to_process(config, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    todo, done = filter_already_processed_scenes(valid_scenes.copy(), output_dir)
    print(f"Shard {args.shard}/{args.num_shards}: {len(todo)} to process, {len(done)} already done")
    if not todo:
        return

    from revisit.proposals.scenediff import SceneDiff

    model = SceneDiff(config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    gt_dir = Path(config["dataset"]["gt_dir"])
    ok = fail = 0
    for scene_name in todo:
        scene_dir = gt_dir / scene_name
        if not scene_dir.exists():
            print(f"Scene directory not found: {scene_dir}")
            continue
        if process_single_scene(scene_name, scene_dir, model, config):
            ok += 1
        else:
            fail += 1
    model.cleanup()
    print(f"Shard {args.shard}/{args.num_shards}: {ok} processed, {fail} failed, {len(done)} skipped")


if __name__ == "__main__":
    main()
