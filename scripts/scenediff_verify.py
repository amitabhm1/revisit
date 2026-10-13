#!/usr/bin/env python3
"""Run one verifier over SceneDiff Stage-1 proposals and score it with the official metrics.

  python scripts/scenediff_verify.py --verifier qwen-gdpo-rlcr --split test
  python scripts/scenediff_verify.py --verifier none --split test        # proposals only

Several GPUs: start one process per GPU with --shard i --num-shards n, then run once with --score-only.
Verifiers with an adapter also need the base model's answers. --base-outputs points at a finished
zero-shot run of the same base model and reuses its answers instead of generating them again
(this is how the reported rows were produced).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from revisit import paths

METRIC_PATTERNS = {
    "scenes": r"[Ss]cenes: (\d+)",
    "px_im_iou": r"px/im IoU \(merged masks, all views\): ([0-9.]+)",
    "obj_im_ap": r"obj/im AP \(IoU-based matching\): ([0-9.]+)",
    "obj_sc_ap": r"obj/sc AP \(without class requirement\): ([0-9.]+)",
    "obj_sc_ap_class": r"obj/sc AP \(with class requirement\): ([0-9.]+)",
    "tp": r"True Positives: (\d+)",
    "fp": r"False Positives: (\d+)",
    "fn": r"False Negatives: (\d+)",
}


def split_scenes(gt_dir: Path, split: str):
    splits = gt_dir.parent / "splits"
    val = json.load(open(splits / "val_split.json"))
    test = json.load(open(splits / "test_split.json"))
    val_all, test_all = val["varied"] + val["kitchen"], test["varied"] + test["kitchen"]
    return val_all, test_all, (val_all if split == "val" else test_all)


def load_base_answers(base_dir: Path, sname: str) -> dict:
    f = base_dir / sname / "verifier_results.json"
    if not f.exists():
        return {}
    return {int(r["obj_id"]): r["vlm"] for r in json.load(open(f)).get("records", [])}


def verify(args, preset, scenes, all_scenes):
    from revisit.eval.scenediff_verifier import verify_scene

    shard = [s for s in all_scenes[args.shard::args.num_shards] if s in set(scenes)]
    vlm = None
    has_adapter = preset["adapter"] is not None
    for sname in shard:
        s_pred, s_out = args.proposals / sname, args.output_dir / sname
        if not (s_pred / "object_masks.pkl").exists():
            continue
        if (s_out / "verifier_results.json").exists() and (s_out / "object_masks.pkl").exists():
            recs = json.load(open(s_out / "verifier_results.json")).get("records", [])
            if not has_adapter or (recs and "rank_p" in recs[0].get("vlm", {})):
                continue
        if preset["model"] is not None and vlm is None:
            from revisit.verifier import VLMVerifier

            adapter = str(preset["adapter"]) if has_adapter else None
            vlm = VLMVerifier(str(preset["model"]), adapter_path=adapter, rl_mode=preset["rl_mode"])
        base = load_base_answers(args.base_outputs, sname) if (has_adapter and args.base_outputs) else {}
        verify_scene(sname, s_pred, s_out, args.gt_dir, vlm, rl_mode=preset["rl_mode"] if has_adapter else None,
                     base_vlm_map=base)
        print(f"[verify] {sname}: done")


def score(args, preset, scenes):
    from revisit.eval.scenediff_verifier import verifier_metrics

    records = []
    for s in scenes:
        f = args.output_dir / s / "verifier_results.json"
        if f.exists():
            records.extend(json.load(open(f)).get("records", []))
    vm = verifier_metrics(records)
    (args.output_dir / f"verifier_metrics_{args.split}.json").write_text(json.dumps(vm, indent=2))

    out_txt = args.output_dir / f"result_{args.split}.txt"
    subprocess.run([sys.executable, "-m", "revisit.eval.scenediff_metrics", "--pred_dir", str(args.output_dir),
                    "--gt_dir", str(args.gt_dir), "--output_path", str(out_txt), "--splits", args.split, "--sets", "all"],
                   check=True)
    txt = (args.output_dir / f"result_{args.split}_frame_dup_1_obj_dup_1.txt").read_text()
    row = {"verifier": preset["name"], "split": args.split}
    for k, pat in METRIC_PATTERNS.items():
        m = re.search(pat, txt)
        row[k] = float(m.group(1)) if m else None
    key = "vlm" if preset["model"] is not None else "stage1"
    row.update({
        "candidates": vm.get("num_candidates"),
        "auroc": vm.get(f"auroc_{key}"),
        "brier": vm.get(f"brier_{key}"),
        "ece": vm.get(f"ece_{key}"),
        "acc_4way": vm.get("verdict_acc_4way") if key == "vlm" else None,
        "macro_f1_4way": vm.get("verdict_macro_f1_4way") if key == "vlm" else None,
    })
    (args.output_dir / f"metrics_{args.split}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verifier", required=True, help="preset from configs/checkpoints.yaml, 'none', or 'custom'")
    p.add_argument("--model", type=Path, help="custom: base VLM folder")
    p.add_argument("--adapter", type=Path, help="custom: LoRA adapter folder")
    p.add_argument("--rl-mode", choices=["grpo", "gdpo", "gdpo_rlcr"], help="custom: RL variant of the adapter")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--proposals", type=Path, default=paths.DATA_DIR / "artifacts/scenediff/proposals")
    p.add_argument("--gt-dir", type=Path, default=paths.DATA_DIR / "scenediff/scenediff_benchmark/data")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--base-outputs", type=Path, help="zero-shot run of the same base model to reuse answers from")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    if args.verifier == "none":
        preset = {"name": "none", "model": None, "adapter": None, "rl_mode": None}
    elif args.verifier == "custom":
        preset = {"name": "custom", "model": args.model, "adapter": args.adapter, "rl_mode": args.rl_mode}
    else:
        preset = paths.verifier(args.verifier)
    args.output_dir = args.output_dir or Path("outputs/scenediff") / args.verifier
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_all, test_all, scenes = split_scenes(args.gt_dir, args.split)

    if not args.score_only:
        verify(args, preset, scenes, val_all + test_all)
    if args.score_only or args.num_shards == 1:
        score(args, preset, scenes)


if __name__ == "__main__":
    main()
