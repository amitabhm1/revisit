#!/usr/bin/env python3
"""Per-stage runtime of room runs: median [min-max] over repeats.

  python scripts/summarize_timing.py --verifier qwen outputs/runtime/r1 outputs/runtime/r2 outputs/runtime/r3

Each folder is an --output-dir of reproduce/room.sh (OUT=...), holding <verifier>/<comparison>/timing.json
and logs/wall.tsv.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PAIRS = {"Take_1_vs_Take_2": "no-change pair", "Take_1_vs_Take_3": "before/after",
         "Take_1_vs_Take_3_relook_Take_4": "with re-look"}
ROWS = [
    ("stage0_keyframes_s", "Stage 0: keyframes + LiDAR"),
    ("stage12_pi3_sam_dinov3_s", "Stage 1-2: Pi3 + SAM + DINOv3"),
    ("sd:Depth and pose estimation", "  Pi3 depth and poses"),
    ("sd:SAM masks", "  SAM masks"),
    ("sd:Cost map computation", "  cost maps"),
    ("sd:Voxelization and features", "  voxelization and features"),
    ("sd:rest", "  other SceneDiff stages"),
    ("stage1b_metric_scale_s+stage3_ledger_s", "Stage 1B + 3: metric scale, ledger"),
    ("stage4_vlm_load_s", "Stage 4: load VLM + adapter"),
    ("stage4_vlm_verify_s", "Stage 4: verify proposals"),
    ("per_card", "  per proposal"),
    ("stage5_report_export_s", "Stage 5: report + Rerun"),
    ("total_in_process_s", "Total in process"),
    ("wall_s", "Wall clock, process start to exit"),
    ("peak_host_rss_gb", "Peak host RAM (GB)"),
    ("peak_gpu_mem_gb", "Peak GPU memory, PyTorch (GB)"),
]


def collect(run_dirs, verifier):
    data = defaultdict(lambda: defaultdict(list))
    meta = {}
    for run in map(Path, run_dirs):
        walls = {}
        wall_file = run / "logs" / "wall.tsv"
        if wall_file.exists():
            for line in open(wall_file):
                v, pair, secs = line.strip().split("\t")[:3]
                walls[(v, pair)] = float(secs)
        for pair in PAIRS:
            tj = run / verifier / pair / "timing.json"
            if not tj.exists():
                continue
            t = json.load(open(tj))
            meta = {"gpu": t.get("gpu"), "torch": t.get("torch"), "sam_backend": t.get("sam_backend"),
                    "n_frames": t.get("n_frames")}
            d = data[pair]
            for k in ("stage0_keyframes_s", "stage12_pi3_sam_dinov3_s", "stage4_vlm_load_s", "stage4_vlm_verify_s",
                      "stage5_report_export_s", "total_in_process_s", "peak_host_rss_gb", "peak_gpu_mem_gb"):
                if k in t:
                    d[k].append(float(t[k]))
            d["stage1b_metric_scale_s+stage3_ledger_s"].append(t.get("stage1b_metric_scale_s", 0) + t.get("stage3_ledger_s", 0))
            if t.get("vlm_cards"):
                d["per_card"].append(t["stage4_vlm_verify_s"] / t["vlm_cards"])
                d["cards"] = [t["vlm_cards"]]
            sd = t.get("scenediff_stages_s", {})
            for name, secs in sd.items():
                d[f"sd:{name}"].append(secs)
            if sd:
                main = ("Depth and pose estimation", "SAM masks", "Cost map computation", "Voxelization and features")
                d["sd:rest"].append(t["stage12_pi3_sam_dinov3_s"] - sum(sd.get(k, 0) for k in main))
            if (verifier, pair) in walls:
                d["wall_s"].append(walls[(verifier, pair)])
    return data, meta


def fmt(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{statistics.median(vals):.1f} [{min(vals):.1f}-{max(vals):.1f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--verifier", default="qwen")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    data, meta = collect(args.runs, args.verifier)
    print(f"{args.verifier}: {meta.get('gpu')}, torch {meta.get('torch')}, {len(args.runs)} repeat(s)\n")
    print("| Seconds | " + " | ".join(PAIRS.values()) + " |")
    print("|:--|" + "--:|" * len(PAIRS))
    print("| Proposals verified | " + " | ".join(str(data[p].get("cards", ["-"])[0]) for p in PAIRS) + " |")
    for key, label in ROWS:
        print(f"| {label} | " + " | ".join(fmt(data[p].get(key, [])) for p in PAIRS) + " |")
    if args.out:
        summary = {p: {k: (statistics.median(v) if isinstance(v, list) and v else None) for k, v in d.items()}
                   for p, d in data.items()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"meta": meta, "median": summary,
                                        "all": {p: dict(d) for p, d in data.items()}}, indent=2))


if __name__ == "__main__":
    main()
