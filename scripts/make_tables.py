#!/usr/bin/env python3
"""Render the result tables from JSON, and compare fresh outputs with the committed references.

  python scripts/make_tables.py                     # all tables from results/
  python scripts/make_tables.py scenediff --outputs outputs/scenediff --compare results/scenediff.json
"""
import argparse
import json
import sys
from pathlib import Path

ROWS = [("none", "Proposals only (Pi3 + DINOv3 + SAM 1)"), ("qwen-zeroshot", "Qwen3-VL-8B, zero-shot"),
        ("sensenova-zeroshot", "SenseNova-SI-8B, zero-shot"), ("qwen-grpo", "Qwen3-VL-8B + GRPO"),
        ("qwen-gdpo", "Qwen3-VL-8B + GDPO"), ("qwen-gdpo-rlcr", "Qwen3-VL-8B + GDPO+RLCR")]
KEYS = ["px_im_iou", "obj_im_ap", "obj_sc_ap", "obj_sc_ap_class", "tp", "fp", "fn", "auroc", "brier", "ece", "acc_4way"]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def num(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def scenediff_table(rows, split):
    if not rows:
        sys.exit(f"no metrics_{split}.json found for any row")
    scenes = int(next(r["scenes"] for r in rows.values()))
    out = [f"SceneDiff {split} split ({scenes} scenes)\n",
           "| Method | px/im IoU | obj/im AP | obj/sc AP | AUROC | 4-way acc | Brier | ECE |",
           "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for key, label in ROWS:
        r = rows.get(key)
        if r is None:
            out.append(f"| {label} | missing | | | | | | |")
            continue
        out.append(f"| {label} | {pct(r['px_im_iou'])} | {pct(r['obj_im_ap'])} | {pct(r['obj_sc_ap'])} | "
                   f"{num(r['auroc'])} | {pct(r['acc_4way'])} | {num(r['brier'])} | {num(r['ece'])} |")
    return "\n".join(out)


def load_outputs(outputs: Path, split: str):
    rows = {}
    for key, _ in ROWS:
        f = outputs / key / f"metrics_{split}.json"
        if f.exists():
            rows[key] = json.load(open(f))
    return rows


def compare_rows(new, ref):
    bad = []
    for key, _ in ROWS:
        if key not in new:
            continue
        for k in KEYS:
            a, b = new[key].get(k), ref.get(key, {}).get(k)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 5e-4:
                bad.append(f"{key}.{k}: {a} vs reference {b}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", nargs="?", default="all", choices=["all", "scenediff"])
    p.add_argument("--outputs", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--compare", type=Path)
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()

    if args.table == "scenediff" and args.outputs:
        rows = load_outputs(args.outputs, args.split)
        print(scenediff_table(rows, args.split))
        if args.compare:
            bad = compare_rows(rows, json.load(open(args.compare))[args.split])
            print("\nmatches the reference" if not bad else "\nDIFFERS from the reference:\n  " + "\n  ".join(bad))
            sys.exit(1 if bad else 0)
        return

    res = args.results
    sd = json.load(open(res / "scenediff.json"))
    for split in ("test", "val"):
        if split in sd:
            print(scenediff_table(sd[split], split) + "\n")
    board = json.load(open(res / "room" / "scoreboard.json"))
    print("Room benchmark (final configuration)\n")
    print("| Verifier | No-change pair: false changes / proposals | Before/after: found / static false flags | "
          "With re-look: found / static false flags | Change 7, keyword rule |")
    print("|:--|--:|--:|--:|:--|")
    for name, s in board.items():
        print(f"| {name} | {s['null_pair']['false_changes']} / {s['null_pair']['proposals']} | "
              f"{s['before_after']['found']} / {s['before_after']['static_false_flags']} | "
              f"{s['with_relook']['found']} / {s['with_relook']['static_false_flags']} | {s['hidden_plant']} |")


if __name__ == "__main__":
    main()
