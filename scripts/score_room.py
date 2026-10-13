#!/usr/bin/env python3
"""Score room runs against the staged changes (docs/room_benchmark.md).

  python scripts/score_room.py qwen=outputs/room/qwen sensenova=outputs/room/sensenova

Each run folder must hold the three comparisons written by reproduce/room.sh.
--compare checks the scores against a reference scoreboard. --reference-dir also checks every
per-proposal verdict against reference reports (results/room/<name>/<comparison>/change_report.json).
The exit code is 1 if anything differs.
"""
import argparse
import json
import sys
from pathlib import Path

from revisit.eval.room_score import PAIRS, markdown, score_run


def verdicts(report):
    d = json.load(open(report))
    return sorted((k, v.get("verdict")) for k, v in d.get("vlm_verdicts", {}).items())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="name=run_dir")
    p.add_argument("--out", type=Path, help="write scores as JSON")
    p.add_argument("--compare", type=Path, help="reference scoreboard JSON")
    p.add_argument("--reference-dir", type=Path, help="folder with reference change reports")
    args = p.parse_args()

    scores, dirs = {}, {}
    for item in args.runs:
        name, run_dir = item.split("=", 1)
        scores[name], dirs[name] = score_run(run_dir), Path(run_dir)
    print(markdown(scores))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scores, indent=2))

    failed = False
    if args.compare:
        ref = json.load(open(args.compare))
        for name in scores:
            if name in ref:
                same = scores[name] == ref[name]
                failed |= not same
                print(f"scores    {name:28s} {'match' if same else 'DIFFER'}")
    if args.reference_dir:
        for name, run_dir in dirs.items():
            for pair in PAIRS.values():
                ref_report = args.reference_dir / name / pair / "change_report.json"
                new_report = run_dir / pair / "change_report.json"
                if ref_report.exists() and new_report.exists():
                    same = verdicts(ref_report) == verdicts(new_report)
                    failed |= not same
                    print(f"verdicts  {name:12s} {pair:32s} {'identical' if same else 'DIFFER'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
