"""Mechanical scoreboard for the room benchmark (docs/room_benchmark.md). No manual judgment.

- Take_1 vs Take_2 (nothing changed): every active verdict is a false change.
- Take_1 vs Take_3, and the same with the Take_4 re-look: a change counts as found when an active verdict's
  label matches its keywords (a proxy for recall, since labels come from the VLM). Active verdicts on
  categories that never changed (doors, paintings, sofa, rug, tables, lamps, walls, floor) are counted
  as static false flags.
"""
import json
import re
from pathlib import Path

PAIRS = {
    "null": "Take_1_vs_Take_2",
    "before_after": "Take_1_vs_Take_3",
    "with_relook": "Take_1_vs_Take_3_relook_Take_4",
}
ACTIVE = {"moved", "added", "removed", "replaced", "swapped"}
STATIC = re.compile(r"door|painting|poster|frame|art\b|picture|plaque|sign|placard|sofa|couch|coffee table|\brug\b|carpet|light|lamp|ceiling|wall|\btable\b|floor", re.I)
CHAIR = re.compile(r"chair|stool|seat", re.I)
BOOK = re.compile(r"book", re.I)
FLAT = re.compile(r"book|magazine|\bmat\b|paper|notebook|folder|cloth|towel|card|tablet", re.I)
CUP = re.compile(r"\bcup\b|\bmug\b", re.I)
PLANT = re.compile(r"plant|\bpot\b|vase", re.I)
PILLOW = re.compile(r"pillow|cushion", re.I)
HAT = re.compile(r"\bhat\b|\bcap\b", re.I)


def load_rows(run_dir, scene):
    p = Path(run_dir) / scene / "change_report.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    v = d.get("vlm_verdicts", {})
    return [{"verdict": v.get(str(c["obj_id"]), {}).get("verdict", "none"),
             "label": str(v.get(str(c["obj_id"]), {}).get("label", ""))} for c in d.get("candidates", [])]


def gt_hits(rows, relook):
    act = [r for r in rows if r["verdict"] in ACTIVE]
    moved_chairs = sum(1 for r in act if r["verdict"] == "moved" and CHAIR.search(r["label"]))
    hits = {
        "1+4 moved chairs": min(moved_chairs, 2),
        "2 added books": int(any(r["verdict"] == "added" and BOOK.search(r["label"]) for r in act)),
        "3 removed chair": int(any(r["verdict"] == "removed" and CHAIR.search(r["label"]) for r in act)),
        "5 removed flat item": int(any(r["verdict"] == "removed" and FLAT.search(r["label"]) for r in act)),
        "6 cup": int(any(CUP.search(r["label"]) for r in act)),
        "7 plant": int(any(r["verdict"] == "removed" and PLANT.search(r["label"]) for r in act)) if relook
        else int(any(r["verdict"] == "unverified" for r in rows)),
        "8 pillow": int(any(PILLOW.search(r["label"]) for r in act)),
        "9 hat": int(any(HAT.search(r["label"]) for r in act)),
    }
    return hits, sum(1 for r in act if STATIC.search(r["label"]))


def score_run(run_dir):
    null, ab, abr = (load_rows(run_dir, PAIRS[k]) for k in ("null", "before_after", "with_relook"))
    if not (null and ab and abr):
        return None
    h_ab, s_ab = gt_hits(ab, relook=False)
    h_abr, s_abr = gt_hits(abr, relook=True)
    return {
        "null_pair": {"proposals": len(null), "false_changes": sum(1 for r in null if r["verdict"] in ACTIVE)},
        "before_after": {"found": sum(h_ab.values()), "static_false_flags": s_ab, "per_change": h_ab},
        "with_relook": {"found": sum(h_abr.values()), "static_false_flags": s_abr, "per_change": h_abr},
        "hidden_plant": f"{'UNVERIFIED' if h_ab['7 plant'] else '-'} -> {'REMOVED' if h_abr['7 plant'] else '-'}",
    }


def markdown(scores):
    lines = ["| Run | No-change pair: false changes / proposals | Before/after: found (of 9) / static false flags "
             "| With re-look: found / static false flags | Change 7, keyword rule |", "|:--|--:|--:|--:|:--|"]
    for name, s in scores.items():
        if s is None:
            lines.append(f"| {name} | missing reports | | | |")
            continue
        lines.append(f"| {name} | {s['null_pair']['false_changes']} / {s['null_pair']['proposals']} | "
                     f"{s['before_after']['found']} / {s['before_after']['static_false_flags']} | "
                     f"{s['with_relook']['found']} / {s['with_relook']['static_false_flags']} | {s['hidden_plant']} |")
    return "\n".join(lines) + "\n"
