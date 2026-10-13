#!/usr/bin/env python3
"""Metric scale check for room runs: ARKit trajectory scale per take vs the LiDAR-median estimate.

  python scripts/check_metric_scale.py outputs/room/qwen --out results/metric_scale.json

Needs a finished run folder (scene_candidates.pkl from Stage 1-2) and the captures in data/room.
CPU only, under a minute.
"""
import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from revisit import paths
from revisit.capture import load_arkit_positions, load_stray_scanner_frames

PAIRS = {
    "Take_1_vs_Take_2": ["Take_1", "Take_2"],
    "Take_1_vs_Take_3": ["Take_1", "Take_3"],
    "Take_1_vs_Take_3_relook_Take_4": ["Take_1", "Take_3", "Take_4"],
}


def check(run_dir: Path, captures: Path, scene: str, takes: list) -> dict:
    cand = pickle.load(open(run_dir / scene / "scene_candidates.pkl", "rb"))
    meta = json.load(open(run_dir / scene / "stage12_meta.json"))
    kf = int(meta["keyframes"])
    centers = np.asarray(cand["poses"])[:, :3, 3]
    tmp = Path(tempfile.mkdtemp())
    seq, depths = [], []
    for i, t in enumerate(takes):
        frames, dm = load_stray_scanner_frames(captures / t, tmp / f"{t}_{i}", target_keyframes=kf if i < 2 else max(12, kf // 2))
        seq += [(t, int(Path(f).stem)) for f in frames]
        depths += dm
    if len(seq) != len(centers):
        raise SystemExit(f"{scene}: {len(seq)} keyframes but {len(centers)} Pi3 cameras")

    per_take, all_ratios = {}, []
    for t in dict.fromkeys(takes):
        pos = load_arkit_positions(captures / t)
        idx = [k for k, (tt, _) in enumerate(seq) if tt == t]
        a = np.stack([pos[seq[k][1]] for k in idx])
        c = centers[idx]
        iu = np.triu_indices(len(idx), 1)
        da = np.linalg.norm(a[:, None] - a[None], axis=-1)[iu]
        dc = np.linalg.norm(c[:, None] - c[None], axis=-1)[iu]
        ok = (da >= 0.3) & (dc > 1e-6)
        r = da[ok] / dc[ok]
        all_ratios.append(r)
        per_take[t] = {"scale": round(float(np.median(r)), 3),
                       "rel_spread": round(float(np.median(np.abs(r - np.median(r))) / np.median(r)), 3),
                       "pairs": int(ok.sum())}
    pooled = float(np.median(np.concatenate(all_ratios)))

    pts = np.concatenate([np.asarray(m["pc"], np.float32) for m in cand.get("merged_1", []) + cand.get("merged_2", [])
                          if len(m.get("pc", [])) > 0])
    z = np.min(np.linalg.norm(pts[:, None, :] - centers[None], axis=-1), axis=-1)
    z = z[(z > 0.05) & (z < 10.0)]
    lid = np.concatenate([d[d > 0.35] for d in depths if (d > 0.35).any()])
    lidar_raw = float(np.median(lid) / max(np.median(z), 1e-3))
    return {"arkit_per_take": per_take, "arkit_pooled": round(pooled, 3),
            "lidar_median_uncapped": round(lidar_raw, 2), "lidar_median_capped": round(float(np.clip(lidar_raw, 0.25, 12.0)), 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--captures", type=Path, default=paths.DATA_DIR / "room")
    p.add_argument("--out", type=Path)
    p.add_argument("--compare", type=Path, help="reference JSON; exit code 1 if any value differs")
    args = p.parse_args()
    res = {scene: check(args.run_dir, args.captures, scene, takes) for scene, takes in PAIRS.items()}
    print("| Comparison | ARKit scale per take (m / Pi3 unit) | ARKit pooled (used) | LiDAR median, uncapped | LiDAR median, capped |")
    print("|:--|:--|--:|--:|--:|")
    for scene, r in res.items():
        per = ", ".join(f"{t} {v['scale']:.3f}" for t, v in r["arkit_per_take"].items())
        print(f"| {scene} | {per} | {r['arkit_pooled']:.3f} | {r['lidar_median_uncapped']:.2f} | {r['lidar_median_capped']:.2f} |")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
    if args.compare:
        same = json.loads(json.dumps(res)) == json.load(open(args.compare))
        print("matches the reference" if same else "DIFFERS from the reference")
        raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
