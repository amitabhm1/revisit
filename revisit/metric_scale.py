"""Stage 1B: metres per Pi3 unit.

The scale comes from the phone's ARKit trajectory: within each take, distances between keyframe
camera centres in ARKit (metres) are divided by the same distances in the Pi3 reconstruction, and
the median over all pairs is used. The older LiDAR-median estimate is kept as a logged cross-check;
it compares wall-dominated depth with object distances and reads about 1.8x too high.
"""
import pickle
from pathlib import Path

import numpy as np

from .capture import load_arkit_positions


def lidar_median_scale(
    candidates_pkl: Path,
    depths_v1: list[np.ndarray],
    depths_v2: list[np.ndarray],
    lidar_mode: str = "auto",
) -> tuple[float, dict]:
    """
    Legacy estimate s_hat = median(z_lidar) / median(z_pi3) over confidence-2 LiDAR pixels in
    [0.35 m, 4.2 m], clipped to [0.25, 12.0]. Only logged; the ARKit estimate is used.
    """
    info = {"lidar_mode": lidar_mode, "scale_factor": 1.0, "status": "rgb_only_default"}
    if lidar_mode == "off" or (not depths_v1 and not depths_v2) or not candidates_pkl.exists():
        return 1.0, info

    try:
        cand_data = pickle.load(open(candidates_pkl, "rb"))
        pts_all = []
        for m in cand_data.get("merged_1", []) + cand_data.get("merged_2", []):
            if "pc" in m and len(m["pc"]) > 0:
                pts_all.append(np.asarray(m["pc"], dtype=np.float32))
        poses = cand_data.get("poses", None)
        if not pts_all:
            return 1.0, {**info, "status": "no_3d_points"}
        pts_cat = np.concatenate(pts_all, axis=0)
        if poses is not None and len(poses) > 0:
            cam_centers = poses[:, :3, 3]
            dists = np.linalg.norm(pts_cat[:, None, :] - cam_centers[None, :, :], axis=-1)
            z_pi3 = np.min(dists, axis=-1)
        else:
            z_pi3 = np.linalg.norm(pts_cat, axis=-1)
        z_pi3 = z_pi3[(z_pi3 > 0.05) & (z_pi3 < 10.0)]

        valid_lidar = []
        for d in depths_v1 + depths_v2:
            v = d[d > 0.35]
            if v.size > 0:
                valid_lidar.append(v)
        if not valid_lidar or z_pi3.size < 50:
            return 1.0, {**info, "status": "insufficient_valid_lidar_pixels"}

        lidar_cat = np.concatenate(valid_lidar)
        total_px = sum(d.size for d in depths_v1 + depths_v2)
        valid_ratio = float(lidar_cat.size) / float(max(total_px, 1))
        med_lidar = float(np.median(lidar_cat))
        mad_lidar = float(np.median(np.abs(lidar_cat - med_lidar)))
        rel_mad = mad_lidar / max(med_lidar, 1e-4)

        if valid_ratio < 0.05 or rel_mad > 0.55:
            return 1.0, {
                "lidar_mode": lidar_mode,
                "scale_factor": 1.0,
                "valid_ratio": round(valid_ratio, 4),
                "rel_mad": round(rel_mad, 4),
                "status": "fallback_noisy_lidar_ignored",
            }

        s_hat = float(np.clip(med_lidar / float(max(np.median(z_pi3), 1e-3)), 0.25, 12.0))
        return s_hat, {
            "lidar_mode": lidar_mode,
            "scale_factor": round(s_hat, 4),
            "valid_ratio": round(valid_ratio, 4),
            "rel_mad": round(rel_mad, 4),
            "status": "applied_1dof_metric_scale",
        }
    except Exception as e:
        return 1.0, {**info, "status": f"fallback_exception: {e}"}


def arkit_trajectory_scale(candidates_pkl: Path, segments: list) -> tuple:
    """
    Metres per Pi3 unit from the phone's metric VIO trajectory. `segments` = [(scan_dir, frame_paths), ...]
    in Pi3 file order. Each take has its own ARKit world frame, so ratios of camera-center distances are taken
    within a take (pairs with >= 0.3 m ARKit baseline) and pooled with a median.
    """
    cand = pickle.load(open(candidates_pkl, "rb"))
    poses = cand.get("poses", None)
    if poses is None or len(poses) == 0:
        return None, {"status": "arkit_no_pi3_poses"}
    centers = np.asarray(poses)[:, :3, 3]
    offset, ratios, per_take = 0, [], {}
    for scan_dir, frames in segments:
        n = len(frames)
        pos = load_arkit_positions(scan_dir)
        ids = [int(Path(fp).stem) for fp in frames]
        if n >= 3 and pos and all(i in pos for i in ids) and offset + n <= len(centers):
            a = np.stack([pos[i] for i in ids])
            c = centers[offset:offset + n]
            iu = np.triu_indices(n, 1)
            da = np.linalg.norm(a[:, None] - a[None], axis=-1)[iu]
            dc = np.linalg.norm(c[:, None] - c[None], axis=-1)[iu]
            ok = (da >= 0.3) & (dc > 1e-6)
            if ok.sum() >= 10:
                r = da[ok] / dc[ok]
                ratios.append(r)
                per_take[Path(scan_dir).name] = round(float(np.median(r)), 4)
        offset += n
    if not ratios:
        return None, {"status": "arkit_unavailable"}
    r = np.concatenate(ratios)
    s = float(np.median(r))
    return s, {
        "scale_source": "arkit_trajectory",
        "scale_factor": round(s, 4),
        "per_take_scale": per_take,
        "rel_spread": round(float(np.median(np.abs(r - s)) / s), 4),
        "status": "applied_arkit_metric_scale",
    }
