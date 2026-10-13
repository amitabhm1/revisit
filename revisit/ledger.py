"""Stage 3: evidence ledger.

1. A "removed" object whose location the second scan never looked at becomes UNVERIFIED, with a
   suggested look-from pose 0.85 m away.
2. A "moved" object that stayed in place but changed appearance becomes REPLACED.
3. Two moved objects that traded places become SWAPPED.
"""
import math
import pickle
from pathlib import Path

import numpy as np


def apply_evidence_ledger(
    candidates_pkl: Path,
    cards: list[dict],
    scale_factor: float,
    has_relook: bool = False,
) -> list[dict]:
    """
    Stage 3 (3D Evidence Ledger & Cross-Scan Matching):
      1. Frustum/visibility check on Pi3 poses: if a Before-only object (`removed`) has
         low visibility coverage in Scan 2 (`vis_coverage < 0.14` or < 12% of Scan-2 cameras
         contain its 3D centroid inside their frustum), reclassify `candidate_type` as
         `unverified` and compute an actionable metric 3D `look_from_pose`.
      2. Same-footprint check: if a `moved` object stayed within 0.28 m (`delta_3d_m < 0.28`)
         but its appearance changed (`c_render > 0.22` and `dino_similarity < 0.82`),
         mark `candidate_type` as `replaced`.
      3. Reciprocal move check: if two `moved` candidates swap 3D locations, mark `swapped`.
    """
    if not candidates_pkl.exists() or not cards:
        return cards

    cand_data = pickle.load(open(candidates_pkl, "rb"))
    poses = cand_data.get("poses", None)
    intrinsics = cand_data.get("intrinsic", None)
    H = int(cand_data.get("H", 518))
    W = int(cand_data.get("W", 392))
    n_v1 = int(cand_data.get("img_1_length", 0))
    n_v2 = int(cand_data.get("img_2_length", 0))
    s_f = float(scale_factor)

    # 1. Check Scan 2 camera frustum visibility for Before-only objects
    for card in cards:
        c1 = card.get("centroid_3d_v1")
        c2 = card.get("centroid_3d_v2")
        if c1 is not None and c2 is not None:
            card["delta_3d_m"] = float(np.linalg.norm(np.asarray(c1) - np.asarray(c2)) * s_f)
        else:
            card["delta_3d_m"] = float(card.get("delta_3d", 0.0) * s_f)

        if card.get("candidate_type") == "removed" and c1 is not None and poses is not None and intrinsics is not None and n_v2 > 0:
            pt_w = np.array([c1[0], c1[1], c1[2], 1.0], dtype=np.float32)
            in_frustum_count = 0
            for j in range(n_v2):
                cam_idx = n_v1 + j
                if cam_idx >= len(poses):
                    break
                c2w = poses[cam_idx]
                K = intrinsics[cam_idx]
                w2c = np.linalg.inv(c2w)
                pt_c = w2c @ pt_w
                z = float(pt_c[2])
                if z <= 0.15:
                    continue
                uv = K @ pt_c[:3]
                u, v = float(uv[0] / z), float(uv[1] / z)
                margin_u, margin_v = 0.08 * W, 0.08 * H
                if margin_u <= u <= (W - margin_u) and margin_v <= v <= (H - margin_v):
                    in_frustum_count += 1
            frustum_ratio = float(in_frustum_count) / float(max(n_v2, 1))
            card["scan2_frustum_ratio"] = round(frustum_ratio, 4)

            if not has_relook and (frustum_ratio < 0.12 or float(card.get("vis_coverage", 1.0)) < 0.14):
                card["candidate_type"] = "unverified"
                # Compute actionable look_from_pose from nearest Scan 2 trajectory camera
                cam2_positions = poses[n_v1 : n_v1 + n_v2, :3, 3]
                c1_arr = np.asarray(c1, dtype=np.float32)
                dists = np.linalg.norm(cam2_positions - c1_arr[None, :], axis=-1)
                nearest_cam = cam2_positions[int(np.argmin(dists))]
                dir_vec = c1_arr - nearest_cam
                norm_d = float(np.linalg.norm(dir_vec)) + 1e-6
                unit_dir = dir_vec / norm_d
                # Position robot 0.85m back from target along approach ray
                stand_pos = c1_arr * s_f - unit_dir * 0.85
                targ_pos = c1_arr * s_f
                yaw_deg = float(math.degrees(math.atan2( float(unit_dir[0]), float(unit_dir[2]) )))
                card["look_from_pose"] = {
                    "position_room_m": [round(float(x), 3) for x in stand_pos],
                    "look_at_target_m": [round(float(x), 3) for x in targ_pos],
                    "yaw_deg": round(yaw_deg, 1),
                    "standoff_m": 0.85,
                }

        # 2. Same-footprint in-place appearance change -> replaced
        if (
            card.get("candidate_type") == "moved"
            and card.get("delta_3d_m", 99.0) < 0.28
            and float(card.get("c_render", 0.0)) > 0.22
            and float(card.get("dino_similarity", 1.0)) < 0.82
        ):
            card["candidate_type"] = "replaced"

    # 3. Two moved objects whose start and end points trade places (within 0.45 m) -> swapped
    moved_cards = [c for c in cards if c.get("candidate_type") == "moved" and c.get("centroid_3d_v1") and c.get("centroid_3d_v2")]
    for i in range(len(moved_cards)):
        for j in range(i + 1, len(moved_cards)):
            ci, cj = moved_cards[i], moved_cards[j]
            c1_i = np.asarray(ci["centroid_3d_v1"]) * s_f
            c2_i = np.asarray(ci["centroid_3d_v2"]) * s_f
            c1_j = np.asarray(cj["centroid_3d_v1"]) * s_f
            c2_j = np.asarray(cj["centroid_3d_v2"]) * s_f
            if np.linalg.norm(c1_i - c2_j) < 0.45 and np.linalg.norm(c2_i - c1_j) < 0.45:
                ci["candidate_type"] = "swapped"
                cj["candidate_type"] = "swapped"

    return cards
