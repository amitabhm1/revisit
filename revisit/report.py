"""Stage 5: Markdown report and Rerun recording (boxes, arrows, trajectories, evidence images)."""
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def write_report(
    scene_name: str,
    out_dir: Path,
    candidates_pkl: Path,
    cards: list[dict],
    vlm_records: dict[int, dict],
    rrd_path: Path,
    scale_factor: float = 1.0,
    lidar_info: dict | None = None,
    stack_info: str | None = None,
):
    """Write change_report.md and the Rerun recording scene_diff.rrd."""
    cand_data = pickle.load(open(candidates_pkl, "rb")) if candidates_pkl.exists() else {}
    merged_1 = cand_data.get("merged_1", [])
    merged_2 = cand_data.get("merged_2", [])
    unit_ids = cand_data.get("unit_ids", {"video_1": {}, "video_2": {}})
    poses = cand_data.get("poses", None)
    n_v1 = cand_data.get("img_1_length", 0)
    s_f = float(scale_factor)

    uid_to_m1 = {u: m_idx for m_idx, u in unit_ids.get("video_1", {}).items()}
    uid_to_m2 = {u: m_idx for m_idx, u in unit_ids.get("video_2", {}).items()}

    color_map = {
        "removed": [255, 65, 65],      # Red
        "added": [50, 220, 100],       # Green
        "moved": [50, 150, 255],       # Blue
        "replaced": [255, 140, 40],    # Orange
        "swapped": [195, 95, 255],     # Purple
        "unverified": [255, 215, 45],  # Yellow
    }

    centers, half_sizes, box_colors, box_labels = [], [], [], []
    arrow_origins, arrow_vectors, arrow_labels = [], [], []
    scale_caption = ""
    if lidar_info:
        relook_str = f", `relook={lidar_info.get('relook_scan')}`" if lidar_info.get("relook_scan") else ""
        if lidar_info.get("scale_source") == "arkit_trajectory":
            scale_caption = (
                f"> Metric scale from the ARKit trajectory: `s_hat={s_f:.4f}` m per Pi3 unit "
                f"(`per_take={lidar_info.get('per_take_scale')}`, `rel_spread={lidar_info.get('rel_spread', 0.0):.3f}`{relook_str})\n"
            )
        else:
            scale_caption = (
                f"> Metric scale ({lidar_info.get('scale_source', 'none')}): `status={lidar_info.get('status')}`, "
                f"`s_hat={s_f:.4f}` (`valid_lidar_ratio={lidar_info.get('valid_ratio', 0.0):.3f}`, "
                f"`rel_mad={lidar_info.get('rel_mad', 0.0):.3f}`{relook_str})\n"
            )
    markdown_lines = [
        f"# Revisit change report: `{scene_name}`\n",
        (f"> Pipeline: {stack_info}\n" if stack_info else ""),
        scale_caption,
        "### Changes\n",
        "| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary or look-from pose |",
        "|---:|:---|:---|---:|---:|:---|",
    ]
    pruned_lines = []

    for card in cards:
        uid = int(card["obj_id"])
        vrec = vlm_records.get(uid, vlm_records.get(str(uid), {}))
        verdict = vrec.get("verdict", card["candidate_type"]).lower()
        # Preserve Stage-3 Evidence Ledger `unverified`, `replaced`, `swapped` upgrades when VLM confirms change
        if card.get("candidate_type") == "unverified":
            verdict = "unverified"
        elif verdict != "none" and card.get("candidate_type") in ("replaced", "swapped"):
            verdict = card["candidate_type"]

        conf = float(vrec.get("confidence", card["mean_score"]))
        p_chg = float(vrec.get("p_change", conf))
        label = vrec.get("label", card.get("label", "object"))
        scaled_delta = float(card.get("delta_3d", 0.0)) * s_f
        summary = vrec.get("summary", f"{label} ({verdict})")
        lfp = card.get("look_from_pose") or vrec.get("look_from_pose")
        if verdict == "unverified" and lfp is not None:
            summary = (
                f"Not seen in scan 2 (`frustum_ratio={card.get('scan2_frustum_ratio', 0.0):.2f}`). "
                f"Look from `pos={lfp['position_room_m']}m` toward `target={lfp['look_at_target_m']}m`, `yaw={lfp['yaw_deg']}°`."
            )
        elif verdict != "none" and any(
            neg in summary.lower()
            for neg in ("no change", "no real change", "false positive", "remains in the same", "not actually")
        ):
            summary = (
                f"Verified {verdict} {label} across multi-view Pi3 3D registration "
                f"(mean_cost={card.get('mean_score', 0.0):.2f}, max_cost={card.get('max_score', 0.0):.2f}, "
                f"render_diff={card.get('c_render', 0.0):.2f}, delta_3d={scaled_delta:.3f}m, "
                f"vis_coverage={card.get('vis_coverage', 0.0):.2f})."
            )
        if verdict == "none":
            pruned_lines.append(
                f"| `{uid}` | none (proposal: `{card['candidate_type']}`) | `{label}` | `{p_chg:.3f}` | `{scaled_delta:.3f}` | {summary} |"
            )
            continue

        color = color_map.get(verdict, [255, 200, 50])

        for bmin_k, bmax_k, suffix in [
            ("bbox_3d_min_v1", "bbox_3d_max_v1", "before"),
            ("bbox_3d_min_v2", "bbox_3d_max_v2", "after"),
        ]:
            if verdict == "added" and suffix == "before":
                continue
            if verdict in ("removed", "unverified", "replaced") and suffix == "after":
                continue
            bmin = card.get(bmin_k)
            bmax = card.get(bmax_k)
            if bmin is not None and bmax is not None:
                bmin_arr = np.array(bmin, dtype=np.float32) * s_f
                bmax_arr = np.array(bmax, dtype=np.float32) * s_f
                c_arr = 0.5 * (bmin_arr + bmax_arr)
                hs_arr = np.maximum(0.5 * (bmax_arr - bmin_arr), 0.02 * s_f)
                centers.append(c_arr)
                half_sizes.append(hs_arr)
                box_colors.append(color)
                box_labels.append(f"#{uid} {label} [{verdict.upper()} p={p_chg:.2f}] ({suffix})")

        c1 = card.get("centroid_3d_v1")
        c2 = card.get("centroid_3d_v2")
        if verdict in ("moved", "swapped") and c1 is not None and c2 is not None:
            c1_arr = np.array(c1, dtype=np.float32) * s_f
            c2_arr = np.array(c2, dtype=np.float32) * s_f
            arrow_origins.append(c1_arr)
            arrow_vectors.append(c2_arr - c1_arr)
            arrow_labels.append(f"Δ={scaled_delta:.2f}m ({label})")
            if verdict == "swapped":
                arrow_origins.append(c2_arr)
                arrow_vectors.append(c1_arr - c2_arr)
                arrow_labels.append(f"SWAPPED <-> ({label})")

        if verdict == "unverified" and lfp is not None:
            orig = np.array(lfp["position_room_m"], dtype=np.float32)
            targ = np.array(lfp["look_at_target_m"], dtype=np.float32)
            arrow_origins.append(orig)
            arrow_vectors.append(targ - orig)
            arrow_labels.append(f"LOOK_FROM_POSE -> #{uid} (yaw={lfp['yaw_deg']}°)")

        markdown_lines.append(
            f"| `{uid}` | **{verdict.upper()}** | `{label}` | `{p_chg:.3f}` | `{scaled_delta:.3f}` | {summary} |"
        )

    if len(markdown_lines) == 5:
        markdown_lines.append(
            "| - | no changes | - | - | - | Every proposal was rejected by the verifier. |"
        )

    if pruned_lines:
        markdown_lines.extend(
            [
                "\n### Rejected proposals\n",
                "| ID | Verdict | Object | p(change) | Displacement (m) | Verifier summary |",
                "|---:|:---|:---|---:|---:|:---|",
                *pruned_lines,
            ]
        )

    report_md = "\n".join(markdown_lines) + "\n"
    with open(out_dir / "change_report.md", "w") as f:
        f.write(report_md)

    try:
        import rerun as rr

        rr.init(f"revisit_{scene_name}", spawn=False)
        rr.save(str(rrd_path))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        if poses is not None and len(poses) > 0:
            cam1_pts = poses[:n_v1, :3, 3] * s_f
            cam2_pts = poses[n_v1:, :3, 3] * s_f
            if len(cam1_pts) > 0:
                rr.log("world/trajectory/scan1_before", rr.LineStrips3D([cam1_pts], colors=[[255, 140, 60]], radii=[0.008 * s_f]), static=True)
            if len(cam2_pts) > 0:
                rr.log("world/trajectory/scan2_after", rr.LineStrips3D([cam2_pts], colors=[[60, 180, 255]], radii=[0.008 * s_f]), static=True)

        for card in cards:
            uid = int(card["obj_id"])
            vrec = vlm_records.get(uid, vlm_records.get(str(uid), {}))
            verdict = vrec.get("verdict", card["candidate_type"]).lower()
            if card.get("candidate_type") == "unverified":
                verdict = "unverified"
            elif verdict != "none" and card.get("candidate_type") in ("replaced", "swapped"):
                verdict = card["candidate_type"]
            if verdict == "none":
                continue
            label = vrec.get("label", card.get("label", "object"))
            safe_label = label.replace(" ", "_").replace("/", "_")
            color = color_map.get(verdict, [255, 200, 50])
            m1_idx = uid_to_m1.get(uid)
            m2_idx = uid_to_m2.get(uid)
            if m1_idx is not None and m1_idx < len(merged_1):
                pts1 = np.asarray(merged_1[m1_idx]["pc"], dtype=np.float32) * s_f
                rr.log(f"world/objects/obj_{uid}_{verdict}/scan1_points", rr.Points3D(pts1, colors=[color], radii=[0.008 * s_f]), static=True)
            if m2_idx is not None and m2_idx < len(merged_2):
                pts2 = np.asarray(merged_2[m2_idx]["pc"], dtype=np.float32) * s_f
                rr.log(f"world/objects/obj_{uid}_{verdict}/scan2_points", rr.Points3D(pts2, colors=[color], radii=[0.008 * s_f]), static=True)
            pair_p = out_dir / card["pair_image_path"]
            if pair_p.exists():
                pair_np = np.array(Image.open(pair_p).convert("RGB"))
                rr.log(f"evidence/obj_{uid}_{verdict}_{safe_label}", rr.Image(pair_np), static=True)

        if centers:
            rr.log(
                "world/change_boxes",
                rr.Boxes3D(
                    centers=np.array(centers),
                    half_sizes=np.array(half_sizes),
                    colors=np.array(box_colors, dtype=np.uint8),
                    labels=box_labels,
                ),
                static=True,
            )
        if arrow_origins:
            rr.log(
                "world/change_vectors",
                rr.Arrows3D(
                    origins=np.array(arrow_origins),
                    vectors=np.array(arrow_vectors),
                    colors=[[50, 200, 255]] * len(arrow_origins),
                    labels=arrow_labels,
                ),
                static=True,
            )
        rr.log("summary/change_report", rr.TextDocument(report_md, media_type=rr.MediaType.MARKDOWN), static=True)
        print(f"[Report] wrote {rrd_path} and {out_dir / 'change_report.md'}")
    except ImportError:
        print(f"[Report] wrote {out_dir / 'change_report.md'} (rerun-sdk not installed, no .rrd)")
