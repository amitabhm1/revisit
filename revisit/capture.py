"""Stage 0: read a Stray Scanner capture (iPhone Pro with LiDAR).

Picks the steadiest frame (lowest gyro rate) in each of N uniform windows, rotates frames upright
when the phone was held in portrait, and loads confidence-gated LiDAR depth plus the ARKit
camera trajectory from odometry.csv.
"""
import csv
import math
from pathlib import Path

import numpy as np


def detect_stray_gravity_rotation(scan_dir: Path) -> bool:
    """
    Check Stray Scanner `imu.csv` to determine if raw sensor frames (1920x1440)
    need a 90-degree clockwise rotation to be gravity-upright.
    When an iPhone is held upright in portrait while Stray Scanner records raw
    landscape sensor buffers, median(a_y) is approximately -1.0 g (-9.8 m/s^2 or -1.0 G).
    """
    imu_path = scan_dir / "imu.csv"
    if not imu_path.exists():
        return False
    ays = []
    try:
        with open(imu_path, "r") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader, [])]
            ay_idx = header.index("a_y") if "a_y" in header else 2
            for idx, row in enumerate(reader):
                if idx > 600:
                    break
                if len(row) > ay_idx:
                    ays.append(float(row[ay_idx]))
    except Exception:
        return False
    if not ays:
        return False
    med_ay = float(np.median(ays))
    return med_ay < -0.65


def load_stray_imu_omega(scan_dir: Path, n_total_frames: int) -> np.ndarray:
    """Return per-frame rotational speed ||omega||_2 mapped to video frame indices."""
    imu_path = scan_dir / "imu.csv"
    if not imu_path.exists() or n_total_frames <= 0:
        return np.zeros(max(n_total_frames, 1), dtype=np.float32)
    omegas = []
    try:
        with open(imu_path, "r") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader, [])]
            gx_i = header.index("alpha_x") if "alpha_x" in header else 4
            gy_i = header.index("alpha_y") if "alpha_y" in header else 5
            gz_i = header.index("alpha_z") if "alpha_z" in header else 6
            for row in reader:
                if len(row) > gz_i:
                    wx, wy, wz = float(row[gx_i]), float(row[gy_i]), float(row[gz_i])
                    omegas.append(math.sqrt(wx * wx + wy * wy + wz * wz))
    except Exception:
        return np.zeros(max(n_total_frames, 1), dtype=np.float32)
    if not omegas:
        return np.zeros(max(n_total_frames, 1), dtype=np.float32)
    omega_arr = np.asarray(omegas, dtype=np.float32)
    idx_map = np.linspace(0, len(omega_arr) - 1, n_total_frames).astype(np.int64)
    return omega_arr[idx_map]


def load_stray_scanner_frames(
    scan_dir: Path,
    out_frames_dir: Path,
    target_keyframes: int = 36,
    max_side: int = 640,
) -> tuple[list[Path], list[np.ndarray]]:
    """
    Extract upright, low-motion RGB keyframes and confidence-gated metric LiDAR depth maps
    from a Stray Scanner folder.
    """
    import cv2

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    rgb_video = scan_dir / "rgb.mp4"
    depth_dir = scan_dir / "depth"
    conf_dir = scan_dir / "confidence"
    rotate_cw = detect_stray_gravity_rotation(scan_dir)

    depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
    conf_files = sorted(conf_dir.glob("*.png")) if conf_dir.exists() else []

    cap = cv2.VideoCapture(str(rgb_video)) if rgb_video.exists() else None
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap is not None and cap.isOpened()) else len(depth_files)
    if n_video <= 0 and depth_files:
        n_video = len(depth_files)

    omega_per_frame = load_stray_imu_omega(scan_dir, n_video)

    # Trim the first/last 1.0s (30 frames) static anchor holds if clip is > 120 frames,
    # and select the steadiest (lowest angular velocity) frame in each uniform window.
    start_f = 20 if n_video > 120 else 0
    end_f = (n_video - 20) if n_video > 120 else n_video
    n_usable = max(1, end_f - start_f)
    k = min(target_keyframes, n_usable)
    edges = np.linspace(start_f, end_f - 1, k + 1, dtype=np.int64)
    selected_indices = []
    for i in range(k):
        w0, w1 = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
        best_rel = int(np.argmin(omega_per_frame[w0:w1]))
        selected_indices.append(w0 + best_rel)

    existing_jpgs = sorted(out_frames_dir.glob("*.jpg"))
    need_extract = len(existing_jpgs) != len(selected_indices)
    if need_extract:
        for old_p in existing_jpgs:
            old_p.unlink()

    frames_out = []
    depths_out = []

    if need_extract and cap is not None and cap.isOpened():
        sel_set = set(selected_indices)
        max_sel = max(selected_indices) if selected_indices else 0
        f_idx = 0
        saved_map = {}
        while f_idx <= max_sel:
            ret, bgr = cap.read()
            if not ret:
                break
            if f_idx in sel_set:
                if rotate_cw:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                h0, w0 = bgr.shape[:2]
                if max(h0, w0) > max_side:
                    scale = float(max_side) / float(max(h0, w0))
                    bgr = cv2.resize(
                        bgr,
                        (int(round(w0 * scale)), int(round(h0 * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                out_p = out_frames_dir / f"{f_idx:06d}.jpg"
                cv2.imwrite(str(out_p), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                saved_map[f_idx] = out_p
            f_idx += 1
        cap.release()
        frames_out = [saved_map[i] for i in selected_indices if i in saved_map]
    else:
        if cap is not None:
            cap.release()
        frames_out = sorted(out_frames_dir.glob("*.jpg"))

    # Load corresponding confidence-gated metric LiDAR depth maps
    if depth_files:
        for f_idx in selected_indices[: len(frames_out)]:
            d_idx = min(f_idx, len(depth_files) - 1)
            d_mm = cv2.imread(str(depth_files[d_idx]), cv2.IMREAD_UNCHANGED)
            if d_mm is None:
                continue
            d_m = d_mm.astype(np.float32) / 1000.0
            if d_idx < len(conf_files):
                c_img = cv2.imread(str(conf_files[d_idx]), cv2.IMREAD_UNCHANGED)
                if c_img is not None:
                    d_m[c_img < 2] = 0.0
            if rotate_cw:
                d_m = cv2.rotate(d_m, cv2.ROTATE_90_CLOCKWISE)
            d_m[(d_m < 0.35) | (d_m > 4.2)] = 0.0
            depths_out.append(d_m)

    print(
        f"[Stage 0] {scan_dir.name}: selected {len(frames_out)} low-motion keyframes "
        f"(rotate_90_cw={rotate_cw}, lidar_frames={len(depths_out)})"
    )
    return frames_out, depths_out


def load_arkit_positions(scan_dir: Path) -> dict:
    """Stray Scanner odometry.csv: per-frame ARKit camera position (metres) in that take's own world frame."""
    p = Path(scan_dir) / "odometry.csv"
    pos = {}
    if not p.exists():
        return pos
    with open(p) as f:
        rows = csv.reader(f)
        next(rows, None)
        for row in rows:
            try:
                pos[int(row[1])] = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=np.float64)
            except (ValueError, IndexError):
                continue
    return pos
