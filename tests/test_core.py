"""CPU checks for the pieces that decide verdicts and rewards. Run with `pytest -q`."""
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from revisit.ledger import apply_evidence_ledger
from revisit.metric_scale import arkit_trajectory_scale
from revisit.rl import compute_group_advantages, compute_reward_components
from revisit.verifier import parse_vlm_response

REPO = Path(__file__).resolve().parents[1]


def answer(verdict, conf, think="3d geometry c_geo and dino appearance with good coverage of the object"):
    return f"<think>{think}</think><answer>{{\"verdict\": \"{verdict}\", \"confidence\": {conf}, \"label\": \"chair\", \"summary\": \"x\"}}</answer>"


def test_parse_answer_and_p_change():
    p = parse_vlm_response(answer("none", 0.9), "moved", 0.5)
    assert p["format_ok"] and p["verdict"] == "none"
    assert p["p_change"] == pytest.approx(0.1)
    bad = parse_vlm_response("no json here", "moved", 0.4)
    assert not bad["format_ok"] and bad["verdict"] == "moved"


def test_rewards():
    oracle_tp = {"gt_verdict": "moved", "gt_is_tp": True}
    oracle_fp = {"gt_verdict": "none", "gt_is_tp": False}
    card = {"c_geo": 0.1}
    _, r_vrd, r_cal = compute_reward_components(parse_vlm_response(answer("moved", 0.9), "moved", 0.5), oracle_tp, card)
    assert r_vrd == 1.0 and r_cal > 0.98
    _, r_vrd, _ = compute_reward_components(parse_vlm_response(answer("added", 0.9), "moved", 0.5), oracle_tp, card)
    assert r_vrd == 0.35
    _, r_vrd, _ = compute_reward_components(parse_vlm_response(answer("moved", 0.9), "moved", 0.5), oracle_fp, card)
    assert r_vrd == -1.0  # hallucinated change with low 3D cost
    r_fmt, _, _ = compute_reward_components(parse_vlm_response("garbage", "moved", 0.5), oracle_tp, card)
    assert r_fmt < 0


def test_group_advantages():
    fmt = torch.tensor([0.8, 0.8, 0.8, 0.8])
    vrd = torch.tensor([1.0, -1.0, 1.0, -1.0])
    cal = torch.tensor([0.9, 0.1, 0.5, 0.5])
    for mode in ("grpo", "gdpo", "gdpo_rlcr"):
        a = compute_group_advantages(fmt, vrd, cal, mode=mode)
        assert a.shape == (4,) and abs(float(a.mean())) < 1e-5
    # calibration only enters GDPO+RLCR
    assert torch.allclose(compute_group_advantages(fmt, vrd, cal, mode="gdpo"),
                          compute_group_advantages(fmt, vrd, cal * 0, mode="gdpo"))
    assert not torch.allclose(compute_group_advantages(fmt, vrd, cal, mode="gdpo_rlcr"),
                              compute_group_advantages(fmt, vrd, cal * 0, mode="gdpo_rlcr"))
    # identical answers give no learning signal
    same = torch.ones(4)
    assert torch.count_nonzero(compute_group_advantages(same, same, same, mode="gdpo_rlcr")) == 0


def _write_take(root, name, positions):
    d = root / name
    d.mkdir()
    with open(d / "odometry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for i, p in enumerate(positions):
            w.writerow([i, f"{i:06d}", *p, 0, 0, 0, 1])
    return d


def test_arkit_scale_recovers_known_scale(tmp_path):
    rng = np.random.default_rng(0)
    true_scale = 8.3  # metres per Pi3 unit
    pos_a, pos_b = rng.uniform(-2, 2, (12, 3)), rng.uniform(-2, 2, (12, 3))
    take_a, take_b = _write_take(tmp_path, "A", pos_a), _write_take(tmp_path, "B", pos_b)
    poses = np.tile(np.eye(4), (24, 1, 1))
    poses[:12, :3, 3] = pos_a / true_scale
    poses[12:, :3, 3] = (pos_b + 5.0) / true_scale  # each take has its own ARKit origin
    pkl = tmp_path / "scene_candidates.pkl"
    pickle.dump({"poses": poses}, open(pkl, "wb"))
    frames = [f"{i:06d}.jpg" for i in range(12)]
    s, info = arkit_trajectory_scale(pkl, [(take_a, frames), (take_b, frames)])
    assert s == pytest.approx(true_scale, rel=1e-6)
    assert info["status"] == "applied_arkit_metric_scale"


def test_ledger_unverified_and_swapped(tmp_path):
    K = np.array([[300.0, 0, 196], [0, 300.0, 259], [0, 0, 1]])
    poses = np.tile(np.eye(4), (4, 1, 1))  # cameras at the origin looking down +z
    pkl = tmp_path / "scene_candidates.pkl"
    pickle.dump({"poses": poses, "intrinsic": np.tile(K, (4, 1, 1)), "H": 518, "W": 392,
                 "img_1_length": 2, "img_2_length": 2}, open(pkl, "wb"))
    def behind():
        return {"obj_id": 0, "candidate_type": "removed", "centroid_3d_v1": [0.0, 0.0, -2.0], "vis_coverage": 0.5}

    a = {"obj_id": 1, "candidate_type": "moved", "centroid_3d_v1": [0.0, 0.0, 1.0], "centroid_3d_v2": [0.5, 0.0, 1.0]}
    b = {"obj_id": 2, "candidate_type": "moved", "centroid_3d_v1": [0.5, 0.0, 1.0], "centroid_3d_v2": [0.0, 0.0, 1.0]}
    cards = apply_evidence_ledger(pkl, [behind(), a, b], scale_factor=0.1, has_relook=False)
    assert cards[0]["candidate_type"] == "unverified" and "look_from_pose" in cards[0]
    assert cards[1]["candidate_type"] == cards[2]["candidate_type"] == "swapped"
    relook = apply_evidence_ledger(pkl, [behind()], scale_factor=0.1, has_relook=True)
    assert relook[0]["candidate_type"] == "removed"


def test_registry_is_consistent():
    import yaml

    reg = yaml.safe_load(open(REPO / "configs" / "checkpoints.yaml"))
    for name, v in reg["verifiers"].items():
        assert v["base"] in reg["models"], name
        if "adapter" in v:
            assert v["adapter"] in reg["adapters"]["items"], name
    for cfg in (REPO / "configs" / "train").glob("*.yaml"):
        c = yaml.safe_load(open(cfg))
        assert c["base_model"] in reg["models"] and c["mode"] in ("grpo", "gdpo", "gdpo_rlcr"), cfg.name


def test_room_scoreboard_matches_reference_reports():
    from revisit.eval.room_score import score_run

    ref = json.load(open(REPO / "results" / "room" / "scoreboard.json"))
    for name in ref:
        assert score_run(REPO / "results" / "room" / name) == ref[name], name
