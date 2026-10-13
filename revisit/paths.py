"""Locations of checkpoints, data and configs, plus the model registry in configs/checkpoints.yaml.

Defaults are folders inside the repository. Override them with REVISIT_CKPT_DIR and REVISIT_DATA_DIR.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
CKPT_DIR = Path(os.environ.setdefault("REVISIT_CKPT_DIR", str(REPO_ROOT / "checkpoints")))
DATA_DIR = Path(os.environ.setdefault("REVISIT_DATA_DIR", str(REPO_ROOT / "data")))
HF_ORG_PLACEHOLDER = "<HF_ORG>"


@lru_cache(maxsize=1)
def registry() -> dict:
    with open(CONFIG_DIR / "checkpoints.yaml") as f:
        return yaml.safe_load(f)


def hf_org(required: bool = True) -> str | None:
    org = str(registry().get("hf_org", HF_ORG_PLACEHOLDER)).strip()
    if org == HF_ORG_PLACEHOLDER:
        if required:
            raise SystemExit(
                "hf_org is not set in configs/checkpoints.yaml. The Revisit adapters and datasets are not "
                "public yet; set hf_org once they are, or place the files by hand (see README)."
            )
        return None
    return org


def model_dir(name: str) -> Path:
    """Local folder (or file) for a public model listed under `models:` in checkpoints.yaml."""
    entry = registry()["models"][name]
    if "repo" in entry:
        base = CKPT_DIR / entry["repo"]
        return base / entry["file"] if "file" in entry else base
    return CKPT_DIR / entry["path"]


def adapter_dir(name: str) -> Path:
    """Local folder of one of our LoRA adapters (listed under `adapters:`)."""
    reg = registry()["adapters"]
    return CKPT_DIR / reg["repo"] / reg["items"][name]["subfolder"]


def verifier(name: str) -> dict:
    """Resolve a verifier preset to {model, adapter, rl_mode} local paths."""
    v = registry()["verifiers"][name]
    out = {"name": name, "model": model_dir(v["base"]), "adapter": None, "rl_mode": None}
    if v.get("adapter"):
        out["adapter"] = adapter_dir(v["adapter"])
        out["rl_mode"] = registry()["adapters"]["items"][v["adapter"]]["rl"]
    return out


def read_adapter_meta(adapter_path) -> dict:
    """revisit.json written next to every adapter: base model, RL variant and training steps."""
    p = Path(adapter_path) / "revisit.json"
    return json.loads(p.read_text()) if p.exists() else {}


def relative_to_roots(path) -> str:
    """Express a path relative to the checkpoint, data or repo root, so reports carry no machine paths."""
    if path is None:
        return None
    p = Path(path).resolve()
    for root in (CKPT_DIR, DATA_DIR, REPO_ROOT):
        try:
            return str(p.relative_to(Path(root).resolve()))
        except ValueError:
            continue
    return p.name
