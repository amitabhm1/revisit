#!/usr/bin/env python3
"""Download datasets into $REVISIT_DATA_DIR (default ./data).

  python scripts/download_data.py --room        # our four Stray Scanner captures (0.9 GB)  -> data/room/Take_1..4
  python scripts/download_data.py --scenediff   # SceneDiff benchmark (11.9 GB)            -> data/scenediff/
  python scripts/download_data.py --artifacts   # our SceneDiff proposals, verifier outputs and RL card pool
                                                #                                          -> data/artifacts/

--room and --artifacts come from the Revisit Hugging Face account set as hf_org in configs/checkpoints.yaml.
"""
import argparse
import sys
import tarfile
import zipfile

from revisit import paths


def scenediff():
    from huggingface_hub import hf_hub_download

    d = paths.registry()["datasets"]["scenediff"]
    root = paths.DATA_DIR / d["local"]
    if (root / "scenediff_benchmark" / "data").exists():
        print(f"ok    scenediff  {root}")
        return True
    print(f"get   scenediff  {d['repo']}/{d['file']} -> {root}")
    z = hf_hub_download(d["repo"], d["file"], repo_type="dataset", local_dir=root)
    with zipfile.ZipFile(z) as f:
        f.extractall(root)
    return True


def ours(key):
    from huggingface_hub import snapshot_download

    d = paths.registry()["datasets"][key]
    org = paths.hf_org(required=False)
    if org is None:
        print(f"skip  {key:10s} hf_org is not set in configs/checkpoints.yaml (the Revisit datasets are not public yet)")
        return False
    root = paths.DATA_DIR / d["local"]
    repo = d["repo"].format(hf_org=org)
    print(f"get   {key:10s} {repo} -> {root}")
    snapshot_download(repo, repo_type="dataset", local_dir=root)
    for tar in sorted(root.glob("*.tar")):
        print(f"      extract {tar.name}")
        with tarfile.open(tar) as f:
            f.extractall(root)
        tar.unlink()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--room", action="store_true")
    p.add_argument("--scenediff", action="store_true")
    p.add_argument("--artifacts", action="store_true")
    args = p.parse_args()
    if not (args.room or args.scenediff or args.artifacts):
        p.error("choose at least one of --room, --scenediff, --artifacts")
    ok = True
    if args.scenediff:
        ok = scenediff() and ok
    if args.room:
        ok = ours("room") and ok
    if args.artifacts:
        ok = ours("artifacts") and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
