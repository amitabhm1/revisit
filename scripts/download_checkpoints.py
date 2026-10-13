#!/usr/bin/env python3
"""Download model weights into $REVISIT_CKPT_DIR (default ./checkpoints).

  python scripts/download_checkpoints.py --set room        # room pipeline, both verifiers
  python scripts/download_checkpoints.py --set scenediff   # SceneDiff table
  python scripts/download_checkpoints.py --set ablation    # extra SAM checkpoints for the mask ablation
  python scripts/download_checkpoints.py --set train       # base VLMs for training
  python scripts/download_checkpoints.py --set all
  python scripts/download_checkpoints.py --list            # print what goes where, download nothing

Gated models (DINOv3, SAM 3.1): accept the license on the model page, then `hf auth login`.
"""
import argparse
import sys
import urllib.request

from revisit import paths

SETS = ("room", "scenediff", "ablation", "train")


def wanted(entry, chosen):
    return chosen == "all" or chosen in entry.get("sets", [])


def fetch_model(name, entry):
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    dest = paths.model_dir(name)
    if "url" in entry:
        if dest.exists():
            print(f"ok    {name:12s} {dest}")
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"get   {name:12s} {entry['url']}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(entry["url"], tmp)
        tmp.rename(dest)
        return True
    local = paths.CKPT_DIR / entry["repo"]
    print(f"get   {name:12s} {entry['repo']} -> {local}")
    try:
        if "file" in entry:
            hf_hub_download(entry["repo"], entry["file"], local_dir=local)
        else:
            snapshot_download(entry["repo"], local_dir=local)
    except (GatedRepoError, RepositoryNotFoundError):
        print(f"  {entry['repo']} is gated: accept its license at https://huggingface.co/{entry['repo']} "
              "and run `hf auth login`.")
        return False
    return True


def fetch_adapters(chosen):
    from huggingface_hub import snapshot_download

    reg = paths.registry()["adapters"]
    items = [k for k, v in reg["items"].items() if wanted(v, chosen)]
    if not items:
        return True
    org = paths.hf_org(required=False)
    if org is None:
        print("skip  adapters     hf_org is not set in configs/checkpoints.yaml (the Revisit adapters are not public yet)")
        return False
    local = paths.CKPT_DIR / reg["repo"]
    print(f"get   adapters     {org}/{reg['repo']} ({', '.join(items)}) -> {local}")
    snapshot_download(f"{org}/{reg['repo']}", local_dir=local,
                      allow_patterns=[f"{reg['items'][k]['subfolder']}/*" for k in items])
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=SETS + ("all",), default="room")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    reg = paths.registry()

    if args.list:
        for name, e in reg["models"].items():
            src = e.get("repo", e.get("url"))
            print(f"{name:12s} {src:48s} -> {paths.model_dir(name)}   sets: {', '.join(e['sets'])}")
        for name, e in reg["adapters"]["items"].items():
            print(f"{name:32s} <hf_org>/{reg['adapters']['repo']}/{e['subfolder']} -> {paths.adapter_dir(name)}")
        return

    ok = all([fetch_model(n, e) for n, e in reg["models"].items() if wanted(e, args.set)])
    ok = fetch_adapters(args.set) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
