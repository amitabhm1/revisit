#!/usr/bin/env python3
"""Train a verifier LoRA adapter with GRPO, GDPO or GDPO+RLCR on one node.

  torchrun --standalone --nproc_per_node 8 scripts/train_verifier.py --config configs/train/qwen3vl8b_gdpo_rlcr_step45.yaml
  python scripts/train_verifier.py --config ...   # one GPU without torch.distributed, for debugging

Each GPU takes one evidence card per step, samples a group of answers, scores them (revisit/rl.py) and
computes a KL-regularised policy-gradient loss. LoRA gradients are averaged over all GPUs before every
optimizer step. Checkpoints go to <output_dir>/checkpoints; rerunning the command resumes from the
latest one. Any config value can be overridden on the command line, e.g. --max_steps 100.
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image

from revisit import paths
from revisit.rl import augment_multiview_sample, compute_group_advantages, compute_reward_components, load_training_cards
from revisit.verifier import SYSTEM_PROMPT, build_user_prompt, parse_vlm_response

DEFAULTS = {
    "mode": "gdpo_rlcr", "group_size": 8, "max_steps": 45, "lr": 3e-5, "lr_schedule": "constant",
    "lr_decay_steps": 1500, "lr_floor": 0.25, "lora_r": 16, "lora_alpha": 32, "kl_beta": 0.02,
    "balance": True, "sampling": "sequential", "augment": False, "checkpoint_every": 0, "milestone_every": 0, "seed": 42,
    "max_new_tokens": 192, "temperature": 0.95, "top_p": 0.95,
}


def load_cfg():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    known, extra = p.parse_known_args()
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(open(known.config)))
    for i in range(0, len(extra), 2):  # --key value overrides
        key = extra[i].lstrip("-")
        if key not in cfg:
            raise SystemExit(f"unknown option {extra[i]}")
        cfg[key] = type(cfg[key])(extra[i + 1]) if not isinstance(cfg[key], bool) else extra[i + 1].lower() == "true"
    cfg["card_pool"] = [os.path.expandvars(d) for d in cfg["card_pool"]]
    cfg["gt_dir"] = os.path.expandvars(cfg["gt_dir"])
    base = cfg["base_model"]
    cfg["model_path"] = str(paths.model_dir(base)) if base in paths.registry()["models"] else base
    return cfg


def lora_state(model):
    return {k: v.detach().cpu().to(torch.bfloat16) for k, v in model.named_parameters() if v.requires_grad}


def set_lora_state(model, state, device):
    with torch.no_grad():
        for k, v in model.named_parameters():
            if v.requires_grad and k in state:
                v.copy_(state[k].to(device=device, dtype=v.dtype))


def save_checkpoint(step, model, processor, optimizer, history, out_dir, cfg, milestone):
    ckpt_root = out_dir / "checkpoints"
    tmp = ckpt_root / f"tmp_step_{step:05d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / "adapter"))
    processor.save_pretrained(str(tmp / "adapter"))
    write_meta(tmp / "adapter", cfg, step)
    torch.save({"step": step, "lora_state_dict": lora_state(model), "optimizer_state_dict": optimizer.state_dict(),
                "history": history}, tmp / "trainer_state.pt")
    (tmp / "training_log.json").write_text(json.dumps(history, indent=2))
    latest = ckpt_root / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    os.replace(tmp, latest)
    if milestone:
        mile = ckpt_root / f"step_{step:05d}"
        shutil.rmtree(mile, ignore_errors=True)
        shutil.copytree(latest / "adapter", mile / "adapter")
        shutil.copy(latest / "training_log.json", mile / "training_log.json")


def write_meta(adapter_dir, cfg, step):
    meta = {"base_model": cfg["base_model"], "rl": cfg["mode"], "lora_r": cfg["lora_r"], "steps": step,
            "group_size": cfg["group_size"]}
    (Path(adapter_dir) / "revisit.json").write_text(json.dumps(meta, indent=2))


def main():
    cfg = load_cfg()
    distributed = "RANK" in os.environ  # set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1

    def barrier():
        if distributed:
            dist.barrier()
    device = f"cuda:{local_rank}"
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "train_cards_cache.json"
    if rank == 0:
        dataset = load_training_cards(",".join(cfg["card_pool"]), Path(cfg["gt_dir"]), balance=cfg["balance"])
        cache.write_text(json.dumps(dataset))
        print(f"[{cfg['mode']} | {world} GPUs] {len(dataset)} training cards", flush=True)
    barrier()
    dataset = json.load(open(cache))
    if not dataset:
        raise RuntimeError("no training cards found; check card_pool and gt_dir")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg["model_path"], local_files_only=True, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_path"], torch_dtype=torch.bfloat16, device_map={"": local_rank}, local_files_only=True,
        trust_remote_code=True)
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    model = get_peft_model(base_model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.01)

    start_step, history = 1, []
    state_file = out_dir / "checkpoints" / "latest" / "trainer_state.pt"
    if state_file.exists():
        ckpt = torch.load(state_file, map_location="cpu", weights_only=False)
        set_lora_state(model, ckpt["lora_state_dict"], device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step, history = int(ckpt["step"]) + 1, ckpt.get("history", [])
        if rank == 0:
            print(f"resuming from step {ckpt['step']}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    start_t = time.time()
    prev_elapsed = float(history[-1]["elapsed_sec"]) if history else 0.0

    for step in range(start_step, cfg["max_steps"] + 1):
        if cfg["lr_schedule"] == "cosine_floor":
            progress = min(float(step - 1) / cfg["lr_decay_steps"], 1.0)
            fl = cfg["lr_floor"]
            cur_lr = cfg["lr"] * (fl + (1.0 - fl) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            cur_lr = cfg["lr"]
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        if cfg["sampling"] == "shuffled":  # one fixed permutation per epoch, shared by all GPUs
            global_idx = (step - 1) * world + rank
            perm = list(range(len(dataset)))
            random.Random(cfg["seed"] + (global_idx // len(dataset)) * 7919).shuffle(perm)
            item = dataset[perm[global_idx % len(dataset)]]
        else:
            item = dataset[(step * world + rank) % len(dataset)]

        scene_dir, card, oracle = Path(item["scene_dir"]), item["card"], item["oracle"]
        v1, v2, pair, aug_card = augment_multiview_sample(
            Image.open(scene_dir / card["v1_crop_path"]).convert("RGB"),
            Image.open(scene_dir / card["v2_crop_path"]).convert("RGB"),
            Image.open(scene_dir / card["pair_image_path"]).convert("RGB"),
            card, enable_aug=bool(cfg["augment"]) and step > 1)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": v1}, {"type": "image", "image": v2},
                                         {"type": "image", "image": pair},
                                         {"type": "text", "text": build_user_prompt(aug_card)}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[v1, v2, pair], padding=True, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # 1. Sample a group of answers.
        model.eval()
        with torch.no_grad():
            seqs = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=True,
                                  temperature=cfg["temperature"], top_p=cfg["top_p"],
                                  num_return_sequences=cfg["group_size"])
        texts = processor.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rewards = [compute_reward_components(parse_vlm_response(t, aug_card["candidate_type"], aug_card["mean_score"]),
                                             oracle, aug_card) for t in texts]
        r_fmt, r_vrd, r_cal = (torch.tensor(c, dtype=torch.float32, device=device) for c in zip(*rewards))
        adv = compute_group_advantages(r_fmt, r_vrd, r_cal, mode=cfg["mode"])

        # 2. Policy gradient with a per-token KL penalty to the base model (adapter disabled).
        model.train()
        optimizer.zero_grad()
        loss_val = 0.0
        if adv.abs().max().item() >= 1e-4:
            for g in range(cfg["group_size"]):
                seq = seqs[g:g + 1]
                comp = seq[:, prompt_len:]
                valid = (comp != pad_id).float()
                if valid.sum() == 0:
                    continue
                kw = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()
                      if k not in ("input_ids", "attention_mask")}
                attn = torch.ones_like(seq, device=device)
                logits = model(input_ids=seq, attention_mask=attn, **kw).logits[:, prompt_len - 1:-1, :]
                logp = F.log_softmax(logits, dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad(), model.disable_adapter():
                    ref = model(input_ids=seq, attention_mask=attn, **kw).logits[:, prompt_len - 1:-1, :]
                    logp_ref = F.log_softmax(ref, dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
                log_ratio = logp_ref - logp
                kl = torch.exp(log_ratio.clamp(-10, 10)) - log_ratio - 1.0
                loss = (-(adv[g].detach() * logp - cfg["kl_beta"] * kl) * valid).sum() / (valid.sum() * cfg["group_size"])
                loss.backward()
                loss_val += float(loss.item())

        # 3. Average LoRA gradients and metrics over all GPUs (bf16 gradients, summed in fp32).
        flat = torch.cat([(p.grad.detach().to(torch.bfloat16).view(-1) if p.grad is not None
                           else torch.zeros(p.numel(), device=device, dtype=torch.bfloat16)) for p in trainable]).float()
        metrics = torch.tensor([r_fmt.mean().item(), r_vrd.mean().item(), r_cal.mean().item(), loss_val],
                               dtype=torch.float32, device=device)
        if distributed:
            dist.all_reduce(flat)
            dist.all_reduce(metrics)
        flat = (flat / world).to(torch.bfloat16)
        metrics = (metrics / world).cpu()
        offset = 0
        for p in trainable:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        is_ckpt = (cfg["checkpoint_every"] > 0 and step % cfg["checkpoint_every"] == 0) or step == cfg["max_steps"]
        is_mile = (cfg["milestone_every"] > 0 and step % cfg["milestone_every"] == 0) or step == cfg["max_steps"]
        if rank == 0:
            elapsed = round(prev_elapsed + time.time() - start_t, 2)
            rec = {"step": step, "mode": cfg["mode"], "world_size": world, "cards_seen": step * world,
                   "epochs": round(step * world / len(dataset), 3), "lr": round(cur_lr, 8),
                   "r_format": float(metrics[0]), "r_verdict": float(metrics[1]), "r_cal": float(metrics[2]),
                   "loss": float(metrics[3]), "elapsed_sec": elapsed}
            history.append(rec)
            print(f"[{cfg['mode']} | step {step}/{cfg['max_steps']} | ep {rec['epochs']:.2f} | {world} GPUs] "
                  f"R_fmt={rec['r_format']:.3f} R_vrd={rec['r_verdict']:.3f} R_cal={rec['r_cal']:.3f} "
                  f"loss={rec['loss']:.4f} ({elapsed}s)", flush=True)
            if is_ckpt or is_mile:
                save_checkpoint(step, model, processor, optimizer, history, out_dir, cfg, is_mile)
        if is_ckpt or is_mile:
            barrier()

    if rank == 0:
        final = out_dir / "adapter"
        shutil.rmtree(final, ignore_errors=True)
        model.save_pretrained(str(final))
        processor.save_pretrained(str(final))
        write_meta(final, cfg, cfg["max_steps"])
        (out_dir / "training_log.json").write_text(json.dumps(history, indent=2))
        print(f"saved {final}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
