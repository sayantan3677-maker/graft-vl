"""Train the hybrid (Qwen3-VL architecture, text-only Qwen3 LM) — alignment and S2 stages.

Trainable groups (config): merger (+DeepStack mergers), top-k ViT blocks, LoRA on the LM.
The language model's base weights stay frozen (LoRA only), so text behaviour is recoverable by
scaling/merging adapters (roadmap §5.7).

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/train_hybrid.py --config configs/train/hybrid_align.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm.hybrid.data import HybridProcessor, make_hybrid_loader  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lr_lambda(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    run_dir = ROOT / "experiments" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(cfg.get("seed", 0))

    model = Qwen3VLForConditionalGeneration.from_pretrained(ROOT / cfg["init"], dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").cuda()
    model.requires_grad_(False)
    visual = model.model.visual
    groups = []
    merger = list(visual.merger.parameters()) + list(visual.deepstack_merger_list.parameters())
    groups.append({"name": "merger", "params": merger, "lr": cfg["lr_merger"]})
    if cfg.get("vit_top_blocks"):
        vit = [p for b in visual.blocks[-cfg["vit_top_blocks"]:] for p in b.parameters()]
        groups.append({"name": "vit", "params": vit, "lr": cfg["lr_vit"]})
    if cfg.get("lora"):
        from peft import LoraConfig, inject_adapter_in_model

        inject_adapter_in_model(LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                                           lora_dropout=cfg["lora"].get("dropout", 0.05), target_modules=LORA_TARGETS),
                                model.model.language_model)
        lora = [p for n, p in model.model.language_model.named_parameters() if "lora_" in n]
        for p in lora:
            p.data = p.data.float()
        groups.append({"name": "lora", "params": lora, "lr": cfg["lr_lora"]})
    trainable = [p for g in groups for p in g["params"]]
    for p in trainable:
        p.requires_grad_(True)
        if p.dtype == torch.bfloat16:  # FP32 master copies for trainable non-LoRA params (roadmap §1.5)
            p.data = p.data.float()
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    print(f"[{cfg['run_id']}] trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M in {[g['name'] for g in groups]}", flush=True)

    proc = HybridProcessor(AutoProcessor.from_pretrained(ROOT / cfg["init"]), max_pixels=cfg.get("max_pixels", 1_200_000),
                           max_length=cfg.get("max_length", 6144))
    data = [{"path": str(ROOT / d["path"]), "weight": d["weight"], "normalize_format": d.get("normalize_format", False)}
            for d in cfg["data"]]
    loader = iter(make_hybrid_loader(data, proc, seed=cfg.get("seed", 0), num_workers=cfg.get("num_workers", 12)))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    total, warm = cfg["max_steps"], max(1, int(cfg.get("warmup_ratio", 0.03) * cfg["max_steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warm))
    accum = cfg["grad_accum"]
    mf = open(run_dir / "metrics.jsonl", "a")
    step, t0, acc = 0, time.time(), {"loss": 0.0, "tok": 0, "img_tok": 0, "n": 0, "think": 0, "text_only": 0}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    while step < total:
        for _ in range(accum):
            b = None
            while b is None:  # images are preprocessed here, on the GPU (CPU is shared and contended)
                b = proc.finalize(next(loader), device="cuda")
            kw = {"input_ids": b["input_ids"][None].cuda(), "labels": b["labels"][None].cuda()}
            if "pixel_values" in b:
                kw["pixel_values"] = b["pixel_values"].to("cuda", torch.bfloat16)
                kw["image_grid_thw"] = b["image_grid_thw"].cuda()
            if "mm_token_type_ids" in b:
                kw["mm_token_type_ids"] = b["mm_token_type_ids"][None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**kw).loss
            (loss / accum).backward()
            acc["loss"] += loss.item() / accum
            acc["tok"] += len(b["input_ids"])
            acc["img_tok"] += b["n_image_tokens"]
            acc["n"] += 1
            acc["think"] += int(b["thinking"])
            acc["text_only"] += int("pixel_values" not in b)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.get("log_every", 25) == 0 or step == total:
            dt = time.time() - t0
            n = cfg.get("log_every", 25) if step % cfg.get("log_every", 25) == 0 else step % cfg.get("log_every", 25)
            rec = {"step": step, "loss": acc["loss"] / n, "grad_norm": float(gn), "lr": {g["name"]: g["lr"] for g in opt.param_groups},
                   "step_time_s": dt / n, "tokens_per_s": acc["tok"] / dt, "samples_per_s": acc["n"] / dt,
                   "frac_thinking": acc["think"] / max(1, acc["n"]), "frac_text_only": acc["text_only"] / max(1, acc["n"]),
                   "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30}
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "lr"}), flush=True)
            t0, acc = time.time(), {k: 0 for k in acc}
        if step % cfg["save_every"] == 0 or step == total:
            ck = run_dir / "checkpoints" / f"step-{step:07d}"
            ck.mkdir(parents=True, exist_ok=True)
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, ck / "trainable.pt")
            (ck / "info.json").write_text(json.dumps({"step": step, "init": cfg["init"], "lora": cfg.get("lora")}))
    mf.close()


if __name__ == "__main__":
    main()
