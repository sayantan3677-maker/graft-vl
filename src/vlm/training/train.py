"""Stage trainer (roadmap §5). One script, stage selected by config.

    python scripts/train.py --config configs/train/s1_q3-1b7_lr3e-4.json

Writes experiments/<run_id>/{config.json, metrics.jsonl, checkpoints/step-*/}.
Checkpoints contain only trainable parameters (plus optimizer state), so S1
checkpoints are a few tens of MB.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..data.loader import make_loader
from ..data.packing import flex_block_mask
from ..data.processor import ProcessorConfig, VLMProcessor
from ..model.multimodal_qwen import QwenVLM, VLMConfig

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    run_id: str
    stage: str  # "S1" | "S2" | "S3"
    data: str | list  # manifest dir, or [{"path": dir, "weight": w}, ...] for a weighted mixture
    model: dict = field(default_factory=dict)  # VLMConfig overrides
    processor: dict = field(default_factory=dict)  # ProcessorConfig overrides
    # trainability
    train_llm: bool = False
    train_vision: bool = False
    vision_top_layers: int | None = None  # with train_vision: train only the last k SigLIP layers
    special_rows: list[str] = field(default_factory=lambda: ["vision"])
    init_from: str | None = None  # checkpoint dir (e.g. the S1 run's step-XXXXXXX) whose trainable.pt seeds this run
    llm_dtype: str = "bfloat16"  # "float32" for full FT (FP32 master weights, §1.5)
    # optimization
    lr_projector: float = 1e-4
    lr_llm: float = 5e-6
    lr_vision: float = 1e-6
    lr_special_rows: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.04
    max_steps: int = 1000
    grad_accum: int = 16
    max_grad_norm: float = 1.0
    # batching
    pack_tokens: int = 8192
    pack_tiles: int = 64
    num_workers: int = 8
    # runtime
    attn_implementation: str = "flex_attention"  # packed rows: sparse BlockMask; "sdpa" uses a dense bool mask
    gradient_checkpointing: bool = True
    seed: int = 0
    log_every: int = 10
    save_every: int = 500
    wandb: bool = False


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def param_groups(model: QwenVLM, cfg: TrainConfig) -> list[dict]:
    groups = {
        "projector": (list(model.projector.parameters()), cfg.lr_projector, cfg.weight_decay),
        "special_rows": (list(model.special_rows.parameters()), cfg.lr_special_rows, 0.0),
        "llm": (list(model.llm.parameters()), cfg.lr_llm, cfg.weight_decay),
        "vision": (list(model.vision.parameters()), cfg.lr_vision, cfg.weight_decay),
    }
    out = []
    for name, (params, lr, wd) in groups.items():
        params = [p for p in params if p.requires_grad]
        if params:
            out.append({"name": name, "params": params, "lr": lr, "weight_decay": wd})
    return out


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def save_checkpoint(model: QwenVLM, optimizer, step: int, run_dir: Path) -> None:
    ckpt = run_dir / "checkpoints" / f"step-{step:07d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(trainable, ckpt / "trainable.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    (ckpt / "info.json").write_text(json.dumps({"step": step, "n_tensors": len(trainable)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, help="override (e.g. for the 1000-step calibration)")
    ap.add_argument("--device", default="cuda", help="'cpu' for pipeline dry runs only")
    args = ap.parse_args()
    cfg = TrainConfig(**json.loads(Path(args.config).read_text()))
    if args.max_steps:
        cfg.max_steps = args.max_steps
    torch.manual_seed(cfg.seed)
    device = args.device
    on_cuda = device.startswith("cuda")

    run_dir = ROOT / "experiments" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"train": asdict(cfg), "git_sha": git_sha(),
                                                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, indent=2))

    vcfg = VLMConfig(**cfg.model)
    llm_dtype = getattr(torch, cfg.llm_dtype) if on_cuda else torch.float32
    model = QwenVLM.from_pretrained(vcfg, dtype=llm_dtype, attn_implementation=cfg.attn_implementation).to(device)
    use_flex = cfg.attn_implementation == "flex_attention"
    if cfg.init_from:
        init_dir = Path(cfg.init_from) if Path(cfg.init_from).is_absolute() else ROOT / cfg.init_from
        state = torch.load(init_dir / "trainable.pt", map_location="cpu")
        params = dict(model.named_parameters())
        unknown = [k for k in state if k not in params]
        if unknown:
            raise KeyError(f"init_from has unknown keys: {unknown[:5]}")
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v.to(params[k].dtype))
        print(f"[{cfg.run_id}] initialized {len(state)} tensors from {init_dir}", flush=True)
    model.set_trainable(llm=cfg.train_llm, vision=cfg.train_vision, projector=True, special_rows=tuple(cfg.special_rows),
                        vision_top_layers=cfg.vision_top_layers)
    if cfg.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if cfg.train_vision:
            model.vision.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(vcfg.llm_name)
    pcfg = ProcessorConfig(tile_size=model.vision.image_size, **cfg.processor)
    processor = VLMProcessor(tok, pcfg)
    resolve = lambda d: str(Path(d) if Path(d).is_absolute() else ROOT / d)
    data = [{"path": resolve(d["path"]), "weight": d["weight"]} for d in cfg.data] if isinstance(cfg.data, list) else resolve(cfg.data)
    loader = make_loader(data, processor, cfg.pack_tokens, cfg.pack_tiles, tok.pad_token_id,
                         seed=cfg.seed, num_workers=cfg.num_workers, dense_mask=not use_flex)

    groups = param_groups(model, cfg)
    optimizer = torch.optim.AdamW(groups, betas=cfg.betas, fused=on_cuda)
    warmup = max(1, int(cfg.warmup_ratio * cfg.max_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg.max_steps, warmup))
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"[{cfg.run_id}] trainable params: {n_trainable / 1e6:.2f}M in groups {[g['name'] for g in groups]}", flush=True)

    if cfg.wandb:
        import wandb

        wandb.init(project="qwen3-vlm", name=cfg.run_id, config=asdict(cfg))

    metrics_f = open(run_dir / "metrics.jsonl", "a")
    data_iter = iter(loader)
    step, t_last = 0, time.time()
    acc = {"loss": 0.0, "tokens": 0, "image_tokens": 0, "target_tokens": 0, "samples": 0, "data_wait": 0.0,
           "thinking": 0, "text_only": 0}
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    while step < cfg.max_steps:
        for _ in range(cfg.grad_accum):
            t_wait = time.time()
            batch = next(data_iter)
            acc["data_wait"] += time.time() - t_wait
            if use_flex:
                attention_mask = flex_block_mask(batch["seq_lens"], device)
            else:
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda" if on_cuda else "cpu", dtype=torch.bfloat16, enabled=on_cuda):
                out = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True),
                    pixel_values=None if batch["pixel_values"] is None else batch["pixel_values"].to(device, non_blocking=True),
                    attention_mask=attention_mask,
                    position_ids=batch["position_ids"].to(device, non_blocking=True),
                )
            (out["loss"] / cfg.grad_accum).backward()
            acc["loss"] += out["loss"].item() / cfg.grad_accum
            acc["tokens"] += batch["input_ids"].numel()
            acc["image_tokens"] += batch["n_image_tokens"]
            acc["target_tokens"] += batch["n_target_tokens"]
            acc["samples"] += batch["n_samples"]
            acc["thinking"] += batch["n_thinking"]
            acc["text_only"] += batch["n_text_only"]
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            if on_cuda:
                torch.cuda.synchronize()
            dt = time.time() - t_last
            n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
            rec = {
                "step": step,
                "loss": acc["loss"] / n,
                "grad_norm": float(grad_norm),
                "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
                "step_time_s": dt / n,
                "tokens_per_s": acc["tokens"] / dt,
                "image_tokens_per_s": acc["image_tokens"] / dt,
                "samples_per_s": acc["samples"] / dt,
                "frac_thinking": acc["thinking"] / max(1, acc["samples"]),
                "frac_text_only": acc["text_only"] / max(1, acc["samples"]),
                "data_wait_frac": acc["data_wait"] / dt,
                "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30 if on_cuda else None,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if on_cuda else None,
            }
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "lr"}), flush=True)
            if cfg.wandb:
                wandb.log(rec, step=step)
            acc = {k: 0 for k in acc}
            t_last = time.time()
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            save_checkpoint(model, optimizer, step, run_dir)
    metrics_f.close()


if __name__ == "__main__":
    sys.exit(main())
