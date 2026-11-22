"""Day-0 environment gate (roadmap §1.3).

Runs every backend we depend on through forward, backward and optimizer steps on
the real GPU and a real Qwen3 checkpoint, and writes a JSON report to
reports/day0/. "It installs" is not a pass.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/day0_check.py --model Qwen/Qwen3-0.6B
    CUDA_VISIBLE_DEVICES=0 python scripts/day0_check.py --model Qwen/Qwen3-1.7B --only optim_fp32_master
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm.data.packing import block_causal_mask, pack_samples  # noqa: E402
from vlm.training.losses import (  # noqa: E402
    IGNORE_INDEX,
    chunked_linear_cross_entropy,
    shift_for_causal_lm,
)

DEVICE = "cuda"


# --------------------------------------------------------------------------- helpers


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.split("\n")[0]
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 2**30, 1),
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        **{p: pkg_version(p) for p in ["transformers", "triton", "torchao", "liger-kernel", "peft", "accelerate", "bitsandbytes", "vllm"]},
    }


def gpu_availability(min_free_gb: float) -> dict:
    free, total = torch.cuda.mem_get_info()
    # Physical index of the visible device (first entry of CUDA_VISIBLE_DEVICES).
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    util = subprocess.run(
        ["nvidia-smi", "-i", physical, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    free_gb = free / 2**30
    return {
        "pass": free_gb >= min_free_gb,
        "free_gb": round(free_gb, 1),
        "total_gb": round(total / 2**30, 1),
        "physical_gpu": physical,
        "gpu_util_pct_snapshot": util,
        "required_free_gb": min_free_gb,
    }


def load_model(name: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    return model.to(DEVICE)


def random_tokens(n: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Stay inside the normal text range; avoid special tokens at the top of the vocab.
    return torch.randint(100, min(vocab, 150_000), (n,), generator=g)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# --------------------------------------------------------------------------- checks


def check_sdpa_fwd_bwd(args) -> dict:
    model = load_model(args.model, torch.bfloat16)
    model.train()
    ids = random_tokens(2 * 512, model.config.vocab_size, 0).view(2, 512).to(DEVICE)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(torch.isfinite(g).all().item() for g in grads)
    res = {"pass": bool(torch.isfinite(out.loss).item() and finite), "loss": out.loss.item(), "n_grads": len(grads)}
    del model
    return res


@torch.no_grad()
def check_packing_isolation(args) -> dict:
    """Leakage must be exactly zero. Packed vs. alone must match tightly in FP32; in BF16 it
    must stay within the noise floor of masked vs. unmasked attention on the same sample."""
    lens = [300, 450, 200]
    res = {}
    for dtype in (torch.float32, torch.bfloat16):
        model = load_model(args.model, dtype).eval()
        vocab = model.config.vocab_size
        samples = [{"input_ids": random_tokens(n, vocab, i), "labels": random_tokens(n, vocab, i)} for i, n in enumerate(lens)]

        def packed_logits(ss):
            p = pack_samples(ss, pad_id=0)
            mask = block_causal_mask(p["seq_lens"], torch.bool, DEVICE)
            return model(input_ids=p["input_ids"].to(DEVICE), attention_mask=mask,
                         position_ids=p["position_ids"].to(DEVICE)).logits[0].float()

        packed = packed_logits(samples)
        perturbed = [samples[0], {"input_ids": random_tokens(lens[1], vocab, 99), "labels": samples[1]["labels"]}, samples[2]]
        packed_pert = packed_logits(perturbed)
        leak = max((packed[: lens[0]] - packed_pert[: lens[0]]).abs().max().item(),
                   (packed[lens[0] + lens[1]:] - packed_pert[lens[0] + lens[1]:]).abs().max().item())

        diffs, noise, offset = [], [], 0
        for s, n in zip(samples, lens):
            ids = s["input_ids"][None].to(DEVICE)
            alone = model(input_ids=ids).logits[0].float()
            alone_masked = model(input_ids=ids, attention_mask=block_causal_mask(torch.tensor([n]), torch.bool, DEVICE)).logits[0].float()
            diffs.append((alone - packed[offset: offset + n]).abs().max().item())
            noise.append((alone - alone_masked).abs().max().item())
            offset += n
        name = "fp32" if dtype == torch.float32 else "bf16"
        ok = leak == 0.0 and (max(diffs) < 1e-3 if dtype == torch.float32 else max(diffs) <= 1.5 * max(noise) + 1e-6)
        res[name] = {"pass": ok, "leak_max_abs": leak, "packed_vs_alone_max_abs": diffs, "kernel_noise_floor": noise}
        del model
        torch.cuda.empty_cache()
    res["pass"] = res["fp32"]["pass"] and res["bf16"]["pass"]
    return res


def check_grad_checkpointing(args) -> dict:
    model = load_model(args.model, torch.float32)
    model.train()
    ids = random_tokens(2 * 512, model.config.vocab_size, 1).view(2, 512).to(DEVICE)

    def loss_and_grad():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return loss.item(), g

    l0, g0 = loss_and_grad()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    l1, g1 = loss_and_grad()
    cos = cosine(g0, g1)
    res = {"pass": abs(l0 - l1) < 1e-3 and cos > 0.999, "loss_plain": l0, "loss_ckpt": l1, "grad_cosine": cos}
    del model, g0, g1
    return res


def check_fused_ce(args) -> dict:
    model = load_model(args.model, torch.bfloat16)
    ids = random_tokens(4096, model.config.vocab_size, 2)[None].to(DEVICE)
    labels = ids.clone()
    labels[:, :1000] = IGNORE_INDEX
    with torch.no_grad():
        hidden = model.model(input_ids=ids).last_hidden_state
    weight = model.lm_head.weight.detach()
    h, y = shift_for_causal_lm(hidden, labels)
    h = h.reshape(-1, h.shape[-1])
    y = y.reshape(-1)

    def run(fn):
        hh = h.detach().clone().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = fn(hh)
        loss.backward()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        return loss.item(), hh.grad.detach(), round(peak, 2)

    naive = run(lambda hh: F.cross_entropy((hh @ weight.t()).float(), y, ignore_index=IGNORE_INDEX))
    chunked = run(lambda hh: chunked_linear_cross_entropy(hh, weight, y, chunk_size=1024))
    res = {
        "naive": {"loss": naive[0], "peak_gb": naive[2]},
        "chunked": {"loss": chunked[0], "peak_gb": chunked[2], "grad_cosine_vs_naive": cosine(naive[1], chunked[1])},
    }
    ok = abs(naive[0] - chunked[0]) / naive[0] < 1e-3 and res["chunked"]["grad_cosine_vs_naive"] > 0.999

    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        liger_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)
        liger = run(lambda hh: liger_fn(weight, hh, y))
        liger_cos = cosine(naive[1], liger[1])
        liger_ok = abs(naive[0] - liger[0]) / naive[0] < 1e-3 and liger_cos > 0.999
        res["liger"] = {"loss": liger[0], "peak_gb": liger[2], "grad_cosine_vs_naive": liger_cos, "pass": liger_ok}
    except Exception as e:  # Liger is optional: chunked CE is the fallback
        res["liger"] = {"pass": False, "error": f"{type(e).__name__}: {e}"}

    res["selected_backend"] = "liger" if res["liger"].get("pass") else "chunked"
    res["pass"] = ok
    del model
    return res


def _train_steps(model, optimizer, steps: int, autocast: bool) -> dict:
    model.train()
    ids = random_tokens(2 * 256, model.config.vocab_size, 3).view(2, 256).to(DEVICE)
    losses, times = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    finite = all(torch.isfinite(torch.tensor(losses)))
    return {
        "pass": bool(finite and losses[-1] < 0.5 * losses[0]),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "median_step_s": sorted(times)[len(times) // 2],
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def check_optim_fp32_master(args) -> dict:
    """1.7B full-FT policy: FP32 weights (= master copy) + FP32 AdamW + BF16 autocast."""
    model = load_model(args.model, torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    res = _train_steps(model, opt, args.steps, autocast=True)
    del model, opt
    return res


def check_optim_bf16_stochastic_round(args) -> dict:
    """4B full-FT policy: BF16 weights + torchao AdamW with stochastic rounding."""
    try:
        from torchao.optim import _AdamW as AOAdamW
    except ImportError:
        from torchao.optim import AdamW as AOAdamW
    model = load_model(args.model, torch.bfloat16)
    opt = AOAdamW(model.parameters(), lr=1e-4, bf16_stochastic_round=True)
    res = _train_steps(model, opt, args.steps, autocast=False)
    res["optimizer"] = f"{AOAdamW.__module__}.{AOAdamW.__name__}"
    del model, opt
    return res


def check_bf16_rounding_demo(args) -> dict:
    """Evidence for §1.5: fraction of BF16 weights a plain AdamW step at LR 5e-6 fails to change."""
    model = load_model(args.model, torch.bfloat16)
    params = [p for p in model.parameters() if p.dim() >= 2]
    before = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.0)
    ids = random_tokens(2 * 256, model.config.vocab_size, 4).view(2, 256).to(DEVICE)
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    unchanged = sum((p.detach() == b).sum().item() for p, b in zip(params, before))
    total = sum(b.numel() for b in before)
    del model, opt, before
    # Informational: always "passes"; the number justifies the precision policy.
    return {"pass": True, "lr": 5e-6, "fraction_unchanged": round(unchanged / total, 4)}


CHECKS = {
    "sdpa_fwd_bwd": check_sdpa_fwd_bwd,
    "packing_isolation": check_packing_isolation,
    "grad_checkpointing": check_grad_checkpointing,
    "fused_ce": check_fused_ce,
    "optim_fp32_master": check_optim_fp32_master,
    "optim_bf16_stochastic_round": check_optim_bf16_stochastic_round,
    "bf16_rounding_demo": check_bf16_rounding_demo,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--only", nargs="*", choices=list(CHECKS))
    ap.add_argument("--force", action="store_true", help="run even if the GPU has less free memory than required")
    args = ap.parse_args()

    report = {"time": datetime.now().isoformat(timespec="seconds"), "model": args.model, "env": environment()}
    report["gpu_availability"] = gpu_availability(args.min_free_gb)
    print(json.dumps(report, indent=2))
    if not report["gpu_availability"]["pass"] and not args.force:
        print(f"\nFAIL: less than {args.min_free_gb} GB free on the visible GPU. Free a GPU or pass --force.")
        sys.exit(2)

    results = {}
    for name in args.only or CHECKS:
        print(f"\n== {name}", flush=True)
        try:
            results[name] = CHECKS[name](args)
        except Exception as e:
            results[name] = {"pass": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        torch.cuda.empty_cache()
        print(json.dumps(results[name], indent=2, default=str))
    report["checks"] = results
    report["all_pass"] = all(r["pass"] for r in results.values())

    out_dir = ROOT / "reports" / "day0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{args.model.split('/')[-1]}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nALL PASS: {report['all_pass']}  →  {out}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
