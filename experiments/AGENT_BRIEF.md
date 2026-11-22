# Agent brief — Qwen3 text-only → VLM project (read fully before acting)

Repo: this repository (paths below are relative to its root). Plan: roadmap.md (§3.5 = current architecture). How-to: README.md.
Today: 2026-11-01. Machine: 2× RTX PRO 6000 Blackwell (97 GB each), 64 CPU cores shared with other users.

## Environments
- PY=~/.conda/envs/vlm-train/bin/python   (torch 2.11 cu128, transformers 5.17, peft, our `vlm` package installed -e)
- EVAL_PY=~/.conda/envs/vlm-eval/bin/python (vLLM 0.30, lmms-eval 0.7.3, lm-eval; torch 2.13 cu130; `vlm` installed -e)
- Always: export VLLM_USE_FLASHINFER_SAMPLER=0 (FlashInfer JIT sampler misdetects sm_120) ; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for training.
- HF token is configured (~/.cache/huggingface/token). GPQA access granted.

## GPU ASSIGNMENT + LOCK PROTOCOL (mandatory)
- GPU 0 → agent "scaling-4b" only (training + its evals). Lock: locks/gpu0.lock
- GPU 1 → split into two memory slots, each job capped at ~45 GB (vLLM gpu_memory_utilization ≤ 0.45; HF evals are small):
    slot A → agent "quality-retention"  (lock locks/gpu1-slotA.lock)
    slot B → agent "serving-bench"      (lock locks/gpu1-slotB.lock)
Every GPU command runs under your lock, e.g.  flock locks/gpu1-slotA.lock env CUDA_VISIBLE_DEVICES=1 <cmd>
Never use another agent's GPU/slot. Never kill processes you did not start; never touch root/other users' processes.
vLLM checks free memory at startup: if it fails because the other slot is busy, wait and retry (do not raise your cap above 0.45).

## Current architecture (hybrid, roadmap §3.5)
Qwen3-VL architecture = Qwen3-VL-2B vision encoder+merger+DeepStack + text-only Qwen3-1.7B LM (all LM weights), RoPE θ=1e6.
Text-only logits identical to Qwen3-1.7B in FP32 (scripts/build_hybrid.py).
- exports/hybrid-q3-1b7-vit2b   untrained hybrid; exports/hybrid-q3-1b7-aligned  after merger alignment
- experiments/HYB-S2-Q3-1B7-LORA32-3000  S2 LoRA run (r32/α64 on LM + merger); checkpoints step-0001000/2000/3000 (trainable.pt)
  Official-split QA (60 Q/source, scripts/eval_hybrid.py): step1000 52.6, step2000 55.5, step3000 54.5 overall.
Key scripts: scripts/train_hybrid.py, scripts/eval_hybrid.py (probe + official QA), scripts/export_hybrid.py
(merges LoRA with --lora-scale, writes <out> = Qwen3-VL checkpoint for vLLM and <out>-llm = plain Qwen3 LM for text evals),
scripts/eval_text.py (vLLM text suite, noise-aware, thinking/nothink), scripts/injection_eval.py (SigLIP-path; needs a hybrid variant),
scripts/run_benchmarks.sh (lmms-eval via vLLM), scripts/zeroshot_qwen3vl_vit.py (Qwen3-VL-2B ceiling), src/vlm/hybrid/data.py.

## Data / eval assets
- manifests/eval-official (1,400 Q from official val/test splits: docvqa_val, infovqa_val, chartqa_test, textvqa_val, vqav2_val, ai2d_test, ocrbench2_screen; meta.golds)
- manifests/injection-eval (120 injection images), manifests/s2/<source> (35 training sources), manifests/synth-reading-200k,
  replay manifests (replay-q3-1b7-20k, replay-q3-1b7-ifsafety-5k), traces (traces-q3-1b7-v3, traces-q3-1b7-s2-v3).
- Text baselines of original Qwen3-1.7B: reports/text/qwen3-1.7b-original.json (think/nothink):
  MMLU-Pro(1K) 56.48/41.38, GSM8K 90.35/81.22, IFEval 70.29/68.02, MATH-500 89.95/72.45, GPQA-D 39.27/30.93, AIME24/25 48.75/33.54 (think, 16 seeds).
  Retention gate (roadmap §7.2): large benchmarks drop ≤ ~1 pt; noisy ones drop < 2σ of baseline seeds.

## Rules
- Do not edit roadmap.md or README.md (the lead consolidates). Write your results to reports/agents/<your-name>.md (+ JSON).
- Prefer new scripts over editing shared ones; if you must fix a shared script, keep it backward compatible and run `CUDA_VISIBLE_DEVICES="" $PY -m pytest -q tests/` after.
- Verify claims with numbers; report failures honestly. Long jobs: run with setsid nohup + log file, poll with until-loops (no bare sleep chains).
