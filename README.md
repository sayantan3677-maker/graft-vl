# Qwen3 → VLM

Converting text-only Qwen3 into a vision-language model. Plan: [roadmap.md](roadmap.md). Data licenses: [LICENSES.md](LICENSES.md).

## Setup
```bash
conda create -n vlm-train python=3.12 -y
~/.conda/envs/vlm-train/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
~/.conda/envs/vlm-train/bin/pip install transformers accelerate peft torchao liger-kernel datasets webdataset pillow safetensors pytest wandb requests
```
Use `PY=~/.conda/envs/vlm-train/bin/python` below.

## Tests (CPU)
```bash
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q tests/
```

## Day 0 (GPU)
```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/day0_check.py --model Qwen/Qwen3-0.6B
CUDA_VISIBLE_DEVICES=1 $PY scripts/day0_check.py --model Qwen/Qwen3-1.7B --only optim_fp32_master fused_ce
$PY scripts/check_qwen_assumptions.py Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B   # CPU
$PY scripts/smoke_real_model.py --device cuda
```

## Data
```bash
$PY scripts/build_shards.py --mixture configs/data/s1.json --n 50000 --name s1-50k
```

Stage 2 sources are built in parallel, one manifest per source (`manifests/s2/<source>`), and mixed by weight at load time:
```bash
scripts/build_s2_parallel.sh 300000 6
$EVAL_PY scripts/gen_replay.py --model Qwen/Qwen3-1.7B --n 20000 --name replay-q3-1b7-20k           # self-distilled text replay
$EVAL_PY scripts/gen_traces.py describe --n 6000 --name traces-src-6k                              # Qwen3-VL-8B descriptions
$EVAL_PY scripts/gen_traces.py reason --src traces-src-6k --model Qwen/Qwen3-1.7B --name traces-q3-1b7  # verified thinking traces
$PY scripts/make_s2_config.py --base configs/train/s2_q3-1b7_full.json --replay manifests/replay-q3-1b7-20k --traces manifests/traces-q3-1b7
```
`EVAL_PY=~/.conda/envs/vlm-eval/bin/python` (vLLM 0.30; `pip install -e .` registers the serving plugin).

## Training
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 $PY scripts/train.py --config configs/train/s1_50k_lr1e-3.json
```
Outputs go to `experiments/<run_id>/` (config, metrics.jsonl, trainable-parameter checkpoints). `init_from` seeds a run from
an earlier checkpoint (S2 starts from the S1 projector). Packed rows use FlexAttention; `data` may be a weighted list of manifests.

## Evaluation
```bash
$PY scripts/eval_s1.py --run experiments/<run> --eval-data manifests/s1-eval300 --train-data manifests/s1-50k   # image reliance
$PY scripts/eval_vqa.py --run experiments/<run> --eval-data manifests/qa-eval-1500 [--thinking]                  # held-out QA accuracy
$PY scripts/injection_eval.py eval --run experiments/<run>                                                         # prompt injection via image
scripts/eval_s2_checkpoint.sh <run_id> <step> [quick|full]                                                         # full S2 gate
$EVAL_PY scripts/eval_text.py --model <hf-id or exports/...-llm> --tasks mmlu_pro gsm8k ifeval math500 --seeds 4 --out reports/text/x.json
```

## Hybrid path (current main line, roadmap §3.5)
Qwen3-VL-2B vision encoder + merger + DeepStack on the **text-only** Qwen3-1.7B language model, inside the Qwen3-VL architecture.
```bash
$PY scripts/build_hybrid.py --out exports/hybrid-q3-1b7-vit2b                     # text-only logits identical to Qwen3-1.7B (FP32)
$PY scripts/train_hybrid.py --config configs/train/hybrid_align.json             # alignment: merger + DeepStack, LM frozen
$PY scripts/export_hybrid.py --run experiments/HYB-ALIGN-... --out exports/hybrid-q3-1b7-aligned --no-llm-only
$PY scripts/train_hybrid.py --config configs/train/hybrid_s2_lora.json           # S2: LoRA on the LM + replay
$PY scripts/eval_hybrid.py --run experiments/<run> --n-qa 60                     # reading probe + official-split QA
$PY scripts/export_hybrid.py --run experiments/<run> --lora-scale 1.0 --out exports/<name>   # + <name>-llm for text evals
```
**S3 (4B, doc/chart/OCR; current release `exports/hyb-4b-s3-s4000-l05`):**
```bash
experiments/s3/build_s3.sh                                                       # shards; FineVision multi-QA rows via build_shards.py --all-turns
$PY experiments/s3/overlap_check.py && $PY experiments/s3/drop_overlap.py 'manifests/s3/*'   # drop eval-image overlap (hash + dHash)
$PY scripts/train_hybrid.py --config configs/train/hybrid_s3_lora_4b.json        # per-source "normalize_format": true | "qa" (src/vlm/hybrid/formats.py)
$PY scripts/export_hybrid.py --run experiments/HYB-S3-Q3-4B-LORA64-4000 --step 4000 --lora-scale 0.5 --out exports/hyb-4b-s3-s4000-l05
$PY scripts/score_clean.py s3-l05=experiments/s3-eval/vision/hyb-4b-s3-s4000-l05.qa.json  # all vs contamination-clean subset + answer-format rates
```
Multi-turn QA rows hold one image with several independent questions; the loader draws one turn per use (`pick_turn`).
The exported hybrid is a stock Qwen3-VL checkpoint, so `vllm serve exports/<name>` works without the plugin.
Image preprocessing runs on the GPU in the trainer (`HybridProcessor.finalize`); prompt/target rendering runs in loader workers.

## Demo (Gradio)
```bash
python -m venv ~/.venvs/vlm-demo && ~/.venvs/vlm-demo/bin/pip install "gradio>=5,<6" "openai>=1.40" pillow
demo/launch.sh exports/hyb-4b-s3-s4000-l05            # vLLM on :8011 + Gradio on :7860
```
The app is a thin OpenAI-API client of `vllm serve` (streaming, TTFT and tok/s per request, thinking toggle, task presets:
QA, OCR, description, chart → table, key-field JSON). Example images are rendered by `demo/make_examples.py` (no third-party images).

## Serving (vLLM)
```bash
$PY scripts/export_vlm.py --run experiments/<run> --out exports/<name>          # self-contained serving checkpoint
$EVAL_PY scripts/test_vllm_parity.py --export exports/<name> --run experiments/<run>   # release gate: prompt ids + greedy parity
vllm serve exports/<name> --max-model-len 16384 --limit-mm-per-prompt '{"image": 1}'   # plugin auto-registers
scripts/run_benchmarks.sh exports/<name> quick                                    # lmms-eval public benchmarks via the plugin
```
Set `VLLM_USE_FLASHINFER_SAMPLER=0` on Blackwell (FlashInfer's JIT sampler misdetects sm_120).
