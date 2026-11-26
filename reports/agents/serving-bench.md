# serving-bench — hybrid VLM in vLLM: parity, latency/throughput, FP8, public benchmarks

Agent: serving-bench · 2026-11-01 · GPU 1 slot B (`locks/gpu1-slotB.lock`, vLLM `gpu_memory_utilization` 0.45) · JSON: `reports/agents/serving-bench.json`

**Bottom line.** The hybrid is a stock `Qwen3VLForConditionalGeneration` checkpoint and serves natively in vLLM 0.30 with no plugin.
- **Parity:** passes on numerics. Prompt ids match on 48/48 cases, and vLLM-BF16 sits as close to an FP32 reference as HF-BF16 does.
- **Chat template:** needs one fix. As exported, the Qwen3 text template does not handle OpenAI image content lists; fixed with a merged template.
- **FP8 (dynamic):** gives small gains at this model size and does not change accuracy measurably.
- **Benchmarks:** the step-2000 hybrid trails Qwen3-VL-2B by 7–36 points on the public benchmarks. The gap is not explained by the image-resolution cap.

Export: `exports/serve-hyb-1b7-s2000` (`scripts/export_hybrid.py --run experiments/HYB-S2-Q3-1B7-LORA32-3000 --step 2000 --lora-scale 1.0`), then `scripts/prepare_hybrid_serving.py`. Also written: `exports/serve-hyb-1b7-s2000-llm`.

## 1. Chat template (step 2)

The export's `chat_template.jinja` is Qwen3-1.7B's text template. It sets `content = ''` for any non-string content.

- **What vLLM does with it:** vLLM detects content format `string`. It then flattens the content list itself and puts the image placeholder before the text, joined by `"\n"`: `<|vision_start|><|image_pad|><|vision_end|>\nQuestion`.
- **What training used:** `<|vision_start|><|image_pad|><|vision_end|>Question`, with no separator (`src/vlm/hybrid/data.py`).
- **Effect:** with the original template, vLLM's prompt ids did not match HF's (checked; `template_checks` in the parity JSON). The original template therefore "works" but serves a prompt format the model never saw in training.

**Fix:** `scripts/prepare_hybrid_serving.py <export>` (new; also applied to this export). It writes:
- `chat_template.jinja`: a merged template that keeps Qwen3's hybrid-thinking logic and handles Qwen3-VL content lists.
  - Image parts (`image`, `image_url`, `image_pil`) become `<|vision_start|><|image_pad|><|vision_end|>` and are joined to the text parts with no separator.
  - Video parts use `<|video_pad|>`. A list-valued system message is supported.
  - User messages with list content count as the "last query" for the thinking-strip logic.
  - String content renders byte-identically to the original. That is tested in `tests/test_hybrid_chat_template.py`, and all 38 CPU tests pass.
  - vLLM detects content format `openai` for this template.
- `chat_template_nothink.jinja`: the same template with `{%- set enable_thinking = false %}` prepended. lmms-eval uses this file.
- `chat_template.qwen3_text_original.jinja`: the original template, kept for reference.

**`enable_thinking`:**
- Passing `chat_template_kwargs={"enable_thinking": false}` works in both `LLM.chat` and the OpenAI server. The prompt ends with `<think>\n\n</think>\n\n`, identical to HF.
- The default is thinking ON, which is Qwen3 behaviour.

## 2. Parity: HF transformers vs vLLM (step 3)

Script: `scripts/test_hybrid_vllm_parity.py`. Report: `reports/serving/parity-serve-hyb-1b7-s2000.json` (+ `.log`).

**Setup**
- 24 images from `manifests/eval-official`, taken round-robin across 7 sources: page 7, photo 6, chart 4, diagram 4, screenshot 3.
- Two prompts per image, so 48 cases:
  - the short-answer QA prompt used by eval_hybrid;
  - "Describe this image in detail."
- 64 greedy tokens, `VLLM_USE_FLASHINFER_SAMPLER=0`, max_pixels 1.6M (the training cap), prompts of 160–1,544 tokens.
- HF side: same export, BF16, prompt built as in `eval_hybrid.generate()`.
- vLLM side: `LLM.chat` with OpenAI content lists `[image, text]` and `enable_thinking=False`.

| Check | Result |
|---|---|
| Prompt token ids equal (after image-token expansion) | **48 / 48** |
| Full 64-token greedy match | 28 / 48 (short answers 23/24, descriptions 5/24) |
| Divergences / near-ties (gap < 0.5 nats in both engines) | 20 / 17 |
| Divergences ≥ 0.5 nats | 3: vLLM gaps 0.5, 0.625, 0.75 nats (HF-side gaps ≤ 0.25) |
| Teacher-forced, mean \|Δ log p\| vs FP32 HF on 1.3–1.4K generated tokens | HF-BF16 **0.032**, vLLM-BF16 **0.032** |
| p99 / max \|Δ log p\| vs FP32 | HF-BF16 0.29 / 1.10, vLLM-BF16 0.35 / 0.70 |
| Top-1 agreement with FP32 | HF-BF16 97.1 %, vLLM-BF16 97.4 % |

**Verdict: PASS (numerics).** The strict criterion "every divergence < 0.5 nats" fails on 3 of 48 cases. In each of the three, HF's own gap is ≤ 0.25 nats and the gap is at most 0.75 nats. The teacher-forced comparison against FP32 is the stronger test: vLLM-BF16 is exactly as close to FP32 as HF-BF16 is. That rules out a systematic bug in the vision path, MRoPE or DeepStack.

Greedy divergences also vary between vLLM runs: the first run gave 23/48 full matches, the second 28/48. They come from BF16 kernel differences that then compound over long generations.

## 3. Latency and throughput, BF16 vs FP8 (step 4)

**Setup**
- Command: `vllm serve exports/serve-hyb-1b7-s2000 --gpu-memory-utilization 0.45 --max-model-len 16384 --limit-mm-per-prompt '{"image":1}' [--quantization fp8]` on port 8011.
- Scripts: runner `scripts/run_hybrid_serving_bench.sh`, client `scripts/bench_hybrid_serving.py` (OpenAI streaming).
- Requests: a mix of document pages (DocVQA, about 1,450–1,530 prompt tokens), phone screenshots (OCRBench-v2 screens, about 1,150–1,340) and photos (VQAv2, about 280).
- Every request uses a distinct image, so there are no prefix or multimodal cache hits.
- Each request generates 64 output tokens (`ignore_eos`) in non-thinking mode.
- The tables below come from run 2: 24 / 48 / 192 requests at concurrency 1 / 8 / 32. Run 1 (12 / 24 / 96 requests) is in the JSON. It agrees except at concurrency 8, where it had one large TTFT outlier.

**Caveat:** GPU 1 was shared throughout with the slot-A co-tenant at about 94 % utilisation, and the CPU was contended. Read these numbers as a lower bound for a dedicated GPU. Decode speed at concurrency 1 (about 9.4 ms per token for a 1.7B model) is clearly time-sliced.

| Precision | Concurrency | TTFT p50 / mean / p90 (ms) | TPOT mean (ms) | Per-request decode (tok/s) | Output tok/s (aggregate) | Req/s |
|---|---:|---|---:|---:|---:|---:|
| BF16 | 1 | 260 / 236 / 359 | 9.4 | 106 | 77 | 1.20 |
| BF16 | 8 | 410 / 415 / 730 | 16.3 | 61 | 343 | 5.36 |
| BF16 | 32 | 946 / 1035 / 1752 | 36.1 | 28 | 609 | 9.52 |
| FP8 (dynamic) | 1 | 234 / 241 / 346 | 9.6 | 104 | 74 | 1.15 |
| FP8 (dynamic) | 8 | 307 / 346 / 584 | 14.1 | 71 | 381 | 5.95 |
| FP8 (dynamic) | 32 | 992 / 1031 / 1832 | 33.7 | 30 | 627 | 9.80 |

**TTFT by image type at concurrency 1 (BF16):** photo 79 ms (about 280 tokens), screenshot 215 ms (about 1,150), page 415 ms (about 1,455).

**Memory:**

| | BF16 | FP8 |
|---|---:|---:|
| Weights | 4.31 GiB | 2.63 GiB |
| KV cache at 0.45 | 334K tokens | 350K tokens |

**FP8 conclusion:** about +5–10 % throughput at concurrency 8–32 and no change at concurrency 1. A 1.7B LM plus a 0.4B ViT is not weight-bandwidth-bound at these batch sizes. Prefill is dominated by the vision encoder and attention, which `--quantization fp8` does not quantize in the ViT, and part of the request time is CPU preprocessing. FP8 is not worth it for latency at this size; it frees memory only.

**FP8 accuracy delta.** Measured on 210 questions from eval-official (30 per source, eval_hybrid prompt and scoring), through the server:

| | Overall | ai2d | chartqa | docvqa | infovqa | ocrbench2_screen | textvqa | vqav2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 52.86 | 56.7 | 46.7 | 56.7 | 23.3 | 60.0 | 66.7 | 60.0 |
| FP8 | 55.24 | 70.0 | 43.3 | 56.7 | 30.0 | 63.3 | 70.0 | 53.3 |

**Delta: +2.4 points (FP8 − BF16), which is not significant.**
- Paired comparison: 170 of 210 predictions are identical. 4 questions are correct only under BF16 and 9 only under FP8; sign test p ≈ 0.27. The standard error of each score is about 3.4 points.
- This subset is too small to resolve a delta below about 5 points. Before shipping FP8, run the full product eval plus the text regression suite, as roadmap §12.3 requires.

## 4. Public benchmarks with lmms-eval (step 5)

**Setup**
- Runner: `scripts/run_benchmarks.sh` with `quick` mode (`--limit 500`), lmms-eval 0.7.3, the vLLM chat backend, `GPU_MEM=0.45`, `CUDA_VISIBLE_DEVICES=1`. Results are in `reports/benchmarks/<model>-quick*/`.
- Table made by `scripts/summarize_benchmarks.py`; outputs in `reports/serving/benchmarks-quick.{md,json}`.
- **Ours:** non-thinking template (`chat_template_nothink.jinja`), max_pixels 1.6M.
- **Qwen3-VL-2B-Instruct:** its own chat template and default preprocessing (max_pixels 16.7M), run with `MAX_LEN=32768`. With 16384 the run failed: an InfoVQA prompt came to 16,413 tokens.
- **Qwen3-VL-2B at 1.6MP:** the same model with max_pixels capped at 1.6M via `exports/serve-bench-qwen3vl2b-1p6mp`, which symlinks the weights and edits only the preprocessor config. This gives a same-budget ceiling.

| Benchmark | Metric | n | **Ours (hybrid s2000)** | Qwen3-VL-2B (default px) | Qwen3-VL-2B (1.6MP cap) | Gap vs ceiling |
|---|---|---:|---:|---:|---:|---:|
| docvqa_val | ANLS | 500 | **62.5** | 93.3 | 92.2 | −30.8 |
| infovqa_val | ANLS | 500 | **34.1** | 70.0 | 66.8 | −35.9 |
| chartqa | relaxed_overall | 500 | **41.2** | 68.2 | 67.8 | −27.0 |
| textvqa_val | exact_match (VQA acc) | 500 | **58.6** | 79.7 | 79.9 | −21.1 |
| ocrbench | ocrbench_accuracy as mean score over n | 500 | **74.4** | 90.0 | 90.0 | −15.6 |
| ai2d | exact_match, flexible-extract (lenient letter) | 500 | **0.0 (57.2)** | 76.2 (76.2) | 75.8 (75.8) | −19.0 (lenient) |
| mmbench_en_dev | gpt_eval_score (no GPT key; rule matching) | 500 | **68.6** | 81.4 | 81.0 | −12.8 |
| pope | pope_f1_score | 500 | **80.8** | 88.0 | 88.2 | −7.2 |
| realworldqa | exact_match | 500 | **51.6** | 65.8 | 67.0 | −14.2 |
| mmmu_val | mmmu_acc (MM_LIMIT=5) | 500 | **38.6** | 39.8 | – | −1.2 |

The "Gap vs ceiling" column is relative to the default-pixel Qwen3-VL-2B run.

**Notes on these numbers:**
- **Resolution does not explain the gap.** Capping Qwen3-VL-2B at our 1.6M budget costs it at most 3 points (InfoVQA), so the deficit comes from the model itself (S2 step 2000).
- **MMMU is close (−1.2)** because it is mostly knowledge and reasoning, which the text LM keeps.
- **The largest gaps are document and infographic reading** (DocVQA −31, InfoVQA −36), consistent with the in-house eval-official numbers.
- **ai2d reads 0 under the official filter** because our model answers every MCQ as `Answer: B` and ignores "Answer with the option's letter … directly". The lmms-eval filter only accepts `B.` or a bare `B`. With lenient letter extraction ai2d is 57.2. This is a real instruction-following and format defect: product parsers expecting a bare letter would break. It should be addressed in S3 SFT data.
- **OCRBench:** lmms-eval divides by 1000 regardless of `--limit` (it reports 0.372 / 0.450), so the table shows the mean score over the 500 evaluated items. The first 500 OCRBench items are skewed toward text recognition, so both models score high. The comparison is still like-for-like.
- **MMBench** was scored without `OPENAI_API_KEY`; no API errors appeared in the log, so rule-based option matching sufficed. **POPE** in the first 500 items is 50 % "yes".
- **MMMU** has multi-image questions, so it cannot run under `limit_mm_per_prompt image=1`. It was run separately with `MM_LIMIT=5` (`reports/benchmarks/*-quick-mmmu`). The first combined attempt failed with "At most 1 image(s)".

**Changes to `scripts/run_benchmarks.sh`** (backward compatible; the default behaviour for local exports is unchanged):
- It accepts HF ids. An HF id uses the model's own template; `CHAT_TEMPLATE=<file>|none` overrides that.
- A local export that has no `chat_template_nothink.jinja` now fails loudly instead of inside lmms-eval.
- New optional env vars: `MM_LIMIT`, `MAX_LEN`, `OUT_TAG`.

## 5. Things that would block or bite production serving

1. **Chat template (fixed for this export; must be applied to every export).** `export_hybrid.py` writes Qwen3's text template. Run `scripts/prepare_hybrid_serving.py <export>` after every export, or fold it into `export_hybrid.py` (I did not edit that shared script). Without it, OpenAI clients get an off-distribution `\n` between the image and the text.
2. **Pixel cap.** The exported `processor_config.json` keeps Qwen3-VL's default max_pixels of 16.7M. A long infographic then produces more than 16K tokens: the request is rejected at `max-model-len` 16384, and TTFT becomes unbounded. `prepare_hybrid_serving.py` sets the training cap of 1.6M (about 1,560 visual tokens at most). Product surfaces may want a lower cap for TTFT; a page at 1.6M already costs about 415 ms TTFT at concurrency 1, measured on a shared GPU.
3. **generation_config.** The export had only `eos_token_id: 151645` and no sampling defaults, so vLLM's default sampling (temperature 1.0) applied to clients that omit parameters. `prepare_hybrid_serving.py` now copies Qwen3-1.7B's generation_config: eos `[151645, 151643]` and temperature 0.6 / top_p 0.95 / top_k 20. All measurements above were taken before this change, with explicit temperature 0, so they are unaffected.
4. **Default thinking is ON** (Qwen3 template behaviour). Interactive product traffic should default to non-thinking. Either serve with `--default-chat-template-kwargs '{"enable_thinking": false}'` or `--chat-template <export>/chat_template_nothink.jinja`, and expose thinking per request. When thinking is used, use `--reasoning-parser qwen3`; that combination was not tested here.
5. **Answer-format compliance** (`Answer: X` for MCQ) is a model-quality issue that downstream parsers will hit.
6. **Quality gap to Qwen3-VL-2B** is 13–36 points on document, chart and OCR benchmarks, at the same pixel budget. This is the real blocker for shipping this checkpoint in place of Qwen3-VL-2B. Serving itself is not a blocker.
7. **FP8 dynamic** loads and runs (CutlassFP8ScaledMMLinearKernel on sm_120) but buys little at 1.7B. Accuracy is unchanged within noise on 210 questions; a full product eval is needed before adopting it.
8. **Perf numbers were taken on a GPU shared with another job** at about 94 % utilisation. Re-measure on a dedicated GPU before setting SLOs.

## 6. Problems hit and how they were handled

- **Parity on greedy agreement alone was inconclusive**: 3–6 divergences sat at 0.5–2.0 nats in the first run. I added a teacher-forced FP32 noise-floor comparison, which settles it (§2).
- **The first BF16 benchmark client crashed with `ServerDisconnectedError`.** Blocking PNG encoding inside the event loop let keep-alive sockets go stale. Fixed with up-front encoding, `force_close` connections and retries. A leftover FP8 server from that run was stopped.
- **lmms-eval failures:**
  - MMMU is multi-image, so it was split out and run with `MM_LIMIT=5`.
  - The Qwen3-VL-2B prompt exceeded 16K tokens, so it was rerun with `MAX_LEN=32768`.
  - In both cases lmms-eval exits 0 on error. `run_benchmarks.sh` caught the failures because it checks for the results JSON.

## Files

- **Scripts:**
  - `scripts/test_hybrid_vllm_parity.py`
  - `scripts/bench_hybrid_serving.py`
  - `scripts/run_hybrid_serving_bench.sh`
  - `scripts/prepare_hybrid_serving.py`
  - `scripts/summarize_benchmarks.py`
  - `scripts/run_benchmarks.sh` (extended)
  - `tests/test_hybrid_chat_template.py`
- **Exports:** `exports/serve-hyb-1b7-s2000` (+`-llm`) and `exports/serve-bench-qwen3vl2b-1p6mp` (symlinks to the HF weights plus a pixel-capped preprocessor config).
- **Reports:**
  - `reports/serving/parity-serve-hyb-1b7-s2000.{json,log}`
  - `reports/serving/bench-serve-hyb-1b7-s2000-{bf16,fp8}{,-rep2}.json`
  - `reports/serving/benchmarks-quick{,-mmmu}.json`, `reports/serving/benchmarks-quick.md`
  - `reports/benchmarks/{serve-hyb-1b7-s2000,Qwen3-VL-2B-Instruct,serve-bench-qwen3vl2b-1p6mp}-quick*/`
- **Logs:** `logs/serving-bench/`
