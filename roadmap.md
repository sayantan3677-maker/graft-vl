# Roadmap: Converting Text-Only Qwen3 into a Production-Grade Vision-Language Model

**Hardware target:** 1× RTX PRO 6000 Blackwell (96 GB card, ~40–50 GB planned for this project)
**Backbones:** Qwen3-1.7B → Qwen3-4B → Qwen3-8B (text-only, hybrid-thinking)
**Vision tower:** SigLIP2 So400m patch16
**Last revised:** 2026-10-07 (v5: hybrid Qwen3-VL-vision architecture adopted, §3.5)

---

## 0. Summary

### Goal

Expand a strong post-trained, text-only Qwen3 model into a vision-language model that:

1. is **good enough to ship** in a product (measured on our own product eval set, not only public benchmarks),
2. keeps the original model's text, reasoning, coding, instruction-following and thinking-mode behavior, and
3. is reproducible on a single workstation GPU.

Research question:

> How far can a post-trained reasoning LLM be expanded into a capable VLM on one GPU without catastrophic regression of its original capabilities?

### What "SOTA" means for this project

Be precise about this, because it drives every decision.

- Native VLMs of the same size (Qwen3-VL-2B/4B/8B, newer Qwen multimodal releases) were trained on **trillions** of multimodal tokens. This plan uses **~1–3B** Stage-2 tokens per model. We should **not** expect to beat them on broad public benchmarks (MMMU, MMBench, OCRBench) at equal size.
- Realistic "state of the art" claims for this project:
  - **Best text retention** of any converted/modality-expanded model at its size (text scores statistically indistinguishable from the original).
  - **Competitive or better on our product's task distribution**, via targeted data (domain SFT is where a small budget wins).
  - **Best quality per training GPU-hour** among published text→VLM conversion recipes.
- Every gate therefore compares against a **reference ceiling**: the best open native VLM of the same size, evaluated on the same suites *and* the product eval set (§7.6). The ship decision (§14) is made against that ceiling.

### Key decisions (frozen unless an ablation overturns them)

| Area | Decision |
|---|---|
| Backbones | Qwen3-1.7B (science + full FT), 4B (main LoRA model), 8B (LoRA scaling check) |
| Checkpoints | Original hybrid-thinking Qwen3; never mix with `-2507` split variants |
| Vision | **Qwen3-VL-2B vision encoder + merger + DeepStack** (Apache-2.0), attached to the text-only Qwen3 LM inside the Qwen3-VL architecture ("hybrid", §3.5). The original SigLIP2 design is kept as the documented ablation that failed to learn reading (§16) |
| Visual tokens | 2×2 reduction → 144 (384 px) or 256 (512 px) |
| Connector | LayerNorm-free 2-layer MLP first; MLP+LN and resampler as ablations |
| Special tokens | Reuse Qwen's `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|box_start|>`, `<|box_end|>`; these rows are untrained and identical in Qwen3, so boundary/box rows get a learnable row-delta (§3.3) |
| Sample construction | Always via `tokenizer.apply_chat_template` |
| Stages | S1 align → M1.5 resolution freeze → S2 mid-train → S3 SFT → (S4 DPO, optional) → retention merge |
| Text retention | Per-size self-distilled replay (20%), noise-aware gates, interpolation / LoRA-strength sweep |
| Thinking | Short answers in non-thinking format; 10–15% verified thinking traces in S2 and SFT |
| Precision | 1.7B: FP32 master + FP32 AdamW. 4B/8B: BF16 + LoRA. No plain-BF16 full FT |
| Loss | Fused/chunked linear cross-entropy (151,936-token vocab) |
| Attention | PyTorch SDPA default; FA2 only if proven; never FA3 (Hopper-only) |
| Primary scaling comparison | Matched Stage-2 token budget, LoRA at all three sizes |
| Product inputs | Any image: screenshots (web/mobile/desktop), photos, page snapshots (incl. long full-page captures) |
| Data | Real images first; synthetic images ≤ 20% of any stage; model-generated answers only if verified |
| Serving | Hybrid is a stock Qwen3-VL checkpoint → **native vLLM** (no plugin); the SigLIP2 plugin remains for the ablation path. FP8 after parity eval |

---

## 1. Hardware and Environment

### 1.1 Observed machine state (2026-09-23)

| Item | Value |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB each |
| Driver / CUDA | 590.48.01 / 13.1 |
| Current usage | **GPU0: ~90.6 GB used, 99% util; GPU1: ~91.6 GB used, 100% util** |
| Occupants | 5 root-owned `VLLM::EngineCore` processes (3 running 18–21 days, 2 started ~3 h earlier) |
| Free right now | **~6 GB per GPU** |
| CPU / RAM / disk | 64 cores, 503 GB RAM, 2.3 TB free on `/home` |
| Python | Miniconda, Python 3.13 base; no torch in base |

**Allocation:** the project owner will free a GPU for this work before the GPU parts of Day 0. Until then, only CPU-side work (environment setup, code, data pipeline) runs. The Day-0 gate re-checks free memory and utilization before any training. If the card ends up shared again, record the contention factor.

Budget both **VRAM** and **compute share**. If average compute availability is ~50%, wall time roughly doubles. Free VRAM says nothing about compute availability.

### 1.2 Software environments

Two separate conda environments, because vLLM pins its own torch version:

| Env | Purpose | Key packages |
|---|---|---|
| `vlm-train` | training, unit tests, custom eval | torch (CUDA ≥ 12.8 wheels, required for Blackwell sm_120), transformers, accelerate, peft, torchao, liger-kernel, datasets, webdataset, pillow, wandb |
| `vlm-eval` | vLLM offline generation, text evals, teacher/replay generation | vllm, lm-eval, lmms-eval / VLMEvalKit |

Python 3.12 for both (3.13 wheel coverage is still patchy for CUDA extensions).

### 1.3 Day-0 environment gate (mandatory)

Record: GPU model, compute capability, driver, CUDA, torch, transformers, triton, torchao, liger-kernel, bitsandbytes (if used), vLLM, attention backend.

Run and pass all of the following:

| Check | Pass criterion |
|---|---|
| SDPA forward/backward, BF16 | no errors, grads finite |
| Variable-length + packed batches | loss identical to unpacked reference within tolerance |
| Packed attention isolation | perturbing sample B does not change sample A's logits |
| Gradient checkpointing | same loss/grad as without, within tolerance |
| Liger fused linear CE (or chunked CE fallback) | loss/grad match naive CE within tolerance |
| Optimizer path (FP32 master AdamW; torchao BF16 stochastic-round AdamW) | 100 steps, no NaN, loss decreases |
| 100-step end-to-end smoke run (Qwen3-0.6B/1.7B) | clean, peak VRAM recorded |

"It installs" is not a pass; every backend must survive forward, backward and 100 optimizer steps.

### 1.4 Backend policy

- **Attention:** PyTorch SDPA first; FlexAttention for packed/custom masks; FlashAttention 2 only if it passes the Day-0 gate on this exact stack. FlashAttention 3 is Hopper-only and is not used.
- **Cross-entropy:** Qwen3 vocab is 151,936. FP32 logits for one 16K-token sequence are `16,384 × 151,936 × 4 B ≈ 10 GB` (8K ≈ 5 GB). Never materialize the full `[B, T, V]` tensor on long sequences: use Liger `FusedLinearCrossEntropy`, or a chunked fallback.
- **bitsandbytes:** optional; only after a Blackwell smoke test; never a dependency of the 1.7B full-FT path.

### 1.5 Optimizer precision rule

At LRs like 2e-6 to 1e-5, most updates to BF16 weights are smaller than BF16 resolution (a weight near 0.02 has a spacing of ~1e-4) and are silently rounded away. **Never do full-parameter training with plain BF16 weights and deterministic updates.**

| Setup | Precision policy | Model-state memory |
|---|---|---|
| 1.7B full FT | BF16 autocast, FP32 master weights, FP32 AdamW | ~16 B/param ≈ 27 GB |
| 4B full FT (optional) | BF16 weights + stochastic rounding (torchao `torchao.optim._AdamW(bf16_stochastic_round=True)`, verified in torchao 0.18) or Kahan (optimi), 8-bit/low-mem states | ~20–24 GB + activations |
| 4B / 8B LoRA | BF16 frozen base; FP32 LoRA + projector params and states | base 8 / 16 GB + small |

---

## 2. Models and Checkpoints

### 2.1 Ladder

| Backbone | Hidden | Tied embeddings | Role | Main adaptation |
|---|---|---|---|---|
| Qwen3-1.7B | 2048 | yes | architecture/debugging, core science | full FT (+ LoRA comparison) |
| Qwen3-4B | 2560 | yes | main research and product-candidate model | BF16 LoRA (optional full FT) |
| Qwen3-8B | 4096 | no | scaling validation / higher-quality product candidate | BF16 LoRA |

14B/32B are out of scope locally; future work with external compute only.

### 2.2 Checkpoint policy

- Use the **original hybrid-thinking** Qwen3 releases at every size. Never mix with the `-2507` Instruct-only / Thinking-only variants inside one experiment series (different post-training, inference behavior and baseline).
- Use the post-trained models, not Base, for the main study. A Base-model ablation is optional and possible only where a Base checkpoint exists at that size.
- Pin for every run: repo, revision SHA, tokenizer revision, chat template hash, generation config.

### 2.3 Reference ceilings (for product decisions)

Evaluate once, with the same harness and the same product eval set:

- Qwen3-VL at 2B / 4B / 8B (Instruct and Thinking), and the latest same-size native Qwen multimodal release if newer.
- Record their text-benchmark scores too. Native VLMs often regress on text; that gap is part of our story.

---

## 3. Architecture

### 3.1 Pipeline

```text
image
  → SigLIP2 So400m patch16 (frozen initially)
  → penultimate hidden states, pre-pooling-head  [N_patches × 1152]
  → 2×2 spatial reduction (pixel-unshuffle concat → 4×1152)
  → MLP projector: Linear(4608 → d_llm) → GELU → Linear(d_llm → d_llm)
  → replaces <|image_pad|> embeddings inside the Qwen3 chat template
  → Qwen3 decoder → text
```

Deliberately **not** in v1: Q-Former, DeepStack / multi-level injection, multi-encoder fusion, MRoPE changes, video, grounding heads.

### 3.2 Resolution and token budget

Use real SigLIP2 patch16 checkpoints. There is no 448 px checkpoint, and patch14-384 gives an odd 27×27 grid that 2×2 reduction can't split evenly.

| Checkpoint | Grid | Patches | After 2×2 | Use |
|---|---|---|---|---|
| `google/siglip2-so400m-patch16-384` | 24×24 | 576 | **144** | default, general images |
| `google/siglip2-so400m-patch16-512` | 32×32 | 1024 | **256** | OCR / documents / charts |
| `google/siglip2-so400m-patch16-naflex` | variable | variable | variable | optional dynamic-res (M1.5-C) |

**Tiling is required for the product.** Screenshots and page snapshots are text-dense and often have extreme aspect ratios (a full-page capture can be 1:5 or taller). A single 384/512 crop can't read them. Plan:

- Measure the single-crop 384 vs 512 gain first (M1.5 A/B). This is still the right first step.
- Then **M1.5-D (required):** 512 px tiles on an aspect-ratio-preserving grid plus a 384 global thumbnail, capped at **N_max tiles** (start with 6 → 144 + 6×256 = 1,680 visual tokens). Long pages are split vertically into tiles at native width; anything beyond N_max is downscaled.
- Per-image token budget becomes an input-dependent variable: photos usually 1 crop (144–256 tokens), screenshots 2–4 tiles, long pages up to N_max. Report accuracy vs. tokens per image type.
- Keep the tiling logic simple and deterministic (fixed tile size, grid search over aspect ratio), because it has to be reimplemented exactly in the vLLM processor (§12.3).

Vision tower memory: ~400M params ≈ 0.8 GB BF16. The published checkpoint file is much larger because it includes the text tower and FP32 storage; don't budget that.

### 3.3 Special tokens

```text
<|im_start|>user
<|vision_start|><|image_pad|>×144<|vision_end|>
What is shown here?<|im_end|>
<|im_start|>assistant
...
```

```python
embeds = llm.get_input_embeddings()(input_ids)
mask = input_ids == image_pad_id
embeds[mask] = projected_visual_tokens.to(embeds.dtype)   # count must match exactly; assert it
```

**Reserved-row finding (Day 0, `scripts/check_qwen_assumptions.py`):** in Qwen3-0.6B and 1.7B, `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|box_start|>` and `<|box_end|>` are **bit-identical vectors** (never trained), with norm ~0.39 vs. a vocabulary mean of ~1.54 (≈ −4.5σ). Consequences:

- The model can't tell `<|vision_start|>` from `<|vision_end|>` by content.
- With tied embeddings (1.7B/4B), the output logits for `<|box_start|>` and `<|box_end|>` are always identical, so the model **cannot learn to emit grounding boxes correctly** unless these rows are changed.

Policy:

1. Implement special rows as a **learnable row-delta**: `E_eff = E_frozen + Δ`, where Δ is non-zero only for `vision_start`, `vision_end`, `box_start`, `box_end`. With tied weights the same `E_eff` also produces the output logits, so input and output stay consistent. The rest of the embedding matrix is untouched.
2. Initialize each delta so the row starts at a **distinct** vector with vocabulary-typical norm (e.g. the mean embedding plus a small distinct random direction, or the embedding of a related word such as "image"/"box").
3. Train `vision_start`/`vision_end` deltas from S1; `box_start`/`box_end` deltas from the first stage that contains grounding data. `<|image_pad|>` is always replaced and stays untouched.
4. For 8B (untied), check the `lm_head` rows separately; if they're also identical, box tokens need an output-row delta too.
5. The same check runs for every new backbone before its S1.

### 3.4 Vision-tower source (product lever, ablation in M1.5)

The research question is about the *LLM* being text-only, so the vision tower is a free choice. Besides SigLIP2, test the **vision encoder from an open Qwen3-VL checkpoint** of matching size (Apache-2.0, already trained heavily for VLM use). Expect a large quality gain for little compute. It needs its own preprocessing, 2D positions and merger, so it's an M1.5 ablation, not the v1 default. Report both variants.

### 3.5 Hybrid: Qwen3-VL vision on the text-only Qwen3 LM (adopted 2026-10-07)

Why: the SigLIP2 path (§3.1) never learned to read (DocVQA ≤ 2.5% after 0.5B tokens; single-word probe plateaued at ~43–53%), although letters are linearly decodable from its features (89%). Caption-only S1 alignment taught a *semantic* channel (BANANA → "CEREAL"); recovering spelling would need millions of OCR samples.

Construction (`scripts/build_hybrid.py`):
- Qwen3-VL-2B architecture; vision encoder, merger and DeepStack mergers from Qwen3-VL-2B-Instruct.
- **Every** language-model weight from text-only Qwen3-1.7B (same shape: hidden 2048, 28 layers, tied), RoPE θ kept at Qwen3's 1e6. Interleaved MRoPE gives text tokens equal (t,h,w) positions → identical to 1D RoPE: **text-only logits match Qwen3-1.7B exactly in FP32 (max |Δ| = 0.0)**.
- Only `<|vision_start|>`/`<|vision_end|>` rows are borrowed from Qwen3-VL (untrained in Qwen3).

Training: (1) alignment — merger + DeepStack mergers, LM frozen (text behaviour unchanged by construction); (2) S2 — LoRA r32 on the LM + merger, 20% self-distilled replay, 5% verified thinking traces, 5% synthetic reading; LoRA strength scaled at export for retention (§5.7). Images use Qwen3-VL's native dynamic resolution (pixel cap 1.2–1.6 MP), preprocessed on GPU (`vlm/hybrid/data.py`).

---

## 4. Data

### 4.1 Pipeline

```text
raw sources → license/commercial-use check → image decode validation
  → exact dedup (SHA256) → perceptual dedup (pHash + SigLIP embedding)
  → text dedup → quality filtering → task + thinking-mode labeling
  → difficulty scoring → benchmark contamination filtering
  → mixture weighting → versioned manifest (hash)
```

Every run references a **manifest hash**, not dataset names. Each record carries at least:

```json
{"source": "...", "license": "...", "commercial_ok": true, "split": "train",
 "task_type": "doc_qa", "thinking": false, "has_ocr": true, "has_grounding": false,
 "quality_score": 0.91, "difficulty": 0.73, "n_text_tokens": 212, "n_visual_tokens": 144}
```

### 4.2 Licensing (required because this ships in a product)

- Qwen3 and SigLIP2 are Apache-2.0. Qwen3-VL open weights are Apache-2.0 (verify per checkpoint).
- Public multimodal datasets are **mixed**: several aggregated collections include non-commercial or research-only sources, GPT-generated data under provider terms, or web images with unclear rights. Filter per source, not per collection.
- Teacher-generated data inherits the teacher model's license terms; use Apache/MIT-licensed teachers.
- Keep a `LICENSES.md` table (source → license → commercial_ok → decision) and get a legal sign-off before the product model is trained. Train research runs on the full mix if useful, but the **product checkpoint trains only on `commercial_ok=true`**.

### 4.3 Candidate sources

| Stage | Sources (subset, filtered, train splits only) |
|---|---|
| S1 alignment | LLaVA-Pretrain-558K, FineVision caption subsets, PixMo-Cap |
| S2 mid-training | FineVision, LLaVA-OneVision data, Cambrian-7M, PixMo (Cap/AskModelAnything/Points), Docmatix, OCR sets, DocVQA/TextVQA/InfoVQA/ChartQA **train**, diagram/STEM sets, RefCOCO-family **train**, verified synthetic |
| S3 SFT | curated high-quality subset of the above + product-domain data + verified thinking traces + replay |

Do not concatenate blindly. Prefer smaller, verified, balanced mixtures.

**Real-data policy (product requirement).** The product takes any image: screenshots, photos, page snapshots. Training data should reflect that with **real images**:

| Image type | Real-image sources to evaluate (verify each license before use) |
|---|---|
| Photos | COCO, Open Images, PixMo (human-written dense captions), Visual Genome, TextVQA/TextCaps (scene text) |
| Mobile UI screenshots | RICO-based sets (Screen2Words, ScreenQA, widget captioning) |
| Web / desktop screenshots | Mind2Web, SeeClick / OS-Atlas-style GUI grounding data, real web captures we collect ourselves |
| Page snapshots / documents | DocVQA, InfoVQA, Docmatix (real PDF pages), our own full-page web captures |
| Charts / tables | ChartQA (real charts split), PlotQA, table images from real documents |
| Our product | real product images (with user consent and PII handling) as soon as they exist |

Rules:

- Synthetic **images** (rendered HTML, generated charts, synthetic documents) are allowed but capped at **≤ 20%** of any stage and tracked in the manifest (`image_origin: real | synthetic`).
- Model-generated **text** (captions, QA, reasoning) on real images is allowed only after verification (§4.9).
- **Own web captures:** a crawler that renders real public pages (full-page + viewport screenshots, desktop and mobile widths) is the cheapest way to get large amounts of real screenshot and page-snapshot data. Respect robots.txt and site terms, strip PII, and log the source URL per image.

### 4.4 Train/eval split policy

Only train splits of any benchmark family we report on. Never validation/test/hidden splits. Contamination filter every training source against all eval images (pHash + embedding similarity) and questions (n-gram / embedding).

### 4.5 Chat-template rules (all stages, including S1)

1. Build every sample with `tokenizer.apply_chat_template(...)`; never hand-assemble.
2. Short-answer data (captions, VQA, OCR, doc QA) is rendered with `enable_thinking=False`, so it includes the exact empty think block Qwen3's template inserts.
3. Thinking data keeps `<think>…</think>` only where the real template keeps it. Earlier assistant turns in multi-turn samples are stripped exactly as the template strips them.
4. `test_processor_roundtrip.py` enforces all three.

### 4.6 Thinking policy by stage

| Stage | Non-thinking | Verified thinking |
|---|---|---|
| S1 | 100% (LLM frozen; format consistency only) | 0% |
| S2 | 85–90% of the multimodal share | 10–15% of the multimodal share |
| S3 SFT | ~75–85% of the multimodal share | ~15–25% of the multimodal share |

Task category and thinking mode are independent labels.

### 4.7 Verified multimodal thinking traces

```text
image → open captioning VLM (dense description) + PaddleOCR (text)
      → original text-only Qwen3 (same size as the student) reasons over them
      → <think>…</think> + answer
      → keep only if the answer is verified (ground truth / programmatic check / verifier VLM / OCR consistency)
```

- Prompt the teacher to reason as if the visual facts are directly perceived. Reject traces mentioning "caption", "description", "OCR", "provided text", and semantic equivalents (regex + small classifier).
- **Per-size generation:** each student gets traces from its own original model to preserve its reasoning distribution. Sample up to n=8 candidates per question and record the acceptance rate. If Qwen3-1.7B's acceptance is below ~20% on a task family, raise n or drop that family for 1.7B. Using 8B-generated traces for 1.7B is an ablation, not the default.
- Captioner: Apache-licensed open VLM (e.g. Qwen3-VL-8B-Instruct). Choose by license, caption quality, throughput.

### 4.8 Text replay (per size)

- Shared prompt set: coding, math, reasoning, instruction following, writing, multi-turn, tool-style prompts; both thinking and non-thinking.
- **Responses regenerated separately by each size's original model** (1.7B, 4B, 8B). Replay must match the exact source model.
- Offline with vLLM; teacher and student are never resident together.
- Max replay length 8K tokens by default; 16K only in controlled experiments with fused CE; no 32K replay in the main stream.
- Optional KL distillation later: store top-k (k=32, compare 64) `(token_id, logprob)` per position, never full-vocab logits.

### 4.9 Synthetic data policy

Teacher VLM → candidate caption/QA → verifier or ground truth → quality filter → manifest. Reject unsupported answers. Never train at scale on unverified generations.

### 4.10 Grounding format (frozen before S2)

0–1000 integer coordinates, relative to the original image:

```text
<|box_start|>(120,220),(760,910)<|box_end|>
```

No grounding data enters S2 before this is frozen.

---

## 5. Training Stages

### 5.1 Overview

| Stage | Vision | Projector | LLM | Data | Goal |
|---|---|---|---|---|---|
| S0 | — | — | original | — | text baseline + noise |
| S1 | frozen | train | frozen | 200K–500K captions | alignment |
| M1.5 | — | — | — | small probes | freeze resolution, feature layer, ViT source, grounding format |
| S2 | frozen (→ top layers/LoRA ablation) | train | full FT (1.7B) / LoRA (4B, 8B) | 80% MM + 20% replay | capability building |
| S3 | frozen / low LR | train | same as S2 | curated SFT + product data + replay | assistant behavior |
| S4 (opt.) | frozen | frozen | LoRA DPO | preference pairs | hallucination / faithfulness |
| Merge | — | — | interpolate / scale LoRA | — | recover text on the Pareto front |

**Every model size runs its own S1**: the projector's output width equals that LLM's hidden size (2048 / 2560 / 4096), so projectors never transfer. The recipe transfers; the weights don't.

### 5.2 S1: alignment

- Trainable: projector only. Frozen: ViT, LLM, all embedding rows.
- Gradients still flow backward through the whole frozen LLM to reach the projector, so S1 costs about ⅔ of full training compute, not inference cost. Don't wrap the LLM in `no_grad`.
- Loss only on assistant tokens (system, user, vision, assistant prefix = −100).
- Funnel: **5K** (pipeline, masking, LR sweep 1e-4 / 3e-4 / 1e-3) → **50K** (image-reliance tests, stability) → **200K** (first real eval) → **500K** only if metrics still move.
- Config: micro-batch 2 (→4 if room), grad-accum 16, effective batch 32, warmup 3–5%, cosine.

### 5.3 M1.5: decisions before S2 (1.7B)

| Ablation | Options |
|---|---|
| Resolution / tokens | A: 384/144, B: 512/256, C (opt.): NaFlex, **D (required): tiles + thumbnail at native res, N_max = 6** (§3.2) |
| Feature layer | penultimate vs final |
| ViT source | SigLIP2 vs Qwen3-VL vision encoder (§3.4) |
| Connector | 2-layer MLP vs MLP+LayerNorm |

Probe subsets: general VQA, OCRBench, TextVQA, DocVQA, ChartQA, plus product eval. Outcome can be mixed (e.g. 384 for natural images, 512 for OCR/docs). Freeze the policy and the grounding format, then S2.

### 5.4 S2: multimodal mid-training

Stream composition:

```text
100% of training stream
├── 80% multimodal
│   ├── by task:  25% natural photos · 15% dense captions (photos + screens) · 15% screenshots/UI
│   │             15% page snapshots/documents/OCR · 8% charts/tables · 8% visual reasoning
│   │             7% grounding (UI elements + objects) · 3% diagrams/STEM · 4% multi-image
│   ├── by origin: ≥ 80% real images · ≤ 20% synthetic images
│   └── by mode:  85–90% non-thinking · 10–15% verified thinking   (independent of task labels)
└── 20% text replay (per-size self-distilled; thinking + non-thinking)
```

Starting hyperparameters (tune after the 1000-step calibration):

| | 1.7B full FT | 4B / 8B LoRA |
|---|---|---|
| LLM LR | 5e-6 | LoRA 1e-4 – 2e-4 (r=32, α=64, dropout 0.05, all attn+MLP proj) |
| Projector LR | 5e-5 | 5e-5 – 1e-4 |
| ViT top layers (if unfrozen) | ~1e-6 | frozen / vision LoRA |
| Micro-batch / accum | 1 / 16–32 | 1 / tuned for stable effective batch |
| Other | grad checkpointing, fused CE, SDPA, packing with isolation | same |

### 5.5 S3: multimodal SFT

Content: general VQA, multi-turn image chat, OCR, documents, charts, tables, diagrams/STEM, visual math, grounding, screenshots/UI, multi-image, **product-domain tasks**, text replay, thinking and non-thinking.

Starting mix (tune from observed weaknesses):

| Category | Share |
|---|---|
| General multimodal instruction (photos) | 20% |
| Page snapshots / documents / OCR | 15% |
| Screenshots / UI (understanding, QA, element grounding) | 15% |
| Charts / tables | 8% |
| Visual reasoning / math | 8% |
| Diagrams / STEM | 4% |
| Grounding (objects) | 5% |
| Multi-image | 5% |
| **Real product traffic** (once available; until then, own web captures) | 5% |
| Text replay | 15% |

### 5.6 S4 (optional): preference optimization

LoRA DPO is cheap on one GPU. Pairs: grounded vs hallucinated, correct vs guessed OCR, correct count vs language-prior count, faithful uncertainty vs fabricated certainty, correct vs unsupported chart reasoning. Run only after S3 is stable; must not regress text or helpfulness.

### 5.7 Retention recovery

- **Full-FT runs:** interpolate LLM weights only: `θ = α·θ_VLM + (1−α)·θ_original`, α ∈ {1.0, 0.95, 0.9, 0.85, 0.8}. Never interpolate projector or ViT.
- **LoRA runs:** scale the adapter: `W = W₀ + α·ΔW`, α ∈ {1.0, 0.75, 0.5, 0.25}.
- After every merge/scale, rerun image-reliance tests: a sudden visual collapse means projector–LLM alignment broke. Pick the checkpoint on the vision/text Pareto front.

---

## 6. Per-Size Plan

| Size | S1 | S2 (best-model run) | S3 SFT | Method |
|---|---|---|---|---|
| 1.7B | 200K–500K | 1–2B tokens (1–3M samples) | 300K–1M samples | full FT (+ LoRA matched run) |
| 4B | 50K sanity → 200K | 0.5–1B tokens | 150M–400M tokens | BF16 LoRA (opt. full FT with SR) |
| 8B | 50K sanity → 100K–200K | 0.2–0.5B tokens (more if product candidate) | 100M–250M tokens | BF16 LoRA |

If the product needs the highest quality, the 8B budget is raised after scaling validation; that decision is made at Gate 8B-S2.

---

## 7. Evaluation

### 7.1 Text suite (S0 and every gate)

MMLU-Pro, IFEval, GPQA-Diamond, AIME, LiveCodeBench (or practical coding subset), optional GSM8K/MATH, BFCL if tool use matters. Run both `enable_thinking=True` and `False`.

- Text evals run in vLLM on the LLM weights exported as a plain `Qwen3ForCausalLM`. This is exact: on text-only input the wrapper is bypassed.
- Decoding: the model-card sampling settings (thinking: temp 0.6 / top-p 0.95 / top-k 20; non-thinking: 0.7 / 0.8 / 20). Never greedy for thinking mode. Record all settings and seeds.

### 7.2 Noise-aware regression gates

- S0: run each benchmark with 4–8 seeds; record mean, std, CI.
- Large/stable (MMLU-Pro, IFEval): ~1-point warning threshold if the measured variance supports it.
- Small/noisy (GPQA-Diamond, AIME): regression flagged if `drop > 2σ_baseline`.
- AIME at gates: ≥16 samples/problem; report mean pass rate and variance.
- Also track thinking-mode adherence, response-length drift, formatting drift, refusal drift, code validity.

### 7.3 Multimodal suite

| Family | Benchmarks |
|---|---|
| General | MMBench, MMMU, MME, RealWorldQA |
| OCR / docs | OCRBench, TextVQA, DocVQA, InfoVQA |
| Charts | ChartQA |
| Reasoning | MathVista, ScienceQA-IMG |
| Hallucination | POPE, HallusionBench |
| Grounding | RefCOCO / RefCOCO+ / RefCOCOg |

Tooling: lmms-eval and VLMEvalKit adapters for our wrapper. vLLM for vision evals only if the Qwen3-VL compatibility spike (§7.7) passes.

### 7.4 Visual-reliance tests (every checkpoint, in CI)

Blind (no image) · random image · black image · zeroed visual embeddings · similar distractor image · cropped-out evidence · OCR text removed. Scores must drop in the expected way. If they barely move, the model is using language priors.

### 7.5 Thinking-with-images checks (every gate)

- Thinking-enabled image prompts still produce a `<think>` section of plausible length.
- Non-thinking image prompts match the template's non-thinking behavior.
- Reasoning has not collapsed into immediate short answers.

### 7.6 Product evaluation (the gate that decides shipping)

- **Product eval set:** 300–1,000 labeled examples drawn from the real product input distribution (images, prompts, expected outputs), frozen and never trained on. A disjoint product-train set feeds S3.
  - Stratified by image type: ~⅓ screenshots (web, mobile, desktop), ~⅓ photos, ~⅓ page snapshots (including long full-page captures).
  - Stratified by task: describe, read/transcribe text, question answering, find/locate a UI element or object, multi-step reasoning, compare images.
  - Report every metric per image type as well as overall. A model that is strong on photos but weak on page snapshots isn't shippable.
- **Metrics:** task accuracy/quality (rubric or programmatic), hallucination rate, format/schema validity, latency and cost per request at the chosen visual-token budget.
- **Safety:** refusal behavior on harmful image+text requests; **prompt injection via text inside images** (instructions embedded in screenshots/documents must not override the system/user prompt); privacy (no transcription of sensitive fields when policy forbids it).
- **Serving parity:** re-evaluate after export and after any serving quantization (e.g. FP8), since both can shift behavior.

### 7.7 Qwen3-VL class compatibility spike (early)

Instantiate a Qwen3-VL-compatible architecture, load text-only Qwen3 decoder weights, list missing/unexpected/mismatched keys, and compare text-only logits with the original (`max_abs_diff`). Pass → use it for vLLM multimodal serving and eval (big product win). Fail → keep the custom wrapper and plan a vLLM model plugin for serving.

### 7.8 Evaluation budget

| Checkpoint type | What runs |
|---|---|
| Intermediate (every N steps) | 100–500 ex/task: general VQA, OCR, DocVQA, product-eval subset, text subset (1–4 samples), image-swap, thinking-on-image |
| Gate | full required suites, noise-aware text eval, all reliance tests, full product eval |
| Final | everything above + 16+ AIME samples + hallucination + safety + serving parity |

Evaluation time is budgeted (§9.3), not free overhead.

---

## 8. Scaling Comparison Design

Two separate experiment families so size effects aren't mixed up with budget or method:

| Family | Purpose | Setup |
|---|---|---|
| **A. Matched-budget scaling (primary for claims)** | isolate model-size effect | 1.7B, 4B, 8B all BF16 **LoRA** (r=32), same S1 size (200K), same **0.3–0.5B** S2 tokens, same manifest and data order, same eval |
| B. Best-achievable per size | product / "what can each size do" | the per-size plan in §6 (1.7B full FT with more tokens, etc.) |
| C. Adaptation method | full FT vs LoRA effect | 1.7B full FT vs 1.7B LoRA at the same token budget; optional 4B full FT vs LoRA |

Always report stage, tokens, method and GPU-hours next to every score. Never compare different stages.

---

## 9. Memory and Compute Budget

### 9.1 Memory (planning values; decisions use measured peaks)

| Configuration | Main concern | Fits in 40–50 GB? |
|---|---|---|
| 1.7B S1 (projector only) | activations | easy |
| 1.7B full FT, FP32 master + FP32 AdamW | ~27 GB states + activations | target |
| 1.7B full FT + top ViT layers | + ViT states | likely, with checkpointing |
| 4B BF16 LoRA | activations | comfortable |
| 4B full FT, conventional FP32-master Adam | ~64 GB states | no |
| 4B full FT, BF16 + stochastic rounding + low-mem states | ~20–24 GB + activations | optional |
| 8B BF16 LoRA | 16 GB base + activations | target |
| 8B QLoRA | quantization changes baseline | fallback only (baseline = quantized model) |
| FP32 logits, 8K / 16K tokens | ~5 / ~10 GB transient | avoid via fused/chunked CE |

### 9.2 Throughput calibration

Before every major stage and model size: 1,000 representative optimizer steps. Record mean/p50/p95 step time, tokens/s (text, visual), samples/s, peak allocated/reserved VRAM, GPU util, power, CPU util, dataloader wait. This replaces all theoretical estimates.

Pre-calibration sanity range: 1.7B full FT ~10–15K tokens/s on an uncontended GPU; 4B ~2.5× slower; 8B LoRA ~3–5× slower than 1.7B.

### 9.3 Project budget (uncontended single GPU, before calibration)

| Milestone | Model | Work | GPU-days |
|---|---|---|---|
| Day 0 | 0.6B/1.7B | environment gate | 0.5–1 |
| M0 | 1.7B/4B/8B + ceilings | text baselines with multi-seed noise, reference-ceiling evals | 1–3 |
| Data gen | all | captioning, OCR, trace generation (n≤8, rejection), **per-size** replay (×3) | 3–6 |
| M1 | 1.7B | S1 funnel + LR sweep | 0.3–2 |
| M1.5 | 1.7B | resolution / feature / ViT-source / connector probes | 1–3 |
| M2 | 1.7B | S2 best run (1–2B tokens) | 1–4 |
| M2-abl | 1.7B | sequential ablations (replay ratio, ViT adaptation, full vs LoRA) | 4–10 |
| M2-SFT | 1.7B | S3 | 1–3 |
| M3 | 4B | S1 + S2 (0.5–1B) + S3 | 3.5–11 |
| M4 | 8B | S1 + S2 (0.2–0.5B) + optional S3 | 3–10 |
| Scaling A | 1.7B/4B/8B | matched-budget LoRA runs not already covered | 2–5 |
| S4 (opt.) | best | LoRA DPO | 1–2 |
| Evaluation | all | gates + final + product + safety | 3–6 |
| **Total** | | | **~25–65 GPU-days** |

That is roughly **1–2.5 months of exclusive single-GPU time**. Multiply by the measured contention factor if the GPU is shared.

---

## 10. Gates

| Gate | Must pass |
|---|---|
| Day 0 | all §1.3 checks; optimizer + fused/chunked CE verified; freed GPU confirmed |
| M0 | text baselines + variance recorded (both thinking modes); reference ceilings evaluated; product eval set frozen |
| S1 (each size) | coherent image-conditioned output; image-swap / zero-feature / blind tests clearly worse; loss mask verified; text-only logits equivalent (unit test) |
| M1.5 | resolution, token budget, feature layer, ViT source, connector, grounding format frozen |
| S2 (each size) | MM score clearly above S1; reliance tests hold; text within noise-aware bounds; thinking-with-images works; OCR/docs improve if hi-res; no hallucination regression |
| S3 | competitive with ceiling on product eval, or gap quantified; instruction following robust; text retention holds |
| S4 | hallucination down, no helpfulness or text regression |
| Ship | see §14 |

---

## 11. Ablations (sequential, mostly at 1.7B)

Order: stable baseline → resolution → feature layer → ViT source → connector → S1 projector LR → text-replay ratio (0/10/20/30%) → ViT adaptation (frozen / top layers / LoRA) → full FT vs LoRA → LoRA rank (16/32/64) → scale to 4B → validate at 8B.

No combinatorial sweeps. Each step fixes one choice before the next.

---

## 12. Engineering

### 12.1 Repository layout

```text
vlm_training/
├── roadmap.md
├── LICENSES.md                 # data/model license decisions
├── configs/{model,data,train,eval}/
├── src/vlm/
│   ├── model/      vision_encoder.py  projector.py  token_reducer.py  multimodal_qwen.py
│   ├── data/       processor.py  datasets.py  collator.py  packing.py  mixture.py
│   ├── training/   train.py  (stage selected by config)  merge.py
│   └── eval/       visual_reliance.py  text_regression.py  product_eval.py  adapters.py
├── scripts/        day0_check.py  export_llm.py  gen_replay.py  gen_traces.py
├── tests/          test_loss_mask.py  test_text_only_equivalence.py  test_image_token_replace.py
│                   test_attention_isolation.py  test_processor_roundtrip.py  test_fused_ce.py
├── manifests/  experiments/  reports/
```

### 12.2 Experiment tracking

Immutable run IDs, e.g. `EXP-M2-Q3-1B7-S2P16-384-144T-FULL-A03`. Store: git SHA, model/tokenizer revisions, vision checkpoint + feature layer, resolution, visual tokens, manifest hash, mixture + thinking ratio, precision policy, optimizer, LRs, batch config, attention + loss backend, tokens trained, wall time, GPU-hours, peak VRAM, all eval results. Results live in `reports/`, not only in W&B.

### 12.3 Serving and inference

**Engine choice**

| Engine | Use | Why |
|---|---|---|
| **vLLM** (primary) | product serving | continuous batching, paged KV cache, prefix caching, OpenAI-compatible API, native multimodal inputs, FP8/NVFP4 on Blackwell. Already the engine running on this machine |
| SGLang | optional alternative | similar performance; only if it benchmarks clearly better for our traffic |
| llama.cpp | on-device / CPU only | needs our projector implemented in its C++ vision code (`mtmd`/`clip`) and GGUF conversion; not worth it unless we ship on-device |
| HF transformers | reference only | correctness baseline for parity tests; too slow to serve |

**How our model runs in vLLM**

Our architecture (SigLIP2 → 2×2 unshuffle → MLP → Qwen3) isn't a stock vLLM architecture. The Qwen3-VL class can't host it either, because it has its own vision tower. So the primary path is a **vLLM out-of-tree model plugin** (registered through vLLM's plugin entry point):

- reuse vLLM's existing Qwen3 decoder and SigLIP vision modules; add our unshuffle + MLP projector (~200–300 lines),
- a vLLM multimodal processor that reproduces our preprocessing and tiling exactly,
- special-token row-deltas are merged into the embedding matrix at export, so serving needs no special handling.

The §7.7 Qwen3-VL spike is still worth running: if we adopt the Qwen3-VL vision encoder (§3.4), the whole model can export as a stock Qwen3-VL checkpoint and needs no plugin.

**Parity and performance (release gate)**

- `test_vllm_parity`: same images + prompts → logits/greedy outputs from HF wrapper vs. vLLM match within tolerance, for each image type and tiling layout.
- Quantization: BF16 → FP8 (W8A8, e.g. via llm-compressor) is the default serving format on Blackwell; NVFP4 is optional. Rerun the full product eval + text regression after quantization.
- Measure at realistic concurrency: time-to-first-token (dominated by visual tokens: 144 for a photo vs. ~1,700 for a long page), output tokens/s, throughput per GPU, memory per request. Set `limit_mm_per_prompt` and a max tiles cap per request.
- Thinking mode costs latency: expose it as a per-request option and set a default per product surface.

**Export artifacts:** HF-format weights (row-deltas merged), processor config, chat template, generation config, vLLM plugin package, model card with data licenses and eval results.

---

## 13. Execution Order

```text
DAY 0
 1. confirm the freed GPU (free memory + utilization) and pin CUDA_VISIBLE_DEVICES
 2. create vlm-train / vlm-eval envs; pin versions
 3. day0_check.py: SDPA, packing isolation, grad ckpt, fused CE, optimizers, 100-step smoke
 4. reserved-row norm check; Qwen3-VL compatibility spike

M0
 5. text baselines (1.7B/4B/8B, thinking on/off, multi-seed) → noise-aware gates
 6. reference-ceiling evals (Qwen3-VL same sizes)
 7. define and freeze the product eval set; start LICENSES.md

DATA
 8. build S1 manifest; start S2/S3 manifests with license + contamination filters
 9. generate per-size replay; generate + verify thinking traces

1.7B
10. S1 funnel 5K → 50K → 200K (→500K) with LR sweep and reliance tests
11. M1.5 decisions (resolution, feature layer, ViT source, connector, grounding format)
12. S2 full FT (1–2B tokens) → Gate
13. sequential ablations
14. S3 SFT (incl. product data) → Gate

4B / 8B
15. per-size S1 → S2 (LoRA) → S3; matched-budget scaling family A
16. optional S4 DPO on the best candidate

FINAL
17. retention merge / LoRA-strength sweep → Pareto choice
18. full text + vision + product + safety + serving-parity evaluation
19. ship decision (§14); ablation report
```

---

## 14. Ship Decision

The product model is the best checkpoint on the **product eval set** that also:

- passes all safety and image-prompt-injection checks,
- keeps text within the noise-aware bounds,
- meets the latency/cost target at its visual-token budget and serving precision,
- was trained only on `commercial_ok` data.

If our best checkpoint is still clearly below the same-size native ceiling (§2.3) on the product eval after S3, the honest options are: (a) ship the fine-tuned native model for now while this research continues, or (b) move up a size (8B with a larger budget) or take the Qwen3-VL vision-encoder route. Decide with numbers, at the S3 gate.

---

## 15. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| GPU shared with inference again later | Day-0 re-check; record contention factor; memory cap on co-tenants |
| Converted model lags native VLMs on broad benchmarks | product-focused SFT; Qwen3-VL ViT lever; explicit ceiling comparison |
| License contamination in product data | per-source filtering, `commercial_ok` gate, legal sign-off |
| Thinking mode collapses on image inputs | S2/S3 thinking shares, thinking-with-images gate |
| Low trace acceptance at 1.7B | n≤8 sampling; drop families; 8B-trace ablation |
| Blackwell kernel issues | Day-0 gate, SDPA default, chunked-CE fallback |
| Benchmark noise hides regressions | multi-seed baselines, 2σ gates |

**Answered:** product inputs are any image: screenshots, photos, page snapshots. Use a mix of real data (§4.3). Serving is vLLM (§12.3).

**Answered (2026-09-23):**

- **Latency:** production-level. Cost scales with local GPU count, which can grow later. Default to non-thinking for interactive requests, N_max tiles tuned to keep time-to-first-token interactive (measured in the vLLM benchmark), thinking as an opt-in per request.
- **Product images:** none available yet. Train on public datasets (§4.3). The product eval set (§7.6) is built from **held-out public data plus our own web captures**, stratified the same way (⅓ screenshots, ⅓ photos, ⅓ page snapshots), and replaced/extended with real product images once they exist.
- **GPU:** GPU 1 may be freed later (not guaranteed). All code is built and unit-tested on CPU in the meantime; the Day-0 GPU gate runs as soon as a GPU is available.

---

## 16. Status Log

| Date | Item | Result |
|---|---|---|
| 2026-09-23 | `vlm-train` env | torch 2.11+cu128, transformers 5.17, torchao 0.18, liger-kernel 0.8.3, peft 0.21 |
| 2026-09-23 | Qwen3 assumption check (CPU) | template behavior as assumed; vision/box special rows untrained and identical → row-delta policy (§3.3) |
| 2026-09-23 | Core code | model wrapper, SigLIP2 encoder (penultimate, pre-head), 2×2 unshuffle, MLP projector, special-row deltas, tiling, chat-template processor, packing (bool block-causal mask), chunked CE, WebDataset loader, stage trainer |
| 2026-09-23 | CPU unit tests | 28/28 pass (template token equality, loss mask, packing isolation, merge equivalence tied/untied, S1 trainability) |
| 2026-09-23 | Real-model smoke (CPU) | Qwen3-1.7B + SigLIP2-384 loads with no missing vision keys; 2.13B params, 13.6M trainable in S1; finite loss |
| 2026-09-23 | S1 debug manifest | `s1-debug200`: 120 photo / 30 screenshot / 50 page, 5% synthetic, hash `163476a76ab6b0d0` |
| 2026-09-24 | Day-0 GPU gate (GPU 1, Qwen3-0.6B) | **ALL PASS**. Packing: zero leakage; FP32 packed≈alone (≤1.4e-4); BF16 diff within masked/unmasked kernel noise floor. Liger fused CE works on sm_120 (peak 0.02 GB vs 6.96 GB naive). Plain BF16 AdamW at LR 5e-6 left **93.7% of weights unchanged** (confirms §1.5) |
| 2026-09-24 | 1.7B full-FT memory | FP32 master + FP32 AdamW: **26.8 GB** peak at 2×256 tokens (roadmap estimate ~27 GB) |
| 2026-09-24 | Real VLM on GPU | fwd+bwd 1.2 s for a 3-tile screenshot; all SigLIP2 vision weights loaded |
| 2026-09-24 | S1 data | `s1-5k` (3000 photo / 750 screenshot / 1250 page, 5% synthetic, hash `1652be6a5bc92528`, 206 long OCR targets rejected); `s1-eval300` held-out (seed 1, hash `0b2f38ff6a1be202`); `s1-50k` building |
| 2026-09-24 | GPU contention | a root vLLM server (PID 628486) started on GPU 1 ~12 min after Day 0, taking 46 GB at ~100% util; S1 calibration stopped until the GPU is free again |
| 2026-09-24 | Loader deadlock | WebDataset split shards twice (node + worker splitter) → only worker 0 had data, training never started. Fixed; regression test added |
| 2026-09-24 | Packed-attention speed | dense 8K×8K bool mask forced slow SDPA kernel (5.0K tok/s). Switched to **FlexAttention** document-causal BlockMask: zero leakage, loss matches SDPA to 2e-4, S1 throughput **13.8K tok/s** (2.7×). Grad checkpointing off for S1 (22 GB peak) |
| 2026-09-24 | SigLIP2 outlier channel | penultimate features: channel 514 reaches \|x\| ≈ 2,112 (median channel max 7.8); constant energy > across-image energy |
| 2026-09-24 | S1 LR sweep on 5K (1 epoch, 170 steps) | 1e-4 / 3e-4 / 1e-3, with and without projector LayerNorm: model needs *an* image (blind gap ≈ 1.0) but ignores *which* image (swap gap 0.002–0.008). **Fails the gate at every setting** |
| 2026-09-24 | Overfit-64 diagnostic | 300 steps on 64 samples: swap gap **+0.98** (photo 1.08, screenshot 1.61, page 0.67), generations image-specific → pipeline transmits image content correctly; 5K/170 steps was simply too little signal |
| 2026-09-24 | S1 50K run (384, MLP, LR 1e-3, 1,700 steps) | swap gap: photo **+0.087 → +0.349 → +0.405 → +0.443** (steps 425→1700), screenshot +0.018, page +0.043. Photo captions now image-grounded. **S1 accepted for 1.7B**; screenshot/page grounding needs LLM training (S2) and higher resolution (M1.5) |
| 2026-09-24 | vLLM on Blackwell | vLLM 0.30 / torch 2.13+cu130 works; FlashInfer's JIT sampler misdetects sm_120 → `VLLM_USE_FLASHINFER_SAMPLER=0` set in all vLLM scripts |
| 2026-09-24 | S2 prep (queued on GPU 1) | replay 40K (Qwen3-1.7B) → trace describe 6K (Qwen3-VL-8B) → trace reason (Qwen3-1.7B) → text baselines; S2 multimodal build `s2-mm-300k` running |
| 2026-09-24 | M1.5 512 alignment | S1 at 512 px: photo swap gap +0.350 (384: +0.443); no screenshot/page gain |
| 2026-10-05 | Data gen results | replay 16.4K (45% thinking) but **no instruction-following/safety prompts** (category keys missed Tülu source names) → fixed, 5K IF+safety top-up generated; traces v1 kept only 14% (filter rejected the reasoner's references to its input) → prompt presents description as the image + reference rewriting + audit log → v2 68% → re-filtered v3: **1,704 clean traces** (min reasoning length, no "provided list"/"user mentioned" leaks) |
| 2026-10-05 | Eval contamination found | "held-out" QA built from the same FineVision train pool shared 1,476/1,500 images with training (many QAs per image). Replaced by **`eval-official`**: 1,400 questions from official val/test splits (DocVQA val, InfoVQA val, ChartQA test, TextVQA val, VQAv2 val, AI2D test, OCRBench-v2 screenshots); exact-hash overlap filter still drops 118 that FineVision includes |
| 2026-10-05 | M1.5 pilots (800 steps S2 full FT), official splits | 384 frozen 14.4 / 512 frozen 15.4 / 384 + top-9 ViT 14.7 overall; DocVQA 2.5–4, TextVQA 12, screens 9–12, VQAv2 ~29 — **no setting reads text at this budget**; differences within noise. Decision: 384 + 6 tiles, top-9 ViT layers trainable, much larger S2 budget, text-rich mix ×1.3 |
| 2026-10-05 | Serving parity | prompt ids identical 24/24; all 17 greedy divergences are near-ties (max gap 0.125 nats = BF16 noise) → **plugin PASS** |
| 2026-10-05 | Full S2 (1.7B) | `EXP-S2-Q3-1B7-384-FULLFT-VIS9-0.5B`: 15.6K steps ≈ 0.5B tokens, 77% MM (text-rich) / 3% traces / 15% replay / 5% IF+safety replay; 4.3 s/step, 38 GB; ETA 2026-10-06 ~03:40 UTC; checkpoints every 3.9K steps auto-scored on `eval-official` |
| 2026-10-05 | lmms-eval smoke | blocked by HF login (fixed with token), then OOM from co-scheduling with S2 start; runner now fails loudly. Rerun after S2 |
| 2026-10-06 | Full S2 (SigLIP2, 0.5B tok) | official-split QA flat at 12–15% across all checkpoints (DocVQA 1–2%); loss rose in 2nd half with grad-norm spikes to 223 |
| 2026-10-06 | Text regression of that S2 | vs original: IFEval −2.9/−3.2 (think/nothink), MATH-500 −4.5/−2.2, MMLU-Pro −1.6/−1.1, GSM8K −0.2/−2.5 → **fails retention gate** |
| 2026-10-06 | Text baselines (Qwen3-1.7B) | MMLU-Pro 56.5/41.4, GSM8K 90.4/81.2, IFEval 70.3/68.0, MATH-500 90.0/72.5, GPQA-D 39.3/30.9, AIME24/25 48.8/33.5 (think/nothink, 4–16 seeds) |
| 2026-10-06 | Reading diagnosis | single-word probe 0% with *semantic* substitutions; linear probe on frozen SigLIP2 features recovers letters at 89% → training, not features, is the bottleneck. LR 2e-5 pilot: no gain. Reading curriculum (200K synthetic, 386 Latin-complete fonts) + projector-only alignment: probe 0 → 43%; S2 pilot from it: no QA gain |
| 2026-10-07 | **Pivot: hybrid (§3.5)** | zero-shot (no training) Qwen3-1.7B + Qwen3-VL-2B ViT w/ DeepStack: 29.5% vs our best 14.4%; Qwen3-VL-2B ceiling 80.5% (30 Q/source) |
| 2026-10-07 | Hybrid alignment (merger only, 1.5K steps) | official splits (60 Q/source): 18.6 → 41.2 → 46.9 → **48.6%** (DocVQA 45, screens 68, TextVQA 68, ChartQA 38); text LM untouched |
| 2026-10-07 | Hybrid S2 (LoRA r32, 3K steps × 32 samples) | 7.9 s/step after moving image preprocessing to GPU (CPU contended: load 85/64 cores) and disabling grad-ckpt/LoRA dropout (1.7× faster). Official splits (60 Q/source): step 1K 52.6, **2K 55.5** (DocVQA 63.3, ChartQA 46.7, InfoVQA 31.7, screens 60, TextVQA 65, VQAv2 61.7, AI2D 60), 3K 54.5 |
| 2026-11-01 | Parallel workstreams (sub-agents) | quality-retention (GPU1 slot A): LoRA-strength sweep + text gate + full eval + thinking/injection; serving-bench (GPU1 slot B): native vLLM parity, BF16/FP8 latency, lmms-eval vs Qwen3-VL-2B; scaling-4b (GPU0): Qwen3-4B + Qwen3-VL-4B hybrid end to end. GPU access via per-slot `flock` locks |
| 2026-11-01 | Serving & benchmarks (serving-bench agent, `reports/agents/serving-bench.md`) | Hybrid serves natively in vLLM 0.30 (no plugin). Parity: prompt ids 48/48; teacher-forced vs FP32 ref vLLM-BF16 |Δlogp| 0.032 / top-1 97.4% = HF-BF16 (0.032 / 97.1%) → no path bug. Latency (shared GPU, lower bounds): TTFT p50 260 ms @c1 (photo 79 / screenshot 215 / page 415 ms), 609 out-tok/s @c32; FP8 only +5–10% throughput, accuracy unchanged (52.9→55.2, n.s.). lmms-eval (limit 500) ours vs Qwen3-VL-2B: DocVQA 62.5/93.3, InfoVQA 34.1/70.0, ChartQA 41.2/68.2, TextVQA 58.6/79.7, OCRBench 74.4/90.0, AI2D 57.2(lenient)/76.2, MMBench 68.6/81.4, POPE 80.8/88.0, RWQA 51.6/65.8, MMMU 38.6/39.8; pixel cap not the cause. Serving fixes: `scripts/prepare_hybrid_serving.py` (VL-aware Qwen3 template, nothink template, 1.6 MP cap, generation_config) must run after every export; thinking default ON → serve with enable_thinking=false for interactive; "Answer: X" format habit to fix in S3. **Main gap is model quality, not serving.** |
| 2026-11-20 | Retention gate, 4B scaling, release candidates (`reports/final-2026-11-20.md`) | 1.7B text (avg Δ vs Qwen3-1.7B over MMLU-Pro/GSM8K/IFEval/MATH-500 × think/nothink): s3000-l1.0 −1.1 (worst −2.4) with best vision 53.9/1400 → 1.7B candidate; LoRA down-scaling gives no clear text gain. 4B (Qwen3-4B + Qwen3-VL-4B ViT; FP32 text logits identical): s2000 vision 55.3 at both LoRA 1.0 and 0.5; text avg Δ −1.2 (l1.0) vs **−0.5 (l0.5, worst −1.7)** → **4B-l0.5 = release candidate**. lmms-eval quick 4B-l0.5 vs 1.7B vs Qwen3-VL-2B: DocVQA 65.1/62.5/93.3, ChartQA 34.6/41.2/68.2, TextVQA 61.7/58.6/79.7, OCRBench 74.2/74.4/90.0, MMBench 79.6/68.6/81.4, POPE 84.1/80.8/88.0, MMMU 43.6/38.6/39.8. Thinking on images: no gain (4B 55.7 = nothink). Injection: 0 instructions obeyed; canary echo 16% (4B-l0.5). Latency BF16 4B-l0.5: TTFT p50 262 ms @c1 (photo 85 / page 296 ms), 502 out-tok/s @c32. `export_hybrid.py` now runs `prepare_hybrid_serving.py` automatically; serving prep reads base LM from export_info.json. **Remaining gap: document/chart reading (−28 DocVQA, −34 ChartQA vs Qwen3-VL-2B) → next: S3 SFT with text-rich data + answer-format fix, ViT top-block unfreeze.** |
| 2026-11-20 | Release selection (quality-retention + scaling-4b results, final tests in `experiments/final-tests/`) | **1.7B:** retention vs Qwen3-1.7B (8 text tasks, think+nothink, subset): s3000-l1.0 mean Δ −1.1 (worst IFEval-think −2.4), official QA (1,400) 53.9 → best 1.7B; LoRA scaling 0.5/0.75 bought no clear retention. **4B** (Qwen3-4B + Qwen3-VL-4B ViT; FP32 text logits identical): align 1500 → S2 LoRA 2000; s2000-l0.5: official QA 55.3 (= l1.0 55.3), text mean Δ −0.5 (worst MMLU-Pro-think −1.7; l1.0: −1.2/−3.3). lmms-eval quick 4B-l0.5 vs 1.7B vs Qwen3-VL-2B: DocVQA 65.1/62.5/93.3, InfoVQA 35.2/34.1/70.0, ChartQA 34.6 (40.4 trailing-'.' stripped)/41.2/68.2, TextVQA 61.7/58.6/79.7, OCRBench 74.2/74.4/90.0, AI2D(lenient) 61.4/57.2/76.2, MMBench 79.6/68.6/81.4, POPE 84.1/80.8/88.0, RWQA 51.4/51.6/65.8, MMMU 43.6/38.6/39.8. Thinking on images: no gain (4B 55.7 vs 55.7; 1.7B −4.6…+1.4) → serve non-thinking for image traffic. Injection canary leak 4B 9–16% vs 1.7B 34%. Latency BF16 (GPU1, less contended), 64 out tok: 4B TTFT p50 262 ms c1 / 1.34 s c32, 56 → 502 out-tok/s; 1.7B-s3000 245 ms / 548 ms, 102 → 730 out-tok/s. **Decision: release candidate = `exports/hyb-4b-s2000-l05`; low-latency fallback = `exports/hyb-1b7-s3000-l10`.** Open issues for S3: answer-format habits (trailing '.', "Answer: X"), doc/chart/OCR gap 20–35 pts vs Qwen3-VL-2B (data, not scale). `export_hybrid.py` now runs `prepare_hybrid_serving.py` automatically (base LM read from export_info). |
| 2026-11-22 | S3 doc/chart/OCR SFT on 4B (`configs/train/hybrid_s3_lora_4b.json`, evals `experiments/s3-eval/`) | **Data:** 62 sources (docs 25%, charts/tables 19%, OCR 13%, photos/screens 17%, text replay 18%); FineVision multi-QA rows now use every turn (`build_shards.py --all-turns` + `pick_turn`; S2 only ever saw the first question per image); answer-format normalization (`src/vlm/hybrid/formats.py`, `normalize_format` per source: strip trailing '.' on short answers, "Answer: B" → "B", 35% standard lmms-style instructions). **Contamination found:** FineVision `ai2d_merged`/tqa contain AI2D test images and screenqa contains our eval screenshots → S3 manifests filtered by exact hash + dHash (`experiments/s3/drop_overlap.py`, `manifests/s3c/`); all official-QA numbers now also reported on the 1,214-item clean subset (`scripts/score_clean.py`). Earlier AI2D/screen scores were inflated (4B-S2-l0.5: 55.3 all → 52.8 clean). **Results (official QA all/clean):** s2000 62.9/60.8, s3000 64.3/62.6, s4000 l1.0 64.4/62.4, l0.75 64.7/62.9, l0.5 64.1/62.3 (clean +9.5 vs S2; DocVQA 45.7→61.8, ChartQA 46.2→63.7, InfoVQA 33.5→45.0, OCRBench 67.5→74.2); format habits gone (trailing '.' 3.3%→0.3%, "Answer:" 13.3%→0%). Text Δ vs Qwen3-4B: l1.0 −1.85 (worst IFEval-think −4.8), l0.75 −1.68 (−3.5), **l0.5 −0.90 (worst IFEval-think −1.8)**. Thinking on images still no gain (l0.75: 59.6 think vs 62.9 nothink). Injection canary echo l0.75 13% (web 37.5, doc 0, photo 2.5). **Decision: release candidate = `exports/hyb-4b-s3-s4000-l05`** (replaces 4B-S2-l0.5). lmms-eval quick (n=500) S3-l0.5 vs S2-l0.5 vs Qwen3-VL-2B (`reports/benchmarks/summary-s3.md`): DocVQA 79.9/65.1/93.3, InfoVQA 49.2/35.2/70.0, ChartQA 57.0/34.6/68.2, TextVQA 70.6/61.7/79.7, OCRBench 77.4/74.2/90.0, AI2D (strict EM, test images filtered from train) 72.0/3.0/76.2, MMBench 79.6/79.6/81.4, POPE 84.6/84.1/88.0, RWQA 58.2/51.4/65.8, MMMU 47.4/43.6/39.8. Latency BF16 GPU1: TTFT p50 220 ms c1 (photo 74 / page 245 ms) / 930 ms c32, 102 → 808 out-tok/s (shorter, format-exact answers). Remaining gap vs Qwen3-VL-2B: DocVQA −13, InfoVQA −21, OCRBench −13, ChartQA −11 → next lever: ViT top-block unfreeze / higher-res doc stage. |
| 2026-11-23 | S4: ViT top-8 unfreeze + 2.56 MP doc-heavy stage (`configs/train/hybrid_s4_vit_hr_4b.json`, evals `experiments/s4/`) | From the S3 release (LoRA 0.5 merged): fresh LoRA r64, top-8 ViT blocks (lr 5e-6), 2.56 MP cap, docs ×1.5 / charts ×1.25 / OCR ×1.1, 2,000 × 32 samples, 12.9 s/step, 25.6 GB. **Resolution alone is neutral** (S3-l0.5 at 2.56 MP: 62.7 clean vs 62.3). **Official QA all/clean:** s1000-l1.0 65.1/63.3, s2000-l1.0 64.6/63.0, **s2000-l0.5 66.1/64.3** (clean vs S3-l0.5: DocVQA 61.8→67.8, InfoVQA 45.0→51.0, ChartQA 63.7→65.4, TextVQA 73.5→77.0, OCRBench 74.2→71.5, VQAv2 61.0→58.2; ±3 is noise at 200 Q/source). **Text Δ vs Qwen3-4B:** s2000-l1.0 −0.84, **s2000-l0.5 −0.65** (S3-l0.5 −0.91) → S4 improves vision without extra text loss. Fixes: `prepare_hybrid_serving.py` crashed on exports initialised from a prepared export (already-merged template) → detects merged templates; trainer kept `metrics.jsonl` open, so lines after a mid-run file replacement were lost (rebuilt from the train log; now reopened per write). **"Answer:" habit traced** to CoSyn document/chart/table + Geo170K targets (explanation + "Answer: X") → `normalize_format: "rationale"` (`formats.py`). **Decision: release run S4F = S4 recipe + rationale normalization** (`configs/train/hybrid_s4f_release_4b.json`, `experiments/s4f/`). Qwen3-VL-4B-Instruct reference (lmms-eval quick): MMMU 48.0 (ours S3 47.4); other tasks rerun at max_len 32768. |
| 2026-11-24 | **S4F release run** (`configs/train/hybrid_s4f_release_4b.json`, `experiments/s4f/`) | S4 recipe + rationale normalization; 2,000 steps, 13 s/step. **Official QA all/clean:** l0.5 65.4/63.8, l1.0 65.9/64.3 (S4-l0.5 66.1/64.3). **Text Δ vs Qwen3-4B:** l0.5 −0.92, l1.0 −0.87. **lmms-eval quick l0.5** (`reports/benchmarks/summary-s4.md`): DocVQA 79.3, InfoVQA 53.0, ChartQA 56.6, TextVQA 72.9, OCRBench 77.2, AI2D 72.0, MMBench 79.0, POPE 83.7, RWQA 56.8, MMMU 47.0 — all within noise of S4 (mean +0.1), MMMU dip of S4 recovered. Qwen3-VL-4B-Instruct reference: 95.5/79.1/73.4/82.1/89.6/86.4/88.8/88.1/70.8/48.0. **Answer-style probe** (160 open-ended doc/chart/diagram questions, bare and 'Explain your reasoning'): 'Answer:' lines 0% for S3, S4 and S4F — the habit was already gone after S3's normalization; S4/S4F explanations are terse (≈29 tokens vs 43 for S3). **Injection** (normalized fact match — exact match penalized '—' vs '-' and paraphrases): canary echo 13.3% (web 30%), 0 instructions followed. Latency BF16 on a GPU shared with other vLLM servers: TTFT p50 274 ms c1 (photo 83 / page 350 ms), 1.13 s c32, 626 out-tok/s. **Decision: release = `exports/hyb-4b-s4f-s2000-l05` (Graft-VL-4B)**; S4 ties within noise, S4F trains from the committed recipe and recovers MMMU. |
| 2026-11-26 | Release + Qwen3-VL text comparison | **Published** https://huggingface.co/sayantan1/Graft-VL-4B (CC BY-NC 4.0, = `hyb-4b-s4f-s2000-l05`). Text gate on Qwen's own VL models (`reports/text/qwen3-vl/`, `experiments/s4f/run_qwen3vl_text.sh`): vs Qwen3-4B, non-thinking Qwen3-VL-4B-Instruct **+2.6** (newer LM, ≈2.5× longer answers) vs Graft −0.3; thinking Qwen3-VL-4B-Thinking **−2.1** (IFEval −7.7, ≈2× tokens) vs Graft −1.5. → Graft's value is keeping a given text model's behaviour + one hybrid-thinking model, not beating Qwen3-VL on text. Next: more/better doc-infographic-chart data (largest gaps vs Qwen3-VL), commercially licensed mix. |

## References

- LLaVA (Visual Instruction Tuning), arXiv:2304.08485 · LLaVA-NeXT / OneVision
- BLIP-2, arXiv:2301.12597
- Cambrian-1, arXiv:2406.16860
- Molmo & PixMo, arXiv:2409.17146
- InternVL 2.5 (staged training, pixel-unshuffle)
- SigLIP 2, arXiv:2502.14786
- Qwen2.5-VL, arXiv:2502.13923 · Qwen3-VL technical report
- FineVision (Hugging Face aggregated VLM data)
- nanoVLM (minimal reference implementation)
