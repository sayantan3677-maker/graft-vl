# Graft-VL: giving a text-only LLM eyes without changing what it says

**Model:** [sayantan1/Graft-VL-4B](https://huggingface.co/sayantan1/Graft-VL-4B) · **Code:** this repository · **License:** CC BY-NC 4.0 (weights)

## The problem

Teams often build products on a specific text LLM: its prompts, its answer style, its fine-tunes and its
evaluations. Adding image input usually means switching to a different vision-language model and re-validating
everything. Graft-VL asks a narrower question: **can we add vision to an existing text model, Qwen3-4B, so that
its text behaviour stays essentially unchanged?**

## Result

Graft-VL-4B reads documents, charts, screenshots and photos, and stays within about 1 point of Qwen3-4B on text:

| | Graft-VL-4B | Qwen3-VL-2B-Instruct | Qwen3-VL-4B-Instruct |
|---|---:|---:|---:|
| DocVQA (ANLS) | 79.3 | 93.3 | 95.5 |
| InfographicVQA (ANLS) | 53.0 | 70.0 | 79.1 |
| ChartQA | 56.6 | 68.2 | 73.4 |
| TextVQA | 72.9 | 79.7 | 82.1 |
| OCRBench | 77.2 | 90.0 | 89.6 |
| MMBench-EN | 79.0 | 81.4 | 88.8 |
| MMMU | 47.0 | 39.8 | 48.0 |

Text ability against the original Qwen3-4B, averaged over MMLU-Pro, GSM8K, IFEval and MATH-500:

| | Non-thinking | Thinking |
|---|---:|---:|
| Graft-VL-4B (one model, both modes) | −0.3 | −1.5 |
| Qwen3-VL-4B, Instruct / Thinking (two models) | +2.6 (2.5× longer answers) | −2.1 (IFEval −7.7, 2× tokens) |

Graft-VL does not match Qwen's own VL models on vision: they were trained with far more multimodal data and compute.
Its value is different: one model with Qwen3-4B's text behaviour, answer length and thinking switch, served by
stock vLLM, trained with a few GPU-days on one shared GPU.

## How it works

1. **Graft.** Take Qwen3-VL-4B's vision encoder, merger and DeepStack layers, and Qwen3-4B's language model, and
   assemble them inside the Qwen3-VL architecture. Before any training, text-only outputs are **bit-identical** to
   Qwen3-4B (FP32 logits, max |Δ| = 0). That invariant is tested, not assumed.
2. **Align.** Train only the merger (1,500 steps) so image features land in the language model's space.
3. **Teach with LoRA, protect with replay.** Train LoRA adapters on the LM with a broad visual instruction mix
   (S2), then a document-, chart- and OCR-heavy mix of 62 sources (S3), then unfreeze the top 8 ViT blocks at
   2.56 MP (S4). About 20% of every batch is Qwen3-4B's own answers to text prompts, so the model keeps rehearsing
   its own behaviour.
4. **Merge at half strength.** Each stage's LoRA is merged at scale 0.5: on a sweep, full strength gained < 0.6 points
   of vision but cost 1.85 points of text ability.

The result is a standard `Qwen3VLForConditionalGeneration` checkpoint, so `vllm serve` and Transformers load it with
no custom code.

## What mattered (and what didn't)

- **The first approach failed, and the measurement said why.** Training a SigLIP2 encoder plus projector from
  scratch reached only 12–15% on official-split QA after 0.5B tokens, and it failed the text-retention gate. A
  linear probe showed the frozen features did contain the letters (89%), so the bottleneck was the amount of
  training, not the features. Switching to Qwen3-VL's already-trained vision tower scored 29.5% **zero-shot**,
  beating every trained run, and 48.6% after alignment alone.
- **Data beat everything else.** The S3 data overhaul gave +16 points on DocVQA (45.7 → 61.8 on our clean set). Doubling resolution alone gave
  nothing (62.3 → 62.7); unfreezing ViT blocks on document-heavy data gave +1.5.
- **Contamination was real and caught twice.** An early "held-out" set shared 1,476 of 1,500 images with training.
  Later, public training sets turned out to contain AI2D test images and our screenshot eval images. All training
  data is now filtered against eval images by SHA-256 and perceptual hash, and internal scores are reported on a
  verified-clean subset.
- **Answer format is a silent accuracy killer.** Some datasets end one-word answers with "." or answer "Answer: B";
  the model copied this and strict scorers marked it wrong. Per-source normalization removed it: trailing "." fell
  from 3.3% to 0.3%, and "Answer:" from 13.3% to 0%.
- **Most multi-question rows were being wasted.** Rows with several questions per image only ever contributed the
  first question; using all of them was a free data multiplier.
- **Serving needs its own tests.** vLLM parity (prompt ids 48/48, BF16 log-prob gaps at noise level), a pixel cap
  matched to training (Qwen3-VL's default 16.7 MP produced 16K-token prompts), and a merged chat template that keeps
  Qwen3's thinking switch.

## Evaluation discipline

- Official validation/test splits only (1,400 questions over 7 sources), plus lmms-eval on 10 public benchmarks
  through the same vLLM path as production.
- Text retention: 4 benchmarks × thinking/non-thinking × 2 seeds, gating every release candidate.
- Prompt-injection tests with canaries planted in images: no embedded instruction was ever followed.
- Latency under concurrency: TTFT p50 274 ms at concurrency 1, 626 output tokens/s at 32, on a shared GPU.

## Limitations

- 14–26 points behind Qwen3-VL on dense documents and infographics.
- One image per prompt; English only; not evaluated for safety-critical use.
- Parts of the training data are research-only or generated by proprietary models, hence the non-commercial
  license. A per-source audit is in `reports/licenses/dataset-audit.md`.

## Next

A v2 with a commercially licensable data mix: document and chart question-answer pairs generated by Qwen3-VL-4B
(Apache-2.0) on permissively licensed pages, aimed at the largest gaps (DocVQA, InfographicVQA, ChartQA).
