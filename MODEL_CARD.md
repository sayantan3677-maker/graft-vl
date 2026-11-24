---
license: cc-by-nc-4.0
base_model:
  - Qwen/Qwen3-4B
  - Qwen/Qwen3-VL-4B-Instruct
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
tags:
  - vision-language
  - multimodal
  - document-understanding
  - ocr
  - chart-understanding
  - qwen3
  - vllm
---

# Graft-VL-4B

Graft-VL-4B converts the **text-only Qwen3-4B** into a vision-language model. It reads documents, charts,
screenshots and photos, and keeps almost all of the text model's ability.

The approach grafts the vision encoder, merger and DeepStack layers of Qwen3-VL-4B onto the unmodified Qwen3-4B
language model inside the Qwen3-VL architecture. Before training, the text-only logits are **bit-identical** to
Qwen3-4B (FP32, max |Δ| = 0). Training then teaches the language model to use the image tokens with LoRA, while
replaying text data to protect reasoning and instruction following.

- **Code, training configs and evaluation:** <https://github.com/sayantan3677-maker/graft-vl>
- **Serving:** a stock Qwen3-VL checkpoint, so `vllm serve` works with no plugin and no custom code.
- **Thinking:** Qwen3's hybrid thinking is kept and toggled with `enable_thinking`. Use non-thinking mode for image
  questions; thinking gave no gain on vision in our tests.

> Release checkpoint: `S4F` (`hyb-4b-s4f-s2000-l05`). All numbers below are for that checkpoint unless marked.

## Results

### Public benchmarks

Measured with lmms-eval through vLLM on 500 samples per task (one standard error ≈ ±2 points), with the same
pipeline for every model. Qwen3-VL-4B-Instruct uses the same vision encoder plus Qwen's full multimodal training, so
it is the natural upper bound; Qwen3-VL-2B-Instruct is shown for scale.

| Benchmark | Metric | **Graft-VL-4B** | Qwen3-VL-2B-Instruct | Qwen3-VL-4B-Instruct |
|---|---|---:|---:|---:|
| DocVQA (val) | ANLS | **79.3** | 93.3 | 95.5 |
| InfographicVQA (val) | ANLS | **53.0** | 70.0 | 79.1 |
| ChartQA | relaxed acc. | **56.6** | 68.2 | 73.4 |
| TextVQA (val) | VQA acc. | **72.9** | 79.7 | 82.1 |
| OCRBench | score | **77.2** | 90.0 | 89.6 |
| AI2D | exact match | **72.0** | 76.2 | 86.4 |
| MMBench-EN (dev) | rule-based match | **79.0** | 81.4 | 88.8 |
| POPE | F1 | **83.7** | 88.0 | 88.1 |
| RealWorldQA | exact match | **56.8** | 65.8 | 70.8 |
| MMMU (val) | accuracy | **47.0** | 39.8 | 48.0 |

AI2D test images also appear in the public FineVision training subsets (`ai2d_merged`, `tqa`). We removed every
eval image from our training data by exact hash and perceptual hash (dHash); see "Contamination control" below.

### Text ability retained

Each figure is the change against the original **Qwen3-4B**. MMLU-Pro and GSM8K use 500-sample subsets and MATH-500
a 200-sample subset; each figure is the mean of 2 seeds.

| Mode | MMLU-Pro | GSM8K | IFEval | MATH-500 |
|---|---:|---:|---:|---:|
| thinking | −0.7 | −1.4 | −2.5 | −1.5 |
| non-thinking | −0.4 | −0.3 | +0.2 | −0.8 |

The average change is **−0.9 points**. The text path is the original Qwen3-4B plus two merged LoRAs (stages S3
and S4), each at strength 0.5.

### Latency

vLLM 0.30, BF16, one data-centre GPU shared with other inference servers (so these are upper bounds), images capped
at 2.56 MP, 64 output tokens.

| Concurrency | TTFT p50 | Output tokens/s | Requests/s |
|---:|---:|---:|---:|
| 1 | 274 ms (photo ≈ 85 ms, full page ≈ 350 ms) | 93 | 1.5 |
| 8 | 491 ms | 380 | 5.9 |
| 32 | 1.13 s | 626 | 9.8 |

## Usage

### vLLM (recommended)

```bash
vllm serve sayantan1/Graft-VL-4B --max-model-len 16384 --limit-mm-per-prompt '{"image": 1}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="sayantan1/Graft-VL-4B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/invoice.png"}},
        {"type": "text", "text": "What is the total amount due?"}]}],
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

### Transformers

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image

model_id = "sayantan1/Graft-VL-4B"
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="bfloat16", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read all the text in the image."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
inputs = processor(text=[prompt], images=[Image.open("page.png")], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

`processor_config.json` caps images at the training resolution, 2.56 MP (about 2,500 image tokens for a full page).
Qwen3-VL's own default of 16.7 MP produces prompts over 16K tokens on long pages and was never seen in training.

## Training

| Stage | What is trained | Data | Steps |
|---|---|---|---|
| Graft | nothing: weights assembled, text logits verified identical | — | — |
| Align | merger + DeepStack mergers (LM and ViT frozen) | captions, OCR and reading data | 1,500 |
| S2 | LoRA r32 on the LM + merger | broad visual instruction mix + text replay | 2,000 |
| S3 | LoRA r64 on the LM + merger (merged at 0.5) | 62 sources, document-, chart- and OCR-heavy + 18% text replay, 1.6 MP | 4,000 × 32 samples |
| S4 | fresh LoRA r64 + merger + **top 8 ViT blocks** (merged at 0.5) | S3 mix re-weighted to docs ×1.5, charts ×1.25, OCR ×1.1; 2.56 MP | 2,000 × 32 samples |

Engineering details:
- **Answer-format normalization.** Short-answer and multiple-choice targets follow their own instruction. Several
  public datasets end one-word answers with "." or answer "Answer: B" to "Answer with the letter."; left as is, the
  model copies this and strict scorers and downstream parsers fail.
- **All questions per image.** Multi-question rows contribute every question, one sampled per use. An earlier stage
  only ever used the first.
- **Text replay.** Qwen3-4B's own responses to instruction and reasoning prompts.
- **Rationale answers.** Some synthetic sources (CoSyn, Geo170K) end every target with "Answer: X" after an
  explanation. Their targets are rewritten into a plain explanation that states the answer, or into a short answer
  with a matching instruction.
- **LoRA strength.** Each stage's LoRA is merged at strength 0.5, chosen on a vision vs. text-retention sweep: at S3,
  1.0 gained < 0.6 points on vision but lost 1.85 points of text ability on average.
- **What helped at S4.** Raising the resolution alone changed nothing (62.3 → 62.7 on our clean set). Training the
  top ViT blocks on document-heavy data at 2.56 MP gave +1.5 points, mostly documents and infographics
  (InfographicVQA +3.8 on the public benchmark), at a cost of up to 1.5 points on some photo benchmarks (POPE, RealWorldQA).

### Contamination control

All training manifests are checked against the evaluation images, by SHA-256 and by perceptual hash (dHash).
Overlapping images are removed from the training shards. Internal evaluations are also reported on a clean subset
of 1,214 of 1,400 items whose images never appeared in any training stage.

## Limitations

- **Gap to Qwen's own VL models.** Dense documents, infographics and charts remain the main gap (DocVQA −14,
  InfoVQA −17, ChartQA −12 vs. Qwen3-VL-2B; larger vs. Qwen3-VL-4B). Qwen3-VL was trained with far more multimodal
  data and compute. Graft-VL's point is keeping an existing text model intact, not beating Qwen3-VL.
- **Single image per prompt.** Multi-image and video input were not trained.
- **English only** for evaluation; multilingual OCR is untested.
- **Prompt injection.** Text written inside an image is sometimes repeated in the answer: 13% of our injection tests,
  30% on web screenshots. No embedded instruction was followed.
- **Short explanations.** Answers are concise by default; asked to "explain your reasoning", the model often gives a
  one-sentence justification (≈ 30 tokens). Use thinking mode for step-by-step reasoning.
- Not evaluated for safety-critical uses (medical, legal, financial decisions).

## License and data

**Weights: CC BY-NC 4.0, for research and non-commercial use.**
- Qwen3-4B and the Qwen3-VL-4B vision encoder are Apache-2.0; see their model cards and keep their notices.
- The training data mixes public datasets with heterogeneous terms:
  - several are research-only or registration-gated (e.g. DocVQA, InfographicVQA, OCR-VQA, IAM);
  - some contain outputs of proprietary models (GPT-4o/4V, Claude) whose terms restrict building competing models.
- A per-source license audit is in the GitHub repository (`reports/licenses/dataset-audit.md`). Users are responsible
  for complying with those terms.
- No training data is redistributed here.

## Citation

```bibtex
@misc{graftvl2026,
  title  = {Graft-VL: Turning a Text-Only Qwen3 LLM into a Vision-Language Model},
  author = {Sayantan},
  year   = {2026},
  url    = {https://github.com/sayantan3677-maker/graft-vl}
}
```
