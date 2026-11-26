# Graft-VL: project status and handover (2026-11-26)

This document records everything done on the project from the start, what was released, where every artifact
lives, what was deleted from the training server, and what is still pending. It is written so that the project can
be resumed from this repository alone.

- **Released model:** <https://huggingface.co/sayantan1/Graft-VL-4B> (public, CC BY-NC 4.0)
- **Code and results:** <https://github.com/sayantan3677-maker/graft-vl> (public)
- **Write-up:** [WRITEUP.md](WRITEUP.md) · **Detailed experiment log:** [../roadmap.md](../roadmap.md) (Progress log table)
- **Model card (source):** [../MODEL_CARD.md](../MODEL_CARD.md)

---

## 1. Goal

Turn the text-only, hybrid-thinking **Qwen3** LLM into a production-grade vision-language model that reads
screenshots, photos and documents, while keeping the original model's text, reasoning and thinking behaviour, trained
on a single shared workstation GPU and served with stock vLLM. Personal project, published openly.

## 2. Timeline: what was done

Dates are when the work happened. Full numbers for every step are in the roadmap's progress log.

### Phase 0: setup and environment gate (2026-09-23 → 09-24)
- Training env (`vlm-train`: torch 2.11, transformers 5.17, peft, liger-kernel) and eval env (`vlm-eval`: vLLM 0.30,
  lmms-eval).
- Checked Qwen3 assumptions (chat template, untrained vision special-token rows), wrote the core library
  (`src/vlm`), CPU unit tests, and a Day-0 GPU gate (packing without leakage, FP32/BF16 equivalence, memory).
- Fixed early infrastructure bugs: WebDataset double-splitting deadlock; dense packed-attention mask replaced by
  FlexAttention; vLLM FlashInfer sampler on Blackwell (`VLLM_USE_FLASHINFER_SAMPLER=0`).

### Phase 1: original design, SigLIP2 + MLP projector on Qwen3-1.7B (2026-09-24 → 10-06): **abandoned**
- S1 alignment sweeps (LR, LayerNorm, 384 vs 512 px), overfit diagnostics, 50K-sample alignment.
- Data built: S1 sets, synthetic reading set (200K rendered images, 386 fonts), text replay of Qwen3's own answers,
  verified multimodal thinking traces, IF/safety replay.
- M1.5 pilots and a full S2 run (0.5B tokens, full fine-tune): official-split QA stayed at 12–15%, DocVQA 1–2%, and
  the text-retention gate failed (IFEval −3, MATH-500 −4.5).
- Diagnosis: a linear probe recovered letters from frozen SigLIP2 features at 89%, so the features were fine and the
  training budget was the bottleneck.
- Found and fixed eval contamination: the first "held-out" set shared 1,476 of 1,500 images with training. Built
  `eval-official` (1,400 questions from official val/test splits) instead.

### Phase 2: pivot to the hybrid ("graft") design (2026-10-07)
- Qwen3-VL's vision encoder + merger + DeepStack attached to the unmodified Qwen3 LM inside the Qwen3-VL
  architecture. Zero-shot it scored 29.5% (vs 14.4% for the best trained SigLIP2 run); merger-only alignment
  reached 48.6%; LoRA S2 reached 55.5% (1.7B).
- Text-only logits verified bit-identical to the original Qwen3 before training (FP32).

### Phase 3: parallel workstreams (2026-11-01 → 11-20)
- **Serving:** native vLLM serving with no plugin; parity tests (prompt ids, BF16 log-prob gaps at noise level);
  `prepare_hybrid_serving.py` (merged chat template keeping the thinking switch, pixel cap, generation config);
  BF16/FP8 latency benchmarks (FP8 gave only +5–10% throughput).
- **Quality retention:** LoRA-strength sweep, text-retention gate (MMLU-Pro, GSM8K, IFEval, MATH-500, thinking and
  non-thinking, 2 seeds), thinking-on-images test (no gain), prompt-injection tests.
- **Scaling to 4B:** Qwen3-4B + Qwen3-VL-4B vision. 4B-S2 at LoRA strength 0.5: QA 55.3, text −0.5.
- Release candidate at that point: 4B-S2 (LoRA 0.5); 1.7B kept as a low-latency fallback.

### Phase 4: S3, document/chart/OCR SFT on 4B (2026-11-22)
- 62 sources (documents 25%, charts/tables 19%, OCR 13%, photos/screens 17%, text replay 18%).
- Found a second contamination problem: public FineVision subsets contained AI2D test images and our eval
  screenshots. All manifests filtered by SHA-256 + perceptual hash; scores reported on a verified-clean subset
  (1,214 of 1,400 items).
- Answer-format normalization per source (trailing ".", "Answer: B"); all questions of multi-question rows used.
- Result: clean QA 52.8 → 62.3; DocVQA (public) 65.1 → 79.9; text −0.9. Dataset license audit written
  (`reports/licenses/`).

### Phase 5: S4 and the release run S4F (2026-11-22 → 11-24)
- S4: top 8 ViT blocks unfrozen, 2.56 MP images, document-heavier mix. Resolution alone gave nothing; ViT training
  gave +2 on the clean set.
- Bugs fixed on the way: serving prep crashed on exports initialised from a prepared export; the trainer lost
  metrics lines when the metrics file was replaced mid-run.
- "Answer:" habit traced to CoSyn/Geo170K targets → `normalize_format: "rationale"`.
- S4F (S4 recipe + rationale normalization) chosen as the release: ties S4 within noise and recovers S4's MMMU dip.
- Qwen3-VL-4B-Instruct reference benchmarks added; prompt-injection scorer fixed (exact match failed on dash
  styles and paraphrases).

### Phase 6: release and publication (2026-11-24 → 11-26)
- Gradio demo (`demo/`) as a thin client over vLLM, with synthetic example images.
- Weights, card, templates, NOTICE and example images uploaded to Hugging Face; checksum verified.
- Text comparison with Qwen's own VL models (same tests): non-thinking Qwen3-VL-4B-Instruct +2.6 vs Graft −0.3;
  thinking Qwen3-VL-4B-Thinking −2.1 vs Graft −1.5.
- Write-up and README results; repository history scanned for credentials (none found) and made public.
- Repository cleaned of one-off job-queue scripts and status files.

### Phase 7: v2 data, started and paused (2026-11-26)
- Goal: close the document gap and build a commercially licensable data mix.
- Teacher-generated, self-verified QA generator (`experiments/v2/gen_doc_qa.py`): Qwen3-VL-8B-Instruct (Apache-2.0)
  writes QA pairs for DocLayNet pages (CDLA-Permissive-1.0) and answers each question again; pairs are kept only
  when both answers agree and, for extractive answers, the answer is in the page's PDF text.
- Pilot (300 pages): 1,168 kept of 1,402 candidates (`experiments/v2/results/pilot-summary.json`).
- The full run (~69K pages) was cancelled at handover before it started, because the shared GPU had no free memory.

## 3. Final results (release `hyb-4b-s4f-s2000-l05` = Graft-VL-4B)

Public benchmarks (lmms-eval, 500 samples per task, same vLLM pipeline for all):

| | DocVQA | InfoVQA | ChartQA | TextVQA | OCRBench | AI2D | MMBench | POPE | RealWorldQA | MMMU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Graft-VL-4B** | 79.3 | 53.0 | 56.6 | 72.9 | 77.2 | 72.0 | 79.0 | 83.7 | 56.8 | 47.0 |
| Qwen3-VL-2B-Instruct | 93.3 | 70.0 | 68.2 | 79.7 | 90.0 | 76.2 | 81.4 | 88.0 | 65.8 | 39.8 |
| Qwen3-VL-4B-Instruct | 95.5 | 79.1 | 73.4 | 82.1 | 89.6 | 86.4 | 88.8 | 88.1 | 70.8 | 48.0 |

Text ability vs Qwen3-4B (average of MMLU-Pro, GSM8K, IFEval, MATH-500): **−0.3 non-thinking, −1.5 thinking.**
Internal official-split QA (clean subset): 63.8. Latency (shared GPU): TTFT p50 274 ms at concurrency 1,
626 output tokens/s at 32. Prompt injection: no embedded instruction followed.

Honest positioning: not on par with Qwen3-VL on vision; the value is adding vision to an existing text model while
keeping its behaviour, in one model with the thinking switch, on a small compute budget.

## 4. Where everything is

| Artifact | Location | Status |
|---|---|---|
| Code, configs, tests, eval scripts | GitHub repo | complete |
| Results (per-item QA, text evals, benchmarks, latency, injection) | `experiments/`, `reports/` in the repo | complete |
| Training metrics per run | `experiments/<RUN>/metrics.jsonl` + `config.json` | complete |
| Dataset manifests (source lists, counts, hashes) | `manifests/**/manifest.json` | complete; data itself not in git |
| Release weights | Hugging Face `sayantan1/Graft-VL-4B` | complete, checksum verified |
| Training data shards, exports, checkpoints | training server only | **deleted at handover** (see §6) |

Rebuilding what was deleted:
- **Data:** `scripts/build_shards.py`, `experiments/s3/build_s3.sh` (+ overlap filter), `scripts/gen_replay.py`,
  `scripts/gen_traces.py`, `scripts/build_synth_reading.py`, `scripts/build_official_eval.py`. Sources are public
  Hugging Face datasets (mainly FineVision); see LICENSES.md for terms.
- **Model:** `scripts/build_hybrid.py` (graft) → align → S2 → S3 → S4F with the configs in `configs/train/`
  (README lists the commands). Starting from the released weights is the cheapest path for further training.

## 5. Pending work

From the original plan (roadmap §0, §5, §6) and later decisions:

| Item | Status | Notes |
|---|---|---|
| v2 document data (DocLayNet, ~270K verified QA) | pilot done, full run cancelled | needs ~28 GB of GPU memory for ~1 day with the 8B teacher; a 4B FP8 teacher fits in ~11 GB (run the pilot first to compare kept rate) |
| v2 chart/table/infographic data | not started | PubTables-1M (CDLA-Permissive-2.0), self-rendered charts, permissively licensed infographics |
| v2 training + evaluation | not started | S4F recipe on the new mix; compare to Graft-VL-4B |
| Commercially licensable release | not started | requires dropping class-C sources (`reports/licenses/commercial_safe_sources.json`) |
| Hugging Face card: add the Qwen3-VL text comparison | done in the repo, not on HF | needs a new write token for the personal account |
| 8B scaling check | not started | roadmap ladder 1.7B / 4B / 8B; only 1.7B and 4B done |
| Product evaluation set (roadmap §7.6) | not started | only public and official-split evals were used |
| Preference optimization (roadmap §5.6, optional) | not started | LoRA DPO on grounded-vs-hallucinated pairs |
| Grounding (box output) | not trained | box tokens exist in the template |
| Multi-image / video, multilingual OCR | not trained / not evaluated | |
| Longer explanations when asked to "explain" | known limitation | answers ~30 tokens |
| Quantized build (FP8/AWQ) | not released | FP8 measured: small speed gain, accuracy unchanged |
| Hosted demo (Hugging Face Space) | not done | needs a GPU Space |

## 6. Server cleanup at handover (2026-11-26)

Done before removing the server user account:

- **Cancelled:** the queued v2 DocLayNet generation run. No project processes left running.
- **Deleted earlier the same day:** unused dataset shards (~100 GB), checkpoints of the abandoned SigLIP2 runs
  (~232 GB) and 41 redundant model exports (~247 GB).
- **Deleted at handover:** everything in the working copy not tracked by git: remaining dataset shards (~142 GB),
  v2 DocLayNet download and pilot outputs (~35 GB), the three kept exports (~25 GB, the release is on Hugging
  Face), training checkpoints (~19 GB), logs and caches.
- **Deleted:** the assistant's temporary/scratch files for this project (`/tmp/claude-1010/-home-sayantan-dev-vlm-training`),
  which held helper scripts, staging copies and command outputs. The personal Hugging Face upload token file had
  already been deleted right after the upload.
- **Left for account removal** (shared with other projects in the same home directory, removed with the user
  account): Hugging Face cache (~123 GB), pip and vLLM caches, conda envs `vlm-train` / `vlm-eval`, venv
  `~/.venvs/vlm-demo`, the assistant's project memory under `~/.claude/projects/`.

Account actions for the owner:
- Revoke the fine-grained Hugging Face token used for the upload (if not already revoked).
- Revoke or log out the GitHub CLI token and the Hugging Face token stored on the server, unless still used
  elsewhere.
- The released model and this repository do not depend on the server.
