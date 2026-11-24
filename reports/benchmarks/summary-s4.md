# lmms-eval quick (limit 500 per task) — S4F (release) vs S4 vs S3 vs Qwen3-VL references

Same pipeline for every model (vLLM 0.30, lmms-eval, `scripts/run_benchmarks.sh ... quick`). Ours: non-thinking template,
2.56 MP (S4, S4F) / 1.6 MP (S3) pixel cap from `processor_config.json`. Qwen3-VL: own template and pixel cap, max_len 32768.
At n=500 one standard error is about ±2 points.

| Benchmark | Metric | S3-l0.5 | S4-l0.5 | Δ S4−S3 | **S4F-l0.5 (release)** | Qwen3-VL-4B-Instruct | Qwen3-VL-2B-Instruct |
|---|---|---:|---:|---:|---:|---:|---:|
| DocVQA (val) | ANLS | 79.9 | 80.4 | +0.5 | **79.3** | 95.5 | 93.3 |
| InfographicVQA (val) | ANLS | 49.2 | 54.5 | +5.3 | **53.0** | 79.1 | 70.0 |
| ChartQA | relaxed acc | 57.0 | 56.8 | −0.2 | **56.6** | 73.4 | 68.2 |
| TextVQA (val) | VQA acc | 70.6 | 72.8 | +2.2 | **72.9** | 82.1 | 79.7 |
| OCRBench | score | 77.4 | 76.2 | −1.2 | **77.2** | 89.6 | 90.0 |
| AI2D | EM | 72.0 | 72.0 | 0.0 | **72.0** | 86.4 | 76.2 |
| MMBench-EN (dev) | rule match | 79.6 | 78.0 | −1.6 | **79.0** | 88.8 | 81.4 |
| POPE | F1 | 84.6 | 82.6 | −2.0 | **83.7** | 88.1 | 88.0 |
| RealWorldQA | EM | 58.2 | 59.0 | +0.8 | **56.8** | 70.8 | 65.8 |
| MMMU (val) | acc | 47.4 | 44.2 | −3.2 | **47.0** | 48.0 | 39.8 |

Reading: S4's gains are on text-dense images (InfoVQA, TextVQA); general/photo tasks (POPE, MMBench, MMMU) dip by
1.5–3 points, at or just above noise. Mean over the 10 tasks: +0.1. Matches the internal official-QA split
(docs/infographics/scene text up, VQAv2 and OCRBench down).

S4F = S4 recipe + rationale answer normalization (release). S4F vs S4: all differences within ±2.8 (noise); mean +0.1.
MMMU recovers (44.2 → 47.0). Release: `hyb-4b-s4f-s2000-l05`.
