| Benchmark | Metric | n | ours (hybrid s2000) | Qwen3-VL-2B (default px) | Qwen3-VL-2B (1.6MP cap) |
|---|---|---:|---:|---:|---:|
| docvqa_val | ANLS | 500 | 62.5 | 93.3 | 92.2 |
| infovqa_val | ANLS | 500 | 34.1 | 70.0 | 66.8 |
| chartqa | relaxed acc | 500 | 41.2 | 68.2 | 67.8 |
| textvqa_val | VQA acc | 500 | 58.6 | 79.7 | 79.9 |
| ocrbench | score/n | 500 | 74.4 | 90.0 | 90.0 |
| ai2d | EM (flexible-extract) | 500 | 0.0 (57.2) | 76.2 (76.2) | 75.8 (75.8) |
| mmbench_en_dev | gpt_eval_score (rule match; no GPT key) | 500 | 68.6 | 81.4 | 81.0 |
| pope | F1 | 500 | 80.8 | 88.0 | 88.2 |
| realworldqa | EM | 500 | 51.6 (51.6) | 65.8 (65.8) | 67.0 (67.0) |

MCQ cells: official metric (lenient letter extraction in parentheses).
