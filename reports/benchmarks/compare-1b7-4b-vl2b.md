| Benchmark | Metric | n | 1b7 | 4b-l05 | 4b-l10 | vl2b |
|---|---|---:|---:|---:|---:|---:|
| docvqa_val | ANLS | 500 | 62.5 | 65.1 | 65.9 | 93.3 |
| infovqa_val | ANLS | 500 | 34.1 | 35.2 | 35.0 | 70.0 |
| chartqa | relaxed acc | 500 | 41.2 | 34.6 | 35.2 | 68.2 |
| textvqa_val | VQA acc | 500 | 58.6 | 61.7 | 61.2 | 79.7 |
| ocrbench | score/n | 500 | 74.4 | 74.2 | 72.6 | 90.0 |
| ai2d | EM (flexible-extract) | 500 | 0.0 (57.2) | 3.0 (61.4) | 0.6 (63.8) | 76.2 (76.2) |
| mmbench_en_dev | gpt_eval_score (rule match; no GPT key) | 500 | 68.6 | 79.6 | 79.0 | 81.4 |
| pope | F1 | 500 | 80.8 | 84.1 | 83.5 | 88.0 |
| realworldqa | EM | 500 | 51.6 (51.6) | 51.4 (51.6) | 54.6 (54.6) | 65.8 (65.8) |

MCQ cells: official metric (lenient letter extraction in parentheses).
