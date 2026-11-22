| Benchmark | Metric | n | 1b7-s2000 | 1b7-s2000-mmmu | 4b-s2-l05 | 4b-s2-l05-mmmu | 4b-s3-l05 | 4b-s3-l05-mmmu | qwen3vl2b | qwen3vl2b-mmmu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| docvqa_val | ANLS | 500 | 62.5 | – | 65.1 | – | 79.9 | – | 93.3 | – |
| infovqa_val | ANLS | 500 | 34.1 | – | 35.2 | – | 49.2 | – | 70.0 | – |
| chartqa | relaxed acc | 500 | 41.2 | – | 34.6 | – | 57.0 | – | 68.2 | – |
| textvqa_val | VQA acc | 500 | 58.6 | – | 61.7 | – | 70.6 | – | 79.7 | – |
| ocrbench | score/n | 500 | 74.4 | – | 74.2 | – | 77.4 | – | 90.0 | – |
| ai2d | EM (flexible-extract) | 500 | 0.0 (57.2) | – | 3.0 (61.4) | – | 72.0 (72.0) | – | 76.2 (76.2) | – |
| mmbench_en_dev | gpt_eval_score (rule match; no GPT key) | 500 | 68.6 | – | 79.6 | – | 79.6 | – | 81.4 | – |
| pope | F1 | 500 | 80.8 | – | 84.1 | – | 84.6 | – | 88.0 | – |
| realworldqa | EM | 500 | 51.6 (51.6) | – | 51.4 (51.6) | – | 58.2 (58.2) | – | 65.8 (65.8) | – |
| mmmu_val | acc | 500 | – | 38.6 | – | 43.6 | – | 47.4 | – | 39.8 |

MCQ cells: official metric (lenient letter extraction in parentheses).
