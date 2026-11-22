# Data and Model Licenses

Decision table for every source (roadmap §4.2). The **product checkpoint trains only on rows with
`commercial_ok = yes`**, and only after legal sign-off. Research runs may use `unverified` sources.

| Source | Used in | Upstream license (as declared) | Image rights | commercial_ok | Notes / action |
|---|---|---|---|---|---|
| Qwen3 (1.7B/4B/8B) | backbone | Apache-2.0 | — | yes | |
| SigLIP2 So400m patch16 | vision (S1 experiments only) | Apache-2.0 | — | yes | |
| Qwen3-VL-2B / Qwen3-VL-4B vision encoder + merger | vision (hybrid, all releases) | Apache-2.0 | — | yes | training data undisclosed upstream |
| PixMo-Cap | S1 | ODC-BY (captions) | web images, third-party rights | unverified | review image-rights position |
| FineVision / localized_narratives | S1 | per-subset (CC BY 4.0 upstream annotations) | COCO/Open Images/Flickr | unverified | check per-image licenses (Open Images: CC BY 2.0) |
| FineVision / textcaps | S1 | per-subset (CC BY 4.0 upstream) | Open Images | unverified | |
| FineVision / screen2words | S1 | per-subset (CC BY 4.0 upstream) | RICO app screenshots | unverified | |
| FineVision / olmOCR-mix-0225-documents | S1 | per-subset (ODC-BY upstream) | web PDFs | unverified | |
| FineVision / ureader_cap | S1 | per-subset | mixed documents | unverified | |
| FineVision / websight | S1 | per-subset (CC BY 4.0 upstream) | synthetic renders | unverified | synthetic image, counts toward ≤20% cap |
| FineVision / S3 doc, chart, table and OCR subsets (docvqa, infographic_vqa, chartqa, plotqa, dvqa, figureqa, textvqa, ocrvqa, pdfvqa, visualmrc, invoices_receipts, ureader_kg, olmocr_books, llavar, textocr_gpt4v, cocotext, rendered_text, iam, infographic_gpt4v, unichart, mmc_instruct, vistext, cosyn_chart/table/document, robut_wtq, tat_dqa, mapqa, …) | S3 | per-subset (mixed; several research-only upstream) | scanned documents, web charts, COCO/Open Images, synthetic renders | unverified | **review each subset before product use** (DocVQA/InfographicVQA/ChartQA/OCR-VQA/IAM have research-only terms upstream) |

Full per-source audit (A/B/C classes, evidence URLs, open items): `reports/licenses/dataset-audit.md`.

Legend: `yes` = verified OK for commercial use · `no` = excluded from product runs · `unverified` = research only until reviewed.
