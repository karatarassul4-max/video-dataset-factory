# Experiment Runbook

This runbook describes the intended end-to-end workflow for producing a portfolio-quality dataset report.

## 1. Prepare Clips

Put raw source videos under `data/raw/`, then split long videos into normalized short clips:

```bash
vdf split-scenes data/raw/example.mp4 --output-dir data/clips
```

Recommended starting settings are in `configs/default.yaml`: 24 FPS, 512 px square output, CRF 18.

## 2. Process Dataset

Install the full default pipeline dependencies before a real run:

```bash
pip install -e .[dev,scene,dashboard,ocr,aesthetic]
```

The default config uses:

- heuristic captions for local CLI smoke runs, unless you switch `captioning.provider` to `groq` or `transformers`;
- EasyOCR for text/watermark filtering;
- LAION-style aesthetic scoring with open_clip embeddings and an auto-downloaded `vit_b_32` linear head;
- optical-flow motion gates, blur/brightness gates, and duration/resolution gates.

```bash
vdf process-folder data/clips --output outputs/manifest.jsonl
```

Current real quality backend settings:

```yaml
ocr:
  provider: easyocr
  languages: [en]
  gpu: false
  max_frames: 4
  min_confidence: 0.35
quality:
  max_ocr_text_area_ratio: 0.08

aesthetic:
  provider: laion
  model_name: ViT-B-32
  pretrained: openai
  head_variant: vit_b_32
  head_path: null
  head_url: https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth
  max_frames: 4
quality:
  min_aesthetic_score: 4.0
```

`easyocr` is the easiest portable OCR backend. `tesseract` is also supported, but it requires the system Tesseract binary in addition to the Python package.

For a lightweight CPU-only smoke run, create a small override config that turns the heavy backends off:

```yaml
ocr:
  provider: proxy
aesthetic:
  provider: none
quality:
  min_aesthetic_score: null
```

Use `aesthetic.provider: clip` when you intentionally want a no-checkpoint CLIP prompt-comparison proxy instead of the LAION linear head.

For VLM captions, install optional dependencies and configure a model:

```bash
pip install -e .[captioning]
```

```yaml
captioning:
  provider: transformers
  model_name: Qwen/Qwen2-VL-2B-Instruct
  model_family: qwen2-vl
  max_keyframes: 4
```

The Streamlit upload path uses Groq vision captioning, so it requires `GROQ_API_KEY` in app secrets.

## 3. Remove Near-Duplicates

```bash
vdf dedupe-manifest outputs/manifest.jsonl --output outputs/manifest_deduped.jsonl --threshold 6
```

Start with threshold 6. Increase only after manual inspection because high thresholds can group visually different clips.

## 4. Generate Report

```bash
vdf summarize-manifest outputs/manifest_deduped.jsonl --output outputs/dataset_summary.md
```

The report should be copied into `examples/` or attached to a GitHub release after a real run.

## 5. Inspect Dataset

```bash
streamlit run dashboards/app.py
```

Use the dashboard to manually inspect accepted/rejected rows, reject reasons, and score distributions.

## 6. Benchmark Throughput

```bash
vdf benchmark-folder data/clips --output outputs/pipeline_benchmark.json
vdf benchmark-folder data/clips --ray --output outputs/pipeline_benchmark_ray.json
```

Ray can be slower for tiny folders. Report the crossover point instead of claiming a universal speedup.

## 7. Benchmark Inference Trade-Offs

CPU-friendly dry run:

```bash
vdf benchmark-inference --dry-run
```

Real CUDA run:

```bash
pip install -e .[inference]
vdf benchmark-inference --real --model runwayml/stable-diffusion-v1-5
```

Report latency, peak VRAM, and a quality proxy side by side.
