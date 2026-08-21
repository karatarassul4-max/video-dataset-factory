# Experiment Runbook

This runbook describes the intended end-to-end workflow for producing a portfolio-quality dataset report.

## 1. Prepare Clips

Put raw source videos under `data/raw/`, then split long videos into normalized short clips:

```bash
vdf split-scenes data/raw/example.mp4 --output-dir data/clips
```

Recommended starting settings are in `configs/default.yaml`: 24 FPS, 512 px square output, CRF 18.

## 2. Process Dataset

For a lightweight CPU run, keep heuristic captioning, proxy OCR, and no aesthetic model:

```bash
vdf process-folder data/clips --output outputs/manifest.jsonl
```

For real OCR/watermark filtering, install OCR dependencies and configure `easyocr` or `tesseract`:

```bash
pip install -e .[ocr]
```

```yaml
ocr:
  provider: easyocr
  languages: [en]
  gpu: false
  max_frames: 4
  min_confidence: 0.35
quality:
  max_ocr_text_area_ratio: 0.08
```

`easyocr` is the easiest portable backend. `tesseract` requires the system Tesseract binary in addition to the Python package.

For a richer run, enable heuristic aesthetic scoring:

```yaml
aesthetic:
  provider: heuristic
quality:
  min_aesthetic_score: 5.0
```

For LAION-style aesthetic scoring, install aesthetic dependencies and provide a linear-head checkpoint trained on normalized CLIP image embeddings:

```bash
pip install -e .[aesthetic]
```

```yaml
aesthetic:
  provider: laion
  model_name: openai/clip-vit-base-patch32
  head_path: models/laion_aesthetic_head.pt
  max_frames: 4
quality:
  min_aesthetic_score: 5.0
```

Use `provider: clip` when you want a no-checkpoint prompt-comparison proxy instead of the LAION linear head.

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
