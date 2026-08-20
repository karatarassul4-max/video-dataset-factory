# Experiment Runbook

This runbook describes the intended end-to-end workflow for producing a portfolio-quality dataset report.

## 1. Prepare Clips

Put raw source videos under `data/raw/`, then split long videos into normalized short clips:

```bash
vdf split-scenes data/raw/example.mp4 --output-dir data/clips
```

Recommended starting settings are in `configs/default.yaml`: 24 FPS, 512 px square output, CRF 18.

## 2. Process Dataset

For a lightweight CPU run, keep heuristic captioning and no aesthetic model:

```bash
vdf process-folder data/clips --output outputs/manifest.jsonl
```

For a richer run, enable heuristic aesthetic scoring:

```yaml
aesthetic:
  provider: heuristic
quality:
  min_aesthetic_score: 5.0
```

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
