# Role Alignment

This project is built for video generative AI research engineering roles. It demonstrates the practical systems around model research: data preparation, filtering, captioning, experiment metrics, throughput, and inference trade-off reporting.

## What It Demonstrates

| Hiring Signal | Project Evidence |
| --- | --- |
| Video data engineering | Scene splitting, ffmpeg normalization, metadata probing, frame sampling, JSONL manifests. |
| Multimodal data quality | Motion scoring, text/watermark proxy, aesthetic scoring, dense captioning prompts, VLM adapter. |
| Research workflow | Reproducible configs, metrics summaries, benchmark harnesses, failed-experiment log. |
| Scaling instincts | Ray adapter and single-process vs Ray throughput benchmark. |
| GPU efficiency awareness | Inference benchmark harness for steps, slicing, dtype, compile, latency, and VRAM. |
| Code quality | Typed modules, optional heavyweight dependencies, CPU-friendly CI, focused tests. |

## Portfolio Story

The strongest way to present this repository is not as a finished data product, but as a research loop:

1. Convert raw videos into normalized clips.
2. Measure quality, motion, duplicate rate, and caption usefulness.
3. Reject bad clips with auditable reasons.
4. Compare throughput and inference settings.
5. Write down failed experiments so future iterations are grounded in evidence.

## Next Real Experiment

Run the pipeline on 50-100 Creative Commons clips and replace the synthetic demo fixture with a real dataset summary:

```bash
vdf split-scenes data/raw/source.mp4 --output-dir data/clips
vdf process-folder data/clips --output outputs/manifest.jsonl
vdf dedupe-manifest outputs/manifest.jsonl --output outputs/manifest_deduped.jsonl
vdf summarize-manifest outputs/manifest_deduped.jsonl --output examples/real_dataset_summary.md
```
