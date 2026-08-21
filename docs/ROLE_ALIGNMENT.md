# Role Alignment

This project is built for video generative AI research engineering roles. It demonstrates the practical systems around model research: data preparation, filtering, captioning, experiment metrics, throughput, GPU training mechanics, and inference trade-off reporting.

## What It Demonstrates

| Hiring Signal | Project Evidence |
| --- | --- |
| Video data engineering | Scene splitting, ffmpeg normalization, metadata probing, frame sampling, JSONL manifests. |
| Multimodal data quality | Motion scoring, EasyOCR/Tesseract text and watermark detection, LAION-style aesthetic scoring, dense VLM captioning. |
| Research workflow | Reproducible configs, metrics summaries, benchmark harnesses, failed-experiment log. |
| Scaling instincts | Ray adapter, single-process vs Ray throughput benchmark, and Accelerate launch configs. |
| GPU training mechanics | PyTorch manifest-caption contrastive training benchmark with CUDA memory and throughput reporting. |
| Distributed training awareness | Hugging Face Accelerate config for DDP-style multi-GPU runs and optional DeepSpeed ZeRO-2 config. |
| GPU efficiency awareness | Inference benchmark harness for steps, slicing, dtype, compile, latency, and VRAM. |
| Code quality | Typed modules, optional heavyweight dependencies, CPU-friendly CI, focused tests. |

## Portfolio Story

The strongest way to present this repository is not as a finished data product, but as a research loop:

1. Convert raw videos into normalized clips.
2. Measure quality, motion, duplicate rate, OCR/watermark rate, aesthetic score, and caption usefulness.
3. Reject bad clips with auditable reasons.
4. Train a small manifest-caption contrastive benchmark to test CUDA, mixed precision, and distributed launch mechanics.
5. Compare preprocessing throughput and inference settings.
6. Write down failed experiments so future iterations are grounded in evidence.

## Honest Resume Boundary

This project supports claims like:

```text
Implemented GPU training and benchmarking workflows with PyTorch, Hugging Face Accelerate, and optional DeepSpeed ZeRO-2 configuration, measuring throughput, mixed precision behavior, and peak CUDA memory on curated video-caption manifests.
```

It does not, by itself, prove production GPU cluster ownership. Avoid wording like "managed GPU clusters" unless that experience comes from separate real work.

## Next Real Experiment

Run the pipeline on 50-100 Creative Commons clips and replace the synthetic demo fixture with a real dataset summary:

```bash
vdf split-scenes data/raw/source.mp4 --output-dir data/clips
vdf process-folder data/clips --output outputs/manifest.jsonl
vdf dedupe-manifest outputs/manifest.jsonl --output outputs/manifest_deduped.jsonl
vdf summarize-manifest outputs/manifest_deduped.jsonl --output examples/real_dataset_summary.md
```

For the strongest version of the experiment, run with real OCR and LAION-style aesthetic scoring, then report manual precision for rejected watermark/text clips and low-aesthetic clips.

Then run the Kaggle GPU benchmark:

```bash
vdf benchmark-training --real --manifest outputs/manifest_deduped.jsonl \
  --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_manifest_cuda.json \
  --markdown-output outputs/training_manifest_cuda.md
```

For a two-GPU notebook, use `accelerate launch --config_file configs/accelerate_kaggle.yaml`; for the DeepSpeed experiment, use `configs/accelerate_deepspeed_zero2.yaml`.
