# Kaggle GPU Training Runbook

This runbook turns the dataset curation project into a real GPU training and profiling exercise. It is intentionally small enough for Kaggle notebooks, but it exercises the same knobs used in larger ML systems: CUDA placement, mixed precision, DDP through Accelerate, optional DeepSpeed ZeRO-2, throughput, and peak VRAM reporting.

## What This Proves

- PyTorch training loop design for a manifest-conditioned multimodal objective.
- CUDA execution and memory measurement.
- Hugging Face Accelerate launch flow for single-GPU and multi-GPU notebooks.
- Optional DeepSpeed ZeRO-2 configuration through Accelerate.
- Benchmark reporting: steps, samples/sec, final loss, GPU count, distributed backend, mixed precision, and peak VRAM.

This is not a claim of production GPU cluster administration. It is a reproducible hands-on training benchmark that supports honest resume wording around GPU training, Accelerate, DeepSpeed configuration, and CUDA profiling.

## Kaggle Setup

1. Create a Kaggle notebook from the GitHub repository.
2. Enable a GPU accelerator. If a two-GPU accelerator is available, use it for the multi-GPU run; otherwise run the single-GPU baseline.
3. Install the training extras:

```bash
pip install -e .[training]
```

4. Confirm the visible CUDA devices:

```bash
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Single-GPU / CPU-Compatible Smoke Run

```bash
vdf benchmark-training --dry-run --samples 512 --batch-size 32 \
  --output outputs/training_dry_run.json \
  --markdown-output outputs/training_dry_run.md
```

## Real CUDA Run

Use synthetic manifest-like rows when you only want to benchmark training mechanics:

```bash
vdf benchmark-training --real --samples 2048 --epochs 2 --batch-size 64 \
  --mixed-precision fp16 \
  --output outputs/training_cuda.json \
  --markdown-output outputs/training_cuda.md
```

Use the curated dataset manifest when you have already processed videos:

```bash
vdf benchmark-training --real --manifest outputs/manifest_deduped.jsonl \
  --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_manifest_cuda.json \
  --markdown-output outputs/training_manifest_cuda.md
```

## Accelerate Multi-GPU Run

```bash
accelerate launch --config_file configs/accelerate_kaggle.yaml \
  -m video_dataset_factory.training_entrypoint \
  --samples 4096 --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_accelerate.json \
  --markdown-output outputs/training_accelerate.md
```

## DeepSpeed ZeRO-2 Run

Install DeepSpeed only in the Kaggle notebook where you intend to run it:

```bash
pip install deepspeed
accelerate launch --config_file configs/accelerate_deepspeed_zero2.yaml \
  -m video_dataset_factory.training_entrypoint \
  --samples 4096 --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_deepspeed_zero2.json \
  --markdown-output outputs/training_deepspeed_zero2.md
```

## Report In README Or CV

After a real run, report only measured values from the generated markdown file:

```text
Ran a PyTorch manifest-caption contrastive training benchmark on Kaggle CUDA, using Hugging Face Accelerate for launch and optional DeepSpeed ZeRO-2 config; logged GPU count, distributed backend, mixed precision, throughput, final loss, and peak VRAM.
```

Avoid saying that this project manages production GPU clusters. Kaggle proves hands-on GPU training and distributed-launch mechanics, not cluster operations ownership.
