# Role Alignment

This project is built for video generative AI research engineering roles. It demonstrates the practical systems around model research: data preparation, filtering, captioning, experiment metrics, throughput, GPU training mechanics, diffusion fine-tuning, and inference trade-off reporting.

## What It Demonstrates

| Hiring Signal | Project Evidence |
| --- | --- |
| Video data engineering | Scene splitting, ffmpeg normalization, metadata probing, frame sampling, JSONL manifests. |
| Multimodal data quality | Motion scoring, EasyOCR/Tesseract text and watermark detection, LAION-style aesthetic scoring, dense VLM captioning. |
| Research workflow | Reproducible configs, metrics summaries, benchmark harnesses, failed-experiment log. |
| Scaling instincts | Ray adapter, single-process vs Ray throughput benchmark, and Accelerate launch configs. |
| GPU training mechanics | PyTorch manifest-caption contrastive training benchmark with CUDA memory and throughput reporting. |
| Distributed training awareness | Measured Kaggle T4x2 Accelerate multi-GPU run and optional DeepSpeed ZeRO-2 run. |
| Foundation/diffusion fine-tuning | Manifest-to-Diffusers image-caption dataset export and Stable Diffusion LoRA fine-tuning runbook. |
| GPU efficiency awareness | Inference benchmark harness for steps, slicing, dtype, compile, latency, and VRAM. |
| Code quality | Typed modules, optional heavyweight dependencies, CPU-friendly CI, focused tests. |

## Measured GPU Run

A Kaggle notebook run on NVIDIA Tesla T4 x2 verified:

| Run | Distributed type | GPU count | Samples/sec | Peak VRAM MB |
| --- | --- | ---: | ---: | ---: |
| Single-process CUDA | DistributedType.NO | 2 | 3106.05 | 27.64 |
| Accelerate multi-GPU | DistributedType.MULTI_GPU | 2 | 8461.54 | 29.75 |
| DeepSpeed ZeRO-2 | DistributedType.DEEPSPEED | 2 | 5796.43 | 977.60 |

See [Kaggle T4x2 training results](../examples/kaggle_training_results.md) for the full report.

## Diffusion Fine-Tuning Path

The repository now includes a full LoRA fine-tuning path for a pretrained diffusion foundation model:

1. Curate videos and captions with the main pipeline.
2. Export accepted clips into a Diffusers-compatible image-caption dataset.
3. Run the official Hugging Face Diffusers Stable Diffusion LoRA training script with Accelerate.
4. Save adapter weights and verify them through text-to-image inference.

See [Diffusion LoRA fine-tuning](DIFFUSION_LORA_FINETUNE.md) for the runbook.

This is not full text-to-video model training from scratch. It is a practical fine-tuning workflow for a pretrained diffusion foundation model, which is the honest level of evidence this project can support on Kaggle-class GPUs.

## Portfolio Story

The strongest way to present this repository is not as a finished data product, but as a research loop:

1. Convert raw videos into normalized clips.
2. Measure quality, motion, duplicate rate, OCR/watermark rate, aesthetic score, and caption usefulness.
3. Reject bad clips with auditable reasons.
4. Export accepted clips into a Diffusers LoRA fine-tuning dataset.
5. Train a small manifest-caption contrastive benchmark to test CUDA, mixed precision, and distributed launch mechanics.
6. Run a small Stable Diffusion LoRA fine-tune on curated frame-caption data.
7. Compare preprocessing throughput and inference settings.
8. Write down failed experiments so future iterations are grounded in evidence.

## Honest Resume Boundary

This project supports claims like:

```text
Built an end-to-end diffusion fine-tuning workflow that converts curated video-caption manifests into a Diffusers image-caption dataset and runs a Stable Diffusion LoRA fine-tune with Accelerate, mixed precision, and checkpointed adapter outputs.
```

It also supports:

```text
Ran GPU training benchmarks on Kaggle T4x2 with PyTorch, Hugging Face Accelerate multi-GPU launch, and DeepSpeed ZeRO-2 configuration; measured throughput, mixed precision behavior, final loss, distributed backend, and peak CUDA memory.
```

Use the first claim after the LoRA Kaggle run succeeds and adapter weights or logs are saved. Until then, phrase it as "implemented a LoRA fine-tuning workflow" rather than "ran a diffusion fine-tune".

It does not, by itself, prove production GPU cluster ownership, full foundation model pre-training, or text-to-video training from scratch. Avoid wording like "managed GPU clusters" unless that experience comes from separate real work.

## Next Real Experiment

Run the pipeline on 50-100 Creative Commons clips and replace the synthetic demo fixture with a real dataset summary:

```bash
vdf split-scenes data/raw/source.mp4 --output-dir data/clips
vdf process-folder data/clips --output outputs/manifest.jsonl
vdf dedupe-manifest outputs/manifest.jsonl --output outputs/manifest_deduped.jsonl
vdf summarize-manifest outputs/manifest_deduped.jsonl --output examples/real_dataset_summary.md
```

For the strongest version of the experiment, run with real OCR and LAION-style aesthetic scoring, then report manual precision for rejected watermark/text clips and low-aesthetic clips.

Then run the Kaggle GPU benchmark on the real manifest:

```bash
vdf benchmark-training --real --manifest outputs/manifest_deduped.jsonl \
  --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_manifest_cuda.json \
  --markdown-output outputs/training_manifest_cuda.md
```

For a two-GPU notebook, use `accelerate launch --config_file configs/accelerate_kaggle.yaml`; for the DeepSpeed experiment, use `configs/accelerate_deepspeed_zero2.yaml`.

Finally, export a Diffusers LoRA dataset and run a small Stable Diffusion fine-tune:

```bash
vdf prepare-diffusion-lora-data outputs/manifest_deduped.jsonl \
  --output-dir outputs/diffusion_lora_dataset \
  --frames-per-clip 1 \
  --max-clips 100

accelerate launch scripts/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --train_data_dir=outputs/diffusion_lora_dataset \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=120 \
  --mixed_precision=fp16 \
  --output_dir=outputs/diffusion_lora
```

Save the generated LoRA report, training logs, and one inference sample as portfolio artifacts.
