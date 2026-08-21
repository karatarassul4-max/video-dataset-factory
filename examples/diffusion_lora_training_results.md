# Diffusion LoRA Training Results

## Environment

- Platform: Kaggle
- GPU selected: NVIDIA Tesla T4 x2
- Training launch: single-process Hugging Face Accelerate for LoRA stability
- Base model: `runwayml/stable-diffusion-v1-5`
- Training script: Hugging Face Diffusers `examples/text_to_image/train_text_to_image_lora.py`, pinned to `v0.37.1`
- Dataset: 100 UCF101 sampled video frames exported from a Video Dataset Factory manifest
- Precision: fp16
- LoRA rank: 8
- Batch size per device: 1
- Gradient accumulation steps: 4
- Total optimization steps: 120

## Data Preparation

The production quality pipeline rejected the raw UCF101 clips because of low source resolution. For this GPU smoke run, a relaxed manifest was used to validate the end-to-end Diffusers LoRA fine-tuning path while preserving the original reject reasons for auditability.

| Metric | Value |
| --- | ---: |
| Source videos sampled | 100 |
| Manifest records | 100 |
| Near-duplicates flagged | 8 |
| Production accepted clips | 0 |
| Relaxed LoRA clips | 100 |
| Exported training images | 100 |
| Skipped clips during export | 0 |

## Result

| Metric | Value |
| --- | ---: |
| Training examples | 100 |
| Optimization steps | 120 |
| Final logged step loss | 0.0495 |
| Adapter checkpoint | `outputs/diffusion_lora/pytorch_lora_weights.safetensors` |
| Adapter size | 6.2 MB |

The successful run supports the portfolio claim that this project can take curated video-caption metadata through a real Stable Diffusion LoRA fine-tuning workflow and produce checkpointed adapter weights.

## Notebook

See [`notebooks/kaggle_diffusion_lora_finetune.ipynb`](../notebooks/kaggle_diffusion_lora_finetune.ipynb).
