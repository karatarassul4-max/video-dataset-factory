# Kaggle Notebooks

These notebooks are portfolio-clean records of the GPU experiments used by this project. They keep the important commands and measured outputs while removing noisy package download logs and intermediate failed attempts.

## Notebooks

| Notebook | What it shows |
| --- | --- |
| [`kaggle_t4x2_training_benchmark.ipynb`](kaggle_t4x2_training_benchmark.ipynb) | CUDA availability on Kaggle Tesla T4 x2, PyTorch training benchmark, Accelerate multi-GPU launch, and DeepSpeed ZeRO-2 launch. |
| [`kaggle_diffusion_lora_finetune.ipynb`](kaggle_diffusion_lora_finetune.ipynb) | End-to-end manifest-to-Diffusers dataset export and Stable Diffusion LoRA fine-tuning with saved adapter weights. |

## Key Results

| Experiment | Result |
| --- | ---: |
| Accelerate multi-GPU throughput | 8461.54 samples/sec |
| DeepSpeed ZeRO-2 throughput | 5796.43 samples/sec |
| Diffusion LoRA training examples | 100 |
| Diffusion LoRA optimization steps | 120 |
| Final LoRA logged step loss | 0.0495 |
| Saved LoRA adapter size | 6.2 MB |

The LoRA notebook uses UCF101 as a smoke-test video source. The production filters correctly reject the raw UCF101 clips for low source resolution, so the notebook creates a relaxed manifest only to validate the downstream fine-tuning path while preserving the original reject reasons.
