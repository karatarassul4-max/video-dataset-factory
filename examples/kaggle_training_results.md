# Kaggle T4x2 Training Benchmark Results

This report records a real Kaggle notebook run of the training benchmark on a two-GPU accelerator. The goal is to verify hands-on CUDA execution, Hugging Face Accelerate multi-GPU launch, optional DeepSpeed ZeRO-2 configuration, mixed precision, throughput logging, and peak VRAM measurement.

## Environment

| Item | Value |
| --- | --- |
| Platform | Kaggle Notebook |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Accelerate | 1.13.0 |
| DeepSpeed | 0.19.5 |
| Accelerator | NVIDIA Tesla T4 x2 |
| CUDA available | True |
| GPU count | 2 |
| GPU 0 | Tesla T4 |
| GPU 1 | Tesla T4 |

The downloaded notebook metadata showed `accelerator: none`, but the executed cells verified CUDA directly through PyTorch and both T4 devices were visible.

## Commands

Install project training dependencies:

```bash
git clone https://github.com/karatarassul4-max/video-dataset-factory.git
cd video-dataset-factory
pip install -e .[training]
pip install -e .[training,deepspeed]
```

Verify CUDA:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

Run single-process CUDA benchmark:

```bash
vdf benchmark-training --real --samples 2048 --epochs 2 --batch-size 64 \
  --mixed-precision fp16 \
  --output outputs/training_cuda.json \
  --markdown-output outputs/training_cuda.md
```

Run Accelerate multi-GPU benchmark:

```bash
accelerate launch --config_file configs/accelerate_kaggle.yaml \
  -m video_dataset_factory.training_entrypoint \
  --samples 4096 --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_accelerate.json \
  --markdown-output outputs/training_accelerate.md
```

Run DeepSpeed ZeRO-2 benchmark:

```bash
accelerate launch --config_file configs/accelerate_deepspeed_zero2.yaml \
  -m video_dataset_factory.training_entrypoint \
  --samples 4096 --epochs 2 --batch-size 64 --mixed-precision fp16 \
  --output outputs/training_deepspeed_zero2.json \
  --markdown-output outputs/training_deepspeed_zero2.md
```

## Results

| Run | Mode | Distributed type | GPU count | Mixed precision | Samples/sec | Peak VRAM MB | Final loss |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| Single-process CUDA | accelerate | DistributedType.NO | 2 | fp16 | 3106.05 | 27.64 | n/a |
| Accelerate multi-GPU | accelerate_distributed | DistributedType.MULTI_GPU | 2 | fp16 | 8461.54 | 29.75 | 4.0522 |
| DeepSpeed ZeRO-2 | accelerate_distributed | DistributedType.DEEPSPEED | 2 | fp16 | 5796.43 | 977.60 | 4.0469 |

## Interpretation

- CUDA was available and PyTorch saw both Tesla T4 GPUs.
- The single-process run used CUDA but did not use distributed training; it is a baseline for throughput and VRAM logging.
- The Accelerate run verified a real two-process multi-GPU launch with `DistributedType.MULTI_GPU`.
- The DeepSpeed run verified an optional ZeRO-2 launch path through Accelerate with `DistributedType.DEEPSPEED`.
- The benchmark is intentionally small, so the absolute VRAM numbers are not meant to represent large-model training. The important hiring signal is the reproducible GPU/distributed launch workflow and auditable metrics.

## Honest Resume Claim

This run supports wording like:

```text
Ran GPU training benchmarks on Kaggle T4x2 with PyTorch, Hugging Face Accelerate multi-GPU launch, and DeepSpeed ZeRO-2 configuration; measured throughput, mixed precision behavior, final loss, distributed backend, and peak CUDA memory.
```

It does not prove production GPU cluster administration. Avoid claims like "managed GPU clusters" unless backed by separate real infrastructure experience.
