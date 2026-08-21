# Diffusion LoRA Fine-Tuning Runbook

This runbook closes the end-to-end diffusion fine-tuning path for the project. It starts from the curated video manifest, exports training images and captions, then runs a small Stable Diffusion LoRA fine-tune with Hugging Face Diffusers and Accelerate.

This is intentionally a LoRA fine-tune, not full pre-training. LoRA is a lightweight fine-tuning method that trains a small number of adapter weights, making it practical on Kaggle T4 GPUs.

## What This Proves

- End-to-end path from raw video curation to a diffusion fine-tuning dataset.
- Stable Diffusion LoRA fine-tuning workflow with Diffusers.
- Accelerate launch flow, mixed precision, checkpoints, and output LoRA weights.
- Practical understanding of the boundary between data curation, model fine-tuning, and full foundation model pre-training.

## 1. Install Dependencies

```bash
pip install -e .[diffusion-finetune]
mkdir -p scripts
curl -L \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py \
  -o scripts/train_text_to_image_lora.py
```

The training script is the official Hugging Face Diffusers text-to-image LoRA example.

## 2. Prepare Video-Caption Dataset For Diffusion Fine-Tuning

First create a real manifest with accepted clips:

```bash
vdf process-folder data/clips --output outputs/manifest.jsonl
vdf dedupe-manifest outputs/manifest.jsonl --output outputs/manifest_deduped.jsonl
```

Then export sampled frames plus captions into a Diffusers imagefolder dataset:

```bash
vdf prepare-diffusion-lora-data outputs/manifest_deduped.jsonl \
  --output-dir outputs/diffusion_lora_dataset \
  --frames-per-clip 1 \
  --max-clips 100 \
  --resolution 512 \
  --report outputs/diffusion_lora_plan.md \
  --json-report outputs/diffusion_lora_dataset.json
```

The output structure is:

```text
outputs/diffusion_lora_dataset/
  images/
    clip_000.jpg
    clip_001.jpg
  metadata.jsonl
```

Each `metadata.jsonl` row contains:

```json
{"file_name": "images/example_000.jpg", "text": "A dense VLM caption for the frame."}
```

## 3. Run A Small Kaggle LoRA Fine-Tune

Use a small step count first to verify the full path:

```bash
accelerate launch scripts/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --train_data_dir=outputs/diffusion_lora_dataset \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=120 \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --rank=8 \
  --mixed_precision=fp16 \
  --seed=13 \
  --output_dir=outputs/diffusion_lora
```

Expected output includes LoRA weights in `outputs/diffusion_lora/`, usually as a safetensors file.

## 4. Verify The LoRA Weights With Inference

```python
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_lora_weights("outputs/diffusion_lora")
image = pipe(
    "a cinematic frame with smooth camera motion and natural light",
    num_inference_steps=20,
).images[0]
image.save("outputs/diffusion_lora_sample.png")
```

## Honest Resume Claim

After this run succeeds, the project supports wording like:

```text
Built an end-to-end diffusion fine-tuning workflow that converts curated video-caption manifests into a Diffusers image-caption dataset and runs a Stable Diffusion LoRA fine-tune with Accelerate, mixed precision, and checkpointed adapter outputs.
```

Do not claim full foundation model pre-training or text-to-video model training from scratch. This workflow demonstrates LoRA fine-tuning of a pretrained diffusion foundation model.

## References

- Hugging Face Diffusers LoRA training guide: https://huggingface.co/docs/diffusers/training/lora
- Official Diffusers LoRA training script: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
