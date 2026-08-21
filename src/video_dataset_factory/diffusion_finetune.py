from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from video_dataset_factory.manifest import read_jsonl
from video_dataset_factory.video_io import sample_frames


@dataclass(frozen=True)
class DiffusionDatasetConfig:
    manifest_path: Path
    output_dir: Path = Path("outputs/diffusion_lora_dataset")
    frames_per_clip: int = 1
    max_clips: int | None = None
    resolution: int = 512
    caption_prefix: str = ""
    include_motion_caption: bool = True
    allow_missing_videos: bool = False


@dataclass(frozen=True)
class PreparedDiffusionDataset:
    output_dir: str
    metadata_path: str
    image_count: int
    source_clip_count: int
    skipped_clip_count: int
    resolution: int
    notes: str


@dataclass(frozen=True)
class DiffusionLoraCommandConfig:
    train_data_dir: Path = Path("outputs/diffusion_lora_dataset")
    output_dir: Path = Path("outputs/diffusion_lora")
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    train_script_path: Path = Path("scripts/train_text_to_image_lora.py")
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 120
    learning_rate: float = 1e-4
    mixed_precision: str = "fp16"
    rank: int = 8
    seed: int = 13


def prepare_diffusion_lora_dataset(config: DiffusionDatasetConfig) -> PreparedDiffusionDataset:
    records = [_as_record_dict(record) for record in read_jsonl(config.manifest_path)]
    records = [record for record in records if record.get("keep", True)]
    if config.max_clips is not None:
        records = records[: config.max_clips]

    image_dir = config.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_dir / "metadata.jsonl"

    metadata_rows: list[dict[str, str]] = []
    skipped = 0
    for record in records:
        source_path = _resolve_source_path(config.manifest_path, record)
        if source_path is None or not source_path.exists():
            skipped += 1
            if config.allow_missing_videos:
                continue
            raise FileNotFoundError(f"Missing source video for clip {record.get('clip_id')}: {source_path}")

        frames = sample_frames(source_path, config.frames_per_clip)
        if not frames:
            skipped += 1
            continue

        caption = _build_caption(record, config)
        clip_id = str(record.get("clip_id") or source_path.stem)
        for frame_index, frame in enumerate(frames):
            frame = _resize_square(frame, config.resolution)
            file_name = f"{_safe_name(clip_id)}_{frame_index:03d}.jpg"
            relative_path = Path("images") / file_name
            cv2.imwrite(str(config.output_dir / relative_path), frame)
            metadata_rows.append({"file_name": str(relative_path).replace("\\", "/"), "text": caption})

    metadata_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata_rows) + "\n",
        encoding="utf-8",
    )
    return PreparedDiffusionDataset(
        output_dir=str(config.output_dir),
        metadata_path=str(metadata_path),
        image_count=len(metadata_rows),
        source_clip_count=len(records),
        skipped_clip_count=skipped,
        resolution=config.resolution,
        notes="Diffusers imagefolder dataset with metadata.jsonl file_name/text rows.",
    )


def build_diffusers_lora_command(config: DiffusionLoraCommandConfig) -> list[str]:
    return [
        "accelerate",
        "launch",
        str(config.train_script_path),
        f"--pretrained_model_name_or_path={config.pretrained_model_name_or_path}",
        f"--train_data_dir={config.train_data_dir}",
        f"--resolution={config.resolution}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={config.train_batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--max_train_steps={config.max_train_steps}",
        f"--learning_rate={config.learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--rank={config.rank}",
        f"--mixed_precision={config.mixed_precision}",
        f"--seed={config.seed}",
        f"--output_dir={config.output_dir}",
    ]


def build_diffusers_lora_shell_command(config: DiffusionLoraCommandConfig) -> str:
    return " ".join(shlex.quote(part) for part in build_diffusers_lora_command(config))


def write_diffusion_lora_report(
    path: Path,
    dataset: PreparedDiffusionDataset,
    command: str,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Diffusion LoRA Fine-Tuning Plan",
            "",
            "| Item | Value |",
            "| --- | ---: |",
            f"| Base model | {model_name} |",
            f"| Dataset directory | {dataset.output_dir} |",
            f"| Metadata path | {dataset.metadata_path} |",
            f"| Source clips | {dataset.source_clip_count} |",
            f"| Exported images | {dataset.image_count} |",
            f"| Skipped clips | {dataset.skipped_clip_count} |",
            f"| Resolution | {dataset.resolution} |",
            "",
            "## Training Command",
            "",
            "```bash",
            command,
            "```",
            "",
            dataset.notes,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def write_prepared_dataset_json(path: Path, dataset: PreparedDiffusionDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(dataset), indent=2), encoding="utf-8")


def _as_record_dict(record: Any) -> dict:
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    if hasattr(record, "dict"):
        return record.dict()
    raise TypeError(f"Unsupported manifest record type: {type(record).__name__}")


def _resolve_source_path(manifest_path: Path, record: dict) -> Path | None:
    raw_path = record.get("source_path") or record.get("clip_path") or record.get("video_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        manifest_path.parent / path,
        manifest_path.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_caption(record: dict, config: DiffusionDatasetConfig) -> str:
    parts = []
    if config.caption_prefix:
        parts.append(config.caption_prefix.strip())
    caption = str(record.get("caption") or "A curated video frame.").strip()
    parts.append(caption)
    motion_caption = str(record.get("motion_caption") or "").strip()
    if config.include_motion_caption and motion_caption:
        parts.append(motion_caption)
    return " ".join(part for part in parts if part)


def _resize_square(frame, resolution: int):
    height, width = frame.shape[:2]
    crop = min(height, width)
    y0 = max((height - crop) // 2, 0)
    x0 = max((width - crop) // 2, 0)
    frame = frame[y0 : y0 + crop, x0 : x0 + crop]
    return cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)


def _safe_name(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "clip"
