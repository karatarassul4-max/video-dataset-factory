from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    source_path: str
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0


class QualityConfig(BaseModel):
    min_duration_sec: float = 1.0
    max_duration_sec: float = 20.0
    min_width: int = 256
    min_height: int = 256
    min_blur_score: float = 40.0
    min_brightness: float = 20.0
    max_brightness: float = 235.0
    max_ocr_text_area_ratio: float = 0.08
    min_motion_score: float = 0.2
    max_motion_score: float = 50.0
    min_aesthetic_score: float | None = None


class PipelineConfig(BaseModel):
    sample_frames: int = 8
    output_manifest: Path = Path("outputs/manifest.jsonl")


class SceneSplitConfig(BaseModel):
    detector: str = content
    threshold: float = 27.0
    min_scene_len_frames: int = 15
    output_dir: Path = Path("data/clips")
    output_fps: int = 24
    output_width: int = 512
    output_height: int = 512
    crf: int = 18
    preset: str = "medium"


class AppConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    scene_split: SceneSplitConfig = Field(default_factory=SceneSplitConfig)
    captioning: dict[str, Any] = Field(default_factory=dict)
    ocr: dict[str, Any] = Field(default_factory=dict)
    aesthetic: dict[str, Any] = Field(default_factory=dict)
    ray: dict[str, Any] = Field(default_factory=dict)


class ClipRecord(BaseModel):
    clip_id: str
    source_path: str
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int
    blur_score: float | None = None
    brightness_score: float | None = None
    contrast_score: float | None = None
    colorfulness_score: float | None = None
    motion_score: float | None = None
    motion_p95_score: float | None = None
    motion_stability_score: float | None = None
    ocr_text_area_ratio: float | None = None
    aesthetic_score: float | None = None
    perceptual_hash: str | None = None
    duplicate_of: str | None = None
    caption: str | None = None
    motion_caption: str | None = None
    keep: bool
    reject_reasons: list[str] = Field(default_factory=list)


class SceneSegment(BaseModel):
    source_path: str
    clip_path: str
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
