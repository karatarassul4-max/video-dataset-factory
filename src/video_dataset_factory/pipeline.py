from __future__ import annotations

import hashlib
from pathlib import Path

from video_dataset_factory.caption import Captioner
from video_dataset_factory.motion import motion_caption, motion_reject_reasons, motion_score
from video_dataset_factory.quality import aggregate_quality, quality_reject_reasons
from video_dataset_factory.schema import AppConfig, ClipRecord
from video_dataset_factory.video_io import probe_video, sample_frames


def stable_clip_id(path: Path) -> str:
    resolved = str(path.resolve()).encode("utf-8", errors="ignore")
    return hashlib.sha1(resolved).hexdigest()[:16]


def process_video(path: Path, config: AppConfig, captioner: Captioner | None = None) -> ClipRecord:
    captioner = captioner or Captioner()
    metadata = probe_video(path)
    frames = sample_frames(path, config.pipeline.sample_frames)

    quality = aggregate_quality(frames)
    motion = motion_score(frames)

    reasons = quality_reject_reasons(metadata, quality, config.quality)
    reasons.extend(motion_reject_reasons(motion, config.quality))

    return ClipRecord(
        clip_id=stable_clip_id(path),
        source_path=str(path),
        duration_sec=metadata.duration_sec,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        frame_count=metadata.frame_count,
        blur_score=quality["blur_score"],
        brightness_score=quality["brightness_score"],
        motion_score=motion,
        ocr_text_area_ratio=quality["ocr_text_area_ratio"],
        aesthetic_score=None,
        caption=captioner.caption(frames),
        motion_caption=motion_caption(motion),
        keep=not reasons,
        reject_reasons=reasons,
    )
