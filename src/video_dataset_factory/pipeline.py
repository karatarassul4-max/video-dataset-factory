from __future__ import annotations

import hashlib
from pathlib import Path

from video_dataset_factory.caption import CaptionContext, Captioner, build_captioner
from video_dataset_factory.duplicates import clip_perceptual_hash
from video_dataset_factory.motion import motion_caption, motion_reject_reasons, motion_score
from video_dataset_factory.quality import (
    AestheticScorer,
    TextDetector,
    aggregate_quality,
    build_aesthetic_scorer,
    build_text_detector,
    quality_reject_reasons,
)
from video_dataset_factory.schema import AppConfig, ClipRecord
from video_dataset_factory.video_io import probe_video, sample_frames


def stable_clip_id(path: Path) -> str:
    resolved = str(path.resolve()).encode("utf-8", errors="ignore")
    return hashlib.sha1(resolved).hexdigest()[:16]


def process_video(
    path: Path,
    config: AppConfig,
    captioner: Captioner | None = None,
    aesthetic_scorer: AestheticScorer | None = None,
    text_detector: TextDetector | None = None,
) -> ClipRecord:
    captioner = captioner or build_captioner(config.captioning)
    aesthetic_scorer = aesthetic_scorer or build_aesthetic_scorer(config.aesthetic)
    text_detector = text_detector or build_text_detector(config.ocr)
    metadata = probe_video(path)
    frames = sample_frames(path, config.pipeline.sample_frames)

    quality = aggregate_quality(
        frames,
        aesthetic_scorer=aesthetic_scorer,
        text_detector=text_detector,
    )
    motion = motion_score(frames)
    motion_text = motion_caption(motion)

    reasons = quality_reject_reasons(metadata, quality, config.quality)
    reasons.extend(motion_reject_reasons(motion, config.quality))

    clip_id = stable_clip_id(path)
    context = CaptionContext(clip_id=clip_id, source_path=str(path), motion_caption=motion_text)

    return ClipRecord(
        clip_id=clip_id,
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
        aesthetic_score=quality["aesthetic_score"],
        perceptual_hash=clip_perceptual_hash(frames),
        caption=captioner.caption(frames, context),
        motion_caption=motion_text,
        keep=not reasons,
        reject_reasons=reasons,
    )
