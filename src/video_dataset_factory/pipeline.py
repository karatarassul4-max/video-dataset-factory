from __future__ import annotations

import hashlib
from pathlib import Path

from video_dataset_factory.caption import (
    CaptionContext,
    Captioner,
    build_captioner,
    caption_reject_reasons,
)
from video_dataset_factory.duplicates import clip_perceptual_hash
from video_dataset_factory.motion import motion_caption, motion_metrics, motion_reject_reasons
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
    motion = motion_metrics(frames)
    motion_text = motion_caption(motion["motion_score"])

    reasons = quality_reject_reasons(metadata, quality, config.quality)
    reasons.extend(motion_reject_reasons(motion["motion_score"], config.quality))

    clip_id = stable_clip_id(path)
    context = CaptionContext(clip_id=clip_id, source_path=str(path), motion_caption=motion_text)
    caption = captioner.caption(frames, context)
    reasons.extend(caption_reject_reasons(caption))

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
        contrast_score=quality["contrast_score"],
        colorfulness_score=quality["colorfulness_score"],
        motion_score=motion["motion_score"],
        motion_p95_score=motion["motion_p95_score"],
        motion_stability_score=motion["motion_stability_score"],
        ocr_text_area_ratio=quality["ocr_text_area_ratio"],
        aesthetic_score=quality["aesthetic_score"],
        perceptual_hash=clip_perceptual_hash(frames),
        caption=caption,
        motion_caption=motion_text,
        keep=not reasons,
        reject_reasons=reasons,
    )
