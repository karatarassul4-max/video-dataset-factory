import numpy as np

from video_dataset_factory.quality import (
    HeuristicAestheticScorer,
    aggregate_quality,
    brightness_score,
    build_aesthetic_scorer,
    quality_reject_reasons,
)
from video_dataset_factory.schema import QualityConfig, VideoMetadata


def test_brightness_score_tracks_pixel_intensity():
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    bright = np.full((32, 32, 3), 220, dtype=np.uint8)

    assert brightness_score(bright) > brightness_score(dark)


def test_aggregate_quality_handles_empty_frames():
    result = aggregate_quality([])

    assert result["blur_score"] is None
    assert result["brightness_score"] is None
    assert result["ocr_text_area_ratio"] is None
    assert result["aesthetic_score"] is None


def test_heuristic_aesthetic_score_is_bounded():
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8)]
    scorer = HeuristicAestheticScorer()

    score = scorer.score(frames)

    assert score is not None
    assert 0.0 <= score <= 10.0


def test_build_aesthetic_scorer_returns_none_by_default():
    assert build_aesthetic_scorer({}) is None


def test_aggregate_quality_uses_aesthetic_scorer():
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8)]

    result = aggregate_quality(frames, aesthetic_scorer=HeuristicAestheticScorer())

    assert result["aesthetic_score"] is not None


def test_quality_rejects_low_aesthetic_score_when_threshold_is_enabled():
    metadata = VideoMetadata(
        source_path="clip.mp4",
        duration_sec=2.0,
        fps=24.0,
        width=512,
        height=512,
        frame_count=48,
    )
    quality = {
        "blur_score": 100.0,
        "brightness_score": 128.0,
        "ocr_text_area_ratio": 0.0,
        "aesthetic_score": 3.0,
    }
    config = QualityConfig(min_aesthetic_score=5.0)

    reasons = quality_reject_reasons(metadata, quality, config)

    assert reasons == ["aesthetic_score_too_low"]
