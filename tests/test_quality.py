import numpy as np
import pytest

from video_dataset_factory.quality import (
    HeuristicAestheticScorer,
    _bbox_area_ratio,
    aggregate_quality,
    brightness_score,
    build_aesthetic_scorer,
    build_text_detector,
    quality_reject_reasons,
)
from video_dataset_factory.schema import QualityConfig, VideoMetadata


class FakeTextDetector:
    def text_area_ratio(self, frames):
        return 0.42


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


def test_aggregate_quality_uses_configured_text_detector():
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8)]

    result = aggregate_quality(frames, text_detector=FakeTextDetector())

    assert result["ocr_text_area_ratio"] == 0.42


def test_bbox_area_ratio_uses_confident_easyocr_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [
        ([[0, 0], [20, 0], [20, 10], [0, 10]], "logo", 0.9),
        ([[0, 0], [50, 0], [50, 50], [0, 50]], "weak", 0.1),
    ]

    ratio = _bbox_area_ratio(frame, boxes, min_confidence=0.35)

    assert ratio == pytest.approx(0.02)


def test_build_text_detector_returns_proxy_none_by_default():
    assert build_text_detector({}) is None
    assert build_text_detector({"provider": "proxy"}) is None


def test_build_text_detector_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported OCR provider"):
        build_text_detector({"provider": "unknown"})


def test_heuristic_aesthetic_score_is_bounded():
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8)]
    scorer = HeuristicAestheticScorer()

    score = scorer.score(frames)

    assert score is not None
    assert 0.0 <= score <= 10.0


def test_build_aesthetic_scorer_returns_none_by_default():
    assert build_aesthetic_scorer({}) is None


def test_build_laion_aesthetic_requires_head_path():
    with pytest.raises(ValueError, match="head_path"):
        build_aesthetic_scorer({"provider": "laion"})


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
