import numpy as np
import pytest

from video_dataset_factory.quality import (
    HeuristicAestheticScorer,
    _bbox_area_ratio,
    _get_clip_image_features,
    aggregate_quality,
    brightness_score,
    build_aesthetic_scorer,
    build_text_detector,
    quality_reject_reasons,
    resolve_laion_head_path,
)
from video_dataset_factory.schema import QualityConfig, VideoMetadata


class FakeTextDetector:
    def text_area_ratio(self, frames):
        return 0.42


class FakeTensorLike:
    def norm(self, *args, **kwargs):
        return self


class FakePoolerOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class FakeCLIPModelWithOutput:
    def get_image_features(self, **inputs):
        return FakePoolerOutput(inputs["pixel_values"])

    def visual_projection(self, pooled):
        return pooled + 1


class FakeCLIPModelWithTensor:
    def get_image_features(self, **inputs):
        return inputs["pixel_values"]


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


def test_get_clip_image_features_keeps_tensor_outputs():
    values = FakeTensorLike()

    features = _get_clip_image_features(FakeCLIPModelWithTensor(), {"pixel_values": values})

    assert features is values


def test_get_clip_image_features_projects_pooler_output_objects():
    values = np.asarray([[1.0, 2.0]], dtype=np.float32)

    features = _get_clip_image_features(FakeCLIPModelWithOutput(), {"pixel_values": values})

    np.testing.assert_array_equal(features, values + 1)


def test_resolve_laion_head_path_uses_existing_file(tmp_path):
    head = tmp_path / "head.pth"
    head.write_bytes(b"weights")

    assert resolve_laion_head_path(head_path=head, head_url=None) == head


def test_resolve_laion_head_path_downloads_missing_file(tmp_path, monkeypatch):
    head = tmp_path / "head.pth"

    def fake_urlretrieve(url, filename):
        assert url == "https://example.test/head.pth"
        np.asarray([1], dtype=np.float32).tofile(filename)
        return filename, None

    monkeypatch.setattr("video_dataset_factory.quality.urllib.request.urlretrieve", fake_urlretrieve)

    assert resolve_laion_head_path(head_path=head, head_url="https://example.test/head.pth") == head
    assert head.exists()


def test_resolve_laion_head_path_requires_url_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="LAION aesthetic head not found"):
        resolve_laion_head_path(head_path=tmp_path / "missing.pth", head_url=None)


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
