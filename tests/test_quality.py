import numpy as np

from video_dataset_factory.quality import aggregate_quality, brightness_score


def test_brightness_score_tracks_pixel_intensity():
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    bright = np.full((32, 32, 3), 220, dtype=np.uint8)

    assert brightness_score(bright) > brightness_score(dark)


def test_aggregate_quality_handles_empty_frames():
    result = aggregate_quality([])

    assert result["blur_score"] is None
    assert result["brightness_score"] is None
    assert result["ocr_text_area_ratio"] is None
