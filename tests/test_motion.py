import numpy as np

from video_dataset_factory.motion import motion_metrics, motion_score


def test_motion_score_requires_multiple_frames():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert motion_score([frame]) is None


def test_motion_metrics_empty_without_multiple_frames():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    metrics = motion_metrics([frame])

    assert metrics == {
        "motion_score": None,
        "motion_p95_score": None,
        "motion_stability_score": None,
    }


def test_motion_metrics_include_p95_and_stability():
    frame_a = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_b = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_c = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_a[20:40, 10:30] = 255
    frame_b[20:40, 16:36] = 255
    frame_c[20:40, 22:42] = 255

    metrics = motion_metrics([frame_a, frame_b, frame_c])

    assert metrics["motion_score"] is not None
    assert metrics["motion_p95_score"] is not None
    assert metrics["motion_stability_score"] is not None
    assert metrics["motion_p95_score"] >= metrics["motion_score"]
    assert 0.0 <= metrics["motion_stability_score"] <= 1.0
