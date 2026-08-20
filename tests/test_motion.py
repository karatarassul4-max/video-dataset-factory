import numpy as np

from video_dataset_factory.motion import motion_score


def test_motion_score_requires_multiple_frames():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert motion_score([frame]) is None
