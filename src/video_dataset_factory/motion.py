from __future__ import annotations

import cv2
import numpy as np

from video_dataset_factory.schema import QualityConfig


def motion_score(frames: list[np.ndarray]) -> float | None:
    if len(frames) < 2:
        return None

    scores: list[float] = []
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        scores.append(float(np.median(magnitude)))
        previous = current

    return float(np.median(scores)) if scores else None


def motion_reject_reasons(score: float | None, config: QualityConfig) -> list[str]:
    if score is None:
        return ["motion_unavailable"]
    if score < config.min_motion_score:
        return ["too_static"]
    if score > config.max_motion_score:
        return ["motion_too_fast_or_unstable"]
    return []


def motion_caption(score: float | None) -> str | None:
    if score is None:
        return None
    if score < 0.5:
        return "Mostly static shot with minimal camera or object movement."
    if score < 5.0:
        return "Moderate motion with visible object or camera movement."
    return "Fast or unstable motion; inspect before using for training."
