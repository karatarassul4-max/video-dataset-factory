from __future__ import annotations

import cv2
import numpy as np

from video_dataset_factory.schema import QualityConfig


def motion_score(frames: list[np.ndarray]) -> float | None:
    metrics = motion_metrics(frames)
    return metrics["motion_score"]


def motion_metrics(frames: list[np.ndarray]) -> dict[str, float | None]:
    magnitudes = _motion_magnitudes(frames)
    if not magnitudes:
        return {
            "motion_score": None,
            "motion_p95_score": None,
            "motion_stability_score": None,
        }

    values = np.asarray(magnitudes, dtype=np.float32)
    median = float(np.median(values))
    p95 = float(np.percentile(values, 95))
    spread = float(np.std(values))
    stability = float(1.0 / (1.0 + spread / max(median, 1e-6)))

    return {
        "motion_score": median,
        "motion_p95_score": p95,
        "motion_stability_score": stability,
    }


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


def _motion_magnitudes(frames: list[np.ndarray]) -> list[float]:
    if len(frames) < 2:
        return []

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

    return scores
