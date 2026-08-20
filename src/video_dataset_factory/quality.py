from __future__ import annotations

import cv2
import numpy as np

from video_dataset_factory.schema import QualityConfig, VideoMetadata


def blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def estimate_text_area_ratio(frame: np.ndarray) -> float:
    """Cheap OCR proxy until a real OCR model is configured.

    Text and watermarks often create dense high-contrast edges. This heuristic is not a
    replacement for PaddleOCR/EasyOCR, but gives the pipeline an auditable placeholder.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def aggregate_quality(frames: list[np.ndarray]) -> dict[str, float | None]:
    if not frames:
        return {
            "blur_score": None,
            "brightness_score": None,
            "ocr_text_area_ratio": None,
        }

    return {
        "blur_score": float(np.median([blur_score(frame) for frame in frames])),
        "brightness_score": float(np.median([brightness_score(frame) for frame in frames])),
        "ocr_text_area_ratio": float(np.median([estimate_text_area_ratio(frame) for frame in frames])),
    }


def quality_reject_reasons(
    metadata: VideoMetadata,
    quality: dict[str, float | None],
    config: QualityConfig,
) -> list[str]:
    reasons: list[str] = []

    if metadata.duration_sec < config.min_duration_sec:
        reasons.append("duration_too_short")
    if metadata.duration_sec > config.max_duration_sec:
        reasons.append("duration_too_long")
    if metadata.width < config.min_width or metadata.height < config.min_height:
        reasons.append("resolution_too_low")

    blur = quality.get("blur_score")
    if blur is not None and blur < config.min_blur_score:
        reasons.append("too_blurry")

    brightness = quality.get("brightness_score")
    if brightness is not None:
        if brightness < config.min_brightness:
            reasons.append("too_dark")
        if brightness > config.max_brightness:
            reasons.append("too_bright")

    text_ratio = quality.get("ocr_text_area_ratio")
    if text_ratio is not None and text_ratio > config.max_ocr_text_area_ratio:
        reasons.append("text_or_watermark_likely")

    return reasons
