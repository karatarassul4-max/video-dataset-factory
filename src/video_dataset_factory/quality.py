from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from video_dataset_factory.schema import QualityConfig, VideoMetadata


class AestheticScorer(Protocol):
    def score(self, frames: list[np.ndarray]) -> float | None:
        ...


def blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def colorfulness_score(frame: np.ndarray) -> float:
    blue, green, red = cv2.split(frame.astype(np.float32))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    return float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def estimate_text_area_ratio(frame: np.ndarray) -> float:
    """Cheap OCR proxy until a real OCR model is configured.

    Text and watermarks often create dense high-contrast edges. This heuristic is not a
    replacement for PaddleOCR/EasyOCR, but gives the pipeline an auditable placeholder.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def aggregate_quality(
    frames: list[np.ndarray],
    aesthetic_scorer: AestheticScorer | None = None,
) -> dict[str, float | None]:
    if not frames:
        return {
            "blur_score": None,
            "brightness_score": None,
            "ocr_text_area_ratio": None,
            "aesthetic_score": None,
        }

    return {
        "blur_score": float(np.median([blur_score(frame) for frame in frames])),
        "brightness_score": float(np.median([brightness_score(frame) for frame in frames])),
        "ocr_text_area_ratio": float(np.median([estimate_text_area_ratio(frame) for frame in frames])),
        "aesthetic_score": None if aesthetic_scorer is None else aesthetic_scorer.score(frames),
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

    aesthetic = quality.get("aesthetic_score")
    if (
        config.min_aesthetic_score is not None
        and aesthetic is not None
        and aesthetic < config.min_aesthetic_score
    ):
        reasons.append("aesthetic_score_too_low")

    return reasons


class HeuristicAestheticScorer:
    """Fast CPU aesthetic proxy for smoke tests and offline filtering dry-runs."""

    def score(self, frames: list[np.ndarray]) -> float | None:
        if not frames:
            return None

        scores: list[float] = []
        for frame in frames:
            brightness = brightness_score(frame)
            brightness_component = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
            blur_component = min(1.0, blur_score(frame) / 500.0)
            color_component = min(1.0, colorfulness_score(frame) / 80.0)
            scores.append(10.0 * (0.45 * brightness_component + 0.35 * blur_component + 0.20 * color_component))

        return float(np.median(scores))


class CLIPAestheticScorer:
    """CLIP preference proxy for aesthetic filtering.

    This is a lightweight portfolio-friendly adapter: it compares each keyframe against a
    high-quality visual prompt and a low-quality visual prompt, then maps the probability to
    a 0-10 score. A production version could swap this class for LAION's linear aesthetic
    head without changing the rest of the pipeline.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        max_frames: int = 4,
    ):
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install aesthetic dependencies with `pip install -e .[aesthetic]`.") from exc

        self.torch = torch
        self.image_cls = Image
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device != "auto":
            self.model = self.model.to(self.device)
        self.max_frames = max_frames

    def score(self, frames: list[np.ndarray]) -> float | None:
        if not frames:
            return None

        selected = _select_evenly_spaced(frames, self.max_frames)
        images = [self._to_pil(frame) for frame in selected]
        prompts = [
            "a high quality cinematic image with pleasing composition and lighting",
            "a low quality blurry image with bad composition or watermark artifacts",
        ]
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            logits = self.model(**inputs).logits_per_image
            probabilities = logits.softmax(dim=-1)[:, 0]
        return float(probabilities.median().item() * 10.0)

    def _to_pil(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.image_cls.fromarray(rgb)


def build_aesthetic_scorer(settings: dict | None) -> AestheticScorer | None:
    settings = settings or {}
    provider = settings.get("provider", "none")
    if provider in {"none", None}:
        return None
    if provider == "heuristic":
        return HeuristicAestheticScorer()
    if provider == "clip":
        return CLIPAestheticScorer(
            model_name=settings.get("model_name", "openai/clip-vit-base-patch32"),
            device=settings.get("device", "auto"),
            max_frames=int(settings.get("max_frames", 4)),
        )
    raise ValueError(f"Unsupported aesthetic provider: {provider}")


def _select_evenly_spaced(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
    return [frames[int(index)] for index in indices]
