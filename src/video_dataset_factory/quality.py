from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from video_dataset_factory.schema import QualityConfig, VideoMetadata

DEFAULT_LAION_HEAD_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/"
    "raw/main/sa_0_4_vit_b_32_linear.pth"
)
DEFAULT_LAION_HEAD_PATH = Path.home() / ".cache" / "video_dataset_factory" / "sa_0_4_vit_b_32_linear.pth"


class AestheticScorer(Protocol):
    def score(self, frames: list[np.ndarray]) -> float | None:
        ...


class TextDetector(Protocol):
    def text_area_ratio(self, frames: list[np.ndarray]) -> float | None:
        ...


def blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def contrast_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def colorfulness_score(frame: np.ndarray) -> float:
    blue, green, red = cv2.split(frame.astype(np.float32))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    return float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def estimate_text_area_ratio(frame: np.ndarray) -> float:
    """Cheap OCR proxy used when no real OCR backend is configured."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def aggregate_quality(
    frames: list[np.ndarray],
    aesthetic_scorer: AestheticScorer | None = None,
    text_detector: TextDetector | None = None,
) -> dict[str, float | None]:
    if not frames:
        return {
            "blur_score": None,
            "brightness_score": None,
            "contrast_score": None,
            "colorfulness_score": None,
            "ocr_text_area_ratio": None,
            "aesthetic_score": None,
        }

    text_area_ratio = (
        text_detector.text_area_ratio(frames)
        if text_detector is not None
        else float(np.median([estimate_text_area_ratio(frame) for frame in frames]))
    )

    return {
        "blur_score": float(np.median([blur_score(frame) for frame in frames])),
        "brightness_score": float(np.median([brightness_score(frame) for frame in frames])),
        "contrast_score": float(np.median([contrast_score(frame) for frame in frames])),
        "colorfulness_score": float(np.median([colorfulness_score(frame) for frame in frames])),
        "ocr_text_area_ratio": text_area_ratio,
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
    """CLIP prompt-comparison proxy for aesthetic filtering."""

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
        self.device = _resolve_torch_device(torch, device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
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
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            logits = self.model(**inputs).logits_per_image
            probabilities = logits.softmax(dim=-1)[:, 0]
        return float(probabilities.median().item() * 10.0)

    def _to_pil(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.image_cls.fromarray(rgb)


class LAIONAestheticScorer:
    """LAION-style linear aesthetic head over normalized CLIP image embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        head_path: str | Path | None = None,
        head_url: str | None = DEFAULT_LAION_HEAD_URL,
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
        self.device = _resolve_torch_device(torch, device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_frames = max_frames

        projection_dim = int(getattr(self.model.config, "projection_dim", 512))
        self.linear_head = torch.nn.Linear(projection_dim, 1).to(self.device)
        self._load_linear_head(resolve_laion_head_path(head_path, head_url))
        self.linear_head.eval()

    def score(self, frames: list[np.ndarray]) -> float | None:
        if not frames:
            return None

        selected = _select_evenly_spaced(frames, self.max_frames)
        images = [self._to_pil(frame) for frame in selected]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            features = _get_clip_image_features(self.model, inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = self.linear_head(features).squeeze(-1)
        return float(scores.median().clamp(0.0, 10.0).item())

    def _load_linear_head(self, head_path: Path) -> None:
        state = self.torch.load(head_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and {"weight", "bias"}.issubset(state):
            self.linear_head.load_state_dict({"weight": state["weight"], "bias": state["bias"]})
            return
        if isinstance(state, dict):
            normalized = {
                key.removeprefix("linear.").removeprefix("model."): value
                for key, value in state.items()
            }
            self.linear_head.load_state_dict(normalized, strict=False)
            return
        raise ValueError("Unsupported LAION aesthetic head checkpoint format")

    def _to_pil(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.image_cls.fromarray(rgb)


class EasyOCRTextDetector:
    """Text/watermark detector using EasyOCR bounding boxes."""

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        max_frames: int = 4,
        min_confidence: float = 0.35,
    ):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("Install OCR dependencies with `pip install -e .[ocr]`.") from exc

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)
        self.max_frames = max_frames
        self.min_confidence = min_confidence

    def text_area_ratio(self, frames: list[np.ndarray]) -> float | None:
        ratios = [
            _bbox_area_ratio(frame, self._read_frame(frame), min_confidence=self.min_confidence)
            for frame in _select_evenly_spaced(frames, self.max_frames)
        ]
        return None if not ratios else float(np.median(ratios))

    def _read_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.reader.readtext(rgb)


class TesseractTextDetector:
    """Text/watermark detector using pytesseract word boxes."""

    def __init__(self, max_frames: int = 4, min_confidence: float = 35.0):
        try:
            import pytesseract
            from PIL import Image
            from pytesseract import Output
        except ImportError as exc:
            raise RuntimeError("Install OCR dependencies with `pip install -e .[ocr]`.") from exc

        self.pytesseract = pytesseract
        self.output_dict = Output.DICT
        self.image_cls = Image
        self.max_frames = max_frames
        self.min_confidence = min_confidence

    def text_area_ratio(self, frames: list[np.ndarray]) -> float | None:
        ratios = [self._frame_ratio(frame) for frame in _select_evenly_spaced(frames, self.max_frames)]
        return None if not ratios else float(np.median(ratios))

    def _frame_ratio(self, frame: np.ndarray) -> float:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.image_cls.fromarray(rgb)
        data = self.pytesseract.image_to_data(image, output_type=self.output_dict)
        frame_area = float(frame.shape[0] * frame.shape[1])
        area = 0.0
        for index, text in enumerate(data.get("text", [])):
            if not str(text).strip():
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, ValueError, TypeError):
                continue
            if confidence < self.min_confidence:
                continue
            area += float(data["width"][index]) * float(data["height"][index])
        return min(1.0, area / frame_area)


def resolve_laion_head_path(
    head_path: str | Path | None = None,
    head_url: str | None = DEFAULT_LAION_HEAD_URL,
) -> Path:
    path = Path(head_path).expanduser() if head_path else DEFAULT_LAION_HEAD_PATH
    if path.exists():
        return path
    if not head_url:
        raise FileNotFoundError(f"LAION aesthetic head not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(head_url, path)
    return path


def build_text_detector(settings: dict | None) -> TextDetector | None:
    settings = settings or {}
    provider = settings.get("provider", "proxy")
    if provider in {"proxy", "none", None}:
        return None
    if provider == "easyocr":
        languages = settings.get("languages", ["en"])
        return EasyOCRTextDetector(
            languages=list(languages),
            gpu=bool(settings.get("gpu", False)),
            max_frames=int(settings.get("max_frames", 4)),
            min_confidence=float(settings.get("min_confidence", 0.35)),
        )
    if provider == "tesseract":
        return TesseractTextDetector(
            max_frames=int(settings.get("max_frames", 4)),
            min_confidence=float(settings.get("min_confidence", 35.0)),
        )
    raise ValueError(f"Unsupported OCR provider: {provider}")


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
    if provider in {"laion", "laion-aesthetic"}:
        return LAIONAestheticScorer(
            model_name=settings.get("model_name", "openai/clip-vit-base-patch32"),
            head_path=settings.get("head_path"),
            head_url=settings.get("head_url", DEFAULT_LAION_HEAD_URL),
            device=settings.get("device", "auto"),
            max_frames=int(settings.get("max_frames", 4)),
        )
    raise ValueError(f"Unsupported aesthetic provider: {provider}")


def _select_evenly_spaced(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
    return [frames[int(index)] for index in indices]


def _resolve_torch_device(torch_module: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _get_clip_image_features(model: Any, inputs: dict[str, Any]) -> Any:
    if hasattr(model, "get_image_features"):
        features = model.get_image_features(**inputs)
        if hasattr(features, "norm"):
            return features
        if hasattr(features, "image_embeds"):
            return features.image_embeds
        if hasattr(features, "pooler_output"):
            return _project_clip_pooler(model, features.pooler_output)

    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        raise TypeError("CLIP model does not expose image features or a vision_model output")

    outputs = vision_model(**inputs)
    if not hasattr(outputs, "pooler_output"):
        raise TypeError("CLIP vision output does not include pooler_output")
    return _project_clip_pooler(model, outputs.pooler_output)


def _project_clip_pooler(model: Any, pooled: Any) -> Any:
    projection = getattr(model, "visual_projection", None)
    if projection is None:
        return pooled

    pooled_dim = _last_dim(pooled)
    in_features = getattr(projection, "in_features", None)
    if pooled_dim is not None and in_features is not None and pooled_dim != in_features:
        return pooled

    return projection(pooled)


def _last_dim(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def _bbox_area_ratio(
    frame: np.ndarray,
    boxes: list,
    min_confidence: float,
) -> float:
    frame_area = float(frame.shape[0] * frame.shape[1])
    area = 0.0
    for box in boxes:
        if len(box) < 3:
            continue
        confidence = float(box[2])
        if confidence < min_confidence:
            continue
        points = np.asarray(box[0], dtype=np.float32)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        area += max(0.0, float(x_max - x_min)) * max(0.0, float(y_max - y_min))
    return min(1.0, area / frame_area)
