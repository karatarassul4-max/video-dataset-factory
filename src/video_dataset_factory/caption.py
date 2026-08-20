from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptionContext:
    clip_id: str
    source_path: str
    motion_caption: str | None = None


class Captioner(Protocol):
    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        ...


def build_dense_caption_prompt(context: CaptionContext | None = None) -> str:
    motion_hint = context.motion_caption if context and context.motion_caption else "unknown motion"
    return (
        "Describe this video clip for text-to-video model training. Include subject, action, "
        "camera motion, scene, lighting, visual style, and temporal dynamics. "
        f"Motion estimate: {motion_hint}. Avoid guessing identities or protected attributes."
    )


def frame_signature(frames: list[np.ndarray]) -> str:
    digest = hashlib.sha1()
    for frame in frames:
        digest.update(str(frame.shape).encode("utf-8"))
        digest.update(np.asarray(frame.mean(axis=(0, 1)), dtype=np.float32).tobytes())
    return digest.hexdigest()


class HeuristicCaptioner:
    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        if not frames:
            return "Video clip with unavailable visual content."

        mean_brightness = float(np.mean([np.mean(frame) for frame in frames]))
        if mean_brightness < 70:
            lighting = "dark"
        elif mean_brightness > 180:
            lighting = "bright"
        else:
            lighting = "normally lit"

        motion = context.motion_caption if context and context.motion_caption else "motion not estimated"
        return f"A {lighting} video clip; {motion}. Replace with VLM dense captioning."


class CachedCaptioner:
    def __init__(self, backend: Captioner, cache_path: Path):
        self.backend = backend
        self.cache_path = cache_path
        self.cache: dict[str, str] = self._load_cache()

    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        key = context.clip_id if context else frame_signature(frames)
        if key not in self.cache:
            self.cache[key] = self.backend.caption(frames, context)
            self._save_cache()
        return self.cache[key]

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self.cache, handle, indent=2, ensure_ascii=False)


class TransformersVLMCaptioner:
    """Best-effort Hugging Face VLM adapter.

    Some VLMs require model-specific processors or chat templates. This adapter keeps the
    project interface honest while allowing Qwen/LLaVA experiments behind an optional extra.
    """

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 128):
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install captioning dependencies with `pip install -e .[captioning]`.") from exc

        self.torch = torch
        self.image_cls = Image
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        if not frames:
            return "Video clip with unavailable visual content."

        prompt = build_dense_caption_prompt(context)
        image = self._to_pil(frames[len(frames) // 2])
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def _to_pil(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.image_cls.fromarray(rgb)


def build_captioner(settings: dict) -> Captioner:
    provider = settings.get("provider", "heuristic")
    if provider == "heuristic":
        backend: Captioner = HeuristicCaptioner()
    elif provider == "transformers":
        model_name = settings.get("model_name")
        if not model_name:
            raise ValueError("captioning.model_name is required for provider=transformers")
        backend = TransformersVLMCaptioner(model_name=model_name)
    else:
        raise ValueError(f"Unsupported captioning provider: {provider}")

    cache_path = settings.get("cache_path")
    if cache_path:
        return CachedCaptioner(backend, Path(cache_path))
    return backend
