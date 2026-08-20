from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


class BatchCaptioner(Captioner, Protocol):
    def batch_caption(
        self,
        clips: list[list[np.ndarray]],
        contexts: list[CaptionContext | None] | None = None,
    ) -> list[str]:
        ...


def build_dense_caption_prompt(context: CaptionContext | None = None) -> str:
    motion_hint = context.motion_caption if context and context.motion_caption else "unknown motion"
    return (
        "Describe this video clip for text-to-video model training. Include subject, action, "
        "camera motion, scene, lighting, visual style, and temporal dynamics. "
        f"Motion estimate: {motion_hint}. Avoid guessing identities or protected attributes."
    )


def select_keyframes(frames: list[np.ndarray], max_frames: int = 4) -> list[np.ndarray]:
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if len(frames) <= max_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
    return [frames[int(index)] for index in indices]


def build_vlm_messages(
    frames: list[np.ndarray],
    context: CaptionContext | None = None,
    model_family: str = "qwen2-vl",
) -> list[dict[str, Any]]:
    prompt = build_dense_caption_prompt(context)
    family = model_family.lower()

    if family in {"qwen", "qwen2-vl", "qwen2_5-vl"}:
        content: list[dict[str, str]] = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    if family in {"llava", "llava-next", "llava_onevision"}:
        image_tokens = "\n".join(["<image>" for _ in frames])
        return [{"role": "user", "content": f"{image_tokens}\n{prompt}"}]

    return [{"role": "user", "content": prompt}]


def frame_signature(frames: list[np.ndarray]) -> str:
    digest = hashlib.sha1()
    for frame in frames:
        digest.update(str(frame.shape).encode("utf-8"))
        digest.update(np.asarray(frame.mean(axis=(0, 1)), dtype=np.float32).tobytes())
    return digest.hexdigest()


def batch_caption(
    captioner: Captioner,
    clips: list[list[np.ndarray]],
    contexts: list[CaptionContext | None] | None = None,
) -> list[str]:
    if contexts is not None and len(contexts) != len(clips):
        raise ValueError("contexts must have the same length as clips")
    if isinstance(captioner, BatchCaptioner):
        return captioner.batch_caption(clips, contexts)
    return [
        captioner.caption(frames, None if contexts is None else contexts[index])
        for index, frames in enumerate(clips)
    ]


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

    def batch_caption(
        self,
        clips: list[list[np.ndarray]],
        contexts: list[CaptionContext | None] | None = None,
    ) -> list[str]:
        if contexts is not None and len(contexts) != len(clips):
            raise ValueError("contexts must have the same length as clips")
        return [
            self.caption(frames, None if contexts is None else contexts[index])
            for index, frames in enumerate(clips)
        ]

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
    """Hugging Face VLM adapter for Qwen2-VL/LLaVA-style dense captioning."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 160,
        max_keyframes: int = 4,
        model_family: str = "qwen2-vl",
    ):
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
        self.max_keyframes = max_keyframes
        self.model_family = model_family

    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        return self.batch_caption([frames], [context])[0]

    def batch_caption(
        self,
        clips: list[list[np.ndarray]],
        contexts: list[CaptionContext | None] | None = None,
    ) -> list[str]:
        if contexts is not None and len(contexts) != len(clips):
            raise ValueError("contexts must have the same length as clips")

        prompts: list[str] = []
        image_batches: list[list[Any]] = []
        for index, frames in enumerate(clips):
            keyframes = select_keyframes(frames, self.max_keyframes)
            context = None if contexts is None else contexts[index]
            prompts.append(self._format_prompt(keyframes, context))
            image_batches.append([self._to_pil(frame) for frame in keyframes])

        inputs = self.processor(text=prompts, images=image_batches, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)

    def _format_prompt(self, frames: list[np.ndarray], context: CaptionContext | None) -> str:
        messages = build_vlm_messages(frames, context, self.model_family)
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return build_dense_caption_prompt(context)

    def _to_pil(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.image_cls.fromarray(rgb)


def build_captioner(settings: dict) -> Captioner:
    provider = settings.get("provider", "heuristic")
    model_name = settings.get("model_name")
    if provider == "heuristic" or not model_name:
        backend: Captioner = HeuristicCaptioner()
    elif provider == "transformers":
        backend = TransformersVLMCaptioner(
            model_name=model_name,
            device=settings.get("device", "auto"),
            max_new_tokens=int(settings.get("max_new_tokens", 160)),
            max_keyframes=int(settings.get("max_keyframes", 4)),
            model_family=settings.get("model_family", "qwen2-vl"),
        )
    else:
        raise ValueError(f"Unsupported captioning provider: {provider}")

    cache_path = settings.get("cache_path")
    if cache_path:
        return CachedCaptioner(backend, Path(cache_path))
    return backend
