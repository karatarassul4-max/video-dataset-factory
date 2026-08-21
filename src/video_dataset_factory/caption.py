from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

GROQ_DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"


@dataclass(frozen=True)
class CaptionContext:
    clip_id: str
    source_path: str
    motion_caption: str | None = None


class Captioner(Protocol):
    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        ...


@runtime_checkable
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


def encode_frame_as_data_url(frame: np.ndarray, jpeg_quality: int = 85) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("Could not encode frame as JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_openai_vision_messages(
    frames: list[np.ndarray],
    context: CaptionContext | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_dense_caption_prompt(context)}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_frame_as_data_url(frame), "detail": "low"},
            }
        )
    return [{"role": "user", "content": content}]


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
        return f"A {lighting} video clip; {motion}."


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


class OpenAICompatibleVisionCaptioner:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        api_key_name: str = "API key",
        max_new_tokens: int = 160,
        max_keyframes: int = 4,
        timeout_sec: float = 90.0,
        token_parameter: str = "max_tokens",
    ):
        if not api_key:
            raise ValueError(f"Vision captioning requires {api_key_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.api_key_name = api_key_name
        self.max_new_tokens = max_new_tokens
        self.max_keyframes = max_keyframes
        self.timeout_sec = timeout_sec
        self.token_parameter = token_parameter

    def caption(self, frames: list[np.ndarray], context: CaptionContext | None = None) -> str:
        keyframes = select_keyframes(frames, self.max_keyframes)
        payload = {
            "model": self.model_name,
            "messages": build_openai_vision_messages(keyframes, context),
            self.token_parameter: self.max_new_tokens,
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "video-dataset-factory/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vision captioning failed: {exc.code} {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Vision captioning failed: {exc.reason}") from exc

        try:
            caption = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise RuntimeError(f"Unexpected vision response: {data}") from exc
        if not caption:
            raise RuntimeError("Vision captioning returned an empty caption")
        return caption


class OpenAIVisionCaptioner(OpenAICompatibleVisionCaptioner):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        max_new_tokens: int = 160,
        max_keyframes: int = 4,
        timeout_sec: float = 90.0,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            api_key_name="OPENAI_API_KEY",
            max_new_tokens=max_new_tokens,
            max_keyframes=max_keyframes,
            timeout_sec=timeout_sec,
            token_parameter="max_tokens",
        )


class GroqVisionCaptioner(OpenAICompatibleVisionCaptioner):
    def __init__(
        self,
        api_key: str,
        model_name: str = GROQ_DEFAULT_VISION_MODEL,
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        max_new_tokens: int = 180,
        max_keyframes: int = 4,
        timeout_sec: float = 90.0,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            api_key_name="GROQ_API_KEY",
            max_new_tokens=max_new_tokens,
            max_keyframes=max_keyframes,
            timeout_sec=timeout_sec,
            token_parameter="max_completion_tokens",
        )


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
    if provider == "heuristic":
        backend: Captioner = HeuristicCaptioner()
    elif provider == "openai":
        backend = OpenAIVisionCaptioner(
            api_key=settings.get("api_key", ""),
            model_name=model_name or "gpt-4o-mini",
            base_url=settings.get("base_url", "https://api.openai.com/v1/chat/completions"),
            max_new_tokens=int(settings.get("max_new_tokens", 160)),
            max_keyframes=int(settings.get("max_keyframes", 4)),
            timeout_sec=float(settings.get("timeout_sec", 90.0)),
        )
    elif provider == "groq":
        backend = GroqVisionCaptioner(
            api_key=settings.get("api_key", ""),
            model_name=model_name or GROQ_DEFAULT_VISION_MODEL,
            base_url=settings.get("base_url", "https://api.groq.com/openai/v1/chat/completions"),
            max_new_tokens=int(settings.get("max_new_tokens", 180)),
            max_keyframes=int(settings.get("max_keyframes", 4)),
            timeout_sec=float(settings.get("timeout_sec", 90.0)),
        )
    elif provider == "transformers":
        if not model_name:
            raise ValueError("Transformers VLM captioning requires captioning.model_name")
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
