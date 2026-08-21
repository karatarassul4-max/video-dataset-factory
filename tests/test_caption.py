import json

import numpy as np
import pytest

from video_dataset_factory.caption import (
    GROQ_DEFAULT_VISION_MODEL,
    CachedCaptioner,
    CaptionContext,
    GroqVisionCaptioner,
    HeuristicCaptioner,
    OpenAIVisionCaptioner,
    batch_caption,
    build_captioner,
    build_dense_caption_prompt,
    build_openai_vision_messages,
    build_vlm_messages,
    select_keyframes,
)


class CountingCaptioner:
    def __init__(self):
        self.calls = 0

    def caption(self, frames, context=None):
        self.calls += 1
        return "cached caption"


class NativeBatchCaptioner:
    def caption(self, frames, context=None):
        return "single"

    def batch_caption(self, clips, contexts=None):
        return [f"batch-{index}" for index, _ in enumerate(clips)]


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_dense_caption_prompt_includes_motion_context():
    context = CaptionContext(
        clip_id="abc",
        source_path="clip.mp4",
        motion_caption="Moderate motion with visible object movement.",
    )

    prompt = build_dense_caption_prompt(context)

    assert "temporal dynamics" in prompt
    assert "Moderate motion" in prompt


def test_select_keyframes_spreads_frames_across_clip():
    frames = [np.full((4, 4, 3), fill_value=value, dtype=np.uint8) for value in range(6)]

    keyframes = select_keyframes(frames, max_frames=3)

    assert [int(frame[0, 0, 0]) for frame in keyframes] == [0, 2, 5]


def test_select_keyframes_rejects_invalid_count():
    with pytest.raises(ValueError, match="max_frames"):
        select_keyframes([], max_frames=0)


def test_qwen_vlm_messages_use_image_content_blocks():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    context = CaptionContext(clip_id="abc", source_path="clip.mp4", motion_caption="fast pan")

    messages = build_vlm_messages(frames, context, model_family="qwen2-vl")

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "image"}
    assert messages[0]["content"][1] == {"type": "image"}
    assert "fast pan" in messages[0]["content"][2]["text"]


def test_openai_vision_messages_embed_keyframes_as_images():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    messages = build_openai_vision_messages(frames)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][1]["type"] == "image_url"
    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert messages[0]["content"][2]["type"] == "image_url"


def test_llava_vlm_messages_use_image_tokens():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    messages = build_vlm_messages(frames, model_family="llava")

    assert messages[0]["content"].count("<image>") == 3
    assert "text-to-video" in messages[0]["content"]


def test_batch_caption_uses_native_batch_method():
    clips = [[np.zeros((4, 4, 3), dtype=np.uint8)] for _ in range(2)]

    captions = batch_caption(NativeBatchCaptioner(), clips)

    assert captions == ["batch-0", "batch-1"]


def test_batch_caption_rejects_context_length_mismatch():
    clips = [[np.zeros((4, 4, 3), dtype=np.uint8)] for _ in range(2)]

    with pytest.raises(ValueError, match="same length"):
        batch_caption(HeuristicCaptioner(), clips, contexts=[None])


def test_build_captioner_requires_transformers_model_name():
    with pytest.raises(ValueError, match="model_name"):
        build_captioner({"provider": "transformers", "model_name": None})


def test_build_captioner_requires_openai_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_captioner({"provider": "openai", "api_key": ""})


def test_build_captioner_requires_groq_api_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_captioner({"provider": "groq", "api_key": ""})


def test_openai_captioner_calls_vision_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {"choices": [{"message": {"content": "A person walks across a bright room."}}]}
        )

    monkeypatch.setattr("video_dataset_factory.caption.urllib.request.urlopen", fake_urlopen)
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    captioner = OpenAIVisionCaptioner(
        api_key="test-key",
        model_name="vision-test-model",
        max_keyframes=2,
        timeout_sec=12,
    )

    caption = captioner.caption(frames)

    assert caption == "A person walks across a bright room."
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "vision-test-model"
    assert captured["payload"]["max_tokens"] == 160
    assert len(captured["payload"]["messages"][0]["content"]) == 3


def test_groq_captioner_uses_groq_endpoint_and_token_parameter(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {"choices": [{"message": {"content": "A car drives through a rainy street."}}]}
        )

    monkeypatch.setattr("video_dataset_factory.caption.urllib.request.urlopen", fake_urlopen)
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]
    captioner = GroqVisionCaptioner(
        api_key="test-key",
        max_keyframes=3,
        max_new_tokens=180,
    )

    caption = captioner.caption(frames)

    assert caption == "A car drives through a rainy street."
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["payload"]["model"] == GROQ_DEFAULT_VISION_MODEL
    assert captured["payload"]["max_completion_tokens"] == 180
    assert "max_tokens" not in captured["payload"]
    assert len(captured["payload"]["messages"][0]["content"]) == 4


def test_cached_captioner_reuses_clip_id(tmp_path):
    backend = CountingCaptioner()
    cache_path = tmp_path / "captions.json"
    captioner = CachedCaptioner(backend, cache_path)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    context = CaptionContext(clip_id="same", source_path="clip.mp4")

    assert captioner.caption(frames, context) == "cached caption"
    assert captioner.caption(frames, context) == "cached caption"
    assert backend.calls == 1
