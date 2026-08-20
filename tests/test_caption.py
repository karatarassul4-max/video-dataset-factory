import numpy as np
import pytest

from video_dataset_factory.caption import (
    CachedCaptioner,
    CaptionContext,
    HeuristicCaptioner,
    batch_caption,
    build_captioner,
    build_dense_caption_prompt,
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


def test_build_captioner_falls_back_when_transformers_model_is_not_configured(tmp_path):
    captioner = build_captioner(
        {
            "provider": "transformers",
            "model_name": None,
            "cache_path": str(tmp_path / "cache.json"),
        }
    )

    assert isinstance(captioner.backend, HeuristicCaptioner)


def test_cached_captioner_reuses_clip_id(tmp_path):
    backend = CountingCaptioner()
    cache_path = tmp_path / "captions.json"
    captioner = CachedCaptioner(backend, cache_path)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    context = CaptionContext(clip_id="same", source_path="clip.mp4")

    assert captioner.caption(frames, context) == "cached caption"
    assert captioner.caption(frames, context) == "cached caption"
    assert backend.calls == 1
