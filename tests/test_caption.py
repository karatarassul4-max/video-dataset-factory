import numpy as np

from video_dataset_factory.caption import (
    CachedCaptioner,
    CaptionContext,
    build_dense_caption_prompt,
)


class CountingCaptioner:
    def __init__(self):
        self.calls = 0

    def caption(self, frames, context=None):
        self.calls += 1
        return "cached caption"


def test_dense_caption_prompt_includes_motion_context():
    context = CaptionContext(
        clip_id="abc",
        source_path="clip.mp4",
        motion_caption="Moderate motion with visible object movement.",
    )

    prompt = build_dense_caption_prompt(context)

    assert "temporal dynamics" in prompt
    assert "Moderate motion" in prompt


def test_cached_captioner_reuses_clip_id(tmp_path):
    backend = CountingCaptioner()
    cache_path = tmp_path / "captions.json"
    captioner = CachedCaptioner(backend, cache_path)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    context = CaptionContext(clip_id="same", source_path="clip.mp4")

    assert captioner.caption(frames, context) == "cached caption"
    assert captioner.caption(frames, context) == "cached caption"
    assert backend.calls == 1
