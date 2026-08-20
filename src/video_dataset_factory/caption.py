from __future__ import annotations

import numpy as np


def heuristic_caption(frames: list[np.ndarray]) -> str:
    """Small placeholder captioner for the baseline pipeline.

    Real experiments should replace this with a VLM adapter such as Qwen2-VL or LLaVA.
    Keeping a deterministic fallback makes the manifest pipeline testable on CPU.
    """
    if not frames:
        return "Video clip with unavailable visual content."

    mean_brightness = float(np.mean([np.mean(frame) for frame in frames]))
    if mean_brightness < 70:
        lighting = "dark"
    elif mean_brightness > 180:
        lighting = "bright"
    else:
        lighting = "normally lit"

    return f"A {lighting} video clip. Replace this heuristic caption with VLM dense captioning."


class Captioner:
    def caption(self, frames: list[np.ndarray]) -> str:
        return heuristic_caption(frames)
