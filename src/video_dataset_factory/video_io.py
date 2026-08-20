from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_dataset_factory.schema import VideoMetadata

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    capture.release()

    return VideoMetadata(
        source_path=str(path),
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    if count <= 0:
        return []

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        return []

    indices = np.linspace(0, max(total - 1, 0), num=min(count, total), dtype=int)
    frames: list[np.ndarray] = []
    for frame_idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)

    capture.release()
    return frames
