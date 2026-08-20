from __future__ import annotations

import subprocess
from pathlib import Path

from video_dataset_factory.schema import SceneSegment, SceneSplitConfig
from video_dataset_factory.video_io import probe_video


def detect_scene_boundaries(video: Path, config: SceneSplitConfig) -> list[tuple[float, float]]:
    """Return scene boundaries in seconds using PySceneDetect when available."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise RuntimeError("Install scene splitting dependencies with `pip install -e .[scene]`.") from exc

    if config.detector != "content":
        raise ValueError(f"Unsupported scene detector: {config.detector}")

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=config.threshold, min_scene_len=config.min_scene_len_frames)
    )
    video_stream = open_video(str(video))
    scene_manager.detect_scenes(video_stream)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        metadata = probe_video(video)
        return [(0.0, metadata.duration_sec)]

    boundaries: list[tuple[float, float]] = []
    for start_time, end_time in scene_list:
        boundaries.append((float(start_time.get_seconds()), float(end_time.get_seconds())))
    return boundaries


def build_ffmpeg_clip_command(
    source: Path,
    output: Path,
    start_sec: float,
    end_sec: float,
    config: SceneSplitConfig,
) -> list[str]:
    duration = max(0.0, end_sec - start_sec)
    scale = f"scale={config.output_width}:{config.output_height}:force_original_aspect_ratio=decrease"
    pad = f"pad={config.output_width}:{config.output_height}:(ow-iw)/2:(oh-ih)/2"
    vf = f"fps={config.output_fps},{scale},{pad},setsar=1"

    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        str(output),
    ]


def split_video_into_scenes(
    video: Path,
    config: SceneSplitConfig,
    dry_run: bool = False,
) -> list[SceneSegment]:
    boundaries = detect_scene_boundaries(video, config)
    output_dir = config.output_dir / video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    segments: list[SceneSegment] = []
    for scene_index, (start_sec, end_sec) in enumerate(boundaries):
        output = output_dir / f"{video.stem}_scene_{scene_index:04d}.mp4"
        command = build_ffmpeg_clip_command(video, output, start_sec, end_sec, config)
        if not dry_run:
            subprocess.run(command, check=True)

        segments.append(
            SceneSegment(
                source_path=str(video),
                clip_path=str(output),
                scene_index=scene_index,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=max(0.0, end_sec - start_sec),
            )
        )

    return segments
