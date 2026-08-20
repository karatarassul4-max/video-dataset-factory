from __future__ import annotations

from pathlib import Path

from video_dataset_factory.pipeline import process_video
from video_dataset_factory.schema import AppConfig, ClipRecord


def process_videos_with_ray(paths: list[Path], config: AppConfig) -> list[ClipRecord]:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Install the Ray extra with `pip install -e .[ray]`.") from exc

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    @ray.remote
    def _process(path_str: str, cfg: AppConfig) -> ClipRecord:
        return process_video(Path(path_str), cfg)

    refs = [_process.remote(str(path), config) for path in paths]
    return list(ray.get(refs))
