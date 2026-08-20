from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from video_dataset_factory.pipeline import process_video
from video_dataset_factory.ray_pipeline import process_videos_with_ray
from video_dataset_factory.schema import AppConfig, ClipRecord
from video_dataset_factory.video_io import is_video_file


@dataclass(frozen=True)
class PipelineBenchmarkResult:
    mode: str
    clip_count: int
    accepted_count: int
    rejected_count: int
    seconds: float
    clips_per_minute: float


def collect_video_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if is_video_file(path))


def summarize_records(mode: str, records: list[ClipRecord], seconds: float) -> PipelineBenchmarkResult:
    accepted = sum(1 for record in records if record.keep)
    rejected = len(records) - accepted
    clips_per_minute = (len(records) / seconds * 60.0) if seconds > 0 else 0.0
    return PipelineBenchmarkResult(
        mode=mode,
        clip_count=len(records),
        accepted_count=accepted,
        rejected_count=rejected,
        seconds=seconds,
        clips_per_minute=clips_per_minute,
    )


def benchmark_single_process(paths: list[Path], config: AppConfig) -> PipelineBenchmarkResult:
    started_at = perf_counter()
    records = [process_video(path, config) for path in paths]
    seconds = perf_counter() - started_at
    return summarize_records("single_process", records, seconds)


def benchmark_ray(paths: list[Path], config: AppConfig) -> PipelineBenchmarkResult:
    started_at = perf_counter()
    records = process_videos_with_ray(paths, config)
    seconds = perf_counter() - started_at
    return summarize_records("ray", records, seconds)


def run_pipeline_benchmark(
    folder: Path,
    config: AppConfig,
    include_ray: bool = False,
) -> list[PipelineBenchmarkResult]:
    paths = collect_video_paths(folder)
    results = [benchmark_single_process(paths, config)]
    if include_ray:
        results.append(benchmark_ray(paths, config))
    return results


def write_benchmark_report(path: Path, results: list[PipelineBenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in results], handle, indent=2)
