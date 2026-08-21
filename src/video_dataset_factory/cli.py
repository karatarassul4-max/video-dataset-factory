from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from video_dataset_factory.benchmark_inference import (
    run_diffusers_text_to_image_benchmark,
    run_dry_inference_benchmark,
    write_json_report,
    write_markdown_report,
)
from video_dataset_factory.benchmark_pipeline import run_pipeline_benchmark, write_benchmark_report
from video_dataset_factory.caption import build_captioner
from video_dataset_factory.config import load_config
from video_dataset_factory.duplicates import find_duplicate_pairs, mark_duplicates
from video_dataset_factory.manifest import append_jsonl, read_jsonl, write_jsonl
from video_dataset_factory.pipeline import process_video
from video_dataset_factory.quality import build_aesthetic_scorer, build_text_detector
from video_dataset_factory.reporting import summarize_manifest, write_markdown_summary
from video_dataset_factory.scene_split import split_video_into_scenes
from video_dataset_factory.training_benchmark import (
    TrainingBenchmarkConfig,
    run_training_benchmark,
    write_training_json_report,
    write_training_markdown_report,
)
from video_dataset_factory.video_io import is_video_file

app = typer.Typer(help="Build filtered, captioned video dataset manifests.")
console = Console()


@app.command("process-video")
def process_video_command(
    video: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    config_path: Path | None = typer.Option(None, "--config", "-c", exists=True, readable=True),
) -> None:
    config = load_config(config_path)
    if output is not None:
        config.pipeline.output_manifest = output

    record = process_video(video, config)
    append_jsonl(config.pipeline.output_manifest, [record])
    _print_records([record])


@app.command("process-folder")
def process_folder_command(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    config_path: Path | None = typer.Option(None, "--config", "-c", exists=True, readable=True),
    use_ray: bool = typer.Option(False, "--ray", help="Use Ray for parallel processing."),
) -> None:
    config = load_config(config_path)
    if output is not None:
        config.pipeline.output_manifest = output

    videos = sorted(path for path in folder.rglob("*") if is_video_file(path))
    if not videos:
        console.print("No video files found.", style="yellow")
        return

    if use_ray:
        from video_dataset_factory.ray_pipeline import process_videos_with_ray

        records = process_videos_with_ray(videos, config)
    else:
        captioner = build_captioner(config.captioning)
        aesthetic_scorer = build_aesthetic_scorer(config.aesthetic)
        text_detector = build_text_detector(config.ocr)
        records = [
            process_video(
                path,
                config,
                captioner=captioner,
                aesthetic_scorer=aesthetic_scorer,
                text_detector=text_detector,
            )
            for path in videos
        ]

    append_jsonl(config.pipeline.output_manifest, records)
    _print_records(records)


@app.command("split-scenes")
def split_scenes_command(
    video: Path = typer.Argument(..., exists=True, readable=True),
    config_path: Path | None = typer.Option(None, "--config", "-c", exists=True, readable=True),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Detect scenes but do not run ffmpeg."),
) -> None:
    config = load_config(config_path)
    if output_dir is not None:
        config.scene_split.output_dir = output_dir

    segments = split_video_into_scenes(video, config.scene_split, dry_run=dry_run)
    _print_segments(segments)


@app.command("dedupe-manifest")
def dedupe_manifest_command(
    manifest: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/manifest_deduped.jsonl"), "--output", "-o"),
    threshold: int = typer.Option(6, "--threshold", help="Maximum pHash Hamming distance."),
) -> None:
    records = read_jsonl(manifest)
    pairs = find_duplicate_pairs(records, threshold=threshold)
    deduped = mark_duplicates(records, threshold=threshold)
    write_jsonl(output, deduped)
    _print_dedupe_summary(len(records), pairs, output)


@app.command("summarize-manifest")
def summarize_manifest_command(
    manifest: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/dataset_summary.md"), "--output", "-o"),
) -> None:
    summary = summarize_manifest(read_jsonl(manifest))
    write_markdown_summary(output, summary)
    _print_manifest_summary(summary, output)


@app.command("benchmark-folder")
def benchmark_folder_command(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(Path("outputs/pipeline_benchmark.json"), "--output", "-o"),
    config_path: Path | None = typer.Option(None, "--config", "-c", exists=True, readable=True),
    include_ray: bool = typer.Option(False, "--ray", help="Also benchmark Ray execution."),
) -> None:
    config = load_config(config_path)
    results = run_pipeline_benchmark(folder, config, include_ray=include_ray)
    write_benchmark_report(output, results)
    _print_benchmark(results)


@app.command("benchmark-inference")
def benchmark_inference_command(
    output: Path = typer.Option(Path("outputs/inference_benchmark.json"), "--output", "-o"),
    markdown_output: Path = typer.Option(
        Path("outputs/inference_benchmark.md"), "--markdown-output"
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--real", help="Use estimates instead of model execution."),
    model_name: str = typer.Option("runwayml/stable-diffusion-v1-5", "--model"),
    prompt: str = typer.Option("a cinematic shot of a car driving through rain", "--prompt"),
) -> None:
    if dry_run:
        results = run_dry_inference_benchmark()
    else:
        results = run_diffusers_text_to_image_benchmark(model_name=model_name, prompt=prompt)

    write_json_report(output, results)
    write_markdown_report(markdown_output, results)
    _print_inference_benchmark(results)


@app.command("benchmark-training")
def benchmark_training_command(
    manifest: Path | None = typer.Option(None, "--manifest", exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/training_benchmark.json"), "--output", "-o"),
    markdown_output: Path = typer.Option(
        Path("outputs/training_benchmark.md"), "--markdown-output"
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--real", help="Use estimates instead of GPU training."),
    samples: int = typer.Option(512, "--samples", min=1),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    epochs: int = typer.Option(1, "--epochs", min=1),
    mixed_precision: str = typer.Option("no", "--mixed-precision", help="no, fp16, or bf16."),
    gradient_accumulation_steps: int = typer.Option(1, "--gradient-accumulation-steps", min=1),
) -> None:
    config = TrainingBenchmarkConfig(
        manifest_path=manifest,
        output_path=output,
        markdown_output_path=markdown_output,
        samples=samples,
        batch_size=batch_size,
        epochs=epochs,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dry_run=dry_run,
    )
    result = run_training_benchmark(config)
    write_training_json_report(output, result)
    write_training_markdown_report(markdown_output, result)
    _print_training_benchmark(result)


def _print_records(records) -> None:
    table = Table(title="Video Dataset Factory")
    table.add_column("clip_id")
    table.add_column("keep")
    table.add_column("duration")
    table.add_column("motion")
    table.add_column("reasons")

    for record in records:
        table.add_row(
            record.clip_id,
            "yes" if record.keep else "no",
            f"{record.duration_sec:.2f}s",
            "n/a" if record.motion_score is None else f"{record.motion_score:.2f}",
            ", ".join(record.reject_reasons) or "-",
        )

    console.print(table)


def _print_segments(segments) -> None:
    table = Table(title="Detected Scenes")
    table.add_column("scene")
    table.add_column("start")
    table.add_column("end")
    table.add_column("duration")
    table.add_column("clip_path")

    for segment in segments:
        table.add_row(
            str(segment.scene_index),
            f"{segment.start_sec:.2f}s",
            f"{segment.end_sec:.2f}s",
            f"{segment.duration_sec:.2f}s",
            segment.clip_path,
        )

    console.print(table)


def _print_dedupe_summary(total_records, pairs, output: Path) -> None:
    table = Table(title="Manifest Dedupe")
    table.add_column("records")
    table.add_column("duplicates")
    table.add_column("output")
    table.add_row(str(total_records), str(len(pairs)), str(output))
    console.print(table)


def _print_manifest_summary(summary, output: Path) -> None:
    table = Table(title="Dataset Summary")
    table.add_column("total")
    table.add_column("accepted")
    table.add_column("rejected")
    table.add_column("duplicates")
    table.add_column("output")
    table.add_row(
        str(summary.total_clips),
        str(summary.accepted_clips),
        str(summary.rejected_clips),
        str(summary.duplicate_clips),
        str(output),
    )
    console.print(table)


def _print_benchmark(results) -> None:
    table = Table(title="Pipeline Benchmark")
    table.add_column("mode")
    table.add_column("clips")
    table.add_column("accepted")
    table.add_column("rejected")
    table.add_column("seconds")
    table.add_column("clips/min")

    for result in results:
        table.add_row(
            result.mode,
            str(result.clip_count),
            str(result.accepted_count),
            str(result.rejected_count),
            f"{result.seconds:.2f}",
            f"{result.clips_per_minute:.2f}",
        )

    console.print(table)


def _print_inference_benchmark(results) -> None:
    table = Table(title="Inference Benchmark")
    table.add_column("scenario")
    table.add_column("seconds")
    table.add_column("peak VRAM MB")
    table.add_column("notes")

    for result in results:
        peak_vram = "n/a" if result.peak_vram_mb is None else f"{result.peak_vram_mb:.2f}"
        table.add_row(result.name, f"{result.seconds:.4f}", peak_vram, result.notes)

    console.print(table)


def _print_training_benchmark(result) -> None:
    table = Table(title="Training Benchmark")
    table.add_column("mode")
    table.add_column("device")
    table.add_column("gpus")
    table.add_column("distributed")
    table.add_column("samples/sec")
    table.add_column("peak VRAM MB")
    peak_vram = "n/a" if result.peak_vram_mb is None else f"{result.peak_vram_mb:.2f}"
    table.add_row(
        result.mode,
        result.device,
        str(result.gpu_count),
        result.distributed_type,
        f"{result.samples_per_second:.2f}",
        peak_vram,
    )
    console.print(table)


if __name__ == "__main__":
    app()
