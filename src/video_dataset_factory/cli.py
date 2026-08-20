from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from video_dataset_factory.config import load_config
from video_dataset_factory.manifest import append_jsonl
from video_dataset_factory.pipeline import process_video
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
        records = [process_video(path, config) for path in videos]

    append_jsonl(config.pipeline.output_manifest, records)
    _print_records(records)


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


if __name__ == "__main__":
    app()
