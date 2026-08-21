import json

from video_dataset_factory.training_benchmark import (
    TrainingBenchmarkConfig,
    estimate_dry_training_benchmark,
    load_training_rows,
    write_training_json_report,
    write_training_markdown_report,
)


def test_dry_training_benchmark_reports_throughput():
    result = estimate_dry_training_benchmark(
        TrainingBenchmarkConfig(samples=128, batch_size=32, epochs=2, dry_run=True)
    )

    assert result.mode == "dry_run"
    assert result.steps == 8
    assert result.samples_per_second > 0
    assert result.peak_vram_mb is None


def test_manifest_loader_keeps_accepted_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "a", "keep": True, "caption": "a clean video"}),
                json.dumps({"clip_id": "b", "keep": False, "caption": "bad watermark"}),
                json.dumps({"clip_id": "c", "keep": True, "caption": "smooth motion"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_training_rows(TrainingBenchmarkConfig(manifest_path=manifest, samples=10))

    assert [row["clip_id"] for row in rows] == ["a", "c"]


def test_training_reports_are_written(tmp_path):
    result = estimate_dry_training_benchmark(TrainingBenchmarkConfig(samples=64, batch_size=16))
    json_path = tmp_path / "training.json"
    md_path = tmp_path / "training.md"

    write_training_json_report(json_path, result)
    write_training_markdown_report(md_path, result)

    assert json.loads(json_path.read_text(encoding="utf-8"))["mode"] == "dry_run"
    assert "Training Benchmark Report" in md_path.read_text(encoding="utf-8")
