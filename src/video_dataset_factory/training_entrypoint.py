from __future__ import annotations

import argparse
from pathlib import Path

from video_dataset_factory.training_benchmark import (
    TrainingBenchmarkConfig,
    run_training_benchmark,
    write_training_json_report,
    write_training_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manifest-caption GPU training benchmark.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest path.")
    parser.add_argument("--output", type=Path, default=Path("outputs/training_benchmark.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("outputs/training_benchmark.md"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingBenchmarkConfig(
        manifest_path=args.manifest,
        output_path=args.output,
        markdown_output_path=args.markdown_output,
        samples=args.samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dry_run=args.dry_run,
    )
    result = run_training_benchmark(config)
    write_training_json_report(args.output, result)
    write_training_markdown_report(args.markdown_output, result)
    print(result)


if __name__ == "__main__":
    main()
