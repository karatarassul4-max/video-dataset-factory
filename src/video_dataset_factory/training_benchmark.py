from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    manifest_path: Path | None = None
    output_path: Path = Path("outputs/training_benchmark.json")
    markdown_output_path: Path = Path("outputs/training_benchmark.md")
    samples: int = 512
    feature_dim: int = 16
    vocab_size: int = 4096
    hidden_dim: int = 128
    embedding_dim: int = 64
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    seed: int = 13
    num_workers: int = 0
    max_caption_tokens: int = 32
    dry_run: bool = False


@dataclass(frozen=True)
class TrainingBenchmarkResult:
    mode: str
    device: str
    gpu_count: int
    distributed_type: str
    mixed_precision: str
    samples: int
    batch_size: int
    epochs: int
    steps: int
    seconds: float
    samples_per_second: float
    final_loss: float
    peak_vram_mb: float | None
    notes: str


def estimate_dry_training_benchmark(config: TrainingBenchmarkConfig) -> TrainingBenchmarkResult:
    steps = max(1, math.ceil(config.samples / max(1, config.batch_size)) * config.epochs)
    estimated_seconds = steps * 0.035
    return TrainingBenchmarkResult(
        mode="dry_run",
        device="cpu_estimate",
        gpu_count=0,
        distributed_type="none",
        mixed_precision=config.mixed_precision,
        samples=config.samples,
        batch_size=config.batch_size,
        epochs=config.epochs,
        steps=steps,
        seconds=estimated_seconds,
        samples_per_second=config.samples / estimated_seconds,
        final_loss=0.0,
        peak_vram_mb=None,
        notes="Deterministic estimate for CI and documentation; run with --real on Kaggle/CUDA.",
    )


def run_training_benchmark(config: TrainingBenchmarkConfig) -> TrainingBenchmarkResult:
    if config.dry_run:
        return estimate_dry_training_benchmark(config)

    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - exercised only without training extra
        raise RuntimeError("Install training extras first: pip install -e .[training]") from exc

    try:
        from accelerate import Accelerator
    except ImportError:  # pragma: no cover - accelerate is optional at import time
        Accelerator = None

    _seed_everything(config.seed, torch)
    accelerator = None
    if Accelerator is not None:
        accelerator = Accelerator(
            mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        device = accelerator.device
        distributed_type = str(accelerator.distributed_type)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed_type = "none"

    rows = load_training_rows(config)
    dataset = ManifestContrastiveDataset(rows, config, torch, Dataset)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    model = TinyManifestContrastiveModel(
        feature_dim=config.feature_dim,
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        torch=torch,
        nn=nn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if accelerator is not None:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    else:
        model.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    steps = 0
    final_loss = 0.0
    for _epoch in range(config.epochs):
        for features, tokens in loader:
            if accelerator is None:
                features = features.to(device)
                tokens = tokens.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = _contrastive_loss(model(features, tokens), F)
                loss.backward()
                optimizer.step()
            else:
                with accelerator.accumulate(model):
                    optimizer.zero_grad(set_to_none=True)
                    loss = _contrastive_loss(model(features, tokens), F)
                    accelerator.backward(loss)
                    optimizer.step()
            final_loss = float(loss.detach().cpu())
            steps += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = max(time.perf_counter() - started, 1e-9)
    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    mode = "accelerate" if accelerator is not None else "torch"
    if accelerator is not None and accelerator.num_processes > 1:
        mode = "accelerate_distributed"

    return TrainingBenchmarkResult(
        mode=mode,
        device=str(device),
        gpu_count=gpu_count,
        distributed_type=distributed_type,
        mixed_precision=config.mixed_precision,
        samples=len(dataset),
        batch_size=config.batch_size,
        epochs=config.epochs,
        steps=steps,
        seconds=seconds,
        samples_per_second=(len(dataset) * config.epochs) / seconds,
        final_loss=final_loss,
        peak_vram_mb=peak_vram,
        notes=(
            "Manifest-caption contrastive training benchmark; "
            "use accelerate config for DDP/DeepSpeed."
        ),
    )


def load_training_rows(config: TrainingBenchmarkConfig) -> list[dict]:
    if config.manifest_path is None:
        return _synthetic_rows(config.samples)

    rows: list[dict] = []
    with config.manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("keep", True):
                rows.append(item)
            if len(rows) >= config.samples:
                break

    if not rows:
        raise ValueError(f"No accepted manifest rows found in {config.manifest_path}")
    return rows


class ManifestContrastiveDataset:  # constructed with torch Dataset base to keep import lazy
    def __new__(cls, rows: list[dict], config: TrainingBenchmarkConfig, torch, dataset_base):
        class _Dataset(dataset_base):
            def __init__(self) -> None:
                self.rows = rows
                self.config = config

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                row = self.rows[index]
                features = _row_features(row, self.config.feature_dim)
                tokens = _caption_tokens(row.get("caption", ""), self.config)
                return (
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor(tokens, dtype=torch.long),
                )

        return _Dataset()


class TinyManifestContrastiveModel:  # constructed with lazy torch/nn modules
    def __new__(
        cls,
        feature_dim: int,
        vocab_size: int,
        hidden_dim: int,
        embedding_dim: int,
        torch,
        nn,
    ):
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.video_tower = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )
                self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
                self.text_tower = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, embedding_dim),
                )
                self.logit_scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, features, token_ids):
                video = self.video_tower(features)
                text_hidden = self.text_embedding(token_ids).mean(dim=1)
                text = self.text_tower(text_hidden)
                return video, text, self.logit_scale.exp().clamp(max=100.0)

        return _Model()


def _contrastive_loss(outputs, functional) -> object:
    video, text, logit_scale = outputs
    video = functional.normalize(video, dim=-1)
    text = functional.normalize(text, dim=-1)
    logits = logit_scale * video @ text.t()
    labels = logits.new_tensor(list(range(logits.shape[0]))).long()
    loss_video = functional.cross_entropy(logits, labels)
    loss_text = functional.cross_entropy(logits.t(), labels)
    return (loss_video + loss_text) / 2


def _row_features(row: dict, feature_dim: int) -> list[float]:
    values = [
        _num(row.get("duration_sec"), 10.0) / 30.0,
        _num(row.get("width"), 512.0) / 1024.0,
        _num(row.get("height"), 512.0) / 1024.0,
        _num(row.get("fps"), 24.0) / 60.0,
        _num(row.get("blur_score"), 100.0) / 500.0,
        _num(row.get("brightness"), 128.0) / 255.0,
        _num(row.get("contrast_score"), 40.0) / 128.0,
        _num(row.get("colorfulness_score"), 40.0) / 128.0,
        _num(row.get("motion_score"), 1.0) / 50.0,
        _num(row.get("motion_p95_score"), 1.0) / 100.0,
        _num(row.get("motion_stability_score"), 1.0),
        _num(row.get("aesthetic_score"), 5.0) / 10.0,
        _num(row.get("ocr_text_area_ratio"), 0.0),
        1.0 if row.get("keep", True) else 0.0,
    ]
    if len(values) < feature_dim:
        values.extend([0.0] * (feature_dim - len(values)))
    return values[:feature_dim]


def _caption_tokens(caption: str, config: TrainingBenchmarkConfig) -> list[int]:
    words = caption.lower().replace(".", " ").replace(",", " ").split()
    tokens = [abs(hash(word)) % config.vocab_size for word in words[: config.max_caption_tokens]]
    if len(tokens) < config.max_caption_tokens:
        tokens.extend([0] * (config.max_caption_tokens - len(tokens)))
    return tokens


def _synthetic_rows(count: int) -> list[dict]:
    rows = []
    motions = ["static", "smooth camera pan", "fast handheld motion", "subject walking"]
    scenes = ["street", "gym", "studio", "car interior"]
    for index in range(count):
        caption = (
            f"A {motions[index % len(motions)]} video "
            f"in a {scenes[index % len(scenes)]} scene."
        )
        rows.append(
            {
                "clip_id": f"synthetic_{index:05d}",
                "keep": True,
                "duration_sec": 4.0 + (index % 12) * 0.5,
                "width": 512,
                "height": 512,
                "fps": 24,
                "blur_score": 80.0 + (index % 7) * 8.0,
                "brightness": 80.0 + (index % 90),
                "contrast_score": 35.0 + (index % 20),
                "colorfulness_score": 30.0 + (index % 35),
                "motion_score": 0.4 + (index % 9) * 0.35,
                "motion_p95_score": 1.0 + (index % 11) * 0.45,
                "motion_stability_score": 0.55 + (index % 5) * 0.08,
                "aesthetic_score": 4.5 + (index % 6) * 0.35,
                "ocr_text_area_ratio": 0.0,
                "caption": caption,
            }
        )
    return rows


def _num(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _seed_everything(seed: int, torch) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_training_json_report(path: Path, result: TrainingBenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def write_training_markdown_report(path: Path, result: TrainingBenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = "n/a" if result.peak_vram_mb is None else f"{result.peak_vram_mb:.2f} MB"
    content = "\n".join(
        [
            "# Training Benchmark Report",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Mode | {result.mode} |",
            f"| Device | {result.device} |",
            f"| GPU count | {result.gpu_count} |",
            f"| Distributed type | {result.distributed_type} |",
            f"| Mixed precision | {result.mixed_precision} |",
            f"| Samples | {result.samples} |",
            f"| Batch size | {result.batch_size} |",
            f"| Epochs | {result.epochs} |",
            f"| Steps | {result.steps} |",
            f"| Seconds | {result.seconds:.4f} |",
            f"| Samples/sec | {result.samples_per_second:.2f} |",
            f"| Final loss | {result.final_loss:.4f} |",
            f"| Peak VRAM | {peak} |",
            "",
            result.notes,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
