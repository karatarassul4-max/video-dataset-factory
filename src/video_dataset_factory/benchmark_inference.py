from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    seconds: float
    peak_vram_mb: float | None
    notes: str = ""


@dataclass(frozen=True)
class InferenceScenario:
    name: str
    num_inference_steps: int
    attention_slicing: bool = False
    vae_slicing: bool = False
    torch_compile: bool = False
    dtype: str = "float16"


DEFAULT_SCENARIOS = [
    InferenceScenario(name="baseline_30_steps", num_inference_steps=30),
    InferenceScenario(name="fast_8_steps", num_inference_steps=8),
    InferenceScenario(name="memory_sliced_30_steps", num_inference_steps=30, attention_slicing=True, vae_slicing=True),
    InferenceScenario(name="compile_8_steps", num_inference_steps=8, torch_compile=True),
]


def benchmark_callable(name: str, fn: Callable[[], Any], notes: str = "") -> BenchmarkResult:
    """Benchmark a callable and capture CUDA peak memory when torch is available."""
    peak_vram_mb: float | None = None

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = perf_counter()
            fn()
            torch.cuda.synchronize()
            seconds = perf_counter() - start
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            return BenchmarkResult(name=name, seconds=seconds, peak_vram_mb=peak_vram_mb, notes=notes)
    except ImportError:
        pass

    start = perf_counter()
    fn()
    seconds = perf_counter() - start
    return BenchmarkResult(name=name, seconds=seconds, peak_vram_mb=peak_vram_mb, notes=notes)


def estimate_dry_run_result(scenario: InferenceScenario) -> BenchmarkResult:
    """Deterministic estimate for CI/docs when heavy inference dependencies are absent."""
    base_seconds = 30.0
    base_vram = 6_000.0
    step_factor = scenario.num_inference_steps / 30.0
    seconds = base_seconds * step_factor
    peak_vram = base_vram

    notes: list[str] = []
    if scenario.attention_slicing:
        peak_vram *= 0.82
        seconds *= 1.08
        notes.append("attention slicing lowers VRAM with small latency overhead")
    if scenario.vae_slicing:
        peak_vram *= 0.88
        seconds *= 1.04
        notes.append("VAE slicing lowers decode memory")
    if scenario.torch_compile:
        seconds *= 0.88
        notes.append("torch.compile estimate after warmup")

    return BenchmarkResult(
        name=scenario.name,
        seconds=round(seconds, 4),
        peak_vram_mb=round(peak_vram, 2),
        notes="; ".join(notes) or "dry-run baseline estimate",
    )


def run_dry_inference_benchmark(scenarios: list[InferenceScenario] | None = None) -> list[BenchmarkResult]:
    return [estimate_dry_run_result(scenario) for scenario in (scenarios or DEFAULT_SCENARIOS)]


def run_diffusers_text_to_image_benchmark(
    model_name: str,
    prompt: str,
    scenarios: list[InferenceScenario] | None = None,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Run a real diffusers benchmark when optional inference dependencies are installed."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError("Install inference dependencies with `pip install -e .[inference]`.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Real inference benchmark requires CUDA. Use dry_run=True on CPU machines.")

    results: list[BenchmarkResult] = []
    for scenario in scenarios or DEFAULT_SCENARIOS:
        dtype = torch.float16 if scenario.dtype == "float16" else torch.bfloat16
        pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=dtype).to("cuda")

        if scenario.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if scenario.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if scenario.torch_compile and hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        def _generate() -> Any:
            return pipe(
                prompt=prompt,
                num_inference_steps=scenario.num_inference_steps,
                generator=generator,
            ).images[0]

        results.append(
            benchmark_callable(
                scenario.name,
                _generate,
                notes=(
                    f"steps={scenario.num_inference_steps}, "
                    f"attention_slicing={scenario.attention_slicing}, "
                    f"vae_slicing={scenario.vae_slicing}, compile={scenario.torch_compile}"
                ),
            )
        )
        del pipe
        torch.cuda.empty_cache()

    return results


def write_json_report(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in results], handle, indent=2)


def write_markdown_report(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Inference Benchmark Report",
        "",
        "| Scenario | Seconds | Peak VRAM MB | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for result in results:
        peak_vram = "n/a" if result.peak_vram_mb is None else f"{result.peak_vram_mb:.2f}"
        lines.append(f"| {result.name} | {result.seconds:.4f} | {peak_vram} | {result.notes} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
