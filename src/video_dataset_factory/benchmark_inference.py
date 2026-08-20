from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Any


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    seconds: float
    peak_vram_mb: float | None
    notes: str = ""


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
