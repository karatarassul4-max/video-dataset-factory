# Inference Benchmark Report

This is a deterministic dry-run report for documentation and CI-friendly demos. Replace it with a real CUDA run before making performance claims.

| Scenario | Seconds | Peak VRAM MB | Notes |
| --- | ---: | ---: | --- |
| baseline_30_steps | 30.0000 | 6000.00 | dry-run baseline estimate |
| fast_8_steps | 8.0000 | 6000.00 | dry-run baseline estimate |
| memory_sliced_30_steps | 33.6960 | 4329.60 | attention slicing lowers VRAM with small latency overhead; VAE slicing lowers decode memory |
| compile_8_steps | 7.0400 | 6000.00 | torch.compile estimate after warmup |

## Interpretation

- Fewer denoising steps dominate latency in this dry-run estimate.
- Attention and VAE slicing reduce estimated peak VRAM, with a small latency cost.
- `torch.compile` is modeled after warmup; real results should separate compile warmup from steady-state generation.
