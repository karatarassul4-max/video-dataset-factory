from video_dataset_factory.benchmark_inference import (
    InferenceScenario,
    estimate_dry_run_result,
    run_dry_inference_benchmark,
)


def test_dry_run_shorter_step_scenario_is_faster():
    baseline = estimate_dry_run_result(InferenceScenario(name="baseline", num_inference_steps=30))
    fast = estimate_dry_run_result(InferenceScenario(name="fast", num_inference_steps=8))

    assert fast.seconds < baseline.seconds
    assert fast.peak_vram_mb == baseline.peak_vram_mb


def test_slicing_reduces_estimated_vram():
    baseline = estimate_dry_run_result(InferenceScenario(name="baseline", num_inference_steps=30))
    sliced = estimate_dry_run_result(
        InferenceScenario(
            name="sliced",
            num_inference_steps=30,
            attention_slicing=True,
            vae_slicing=True,
        )
    )

    assert sliced.peak_vram_mb < baseline.peak_vram_mb
    assert "slicing" in sliced.notes


def test_default_dry_run_benchmark_has_named_scenarios():
    results = run_dry_inference_benchmark()

    assert len(results) >= 3
    assert {result.name for result in results} >= {"baseline_30_steps", "fast_8_steps"}
