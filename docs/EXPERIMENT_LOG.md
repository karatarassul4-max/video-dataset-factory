# Experiment Log

This log captures hypotheses, expected failure modes, and what to measure next. It is intentionally part of the project because research engineering is about iteration, not just final code.

## Current Baseline

- Input: short normalized clips.
- Output: JSONL manifest with metadata, quality scores, motion scores, captions, perceptual hashes, keep/reject flags, and reject reasons.
- Demo fixture: `examples/demo_manifest.jsonl`.
- CI status: CPU-only lint and tests on Python 3.10 and 3.11.

## Failed or Risky Experiments To Track

### Scene Threshold Too Low

Hypothesis: lowering PySceneDetect threshold catches more real cuts.

Risk: camera shake and fast movement produce too many micro-scenes.

Measurement: median clip duration, number of clips below 1 second, manual precision over 30 split boundaries.

### OCR/Text Proxy Too Strict

Hypothesis: Canny edge density can remove clips with watermarks and burned-in captions.

Risk: urban scenes, product labels, and high-frequency textures look like text.

Measurement: false positive rate over manually reviewed rejected clips.

### Motion Threshold Too High

Hypothesis: text-to-video data benefits from visible movement.

Risk: slow cinematic shots, product videos, and establishing shots get removed even though they are useful.

Measurement: manual review by category and accepted-caption usefulness.

### Generic VLM Prompt Misses Temporal Dynamics

Hypothesis: dense prompts improve caption usefulness.

Risk: VLMs list objects but fail to describe camera motion or frame-to-frame change.

Measurement: rubric over subject, action, camera motion, scene, lighting, and temporal consistency.

### Ray Overhead on Small Datasets

Hypothesis: Ray improves preprocessing throughput.

Risk: startup and serialization overhead dominate when folders are small.

Measurement: clips/min at 10, 50, 100, and 1,000 clips.

### Aggressive Inference Optimization

Hypothesis: fewer denoising steps and memory slicing improve speed/VRAM.

Risk: temporal smoothness or image quality can degrade even when latency improves.

Measurement: latency, peak VRAM, CLIP proxy, and manual visual inspection.

### pHash Threshold Too High

Hypothesis: perceptual hashes can remove duplicate or near-duplicate clips.

Risk: visually similar but semantically different clips are grouped together.

Measurement: duplicate precision over manually inspected duplicate pairs.
