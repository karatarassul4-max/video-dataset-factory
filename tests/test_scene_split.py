from pathlib import Path

from video_dataset_factory.scene_split import build_ffmpeg_clip_command
from video_dataset_factory.schema import SceneSplitConfig


def test_build_ffmpeg_clip_command_contains_normalization_settings():
    config = SceneSplitConfig(output_fps=12, output_width=320, output_height=240, crf=20)

    command = build_ffmpeg_clip_command(
        Path("input.mp4"),
        Path("output.mp4"),
        start_sec=1.0,
        end_sec=3.5,
        config=config,
    )

    assert command[:4] == ["ffmpeg", "-y", "-ss", "1.000"]
    assert "-t" in command
    assert "2.500" in command
    assert "fps=12" in command
    assert "scale=320:240" in command
    assert "-crf" in command
    assert "20" in command
