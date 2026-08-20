from video_dataset_factory.manifest import append_jsonl, read_jsonl
from video_dataset_factory.schema import ClipRecord


def test_manifest_round_trip(tmp_path):
    record = ClipRecord(
        clip_id="abc",
        source_path="video.mp4",
        duration_sec=2.0,
        fps=24.0,
        width=512,
        height=512,
        frame_count=48,
        keep=True,
        reject_reasons=[],
    )
    path = tmp_path / "manifest.jsonl"

    append_jsonl(path, [record])
    loaded = read_jsonl(path)

    assert loaded == [record]
