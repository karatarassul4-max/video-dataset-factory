from video_dataset_factory.benchmark_pipeline import summarize_records
from video_dataset_factory.schema import ClipRecord


def make_record(keep):
    return ClipRecord(
        clip_id="clip",
        source_path="clip.mp4",
        duration_sec=1.0,
        fps=24.0,
        width=512,
        height=512,
        frame_count=24,
        keep=keep,
        reject_reasons=[] if keep else ["too_static"],
    )


def test_summarize_records_reports_counts_and_rate():
    result = summarize_records("single", [make_record(True), make_record(False)], seconds=30.0)

    assert result.clip_count == 2
    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.clips_per_minute == 4.0
