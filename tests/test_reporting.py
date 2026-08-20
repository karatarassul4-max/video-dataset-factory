from video_dataset_factory.reporting import render_markdown_summary, summarize_manifest
from video_dataset_factory.schema import ClipRecord


def _record(clip_id, keep, reject_reasons=None, duplicate_of=None):
    return ClipRecord(
        clip_id=clip_id,
        source_path=f"{clip_id}.mp4",
        duration_sec=2.0,
        fps=24.0,
        width=512,
        height=512,
        frame_count=48,
        aesthetic_score=6.0,
        motion_score=1.5,
        duplicate_of=duplicate_of,
        keep=keep,
        reject_reasons=reject_reasons or [],
    )


def test_summarize_manifest_counts_yield_and_reasons():
    records = [
        _record("a", True),
        _record("b", False, ["too_blurry"]),
        _record("c", False, ["near_duplicate"], duplicate_of="a"),
    ]

    summary = summarize_manifest(records)

    assert summary.total_clips == 3
    assert summary.accepted_clips == 1
    assert summary.rejected_clips == 2
    assert summary.duplicate_clips == 1
    assert summary.acceptance_rate == 1 / 3
    assert summary.reject_reasons == {"near_duplicate": 1, "too_blurry": 1}


def test_render_markdown_summary_includes_core_metrics():
    summary = summarize_manifest([_record("a", True)])

    markdown = render_markdown_summary(summary)

    assert "# Dataset Summary" in markdown
    assert "| Total clips | 1 |" in markdown
    assert "No rejected clips." in markdown
