from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from video_dataset_factory.schema import ClipRecord


@dataclass(frozen=True)
class ManifestSummary:
    total_clips: int
    accepted_clips: int
    rejected_clips: int
    duplicate_clips: int
    acceptance_rate: float
    average_aesthetic_score: float | None
    average_motion_score: float | None
    reject_reasons: dict[str, int]


def summarize_manifest(records: list[ClipRecord]) -> ManifestSummary:
    total = len(records)
    accepted = sum(1 for record in records if record.keep)
    rejected = total - accepted
    duplicate_count = sum(1 for record in records if record.duplicate_of is not None)
    aesthetic_scores = [record.aesthetic_score for record in records if record.aesthetic_score is not None]
    motion_scores = [record.motion_score for record in records if record.motion_score is not None]

    reason_counts: Counter[str] = Counter()
    for record in records:
        reason_counts.update(record.reject_reasons)

    return ManifestSummary(
        total_clips=total,
        accepted_clips=accepted,
        rejected_clips=rejected,
        duplicate_clips=duplicate_count,
        acceptance_rate=0.0 if total == 0 else accepted / total,
        average_aesthetic_score=_mean_or_none(aesthetic_scores),
        average_motion_score=_mean_or_none(motion_scores),
        reject_reasons=dict(sorted(reason_counts.items())),
    )


def write_markdown_summary(path: Path, summary: ManifestSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_summary(summary), encoding="utf-8")


def render_markdown_summary(summary: ManifestSummary) -> str:
    lines = [
        "# Dataset Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total clips | {summary.total_clips} |",
        f"| Accepted clips | {summary.accepted_clips} |",
        f"| Rejected clips | {summary.rejected_clips} |",
        f"| Acceptance rate | {summary.acceptance_rate:.1%} |",
        f"| Near-duplicate clips | {summary.duplicate_clips} |",
        f"| Average aesthetic score | {_format_optional(summary.average_aesthetic_score)} |",
        f"| Average motion score | {_format_optional(summary.average_motion_score)} |",
        "",
        "## Reject Reasons",
        "",
    ]

    if not summary.reject_reasons:
        lines.append("No rejected clips.")
    else:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in summary.reject_reasons.items():
            lines.append(f"| `{reason}` | {count} |")

    return "\n".join(lines) + "\n"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
