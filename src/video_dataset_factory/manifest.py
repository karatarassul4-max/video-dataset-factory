from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from video_dataset_factory.schema import ClipRecord


def append_jsonl(path: Path, records: Iterable[ClipRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[ClipRecord]:
    with path.open("r", encoding="utf-8") as handle:
        return [ClipRecord.model_validate_json(line) for line in handle if line.strip()]
