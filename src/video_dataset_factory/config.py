from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from video_dataset_factory.schema import AppConfig


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    return AppConfig.model_validate(data)
