from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import PipelineSummary, ProviderProfile, RoutingPolicy, TrainingJob


def load_config(path: str | Path) -> tuple[RoutingPolicy, list[ProviderProfile], list[TrainingJob]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    routing = RoutingPolicy(**payload.get("routing", {}))
    providers = [ProviderProfile(**item) for item in payload.get("providers", [])]
    jobs = [TrainingJob(**item) for item in payload.get("jobs", [])]

    return routing, providers, jobs


def save_summary(path: str | Path, summary: PipelineSummary) -> None:
    Path(path).write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
