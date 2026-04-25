from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RoutingPolicy:
    max_failovers: int = 2
    cost_weight: float = 0.65
    health_weight: float = 0.2
    capacity_weight: float = 0.1
    region_weight: float = 0.05


@dataclass(slots=True)
class ProviderProfile:
    name: str
    mode: str = "simulation"
    enabled: bool = True
    regions: list[str] = field(default_factory=list)
    cost_per_gpu_hour: float = 1.0
    egress_cost_per_gb: float = 0.02
    startup_minutes: int = 5
    health_score: float = 0.95
    capacity_score: float = 0.9
    latency_score: float = 0.9
    spot_discount: float = 0.0
    planned_outcomes: list[str] = field(default_factory=lambda: ["success"])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingJob:
    job_id: str
    model_name: str
    framework: str
    entrypoint: str
    dataset_uri: str
    artifact_uri: str
    dataset_size_gb: float
    expected_gpu_hours: float
    gpu_count: int
    gpu_type: str
    max_budget_usd: float | None = None
    preferred_regions: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderHealth:
    provider: str
    healthy: bool
    health_score: float
    capacity_score: float
    latency_score: float
    detail: str = ""


@dataclass(slots=True)
class CostEstimate:
    provider: str
    total_cost_usd: float
    compute_cost_usd: float
    egress_cost_usd: float
    startup_penalty_usd: float
    within_budget: bool


@dataclass(slots=True)
class RouteDecision:
    provider: str
    rank: int
    score: float
    estimated_cost_usd: float
    reason: str


@dataclass(slots=True)
class TrainingAttempt:
    provider: str
    status: str
    message: str
    estimated_cost_usd: float
    cloud_job_id: str | None = None


@dataclass(slots=True)
class TrainingResult:
    provider: str
    cloud_job_id: str
    status: str
    artifact_uri: str
    metrics: dict[str, float]
    cost_usd: float


@dataclass(slots=True)
class PipelineSummary:
    routed_jobs: list[dict[str, Any]]
    total_estimated_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
