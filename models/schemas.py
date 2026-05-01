"""
Pydantic models for the entire orchestrator pipeline.

These schemas define the contract between all components: providers,
cost optimizer, failover engine, and the dashboard.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    MOCK = "mock"


class JobStatus(str, Enum):
    PENDING = "pending"
    ROUTING = "routing"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    FAILOVER = "failover"
    CANCELLED = "cancelled"


class JobTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    GPU_INTENSIVE = "gpu_intensive"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ── Core Models ────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """Configuration for a single training job."""
    algorithm: str = "xgboost"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 32
    early_stopping: bool = True
    early_stopping_patience: int = 3


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    source: str = "local"  # local, s3, gcs, azure_blob
    path: str = ""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    target_column: str = "target"
    feature_columns: List[str] = Field(default_factory=list)


class TrainingJob(BaseModel):
    """A training job submitted to the pipeline."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "training-job"
    tier: JobTier = JobTier.SMALL
    status: JobStatus = JobStatus.PENDING
    provider: Optional[CloudProvider] = None
    preferred_provider: Optional[CloudProvider] = None

    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset_config: DatasetConfig = Field(default_factory=DatasetConfig)

    instance_type: Optional[str] = None
    use_spot: bool = True
    max_budget: float = 50.0

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    failover_count: int = 0
    max_failovers: int = 2
    attempted_providers: List[CloudProvider] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingResult(BaseModel):
    """Result of a completed training job."""
    job_id: str
    provider: CloudProvider
    status: JobStatus

    # Metrics
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)

    # Cost
    actual_cost: float = 0.0
    training_duration_seconds: float = 0.0
    instance_type: str = ""

    # Artifacts
    model_artifact_path: Optional[str] = None
    model_size_mb: Optional[float] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Failover info
    failover_count: int = 0
    attempted_providers: List[CloudProvider] = Field(default_factory=list)

    error_message: Optional[str] = None


class CostEstimate(BaseModel):
    """Cost estimate for a training job on a specific provider."""
    provider: CloudProvider
    instance_type: str
    on_demand_cost_per_hour: float
    spot_cost_per_hour: float
    estimated_hours: float
    estimated_total_on_demand: float
    estimated_total_spot: float
    use_spot: bool = True
    estimated_total: float = 0.0  # Best price (spot if available)
    gpu: Optional[str] = None
    vcpu: int = 4
    memory_gb: float = 16.0

    def model_post_init(self, __context: Any) -> None:
        if self.estimated_total == 0.0:
            self.estimated_total = (
                self.estimated_total_spot if self.use_spot
                else self.estimated_total_on_demand
            )


class ProviderHealth(BaseModel):
    """Health status of a cloud provider."""
    provider: CloudProvider
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0.0
    error_rate: float = 0.0
    consecutive_failures: int = 0
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    circuit_open: bool = False
    message: str = ""


class PipelineRun(BaseModel):
    """Record of a full pipeline execution."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    jobs: List[TrainingJob] = Field(default_factory=list)
    results: List[TrainingResult] = Field(default_factory=list)
    total_cost: float = 0.0
    status: JobStatus = JobStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    cost_savings: float = 0.0  # vs. most expensive option


class DashboardMetrics(BaseModel):
    """Aggregated metrics for the dashboard."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_cost: float = 0.0
    total_savings: float = 0.0
    avg_training_time_seconds: float = 0.0

    provider_breakdown: Dict[str, int] = Field(default_factory=dict)
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    health_statuses: List[ProviderHealth] = Field(default_factory=list)

    recent_runs: List[PipelineRun] = Field(default_factory=list)
    recent_results: List[TrainingResult] = Field(default_factory=list)
