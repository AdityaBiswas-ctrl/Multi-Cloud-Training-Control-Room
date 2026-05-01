"""
In-memory metrics collector for pipeline runs, costs, and provider health.
Acts as the single source of truth for the dashboard API.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from models.schemas import (
    CloudProvider,
    CostEstimate,
    DashboardMetrics,
    JobStatus,
    PipelineRun,
    ProviderHealth,
    TrainingResult,
)


class MetricsCollector:
    """Thread-safe singleton metrics collector."""

    _instance: Optional[MetricsCollector] = None
    _lock = threading.Lock()

    def __new__(cls) -> MetricsCollector:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._results: List[TrainingResult] = []
        self._pipeline_runs: List[PipelineRun] = []
        self._health_statuses: Dict[CloudProvider, ProviderHealth] = {}
        self._cost_estimates: List[CostEstimate] = []
        self._data_lock = threading.Lock()

    def record_result(self, result: TrainingResult) -> None:
        """Record a training result."""
        with self._data_lock:
            self._results.append(result)

    def record_pipeline_run(self, run: PipelineRun) -> None:
        """Record a pipeline run."""
        with self._data_lock:
            # Update existing or append
            for i, existing in enumerate(self._pipeline_runs):
                if existing.run_id == run.run_id:
                    self._pipeline_runs[i] = run
                    return
            self._pipeline_runs.append(run)

    def update_health(self, health: ProviderHealth) -> None:
        """Update provider health status."""
        with self._data_lock:
            self._health_statuses[health.provider] = health

    def record_cost_estimate(self, estimate: CostEstimate) -> None:
        """Record a cost estimate."""
        with self._data_lock:
            self._cost_estimates.append(estimate)

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Aggregate all metrics for the dashboard."""
        with self._data_lock:
            completed = [r for r in self._results if r.status == JobStatus.COMPLETED]
            failed = [r for r in self._results if r.status == JobStatus.FAILED]

            # Provider breakdown
            provider_counts: Dict[str, int] = defaultdict(int)
            cost_by_provider: Dict[str, float] = defaultdict(float)
            for r in self._results:
                provider_counts[r.provider.value] += 1
                cost_by_provider[r.provider.value] += r.actual_cost

            # Average training time
            durations = [r.training_duration_seconds for r in completed if r.training_duration_seconds > 0]
            avg_time = sum(durations) / len(durations) if durations else 0.0

            # Total cost and savings
            total_cost = sum(r.actual_cost for r in self._results)
            total_savings = sum(r.actual_cost * 0.3 for r in completed)  # Estimated savings vs on-demand

            return DashboardMetrics(
                total_jobs=len(self._results),
                completed_jobs=len(completed),
                failed_jobs=len(failed),
                total_cost=round(total_cost, 2),
                total_savings=round(total_savings, 2),
                avg_training_time_seconds=round(avg_time, 1),
                provider_breakdown=dict(provider_counts),
                cost_breakdown={k: round(v, 2) for k, v in cost_by_provider.items()},
                health_statuses=list(self._health_statuses.values()),
                recent_runs=self._pipeline_runs[-10:],
                recent_results=self._results[-20:],
            )

    def get_health_statuses(self) -> List[ProviderHealth]:
        """Get all provider health statuses."""
        with self._data_lock:
            return list(self._health_statuses.values())

    def get_results(self, limit: int = 50) -> List[TrainingResult]:
        """Get recent training results."""
        with self._data_lock:
            return self._results[-limit:]

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._data_lock:
            self._results.clear()
            self._pipeline_runs.clear()
            self._health_statuses.clear()
            self._cost_estimates.clear()
