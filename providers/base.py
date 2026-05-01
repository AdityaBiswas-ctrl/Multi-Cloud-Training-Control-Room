"""
Abstract base class for all cloud ML providers.

Every provider (AWS, GCP, Azure, Mock) implements this interface to ensure
a uniform contract for the orchestrator and failover engine.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from models.schemas import (
    CloudProvider,
    HealthStatus,
    ProviderHealth,
    TrainingJob,
    TrainingResult,
)


class BaseCloudProvider(abc.ABC):
    """Abstract cloud ML provider."""

    def __init__(self, provider_type: CloudProvider) -> None:
        self.provider_type = provider_type

    @abc.abstractmethod
    async def submit_training_job(self, job: TrainingJob) -> str:
        """
        Submit a training job to the cloud provider.

        Returns:
            Remote job ID assigned by the cloud platform.
        """
        ...

    @abc.abstractmethod
    async def get_job_status(self, remote_job_id: str) -> Dict[str, Any]:
        """
        Poll the status of a running job.

        Returns:
            Dict with keys: status, progress, message
        """
        ...

    @abc.abstractmethod
    async def get_training_result(self, remote_job_id: str, job: TrainingJob) -> TrainingResult:
        """
        Retrieve the result of a completed training job.

        Returns:
            TrainingResult with metrics and artifact paths.
        """
        ...

    @abc.abstractmethod
    async def cancel_job(self, remote_job_id: str) -> bool:
        """Cancel a running training job. Returns True on success."""
        ...

    @abc.abstractmethod
    async def check_health(self) -> ProviderHealth:
        """
        Perform a health check against the cloud provider.

        Returns:
            ProviderHealth with latency, status, and error info.
        """
        ...

    @abc.abstractmethod
    async def estimate_cost(self, job: TrainingJob) -> float:
        """
        Estimate the cost for a training job on this provider.

        Returns:
            Estimated cost in USD.
        """
        ...

    @abc.abstractmethod
    async def cleanup(self, remote_job_id: str) -> None:
        """Clean up any resources created for a job."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_type.value}>"
