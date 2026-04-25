from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain, repeat

from ..exceptions import FatalTrainingError, RecoverableTrainingError
from ..models import CostEstimate, ProviderHealth, ProviderProfile, TrainingJob, TrainingResult


class CloudProvider(ABC):
    def __init__(self, profile: ProviderProfile) -> None:
        self.profile = profile
        self._outcomes = iter(chain(profile.planned_outcomes, repeat("success")))

    @property
    def name(self) -> str:
        return self.profile.name

    def check_health(self) -> ProviderHealth:
        healthy = self.profile.enabled and self.profile.health_score > 0.25
        detail = "ready" if healthy else "provider disabled or unhealthy"
        return ProviderHealth(
            provider=self.name,
            healthy=healthy,
            health_score=self.profile.health_score,
            capacity_score=self.profile.capacity_score,
            latency_score=self.profile.latency_score,
            detail=detail,
        )

    def estimate_cost(self, job: TrainingJob) -> CostEstimate:
        discounted_compute = (
            job.expected_gpu_hours
            * job.gpu_count
            * self.profile.cost_per_gpu_hour
            * (1.0 - self.profile.spot_discount)
        )
        egress = job.dataset_size_gb * self.profile.egress_cost_per_gb
        startup_penalty = self.profile.startup_minutes * 0.05
        total = round(discounted_compute + egress + startup_penalty, 2)
        within_budget = job.max_budget_usd is None or total <= job.max_budget_usd
        return CostEstimate(
            provider=self.name,
            total_cost_usd=total,
            compute_cost_usd=round(discounted_compute, 2),
            egress_cost_usd=round(egress, 2),
            startup_penalty_usd=round(startup_penalty, 2),
            within_budget=within_budget,
        )

    def build_submission_payload(self, job: TrainingJob) -> dict[str, object]:
        return {
            "provider": self.name,
            "job_id": job.job_id,
            "model_name": job.model_name,
            "framework": job.framework,
            "entrypoint": job.entrypoint,
            "dataset_uri": job.dataset_uri,
            "artifact_uri": job.artifact_uri,
            "gpu_type": job.gpu_type,
            "gpu_count": job.gpu_count,
            "hyperparameters": job.hyperparameters,
            "tags": job.tags,
        }

    def submit_training_job(self, job: TrainingJob, estimated_cost_usd: float) -> TrainingResult:
        if self.profile.mode == "simulation":
            return self._submit_simulated_job(job, estimated_cost_usd)
        return self.submit_sdk_job(job, estimated_cost_usd)

    def _submit_simulated_job(self, job: TrainingJob, estimated_cost_usd: float) -> TrainingResult:
        outcome = next(self._outcomes)
        if outcome == "success":
            return TrainingResult(
                provider=self.name,
                cloud_job_id=f"{self.name}-{job.job_id}",
                status="completed",
                artifact_uri=f"{job.artifact_uri}/{self.name}/{job.job_id}",
                metrics={
                    "accuracy": round(0.84 + (self.profile.health_score / 10), 4),
                    "f1_score": round(0.8 + (self.profile.capacity_score / 10), 4),
                },
                cost_usd=estimated_cost_usd,
            )
        if outcome in {"capacity", "timeout", "transient"}:
            raise RecoverableTrainingError(f"{self.name} failed with recoverable error: {outcome}")
        raise FatalTrainingError(f"{self.name} failed with fatal error: {outcome}")

    @abstractmethod
    def submit_sdk_job(self, job: TrainingJob, estimated_cost_usd: float) -> TrainingResult:
        raise NotImplementedError
