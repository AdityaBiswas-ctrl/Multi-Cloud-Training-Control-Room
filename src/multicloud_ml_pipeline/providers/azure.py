from __future__ import annotations

from ..exceptions import FatalTrainingError
from ..models import TrainingJob, TrainingResult
from .base import CloudProvider


class AzureProvider(CloudProvider):
    def build_submission_payload(self, job: TrainingJob) -> dict[str, object]:
        payload = super().build_submission_payload(job)
        payload.update(
            {
                "service": "azure-ml",
                "command_job_name": f"{job.job_id}-azure",
                "compute_name": self.profile.metadata.get("compute_name", "gpu-cluster"),
                "environment": self.profile.metadata.get("environment", "pytorch-2.3-cuda12"),
            }
        )
        return payload

    def submit_sdk_job(self, job: TrainingJob, estimated_cost_usd: float) -> TrainingResult:
        raise FatalTrainingError(
            "Azure SDK submission is a placeholder in this scaffold. "
            "Wire this method to Azure ML command job submission for production use."
        )
