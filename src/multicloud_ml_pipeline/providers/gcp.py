from __future__ import annotations

from ..exceptions import FatalTrainingError
from ..models import TrainingJob, TrainingResult
from .base import CloudProvider


class GCPProvider(CloudProvider):
    def build_submission_payload(self, job: TrainingJob) -> dict[str, object]:
        payload = super().build_submission_payload(job)
        payload.update(
            {
                "service": "vertex-ai",
                "custom_job_name": f"{job.job_id}-gcp",
                "machine_type": self.profile.metadata.get("machine_type", "g2-standard-8"),
                "accelerator_type": self.profile.metadata.get("accelerator_type", "NVIDIA_L4"),
            }
        )
        return payload

    def submit_sdk_job(self, job: TrainingJob, estimated_cost_usd: float) -> TrainingResult:
        raise FatalTrainingError(
            "GCP SDK submission is a placeholder in this scaffold. "
            "Wire this method to Vertex AI CustomJob submission for production use."
        )
