from __future__ import annotations

from ..exceptions import FatalTrainingError
from ..models import TrainingJob, TrainingResult
from .base import CloudProvider


class AWSProvider(CloudProvider):
    def build_submission_payload(self, job: TrainingJob) -> dict[str, object]:
        payload = super().build_submission_payload(job)
        payload.update(
            {
                "service": "sagemaker",
                "training_job_name": f"{job.job_id}-aws",
                "instance_type": self.profile.metadata.get("instance_type", "ml.g5.2xlarge"),
                "role_arn": self.profile.metadata.get("role_arn", "arn:aws:iam::123456789012:role/SageMakerExecution"),
            }
        )
        return payload

    def submit_sdk_job(self, job: TrainingJob, estimated_cost_usd: float) -> TrainingResult:
        raise FatalTrainingError(
            "AWS SDK submission is a placeholder in this scaffold. "
            "Wire this method to boto3 SageMaker create_training_job for production use."
        )
