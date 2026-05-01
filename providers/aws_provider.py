"""
AWS SageMaker cloud provider implementation.

Wraps boto3 SageMaker client to submit, monitor, and retrieve
ML training jobs on AWS infrastructure.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from models.schemas import (
    CloudProvider,
    HealthStatus,
    JobStatus,
    ProviderHealth,
    TrainingJob,
    TrainingResult,
)
from providers.base import BaseCloudProvider
from utils.logging import get_logger

logger = get_logger(__name__)


class AWSProvider(BaseCloudProvider):
    """AWS SageMaker training provider."""

    # Instance mapping from tier → SageMaker instance
    TIER_INSTANCE_MAP = {
        "small": "ml.m5.xlarge",
        "medium": "ml.m5.4xlarge",
        "large": "ml.p3.2xlarge",
        "gpu_intensive": "ml.g5.2xlarge",
    }

    def __init__(self, settings: Any = None) -> None:
        super().__init__(CloudProvider.AWS)
        self._settings = settings
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-initialize the SageMaker client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "sagemaker",
                    region_name=self._settings.aws_default_region if self._settings else "us-east-1",
                    aws_access_key_id=self._settings.aws_access_key_id if self._settings else None,
                    aws_secret_access_key=self._settings.aws_secret_access_key if self._settings else None,
                )
            except Exception as e:
                logger.error("failed_to_init_aws_client", error=str(e))
                raise
        return self._client

    async def submit_training_job(self, job: TrainingJob) -> str:
        """Submit a SageMaker training job."""
        instance_type = job.instance_type or self.TIER_INSTANCE_MAP.get(job.tier.value, "ml.m5.xlarge")
        training_job_name = f"mlorch-{job.job_id}-{uuid.uuid4().hex[:6]}"

        logger.info(
            "submitting_aws_training_job",
            job_id=job.job_id,
            instance_type=instance_type,
            training_job_name=training_job_name,
        )

        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.create_training_job,
                TrainingJobName=training_job_name,
                AlgorithmSpecification={
                    "TrainingImage": self._get_training_image(job.training_config.algorithm),
                    "TrainingInputMode": "File",
                },
                RoleArn=self._settings.aws_sagemaker_role_arn if self._settings else "",
                ResourceConfig={
                    "InstanceType": instance_type,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 50,
                },
                StoppingCondition={"MaxRuntimeInSeconds": 3600},
                HyperParameters={
                    k: str(v) for k, v in job.training_config.hyperparameters.items()
                },
                EnableManagedSpotTraining=job.use_spot,
            )
            logger.info("aws_job_submitted", training_job_name=training_job_name)
            return training_job_name
        except Exception as e:
            logger.error("aws_submit_failed", error=str(e), job_id=job.job_id)
            raise

    async def get_job_status(self, remote_job_id: str) -> Dict[str, Any]:
        """Poll SageMaker training job status."""
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.describe_training_job,
                TrainingJobName=remote_job_id,
            )
            sm_status = response.get("TrainingJobStatus", "Unknown")
            status_map = {
                "InProgress": "running",
                "Completed": "completed",
                "Failed": "failed",
                "Stopping": "running",
                "Stopped": "cancelled",
            }
            return {
                "status": status_map.get(sm_status, "unknown"),
                "progress": response.get("SecondaryStatusTransitions", []),
                "message": response.get("FailureReason", ""),
            }
        except Exception as e:
            return {"status": "failed", "progress": 0, "message": str(e)}

    async def get_training_result(self, remote_job_id: str, job: TrainingJob) -> TrainingResult:
        """Retrieve SageMaker training results."""
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.describe_training_job,
                TrainingJobName=remote_job_id,
            )

            metrics = response.get("FinalMetricDataList", [])
            metric_dict = {m["MetricName"]: m["Value"] for m in metrics}

            duration = 0.0
            if response.get("TrainingStartTime") and response.get("TrainingEndTime"):
                duration = (
                    response["TrainingEndTime"] - response["TrainingStartTime"]
                ).total_seconds()

            return TrainingResult(
                job_id=job.job_id,
                provider=CloudProvider.AWS,
                status=JobStatus.COMPLETED,
                accuracy=metric_dict.get("accuracy"),
                loss=metric_dict.get("loss"),
                f1_score=metric_dict.get("f1_score"),
                custom_metrics=metric_dict,
                actual_cost=self._calculate_cost(response),
                training_duration_seconds=duration,
                instance_type=response.get("ResourceConfig", {}).get("InstanceType", ""),
                model_artifact_path=response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
                started_at=response.get("TrainingStartTime"),
                completed_at=response.get("TrainingEndTime"),
                failover_count=job.failover_count,
                attempted_providers=job.attempted_providers,
            )
        except Exception as e:
            return TrainingResult(
                job_id=job.job_id,
                provider=CloudProvider.AWS,
                status=JobStatus.FAILED,
                error_message=str(e),
                failover_count=job.failover_count,
                attempted_providers=job.attempted_providers,
            )

    async def cancel_job(self, remote_job_id: str) -> bool:
        """Cancel a SageMaker training job."""
        try:
            client = self._get_client()
            await asyncio.to_thread(
                client.stop_training_job,
                TrainingJobName=remote_job_id,
            )
            return True
        except Exception:
            return False

    async def check_health(self) -> ProviderHealth:
        """Check AWS SageMaker API health."""
        start = time.monotonic()
        try:
            client = self._get_client()
            await asyncio.to_thread(
                client.list_training_jobs, MaxResults=1
            )
            latency = (time.monotonic() - start) * 1000

            return ProviderHealth(
                provider=CloudProvider.AWS,
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                last_check=datetime.utcnow(),
                last_success=datetime.utcnow(),
                message="SageMaker API responding normally",
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ProviderHealth(
                provider=CloudProvider.AWS,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 1),
                last_check=datetime.utcnow(),
                last_failure=datetime.utcnow(),
                message=f"Health check failed: {e}",
            )

    async def estimate_cost(self, job: TrainingJob) -> float:
        """Estimate training cost on AWS."""
        # Simplified estimation based on pricing data
        instance_type = job.instance_type or self.TIER_INSTANCE_MAP.get(job.tier.value, "ml.m5.xlarge")
        pricing = {
            "ml.m5.xlarge": {"on_demand": 0.23, "spot": 0.08},
            "ml.m5.4xlarge": {"on_demand": 0.92, "spot": 0.32},
            "ml.p3.2xlarge": {"on_demand": 3.825, "spot": 1.15},
            "ml.g5.xlarge": {"on_demand": 1.41, "spot": 0.50},
            "ml.g5.2xlarge": {"on_demand": 1.52, "spot": 0.53},
        }
        tier_hours = {"small": 0.5, "medium": 2.0, "large": 8.0, "gpu_intensive": 24.0}

        price_info = pricing.get(instance_type, {"on_demand": 0.23, "spot": 0.08})
        hours = tier_hours.get(job.tier.value, 1.0)
        rate = price_info["spot"] if job.use_spot else price_info["on_demand"]

        return round(rate * hours, 2)

    async def cleanup(self, remote_job_id: str) -> None:
        """Clean up SageMaker resources."""
        logger.info("aws_cleanup", remote_job_id=remote_job_id)

    @staticmethod
    def _get_training_image(algorithm: str) -> str:
        """Get the SageMaker training container image URI."""
        images = {
            "xgboost": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
            "sklearn": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1",
        }
        return images.get(algorithm, images["xgboost"])

    @staticmethod
    def _calculate_cost(response: Dict) -> float:
        """Calculate actual cost from SageMaker response."""
        billable = response.get("BillableTimeInSeconds", 0)
        instance = response.get("ResourceConfig", {}).get("InstanceType", "ml.m5.xlarge")
        pricing = {
            "ml.m5.xlarge": 0.23, "ml.m5.4xlarge": 0.92,
            "ml.p3.2xlarge": 3.825, "ml.g5.xlarge": 1.41,
        }
        rate = pricing.get(instance, 0.23)
        return round((billable / 3600) * rate, 4)
