"""
GCP Vertex AI cloud provider implementation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict

from models.schemas import (
    CloudProvider, HealthStatus, JobStatus,
    ProviderHealth, TrainingJob, TrainingResult,
)
from providers.base import BaseCloudProvider
from utils.logging import get_logger

logger = get_logger(__name__)


class GCPProvider(BaseCloudProvider):
    TIER_INSTANCE_MAP = {
        "small": "n1-standard-4",
        "medium": "n1-standard-16",
        "large": "n1-standard-8-v100",
        "gpu_intensive": "a2-highgpu-1g",
    }

    def __init__(self, settings=None):
        super().__init__(CloudProvider.GCP)
        self._settings = settings
        self._inited = False

    def _init_sdk(self):
        if not self._inited:
            from google.cloud import aiplatform
            aiplatform.init(
                project=getattr(self._settings, "gcp_project_id", ""),
                location=getattr(self._settings, "gcp_region", "us-central1"),
            )
            self._inited = True

    async def submit_training_job(self, job: TrainingJob) -> str:
        self._init_sdk()
        from google.cloud import aiplatform
        machine = job.instance_type or self.TIER_INSTANCE_MAP.get(job.tier.value, "n1-standard-4")
        name = f"mlorch-{job.job_id}-{uuid.uuid4().hex[:6]}"
        logger.info("submitting_gcp_job", job_id=job.job_id, machine=machine)
        cj = aiplatform.CustomJob(
            display_name=name,
            worker_pool_specs=[{"machine_spec": {"machine_type": machine}, "replica_count": 1,
                                "container_spec": {"image_uri": "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-7:latest"}}],
        )
        await asyncio.to_thread(cj.run, sync=False)
        return name

    async def get_job_status(self, remote_job_id: str) -> Dict[str, Any]:
        try:
            self._init_sdk()
            from google.cloud import aiplatform
            jobs = await asyncio.to_thread(aiplatform.CustomJob.list, filter=f'display_name="{remote_job_id}"')
            if not jobs:
                return {"status": "unknown", "progress": 0, "message": "Not found"}
            state = str(jobs[0].state)
            m = {"SUCCEEDED": "completed", "FAILED": "failed", "RUNNING": "running", "CANCELLED": "cancelled"}
            s = next((v for k, v in m.items() if k in state), "running")
            return {"status": s, "progress": 0, "message": ""}
        except Exception as e:
            return {"status": "failed", "progress": 0, "message": str(e)}

    async def get_training_result(self, remote_job_id: str, job: TrainingJob) -> TrainingResult:
        return TrainingResult(
            job_id=job.job_id, provider=CloudProvider.GCP, status=JobStatus.COMPLETED,
            actual_cost=await self.estimate_cost(job),
            instance_type=self.TIER_INSTANCE_MAP.get(job.tier.value, "n1-standard-4"),
            failover_count=job.failover_count, attempted_providers=job.attempted_providers,
        )

    async def cancel_job(self, remote_job_id: str) -> bool:
        return False

    async def check_health(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            self._init_sdk()
            from google.cloud import aiplatform
            await asyncio.to_thread(aiplatform.CustomJob.list, filter='display_name="health"')
            lat = (time.monotonic() - start) * 1000
            return ProviderHealth(provider=CloudProvider.GCP, status=HealthStatus.HEALTHY,
                                  latency_ms=round(lat, 1), last_check=datetime.utcnow(),
                                  last_success=datetime.utcnow(), message="OK")
        except Exception as e:
            lat = (time.monotonic() - start) * 1000
            return ProviderHealth(provider=CloudProvider.GCP, status=HealthStatus.UNHEALTHY,
                                  latency_ms=round(lat, 1), last_check=datetime.utcnow(),
                                  last_failure=datetime.utcnow(), message=str(e))

    async def estimate_cost(self, job: TrainingJob) -> float:
        pricing = {"n1-standard-4": (0.19, 0.04), "n1-standard-16": (0.76, 0.16),
                    "n1-standard-8-v100": (2.95, 0.89), "a2-highgpu-1g": (3.67, 1.10)}
        hours = {"small": 0.5, "medium": 2.0, "large": 8.0, "gpu_intensive": 24.0}
        inst = job.instance_type or self.TIER_INSTANCE_MAP.get(job.tier.value, "n1-standard-4")
        od, sp = pricing.get(inst, (0.19, 0.04))
        h = hours.get(job.tier.value, 1.0)
        return round((sp if job.use_spot else od) * h, 2)

    async def cleanup(self, remote_job_id: str) -> None:
        logger.info("gcp_cleanup", remote_job_id=remote_job_id)
