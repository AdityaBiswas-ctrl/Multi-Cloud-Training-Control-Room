import unittest

from multicloud_ml_pipeline.flow import run_training_pipeline
from multicloud_ml_pipeline.models import ProviderProfile, RoutingPolicy, TrainingJob


class FailoverTests(unittest.TestCase):
    def test_failover_moves_to_next_provider_after_recoverable_error(self) -> None:
        job = TrainingJob(
            job_id="job-2",
            model_name="demo",
            framework="pytorch",
            entrypoint="train.py",
            dataset_uri="s3://data/train.parquet",
            artifact_uri="s3://artifacts",
            dataset_size_gb=10,
            expected_gpu_hours=1.5,
            gpu_count=1,
            gpu_type="l4",
            preferred_regions=["us-east-1"],
        )
        providers = [
            ProviderProfile(
                name="aws",
                planned_outcomes=["capacity"],
                cost_per_gpu_hour=2.5,
                health_score=0.96,
                capacity_score=0.95,
                regions=["us-east-1"],
            ),
            ProviderProfile(
                name="gcp",
                planned_outcomes=["success"],
                cost_per_gpu_hour=3.1,
                health_score=0.94,
                capacity_score=0.9,
                regions=["us-central1"],
            ),
            ProviderProfile(
                name="azure",
                planned_outcomes=["success"],
                cost_per_gpu_hour=3.8,
                health_score=0.9,
                capacity_score=0.85,
                regions=["eastus"],
            ),
        ]

        summary = run_training_pipeline([job], providers, RoutingPolicy(max_failovers=2))

        routed_job = summary.routed_jobs[0]
        self.assertEqual(routed_job["selected_provider"], "gcp")
        self.assertEqual(routed_job["attempts"][0]["provider"], "aws")
        self.assertEqual(routed_job["attempts"][0]["status"], "failed-recoverable")
        self.assertEqual(routed_job["attempts"][1]["provider"], "gcp")
        self.assertEqual(routed_job["attempts"][1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
