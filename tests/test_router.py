import unittest

from multicloud_ml_pipeline.models import ProviderProfile, RoutingPolicy, TrainingJob
from multicloud_ml_pipeline.providers.factory import build_provider
from multicloud_ml_pipeline.router import rank_providers


class RouterTests(unittest.TestCase):
    def test_rank_providers_prefers_low_cost_healthy_provider(self) -> None:
        job = TrainingJob(
            job_id="job-1",
            model_name="demo",
            framework="pytorch",
            entrypoint="train.py",
            dataset_uri="s3://data/train.parquet",
            artifact_uri="s3://artifacts",
            dataset_size_gb=20,
            expected_gpu_hours=2,
            gpu_count=1,
            gpu_type="l4",
            max_budget_usd=15,
            preferred_regions=["us-central1"],
        )
        providers = [
            build_provider(
                ProviderProfile(
                    name="aws",
                    regions=["us-east-1"],
                    cost_per_gpu_hour=4.5,
                    health_score=0.9,
                    capacity_score=0.85,
                )
            ),
            build_provider(
                ProviderProfile(
                    name="gcp",
                    regions=["us-central1"],
                    cost_per_gpu_hour=3.0,
                    health_score=0.95,
                    capacity_score=0.9,
                )
            ),
            build_provider(
                ProviderProfile(
                    name="azure",
                    regions=["eastus"],
                    cost_per_gpu_hour=3.7,
                    health_score=0.8,
                    capacity_score=0.75,
                )
            ),
        ]

        ranked = rank_providers(job, providers, RoutingPolicy())

        self.assertEqual(ranked[0].provider, "gcp")
        self.assertEqual([item.rank for item in ranked], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
