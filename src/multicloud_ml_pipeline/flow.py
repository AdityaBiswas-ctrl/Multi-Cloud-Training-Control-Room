from __future__ import annotations

from dataclasses import asdict

from .exceptions import FatalTrainingError, RecoverableTrainingError
from .models import PipelineSummary, ProviderProfile, RoutingPolicy, TrainingAttempt, TrainingJob
from .prefect_compat import flow, get_run_logger, task
from .providers.factory import build_provider
from .router import rank_providers


@task
def route_job(job: TrainingJob, policy: RoutingPolicy, provider_profiles: list[dict]) -> list[dict]:
    providers = [
        build_provider(profile if isinstance(profile, ProviderProfile) else ProviderProfile(**profile))
        for profile in provider_profiles
    ]
    decisions = rank_providers(job, providers, policy)
    return [asdict(decision) for decision in decisions]


@task
def train_job_with_failover(
    job: TrainingJob,
    policy: RoutingPolicy,
    provider_profiles: list[dict],
) -> dict:
    logger = get_run_logger()
    providers = {
        normalized.name: build_provider(normalized)
        for normalized in [
            profile if isinstance(profile, ProviderProfile) else ProviderProfile(**profile)
            for profile in provider_profiles
        ]
    }
    routed = rank_providers(job, list(providers.values()), policy)
    attempts: list[TrainingAttempt] = []

    for failover_index, decision in enumerate(routed):
        if failover_index > policy.max_failovers:
            break

        provider = providers[decision.provider]
        logger.info(
            "Submitting %s to %s (rank=%s, est_cost=%.2f)",
            job.job_id,
            decision.provider,
            decision.rank,
            decision.estimated_cost_usd,
        )

        try:
            result = provider.submit_training_job(job, decision.estimated_cost_usd)
            attempts.append(
                TrainingAttempt(
                    provider=decision.provider,
                    status="completed",
                    message="training completed",
                    estimated_cost_usd=decision.estimated_cost_usd,
                    cloud_job_id=result.cloud_job_id,
                )
            )
            return {
                "job_id": job.job_id,
                "model_name": job.model_name,
                "selected_provider": decision.provider,
                "route": [asdict(item) for item in routed],
                "attempts": [asdict(item) for item in attempts],
                "result": asdict(result),
            }
        except RecoverableTrainingError as exc:
            logger.warning("Recoverable failure on %s: %s", decision.provider, exc)
            attempts.append(
                TrainingAttempt(
                    provider=decision.provider,
                    status="failed-recoverable",
                    message=str(exc),
                    estimated_cost_usd=decision.estimated_cost_usd,
                )
            )
            continue
        except FatalTrainingError as exc:
            attempts.append(
                TrainingAttempt(
                    provider=decision.provider,
                    status="failed-fatal",
                    message=str(exc),
                    estimated_cost_usd=decision.estimated_cost_usd,
                )
            )
            raise FatalTrainingError(
                f"Fatal submission failure for {job.job_id} on {decision.provider}: {exc}"
            ) from exc

    raise RecoverableTrainingError(
        f"All failover targets exhausted for {job.job_id}. Attempts={len(attempts)}"
    )


@flow(name="multicloud-model-training")
def run_training_pipeline(
    jobs: list[TrainingJob],
    provider_profiles: list,
    policy: RoutingPolicy | None = None,
) -> PipelineSummary:
    logger = get_run_logger()
    routing_policy = policy or RoutingPolicy()
    provider_dicts = [asdict(profile) if not isinstance(profile, dict) else profile for profile in provider_profiles]

    routed_jobs = []
    total_cost = 0.0

    for job in jobs:
        logger.info("Starting pipeline for job %s", job.job_id)
        routed = train_job_with_failover(job, routing_policy, provider_dicts)
        routed_jobs.append(routed)
        total_cost += routed["result"]["cost_usd"]

    return PipelineSummary(
        routed_jobs=routed_jobs,
        total_estimated_cost_usd=round(total_cost, 2),
    )
