from __future__ import annotations

from .exceptions import RoutingError
from .models import RouteDecision, RoutingPolicy, TrainingJob
from .providers.base import CloudProvider


def rank_providers(
    job: TrainingJob,
    providers: list[CloudProvider],
    policy: RoutingPolicy,
) -> list[RouteDecision]:
    decisions: list[RouteDecision] = []

    for provider in providers:
        health = provider.check_health()
        if not health.healthy:
            continue

        estimate = provider.estimate_cost(job)
        if not estimate.within_budget:
            continue

        region_match = 1.0 if any(region in provider.profile.regions for region in job.preferred_regions) else 0.0
        score = (
            estimate.total_cost_usd * policy.cost_weight
            - health.health_score * 100 * policy.health_weight
            - health.capacity_score * 100 * policy.capacity_weight
            - region_match * 100 * policy.region_weight
        )
        reason = (
            f"cost={estimate.total_cost_usd:.2f}, "
            f"health={health.health_score:.2f}, "
            f"capacity={health.capacity_score:.2f}, "
            f"region_match={region_match:.0f}"
        )
        decisions.append(
            RouteDecision(
                provider=provider.name,
                rank=0,
                score=round(score, 4),
                estimated_cost_usd=estimate.total_cost_usd,
                reason=reason,
            )
        )

    if not decisions:
        raise RoutingError(f"No eligible providers found for job {job.job_id}")

    ordered = sorted(decisions, key=lambda item: item.score)
    for index, decision in enumerate(ordered, start=1):
        decision.rank = index
    return ordered
