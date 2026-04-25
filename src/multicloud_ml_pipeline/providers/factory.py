from __future__ import annotations

from ..models import ProviderProfile
from .aws import AWSProvider
from .azure import AzureProvider
from .base import CloudProvider
from .gcp import GCPProvider


def build_provider(profile: ProviderProfile) -> CloudProvider:
    key = profile.name.lower()
    if key == "aws":
        return AWSProvider(profile)
    if key == "gcp":
        return GCPProvider(profile)
    if key == "azure":
        return AzureProvider(profile)
    raise ValueError(f"Unsupported provider: {profile.name}")
