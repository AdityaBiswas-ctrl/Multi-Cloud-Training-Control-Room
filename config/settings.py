"""
Global settings loaded from environment variables with Pydantic validation.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class CostStrategy(str, Enum):
    CHEAPEST = "cheapest"
    BALANCED = "balanced"
    GEOGRAPHIC = "geographic"
    ROUND_ROBIN = "round_robin"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # --- General ---
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    # --- AWS ---
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    aws_sagemaker_role_arn: str = ""
    aws_s3_bucket: str = ""

    # --- GCP ---
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    gcp_service_account_key_path: str = ""
    gcp_staging_bucket: str = ""

    # --- Azure ---
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_workspace_name: str = ""
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""

    # --- Prefect ---
    prefect_api_url: str = "http://localhost:4200/api"

    # --- Dashboard ---
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 5050
    dashboard_secret_key: str = "change-me-in-production"

    # --- Cost Optimization ---
    cost_strategy: CostStrategy = CostStrategy.CHEAPEST
    spot_instance_preference: bool = True
    max_budget_per_job: float = 50.00

    model_config = {
        "env_file": str(ROOT_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def aws_configured(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def gcp_configured(self) -> bool:
        return bool(self.gcp_project_id)

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_subscription_id and self.azure_client_id)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
