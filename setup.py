"""Setup script for Multi-Cloud ML Pipeline Orchestrator."""

from setuptools import setup, find_packages

setup(
    name="multi-cloud-ml-orchestrator",
    version="1.0.0",
    description="End-to-end ML pipeline orchestrator with multi-cloud training, "
                "automatic failover, and cost optimization routing",
    author="Multi-Cloud ML Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "prefect>=2.14.0,<3.0.0",
        "boto3>=1.34.0",
        "google-cloud-aiplatform>=1.38.0",
        "azure-ai-ml>=1.12.0",
        "azure-identity>=1.15.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "flask>=3.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "httpx>=0.25.0",
        "structlog>=23.2.0",
        "rich>=13.7.0",
        "tenacity>=8.2.0",
    ],
    entry_points={
        "console_scripts": [
            "mlorch=orchestrator.pipeline:main",
        ],
    },
)
