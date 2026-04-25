from __future__ import annotations

import argparse
import json

from .config import load_config
from .flow import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-cloud Prefect training pipeline.")
    parser.add_argument("--config", required=True, help="Path to the JSON pipeline config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy, providers, jobs = load_config(args.config)
    summary = run_training_pipeline(jobs=jobs, provider_profiles=providers, policy=policy)
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
