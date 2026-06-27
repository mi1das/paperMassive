"""CLI: run a single experiment configuration.

Thin entry point -- argument parsing only. All ML logic lives in ``src``.

Example:
    python -m scripts.run_experiment --config configs/baseline.yaml
    python -m scripts.run_experiment --config configs/baseline.yaml --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.pipeline import run_experiment
from src.results_io import save_result

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fine-tuning configuration.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run: cap training to a few hundred examples and 1 epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.smoke:
        overrides["max_train_samples"] = 256
        overrides["num_epochs"] = 1
        overrides["name"] = "smoke"

    config = load_config(args.config, **overrides)
    result = run_experiment(config, runs_dir=RUNS_DIR)
    path = save_result(result, RESULTS_DIR)

    print(f"\n[{config.run_id}] validation={result.validation} test={result.test}")
    print(f"Saved result to {path}")


if __name__ == "__main__":
    main()
