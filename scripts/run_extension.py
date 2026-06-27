"""CLI: low-resource data-scaling extension (paper Sec. 7, Option A).

Re-runs the strongest hyperparameters on subsamples of the German training set
(10/25/50/100 %) to test whether the optimal setting is conditional on the data
regime. Orchestration only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.pipeline import run_experiment
from src.results_io import save_result

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
RESULTS_DIR = REPO_ROOT / "results" / "extension"

FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data-scaling extension.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "baseline.yaml"),
        help="Base config providing the (strongest) hyperparameters to hold fixed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for fraction in FRACTIONS:
        pct = int(round(fraction * 100))
        config = load_config(
            args.config,
            name=f"scale_{pct:03d}pct",
            train_fraction=fraction,
        )
        print(f"\n=== Running {config.run_id} (train_fraction={fraction}) ===")
        result = run_experiment(config, runs_dir=RUNS_DIR)
        save_result(result, RESULTS_DIR)
        print(f"[{config.run_id}] train_size={result.train_size} test={result.test}")


if __name__ == "__main__":
    main()
