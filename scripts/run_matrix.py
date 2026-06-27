"""CLI: run the controlled hyperparameter matrix.

Runs every config in ``configs/`` over every seed, so that *each* configuration
gets a mean +/- std rather than only the baseline. Because a single gbert-base
run on MASSIVE German is cheap (a few minutes on a mid-range GPU), repeating all
configurations is affordable and lets us judge whether any effect exceeds the
per-configuration seed noise -- not just the baseline's. Orchestration only;
the per-run logic lives in ``src``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.pipeline import run_experiment
from src.results_io import save_result

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "results" / "runs"
RESULTS_DIR = REPO_ROOT / "results"

# Every configuration is repeated over these seeds to estimate the seed noise of
# each configuration independently. Five seeds is the common minimum for credibly
# reporting mean +/- std in NLP while keeping the run count affordable.
DEFAULT_SEEDS = [42, 123, 2024, 7, 1337]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hyperparameter matrix.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run every configuration with (default: 42 123 2024 7 1337).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))  # de-duplicate, keep order

    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    if not config_paths:
        raise SystemExit(f"No configs found in {CONFIG_DIR}")

    total = len(config_paths) * len(seeds)
    done = 0
    for path in config_paths:
        for seed in seeds:
            config = load_config(path, seed=seed)
            done += 1
            print(f"\n=== [{done}/{total}] Running {config.run_id} ===")
            result = run_experiment(config, runs_dir=RUNS_DIR)
            save_result(result, RESULTS_DIR)
            print(f"[{config.run_id}] validation={result.validation} test={result.test}")


if __name__ == "__main__":
    main()
