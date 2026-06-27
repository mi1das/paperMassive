"""Experiment configuration.

Single responsibility: define what fully specifies one fine-tuning run and how to
load it from a YAML file. Each YAML only overrides the fields that differ from the
baseline, which keeps the one-factor-at-a-time design explicit and free of
duplication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce a single run.

    Defaults encode the baseline from the paper's experiment matrix; a config
    file changes one factor at a time relative to these defaults.
    """

    name: str = "baseline"

    # Model / tokenization
    model_checkpoint: str = "deepset/gbert-base"
    max_seq_length: int = 64
    dropout: float = 0.1

    # Optimization
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Reproducibility
    seed: int = 42

    # Data (parquet mirror of MASSIVE German; see src/data.py for rationale)
    dataset_name: str = "mteb/amazon_massive_intent"
    dataset_config: str = "de"
    text_column: str = "text"
    label_column: str = "label"
    # Extension (Option A): fraction of the training split to keep (1.0 = full).
    train_fraction: float = 1.0
    # Debugging only: hard cap on training examples for a fast smoke run.
    max_train_samples: int | None = None

    # Evaluation / checkpoint selection
    metric_for_best_model: str = "accuracy"

    @property
    def run_id(self) -> str:
        """Stable identifier used for output directories and result files."""
        return f"{self.name}_seed{self.seed}"


def _known_field_names() -> set[str]:
    return {f.name for f in fields(ExperimentConfig)}


def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Load a config from YAML, applying defaults and optional CLI overrides.

    Unknown keys raise an error early so a typo in a config never silently
    becomes a no-op.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    merged = {**raw, **{k: v for k, v in overrides.items() if v is not None}}

    unknown = set(merged) - _known_field_names()
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    return ExperimentConfig(**merged)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialise a config for logging alongside results."""
    return asdict(config)
