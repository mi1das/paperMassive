"""End-to-end run of a single configuration.

This module is the high-level policy (SLAP): it reads at one level of
abstraction -- load, subsample, tokenize, build, train, evaluate -- and delegates
every detail to a single-responsibility module. Scripts call ``run_experiment``;
they should contain no ML logic themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, config_to_dict
from .data import load_massive_de, subsample_train
from .model import build_model
from .seeding import set_seed
from .tokenization import build_tokenizer, tokenize_splits
from .trainer import build_trainer


@dataclass(frozen=True)
class RunResult:
    """Scores plus enough metadata to connect a result back to its config."""

    config: dict[str, Any]
    train_size: int
    validation: dict[str, float]
    test: dict[str, float]
    test_predictions: list[int]
    test_labels: list[int]
    label_names: list[str]


def _strip_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Turn Trainer's ``eval_accuracy`` style keys into plain metric names."""
    return {
        key[len(prefix):]: value
        for key, value in metrics.items()
        if key.startswith(prefix) and isinstance(value, (int, float))
    }


def run_experiment(config: ExperimentConfig, runs_dir: Path) -> RunResult:
    """Train and evaluate one configuration; return validation and test scores."""
    set_seed(config.seed)

    data = load_massive_de(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        text_column=config.text_column,
        label_column=config.label_column,
    )
    data = subsample_train(
        data,
        train_fraction=config.train_fraction,
        seed=config.seed,
        label_column=config.label_column,
        max_train_samples=config.max_train_samples,
    )

    tokenizer = build_tokenizer(config.model_checkpoint)
    tokenized = tokenize_splits(
        data.splits,
        tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_seq_length=config.max_seq_length,
    )

    model = build_model(config.model_checkpoint, data.label_names, config.dropout)

    import torch

    output_dir = str(runs_dir / config.run_id)
    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        output_dir=output_dir,
        use_fp16=torch.cuda.is_available(),
    )

    trainer.train()

    validation = _strip_prefix(trainer.evaluate(tokenized["validation"]), "eval_")

    test_output = trainer.predict(tokenized["test"])
    test_metrics = _strip_prefix(test_output.metrics, "test_")
    test_predictions = np.argmax(test_output.predictions, axis=-1).tolist()
    test_labels = list(test_output.label_ids)

    return RunResult(
        config=config_to_dict(config),
        train_size=len(tokenized["train"]),
        validation=validation,
        test=test_metrics,
        test_predictions=test_predictions,
        test_labels=[int(x) for x in test_labels],
        label_names=data.label_names,
    )
