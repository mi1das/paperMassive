"""Persistence of run results.

Single responsibility: write a RunResult to disk and read results back. Keeping
serialisation in one place means scripts never hand-roll JSON layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import RunResult


def save_result(result: RunResult, results_dir: Path) -> Path:
    """Persist a single run as ``<name>_seed<seed>.json``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{result.config['name']}_seed{result.config['seed']}"
    path = results_dir / f"{run_id}.json"

    payload: dict[str, Any] = {
        "config": result.config,
        "train_size": result.train_size,
        "validation": result.validation,
        "test": result.test,
        "label_names": result.label_names,
        "test_predictions": result.test_predictions,
        "test_labels": result.test_labels,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every result JSON in ``results_dir`` (non-recursive)."""
    results: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results
