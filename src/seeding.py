"""Reproducibility helpers.

Single responsibility: make a run deterministic given a seed. Centralising this
avoids seeding random/numpy/torch inconsistently across the codebase.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducible fine-tuning."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Imported lazily so non-training utilities (e.g. result aggregation) do not
    # need torch installed.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
