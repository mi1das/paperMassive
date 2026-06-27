"""Evaluation metrics.

Single responsibility: turn model predictions into the two reported scores.
Accuracy is the primary metric; macro-F1 is reported alongside because the intent
classes are imbalanced and macro-F1 weights rare intents equally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
