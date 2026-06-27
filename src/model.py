"""Model construction.

Single responsibility: build a pretrained Transformer encoder with a
classification head, wiring up the label vocabulary and the dropout rate (one of
the studied hyperparameters).
"""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel


def build_model(
    model_checkpoint: str,
    label_names: list[str],
    dropout: float,
) -> PreTrainedModel:
    """Create an encoder + linear classification head for ``len(label_names)`` intents.

    Dropout is set on both the hidden states and the classifier so the dropout
    factor in the study has a consistent, single meaning across encoders.
    """
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    config = AutoConfig.from_pretrained(
        model_checkpoint,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=dropout,
        classifier_dropout=dropout,
    )

    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        config=config,
    )
