"""Training step.

Single responsibility: translate an ExperimentConfig into a configured
Hugging Face Trainer. Checkpoint selection is part of training, so the
"keep the best epoch on validation" rule lives here.
"""

from __future__ import annotations

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .config import ExperimentConfig
from .metrics import compute_metrics


def build_trainer(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
    use_fp16: bool,
) -> Trainer:
    """Configure a Trainer that selects the best epoch by validation metric.

    ``load_best_model_at_end`` plus per-epoch evaluation/saving implements the
    checkpoint-selection rule from the paper without leaking the test split.
    """
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        seed=config.seed,
        data_seed=config.seed,
        fp16=use_fp16,
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=False,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
