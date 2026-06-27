"""Dataset loading for the German MASSIVE intent-classification task.

Single responsibility: turn the raw MASSIVE German subset into a clean
DatasetDict with a known text column and integer intent labels, plus the label
names. Tokenization, modelling and training live in other modules.

Source note: we use the parquet mirror ``mteb/amazon_massive_intent`` (config
``de``) because the original ``AmazonScience/massive`` repo ships a Python
loading script, which modern ``datasets`` versions no longer execute. The mirror
preserves the official train/validation/test split sizes (11514/2033/2974).
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict, load_dataset


@dataclass(frozen=True)
class LoadedData:
    """The prepared splits together with the label vocabulary."""

    splits: DatasetDict
    label_names: list[str]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)


def _build_label_vocabulary(splits: DatasetDict, label_column: str) -> list[str]:
    """Stable, sorted intent vocabulary built from the (string) labels.

    Sorting makes the integer encoding deterministic and independent of split
    order, so the same intent maps to the same id on every machine.
    """
    names = {value for split in splits.values() for value in split[label_column]}
    return sorted(names)


def _encode_labels(
    splits: DatasetDict, label_column: str, label2id: dict[str, int]
) -> DatasetDict:
    """Replace the string intent column with integer label ids."""

    def encode(batch: dict) -> dict:
        return {label_column: [label2id[value] for value in batch[label_column]]}

    return splits.map(encode, batched=True)


def load_massive_de(
    dataset_name: str,
    dataset_config: str,
    text_column: str,
    label_column: str,
) -> LoadedData:
    """Load the official train/validation/test splits of MASSIVE (German).

    Using the official splits (rather than a custom resplit) guarantees
    comparability with prior work and rules out train/test leakage.
    """
    raw = load_dataset(dataset_name, dataset_config)

    keep = {text_column, label_column}
    cleaned = DatasetDict(
        {
            split: dataset.remove_columns(
                [c for c in dataset.column_names if c not in keep]
            )
            for split, dataset in raw.items()
        }
    )

    label_names = _build_label_vocabulary(cleaned, label_column)
    label2id = {name: idx for idx, name in enumerate(label_names)}
    encoded = _encode_labels(cleaned, label_column, label2id)

    return LoadedData(splits=encoded, label_names=label_names)


def subsample_train(
    data: LoadedData,
    train_fraction: float,
    seed: int,
    label_column: str,
    max_train_samples: int | None = None,
) -> LoadedData:
    """Shrink the training split for the low-resource extension or smoke runs.

    The validation and test splits are never touched so every configuration is
    measured on identical evaluation data. Sampling is seeded and shuffled so the
    subset is reproducible but not biased by the dataset's original ordering.
    """
    train = data.splits["train"]

    if train_fraction < 1.0:
        target = max(1, int(len(train) * train_fraction))
        train = train.shuffle(seed=seed).select(range(target))

    if max_train_samples is not None:
        train = train.shuffle(seed=seed).select(range(min(max_train_samples, len(train))))

    new_splits = DatasetDict({**data.splits})
    new_splits["train"] = train
    return LoadedData(splits=new_splits, label_names=data.label_names)
