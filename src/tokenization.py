"""Tokenization step.

Single responsibility: map raw utterances to model inputs with a fixed maximum
sequence length. The tokenizer is created here so the rest of the pipeline only
deals with tokenized datasets.
"""

from __future__ import annotations

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_checkpoint: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_splits(
    splits: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    label_column: str,
    max_seq_length: int,
) -> DatasetDict:
    """Tokenize every split and expose labels under the name the Trainer expects.

    Truncation/padding to ``max_seq_length`` makes the sequence-length factor a
    controlled variable rather than an uncontrolled side effect of input length.
    """

    def encode(batch: dict) -> dict:
        encoded = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        encoded["labels"] = batch[label_column]
        return encoded

    remove = [c for c in splits["train"].column_names]
    return splits.map(encode, batched=True, remove_columns=remove)
