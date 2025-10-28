from typing import List, Optional

import pydantic
import numpy as np


class SequenceDatasetMetadata(pydantic.BaseModel):
    """Shared metadata contract for Athena corpora."""

    pad_id: int
    eos_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int

    vocab_size: int
    seq_len: int

    num_sequence_identifiers: int
    total_groups: int
    mean_examples_per_group: float
    total_sequences: int

    sets: List[str]
    max_iterations: int
    min_iterations: int


def pad_to_length(arr: np.ndarray, length: int, pad_value: int) -> np.ndarray:
    """Pad or trim a 1D array to a fixed length."""
    if arr.shape[0] >= length:
        return arr[:length]

    result = np.full(length, pad_value, dtype=arr.dtype)
    result[: arr.shape[0]] = arr
    return result


def add_eos(tokens: np.ndarray, eos_id: int, max_len: int, pad_id: int) -> np.ndarray:
    """Append EOS and pad."""
    seq = np.concatenate([tokens, np.array([eos_id], dtype=tokens.dtype)])
    return pad_to_length(seq, max_len, pad_id)
