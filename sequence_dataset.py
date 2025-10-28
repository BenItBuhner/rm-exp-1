import glob
import os
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.common import SequenceDatasetMetadata
from models.losses import IGNORE_LABEL_ID


class SequenceDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int


class SequenceDataset(IterableDataset):
    """Streaming dataset that mirrors TRM's batching logic but supports iteration targets and domains."""

    def __init__(self, config: SequenceDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self._data: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._iters = 0

        metadata = [self._load_metadata(path) for path in config.dataset_paths]
        base = metadata[0]

        for meta in metadata[1:]:
            assert meta.seq_len == base.seq_len
            assert meta.vocab_size == base.vocab_size
            assert meta.pad_id == base.pad_id

        total_groups = sum(m.total_groups for m in metadata)
        total_sequences = sum(m.total_sequences for m in metadata)
        mean_examples = total_sequences / total_groups

        self.metadata = SequenceDatasetMetadata(
            pad_id=base.pad_id,
            eos_id=base.eos_id,
            ignore_label_id=base.ignore_label_id,
            blank_identifier_id=base.blank_identifier_id,
            vocab_size=base.vocab_size,
            seq_len=base.seq_len,
            num_sequence_identifiers=sum(m.num_sequence_identifiers for m in metadata),
            total_groups=total_groups,
            mean_examples_per_group=mean_examples,
            total_sequences=total_sequences,
            sets=base.sets,
            max_iterations=max(m.max_iterations for m in metadata),
            min_iterations=min(m.min_iterations for m in metadata),
        )

        assert self.config.global_batch_size % self.config.num_replicas == 0
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

    def _load_metadata(self, path: str) -> SequenceDatasetMetadata:
        base_dir = os.path.join(path, self.split)
        candidates = [os.path.join(base_dir, "dataset.json")]
        candidates.extend(sorted(glob.glob(os.path.join(base_dir, "*", "dataset.json"))))

        for candidate in candidates:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    return SequenceDatasetMetadata(**json.load(f))

        raise FileNotFoundError(f"dataset.json not found under {base_dir}")

    def _lazy_load(self):
        if self._data is not None:
            return

        self._data = {}
        for idx, dataset_path in enumerate(self.config.dataset_paths):
            set_suffix = "" if idx == 0 else f"_{idx}"
            base_dir = os.path.join(dataset_path, self.split)
            for set_name in self.metadata.sets:
                key = f"{set_name}{set_suffix}"
                new_dir = os.path.join(base_dir, set_name)
                legacy_prefix = os.path.join(base_dir, f"{set_name}__")

                def _load(field: str, mmap: Optional[str] = None):
                    candidates = [
                        os.path.join(new_dir, f"{field}.npy"),
                        legacy_prefix + f"{field}.npy",
                        legacy_prefix + f"all__{field}.npy",
                    ]
                    for path in candidates:
                        if os.path.exists(path):
                            return np.load(path, mmap_mode=mmap)
                    return None

                inputs = _load("inputs", "r")
                labels = _load("labels", "r")
                puzzle_ids = _load("puzzle_identifiers")
                puzzle_idx = _load("puzzle_indices")
                group_idx = _load("group_indices")
                assert inputs is not None and labels is not None
                assert puzzle_ids is not None and puzzle_idx is not None and group_idx is not None

                # Coerce 1D to 2D for single-example corpora
                if inputs.ndim == 1:
                    inputs = inputs.reshape(1, -1)
                if labels.ndim == 1:
                    labels = labels.reshape(1, -1)

                num_examples = puzzle_ids.shape[0]
                iteration_targets = _load("iteration_targets")
                if iteration_targets is None:
                    iteration_targets = np.full(num_examples, self.metadata.min_iterations, dtype=np.int32)
                domain_ids = _load("domain_ids")
                if domain_ids is None:
                    domain_ids = np.zeros(num_examples, dtype=np.int32)

                self._data[key] = {
                    "inputs": inputs,
                    "labels": labels,
                    "puzzle_identifiers": puzzle_ids,
                    "puzzle_indices": puzzle_idx,
                    "group_indices": group_idx,
                    "iteration_targets": iteration_targets,
                    "domain_ids": domain_ids,
                }

    def _collate(self, batch: Dict[str, np.ndarray]):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            mask = batch["labels"] == self.metadata.ignore_label_id
            batch["labels"][mask] = IGNORE_LABEL_ID

        if batch["inputs"].shape[0] < self.local_batch_size:
            pad = self.local_batch_size - batch["inputs"].shape[0]
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
                "iteration_targets": self.metadata.min_iterations,
                "domain_ids": 0,
            }
            for key in ("inputs", "labels"):
                batch[key] = np.pad(batch[key], ((0, pad), (0, 0)), constant_values=pad_values[key])
            for key in ("puzzle_identifiers", "iteration_targets", "domain_ids"):
                batch[key] = np.pad(batch[key], (0, pad), constant_values=pad_values[key])

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        assert self._data is not None
        for set_name, dataset in self._data.items():
            total = dataset["inputs"].shape[0]
            start = 0
            while start < total:
                end = min(total, start + self.config.global_batch_size)
                local_start = start + self.config.rank * self.local_batch_size
                local_end = min(end, local_start + self.local_batch_size)

                puzzle_idx = []
                ptr = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while ptr + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][ptr + 1]:
                        ptr += 1
                    puzzle_idx.append(ptr)

                batch = self._collate(
                    {
                        "inputs": dataset["inputs"][local_start:local_end],
                        "labels": dataset["labels"][local_start:local_end],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_idx],
                        "iteration_targets": dataset["iteration_targets"][puzzle_idx],
                        "domain_ids": dataset["domain_ids"][puzzle_idx],
                    }
                )
                yield set_name, batch, end - start
                start += self.config.global_batch_size

    def _iter_train(self):
        assert self._data is not None
        rng = np.random.default_rng(self.config.seed + self._iters)

        for set_name, dataset in self._data.items():
            self._iters += 1
            group_indices = dataset["group_indices"]

            order = np.concatenate(
                [rng.permutation(group_indices.size - 1) for _ in range(self.config.epochs_per_iter)]
            )
            start = 0

            while start < order.size:
                batch_ids = []
                puzzle_ids = []
                current = 0

                while start < order.size and current < self.config.global_batch_size:
                    group_id = order[start]
                    start += 1
                    p_start = int(group_indices[group_id])
                    p_end = int(group_indices[group_id + 1])
                    choice = rng.integers(p_start, p_end)
                    batch_ids.append(choice)
                    puzzle_ids.append(choice)
                    current += 1

                if current < self.config.global_batch_size:
                    break

                global_effective = len(batch_ids)
                batch_slice = slice(
                    self.config.rank * self.local_batch_size, (self.config.rank + 1) * self.local_batch_size
                )

                ids = np.array(batch_ids, dtype=np.int64)[batch_slice]
                puzzles = np.array(puzzle_ids, dtype=np.int64)[batch_slice]

                batch = self._collate(
                    {
                        "inputs": dataset["inputs"][ids],
                        "labels": dataset["labels"][ids],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][puzzles],
                        "iteration_targets": dataset["iteration_targets"][puzzles],
                        "domain_ids": dataset["domain_ids"][puzzles],
                    }
                )

                yield set_name, batch, global_effective

    def __iter__(self):
        worker = get_worker_info()
        assert worker is None, "Multithreaded workers are not supported"
        self._lazy_load()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
