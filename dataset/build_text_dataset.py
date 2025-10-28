from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Dict

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel, Field, field_validator

from dataset.common import SequenceDatasetMetadata, pad_to_length, add_eos
from dataset.tokenizer_utils import TextTokenizer


cli = ArgParser()


class TextRecord(BaseModel):
    prompt: str
    response: str
    meta: Dict[str, Optional[int]] = Field(default_factory=dict)


class TextDatasetConfig(BaseModel):
    input_files: List[str]
    tokenizer_json: str
    output_dir: str = "data/athena-base"

    max_context_tokens: int = 1024
    max_response_tokens: int = 512

    train_ratio: float = 0.98
    seed: int = 7

    min_iterations: int = 4
    max_iterations: int = 48

    domain_weights: Optional[Dict[str, float]] = None

    @field_validator("train_ratio")
    @classmethod
    def _check_ratio(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")
        return v


def read_examples(files: List[str]) -> Iterable[TextRecord]:
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield TextRecord(**json.loads(line))


def encode_example(
    tokenizer: TextTokenizer,
    prompt: str,
    response: str,
    cfg: TextDatasetConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    max_tokens = max(cfg.max_context_tokens, cfg.max_response_tokens)
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)

    prompt_seq = pad_to_length(prompt_ids, max_tokens, tokenizer.pad_id)
    response_seq = add_eos(response_ids, eos_id=tokenizer.eos_id, max_len=max_tokens, pad_id=tokenizer.pad_id)
    return prompt_seq, response_seq


def write_split(
    name: str,
    data: List[Tuple[np.ndarray, np.ndarray, int, int]],
    tokenizer: TextTokenizer,
    cfg: TextDatasetConfig,
):
    os.makedirs(os.path.join(cfg.output_dir, name), exist_ok=True)

    inputs = np.stack([inp for inp, _, _, _ in data])
    labels = np.stack([lab for _, lab, _, _ in data])
    iteration_targets = np.array([it for _, _, it, _ in data], dtype=np.int32)
    domain_ids = np.array([dom for _, _, _, dom in data], dtype=np.int32)

    indices = np.arange(0, inputs.shape[0] + 1, dtype=np.int32)

    np.save(os.path.join(cfg.output_dir, name, "all__inputs.npy"), inputs)
    np.save(os.path.join(cfg.output_dir, name, "all__labels.npy"), labels)
    np.save(
        os.path.join(cfg.output_dir, name, "all__puzzle_identifiers.npy"),
        np.arange(1, inputs.shape[0] + 1, dtype=np.int32),
    )
    np.save(os.path.join(cfg.output_dir, name, "all__puzzle_indices.npy"), indices)
    np.save(os.path.join(cfg.output_dir, name, "all__group_indices.npy"), indices)
    np.save(os.path.join(cfg.output_dir, name, "all__iteration_targets.npy"), iteration_targets)
    np.save(os.path.join(cfg.output_dir, name, "all__domain_ids.npy"), domain_ids)

    metadata = SequenceDatasetMetadata(
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        ignore_label_id=-100,
        blank_identifier_id=0,
        vocab_size=tokenizer.tokenizer.get_vocab_size(),
        seq_len=max(cfg.max_context_tokens, cfg.max_response_tokens),
        num_sequence_identifiers=inputs.shape[0] + 1,
        total_groups=inputs.shape[0],
        mean_examples_per_group=1.0,
        total_sequences=inputs.shape[0],
        sets=["all"],
        max_iterations=cfg.max_iterations,
        min_iterations=cfg.min_iterations,
    )

    with open(os.path.join(cfg.output_dir, name, "dataset.json"), "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))


def compute_iteration_target(meta: Dict[str, Optional[int]], cfg: TextDatasetConfig) -> int:
    if "iteration_budget" in meta and meta["iteration_budget"] is not None:
        return int(meta["iteration_budget"])

    complexity = int(meta.get("complexity", 0) or 0)
    return int(np.clip(cfg.min_iterations + complexity, cfg.min_iterations, cfg.max_iterations))


def assign_domain(meta: Dict[str, Optional[int]], mapping: Dict[str, int]) -> int:
    name = str(meta.get("domain") or "generic")
    if name not in mapping:
        mapping[name] = len(mapping)
    return mapping[name]


@cli.command()
def main(config: TextDatasetConfig):
    np.random.seed(config.seed)
    tokenizer = TextTokenizer.from_file(config.tokenizer_json)

    entries: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    domain_ids: Dict[str, int] = {}

    for record in read_examples(config.input_files):
        prompt_seq, response_seq = encode_example(tokenizer, record.prompt, record.response, config)
        iteration_target = compute_iteration_target(record.meta, config)
        domain_id = assign_domain(record.meta, domain_ids)
        entries.append((prompt_seq, response_seq, iteration_target, domain_id))

    np.random.shuffle(entries)
    split = int(len(entries) * config.train_ratio)
    train_split = entries[:split]
    test_split = entries[split:]

    write_split("train", train_split, tokenizer, config)
    write_split("test", test_split, tokenizer, config)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w", encoding="utf-8") as f:
        json.dump(["<blank>"] + [f"seq_{i}" for i in range(1, len(train_split) + len(test_split) + 1)], f)


if __name__ == "__main__":
    cli()
