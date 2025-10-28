from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel, field_validator

from dataset.common import SequenceDatasetMetadata
from dataset.tokenizer_utils import TextTokenizer


cli = ArgParser()


class CorpusDatasetConfig(BaseModel):
    input_files: List[str]
    tokenizer_json: str
    output_dir: str = "data/athena-corpus"

    seq_len: int = 2048
    stride: int = 2048
    train_ratio: float = 0.995
    seed: int = 0

    format: str = "text"  # "text" or "jsonl"
    text_key: str = "text"

    min_iterations: int = 4
    max_iterations: int = 48

    domain_labels: Optional[List[str]] = None

    @field_validator("train_ratio")
    @classmethod
    def _ratio(cls, v: float):
        if not 0.0 < v < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")
        return v

    @field_validator("format")
    @classmethod
    def _format(cls, v: str):
        if v not in {"text", "jsonl"}:
            raise ValueError("format must be 'text' or 'jsonl'")
        return v

    @field_validator("domain_labels")
    @classmethod
    def _domain_labels(cls, v, info):
        input_files = info.data.get("input_files", [])
        if v is not None and len(v) != len(input_files):
            raise ValueError("domain_labels must match number of input_files")
        return v


def open_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def iter_documents(path: str, cfg: CorpusDatasetConfig) -> Iterable[str]:
    with open_file(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if cfg.format == "jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(cfg.text_key)
                if isinstance(text, str):
                    yield text
            else:
                yield line


def chunk_tokens(
    tokenizer: TextTokenizer,
    docs: Iterable[str],
    seq_len: int,
    stride: int,
) -> Iterable[np.ndarray]:
    buffer: List[int] = []
    for doc in docs:
        ids = tokenizer.encode(doc).tolist()
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(tokenizer.eos_id)

        while len(buffer) >= seq_len + 1:
            window = buffer[: seq_len + 1]
            inputs = np.array(window[:-1], dtype=np.int32)
            labels = np.array(window[1:], dtype=np.int32)
            yield inputs, labels
            step = stride if stride > 0 else seq_len
            del buffer[:step]


def compute_iteration_target(tokens: np.ndarray, cfg: CorpusDatasetConfig) -> int:
    unique_ratio = len(np.unique(tokens)) / max(1, tokens.shape[0])
    span = cfg.max_iterations - cfg.min_iterations
    return int(np.clip(cfg.min_iterations + unique_ratio * span, cfg.min_iterations, cfg.max_iterations))


def build_examples(cfg: CorpusDatasetConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int, int]], TextTokenizer]:
    tokenizer = TextTokenizer.from_file(cfg.tokenizer_json)
    examples: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    domain_map: Dict[str, int] = {}

    for idx, path in enumerate(cfg.input_files):
        domain_label = cfg.domain_labels[idx] if cfg.domain_labels else os.path.basename(path)
        if domain_label not in domain_map:
            domain_map[domain_label] = len(domain_map)

        docs = iter_documents(path, cfg)
        for inputs, labels in chunk_tokens(tokenizer, docs, cfg.seq_len, cfg.stride):
            iteration = compute_iteration_target(inputs, cfg)
            examples.append((inputs, labels, iteration, domain_map[domain_label]))

    return examples, tokenizer


def write_split(name: str, data, tokenizer: TextTokenizer, output_dir: Path, cfg: CorpusDatasetConfig):
    target_dir = output_dir / name / "all"
    target_dir.mkdir(parents=True, exist_ok=True)
    inputs = np.stack([item[0] for item in data])
    labels = np.stack([item[1] for item in data])
    iteration_targets = np.array([item[2] for item in data], dtype=np.int32)
    domain_ids = np.array([item[3] for item in data], dtype=np.int32)

    count = inputs.shape[0]
    indices = np.arange(0, count + 1, dtype=np.int32)

    np.save(target_dir / "inputs.npy", inputs)
    np.save(target_dir / "labels.npy", labels)
    np.save(target_dir / "puzzle_identifiers.npy", np.arange(1, count + 1, dtype=np.int32))
    np.save(target_dir / "puzzle_indices.npy", indices)
    np.save(target_dir / "group_indices.npy", indices)
    np.save(target_dir / "iteration_targets.npy", iteration_targets)
    np.save(target_dir / "domain_ids.npy", domain_ids)

    metadata = SequenceDatasetMetadata(
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        ignore_label_id=-100,
        blank_identifier_id=0,
        vocab_size=tokenizer.tokenizer.get_vocab_size(),
        seq_len=cfg.seq_len,
        num_sequence_identifiers=count + 1,
        total_groups=count,
        mean_examples_per_group=1.0,
        total_sequences=count,
        sets=["all"],
        max_iterations=cfg.max_iterations,
        min_iterations=cfg.min_iterations,
    )

    metadata_json = metadata.model_dump_json(indent=2)
    with open(target_dir / "dataset.json", "w", encoding="utf-8") as f:
        f.write(metadata_json)
    with open((output_dir / name / "dataset.json"), "w", encoding="utf-8") as f:
        f.write(metadata_json)


def convert_dataset(config: CorpusDatasetConfig):
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(output_dir)
                break
            except PermissionError:
                if attempt == 2:
                    raise
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, tokenizer = build_examples(config)
    if not len(examples):
        raise RuntimeError("No sequences were produced. Check tokenizer or input files.")

    np.random.shuffle(examples)
    split_index = int(len(examples) * config.train_ratio)
    train_split = examples[:split_index]
    test_split = examples[split_index:]

    write_split("train", train_split, tokenizer, output_dir, config)
    write_split("test", test_split, tokenizer, output_dir, config)

    with open(output_dir / "identifiers.json", "w", encoding="utf-8") as f:
        total = len(examples)
        f.write(json.dumps(["<blank>"] + [f"seq_{i}" for i in range(1, total + 1)]))


@cli.command()
def main(config: CorpusDatasetConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
