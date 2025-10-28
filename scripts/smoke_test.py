"""
Athena v2 smoke test helper.

This script downloads a tiny public dataset from Hugging Face, converts it into
Athena's raw corpus format, and (optionally) runs a short training loop to
validate the end-to-end pipeline on a single machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from hydra import initialize_config_dir, compose
import importlib
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dataset.build_corpus_dataset import CorpusDatasetConfig, convert_dataset
from pretrain import run_with_config


def prepare_raw_jsonl(args) -> Path:
    """Download HF dataset splits and save small JSONL files with `{"text": ...}`."""
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    raw_dir = Path(args.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def write_split(split_name: str, hf_split: str, limit: int):
        data = dataset[hf_split]
        limit = min(limit, len(data))
        out_file = raw_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for row in data.select(range(limit)):
                text = row[args.text_key]
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")

    write_split("train", args.train_split, args.max_train_samples)
    write_split("test", args.eval_split, args.max_eval_samples)
    return raw_dir


def fetch_tokenizer(args) -> Path:
    if args.tokenizer_json:
        return Path(args.tokenizer_json)
    downloaded = hf_hub_download(args.hf_tokenizer_repo, "tokenizer.json")
    return Path(downloaded)


def ensure_dependency(module_name: str, pip_hint: Optional[str] = None):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = pip_hint or module_name
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it via `pip install {hint}` or `pip install -r requirements.txt`."
        ) from exc


def build_corpus(raw_dir: Path, tokenizer_json: Path, args) -> Path:
    base_processed = Path(args.output_dir) / "processed"
    processed = base_processed
    if processed.exists():
        suffix = 1
        while True:
            candidate = base_processed.with_name(f"{base_processed.name}_{suffix}")
            if not candidate.exists():
                processed = candidate
                break
            suffix += 1
    processed.mkdir(parents=True, exist_ok=True)
    cfg = CorpusDatasetConfig(
        input_files=[str(raw_dir / "train.jsonl"), str(raw_dir / "test.jsonl")],
        tokenizer_json=str(tokenizer_json),
        output_dir=str(processed),
        seq_len=args.seq_len,
        stride=args.stride,
        format="jsonl",
        text_key=args.text_key,
        train_ratio=0.98,
    )
    convert_dataset(cfg)
    return processed


def run_training(processed_dir: Path, args):
    for module, hint in (
        ("hydra", "hydra-core"),
        ("coolname", "coolname"),
        ("wandb", "wandb"),
        ("einops", "einops"),
    ):
        ensure_dependency(module, hint)
    if not torch.cuda.is_available():
        print("CUDA not detected; training will run on CPU (this will be slow).")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    overrides = [
        f"data_paths=['{processed_dir}']",
        f"global_batch_size={max(1, args.train_batch_size)}",
        f"epochs={args.train_epochs}",
        f"eval_interval={args.train_epochs}",
        f"min_eval_interval={max(1, args.train_epochs // 2)}",
        "arch.halt_max_steps=6",
        "arch.H_cycles=2",
        "arch.L_cycles=2",
        "arch.hidden_size=64",
        "arch.num_heads=2",
        "arch.expansion=2.0",
        "arch.puzzle_emb_ndim=0",
        "arch.reasoning_seq_len=32",
        "arch.forward_dtype=float32",
        "lr=5e-4",
        "lr_warmup_steps=5",
        "ema=false",
        "+run_name=smoke_test",
        "hydra.run.dir=.",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(config_dir=str((PACKAGE_ROOT / "config").resolve()), job_name="smoke"):
        cfg = compose(config_name="cfg_pretrain", overrides=overrides)
    run_with_config(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an Athena v2 smoke test.")
    parser.add_argument("--dataset-name", default="wikitext", help="HF dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="HF dataset config.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--output-dir", default="data_examples/smoke_test")
    parser.add_argument("--tokenizer-json", help="Optional local tokenizer.json path.")
    parser.add_argument("--hf-tokenizer-repo", default="gpt2", help="Repo to pull tokenizer.json if not provided.")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=200)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dependency("argdantic")
    raw_dir = prepare_raw_jsonl(args)
    tokenizer = fetch_tokenizer(args)
    processed_dir = build_corpus(raw_dir, tokenizer, args)
    print(f"Corpus prepared under {processed_dir}")
    if args.run_training:
        print("Starting short training run...")
        run_training(processed_dir, args)
    else:
        print("Skipping training. Rerun with --run-training to execute the loop.")


if __name__ == "__main__":
    main()
