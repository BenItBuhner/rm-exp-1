#!/usr/bin/env python3
"""
Prepare a dataset for Athena v2 training on TPU.
Downloads from HuggingFace and converts to the required format.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np


def prepare_dataset(
    output_dir: str,
    dataset_name: str = "openwebtext",
    dataset_config: str = None,
    split: str = "train",
    max_length: int = 2048,
    num_examples: int = 100000,
):
    """
    Download and prepare a dataset for Athena training.
    
    Args:
        output_dir: Where to save the processed dataset
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (if needed)
        split: Which split to use
        max_length: Maximum sequence length
        num_examples: Number of examples to process
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/5] Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    print(f"[2/5] Loading tokenizer (GPT-2)")
    # Use GPT-2 tokenizer as default
    tokenizer = Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"[3/5] Processing {num_examples} examples...")
    
    # Create train and test splits (90/10)
    train_split = int(num_examples * 0.9)
    
    for split_name, start, end in [("train", 0, train_split), ("test", train_split, num_examples)]:
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\n  Processing {split_name} split ({end - start} examples)...")
        
        sequences = []
        total_tokens = 0
        
        for idx, example in enumerate(tqdm(dataset, total=end)):
            if idx < start:
                continue
            if idx >= end:
                break
            
            # Get text from example
            text = example.get('text', example.get('content', str(example)))
            
            # Tokenize
            encoding = tokenizer.encode(text)
            tokens = encoding.ids[:max_length]
            
            if len(tokens) < 10:  # Skip very short sequences
                continue
            
            sequences.append(tokens)
            total_tokens += len(tokens)
        
        # Save as binary numpy arrays (more efficient for loading)
        print(f"  Saving {len(sequences)} sequences...")
        
        # Create memmap file
        data_file = split_dir / "data.npy"
        offsets_file = split_dir / "offsets.npy"
        
        # Calculate offsets
        offsets = [0]
        for seq in sequences:
            offsets.append(offsets[-1] + len(seq))
        
        # Write all tokens
        all_tokens = []
        for seq in sequences:
            all_tokens.extend(seq)
        
        np.save(data_file, np.array(all_tokens, dtype=np.int32))
        np.save(offsets_file, np.array(offsets, dtype=np.int64))
        
        # Create metadata
        metadata = {
            "vocab_size": vocab_size,
            "num_examples": len(sequences),
            "total_tokens": total_tokens,
            "max_length": max_length,
            "mean_length": total_tokens / len(sequences) if sequences else 0,
        }
        
        with open(split_dir / "dataset.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ {split_name}: {len(sequences)} sequences, {total_tokens:,} tokens")
    
    print(f"\n[5/5] Dataset prepared at: {output_path}")
    print(f"✓ Ready for training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Athena training")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed dataset")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                       help="HuggingFace dataset name (default: tiny_shakespeare for quick test)")
    parser.add_argument("--config", type=str, default=None,
                       help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--num_examples", type=int, default=10000,
                       help="Number of examples to process")
    
    args = parser.parse_args()
    
    prepare_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        max_length=args.max_length,
        num_examples=args.num_examples,
    )

