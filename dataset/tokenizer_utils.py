from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from tokenizers import Tokenizer


@dataclass
class TextTokenizer:
    tokenizer: Tokenizer
    pad_id: int
    eos_id: int

    @classmethod
    def from_file(cls, path: str):
        tokenizer = Tokenizer.from_file(path)
        pad_id = tokenizer.token_to_id("<|pad|>")
        if pad_id is None:
            pad_id = tokenizer.get_vocab_size()
            tokenizer.add_tokens(["<|pad|>"])
        eos_id = tokenizer.token_to_id("<|eos|>")
        if eos_id is None:
            eos_id = tokenizer.get_vocab_size()
            tokenizer.add_tokens(["<|eos|>"])
        return cls(tokenizer=tokenizer, pad_id=pad_id, eos_id=eos_id)

    def encode(self, text: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(text).ids, dtype=np.int32)


def format_prompt(example: dict, template: Callable[[dict], str]) -> str:
    return template(example)
