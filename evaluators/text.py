from typing import Dict, Optional

import torch


class PerplexityEvaluator:
    required_outputs = {"logits", "labels"}

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        self.loss_sum = 0.0
        self.token_count = 0.0

    def begin_eval(self):
        self.loss_sum = 0.0
        self.token_count = 0.0

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        labels = batch["labels"].to(torch.long)
        logits = preds["logits"]
        mask = labels != -100
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="none"
        ).view(labels.shape)
        self.loss_sum += (losses * mask).sum().item()
        self.token_count += mask.sum().item()

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None):
        if rank != 0:
            return None
        if self.token_count == 0:
            return {"perplexity": float("inf")}
        ppl = torch.exp(torch.tensor(self.loss_sum / self.token_count))
        return {"text/perplexity": ppl.item()}
