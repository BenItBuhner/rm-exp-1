from typing import Any, Dict, Sequence, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, dim=-1, index=transformed_labels.long().unsqueeze(-1)).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    return (
        F.cross_entropy(
            logits.to(torch.float32).view(-1, logits.shape[-1]),
            labels.long().view(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view(labels.shape)
    )


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        iteration_targets = new_carry.current_data.get("iteration_targets")

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            mask = labels != IGNORE_LABEL_ID
            token_count = mask.sum(-1)
            divisor = token_count.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == token_count
            valid_metrics = new_carry.halted & (token_count > 0)

            halt_targets = seq_is_correct
            if iteration_targets is not None:
                halt_targets = halt_targets | (new_carry.steps >= iteration_targets)

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.float() / divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == halt_targets)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / divisor).sum()
        target = halt_targets.float()

        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], target, reduction="sum")
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"])

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()
        if q_continue_loss != 0:
            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached = {k: outputs[k].detach() for k in return_keys if k in outputs}
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        return new_carry, total_loss, metrics, detached, new_carry.halted.all()
