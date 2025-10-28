import math
import torch


def trunc_normal_init_(tensor: torch.Tensor, std: float):
    """Truncated normal initialization (LeCun style)."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std)
    return tensor
