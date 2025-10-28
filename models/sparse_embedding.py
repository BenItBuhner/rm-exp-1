from typing import Union

import torch
from torch import nn
import torch.distributed as dist
from torch.optim import Optimizer

from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.weights = nn.Parameter(trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std))
        self.local_weights = nn.Parameter(torch.zeros(batch_size, embedding_dim), requires_grad=True)
        self.local_ids = nn.Parameter(torch.zeros(batch_size, dtype=torch.int32), requires_grad=False)

    def forward(self, ids: torch.Tensor):
        if not self.training:
            return self.weights[ids].to(self.dtype)

        with torch.no_grad():
            self.local_weights.copy_(self.weights[ids])
            self.local_ids.copy_(ids)
        return self.local_weights.to(self.dtype)


class CastedSparseEmbeddingSignSGD(Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, world_size: int):
        super().__init__(params, defaults=dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            world = group["world_size"]

            grads = None
            ids = None
            weights = None

            for p in group["params"]:
                if p.requires_grad:
                    grads = p.grad
                elif p.dtype == torch.int32:
                    ids = p
                else:
                    weights = p

            if grads is None or ids is None or weights is None:
                continue

            N, D = grads.shape
            gathered_grad = grads
            gathered_ids = ids

            if world > 1:
                gathered_grad = torch.empty(N * world, D, device=grads.device, dtype=grads.dtype)
                gathered_ids = torch.empty(N * world, device=ids.device, dtype=ids.dtype)
                dist.all_gather_into_tensor(gathered_grad, grads)
                dist.all_gather_into_tensor(gathered_ids, ids)

            unique_ids, inv = gathered_ids.unique(return_inverse=True)
            merged = torch.zeros(unique_ids.shape[0], D, device=grads.device, dtype=grads.dtype)
            merged.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), gathered_grad)

            slice_weights = weights[unique_ids]
            slice_weights.mul_(1 - lr * wd).add_(torch.sign(merged), alpha=-lr)
            weights[unique_ids] = slice_weights
