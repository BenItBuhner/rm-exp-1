from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CastedEmbedding,
    CastedLinear,
    CosSin,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class AthenaTRMInnerCarry:
    z_high: torch.Tensor
    z_low: torch.Tensor


@dataclass
class AthenaTRMCarry:
    inner: AthenaTRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class AthenaTRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_sequence_identifiers: int

    puzzle_emb_ndim: int = 0
    hidden_size: int = 2048
    num_heads: int = 32
    expansion: float = 4.0

    H_cycles: int = 6
    L_cycles: int = 5

    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    halt_max_steps: int = 48
    halt_exploration_prob: float = 0.05
    force_min_steps: int = 2
    forward_dtype: str = "bfloat16"

    reasoning_seq_len: int = 64
    reasoning_vocab_size: int = 32000
    enable_reasoning_stream: bool = True
    post_refinement_steps: int = 3
    no_ACT_continue: bool = True
    use_gradient_checkpointing: bool = False  # Enable for memory efficiency


class AthenaTRMBlock(nn.Module):
    def __init__(self, config: AthenaTRMConfig):
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor):
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), self.norm_eps)
        return hidden_states


class AthenaTRMReasoning(nn.Module):
    def __init__(self, config: AthenaTRMConfig):
        super().__init__()
        self.layers = nn.ModuleList([AthenaTRMBlock(config) for _ in range(config.L_cycles)])

    def forward(self, hidden_states: torch.Tensor, injection: torch.Tensor, **kwargs):
        hidden_states = hidden_states + injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class AthenaTRMInner(nn.Module):
    def __init__(self, config: AthenaTRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        embed_std = 1.0 / (config.hidden_size**0.5)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, embed_std, self.forward_dtype)
        self.sequence_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_ndim else 0
        if config.puzzle_emb_ndim > 0:
            self.sequence_emb = CastedSparseEmbedding(
                config.num_sequence_identifiers, config.puzzle_emb_ndim, config.batch_size, 0.0, self.forward_dtype
            )

        total_seq = config.seq_len + self.sequence_emb_len
        if config.pos_encodings == "rope":
            self.rotary = RotaryEmbedding(config.hidden_size // config.num_heads, total_seq, config.rope_theta)
        else:
            self.pos_embed = CastedEmbedding(total_seq, config.hidden_size, embed_std, self.forward_dtype)

        self.reasoning = AthenaTRMReasoning(config)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        if config.enable_reasoning_stream:
            self.reasoning_head = CastedLinear(config.hidden_size, config.reasoning_vocab_size, bias=False)

        self.high_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size), std=1), requires_grad=False)
        self.low_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size), std=1), requires_grad=False)

    def _input_embeddings(self, inputs: torch.Tensor, identifiers: torch.Tensor):
        emb = self.embed_tokens(inputs.to(torch.int32))

        if self.sequence_emb_len > 0:
            seq_emb = self.sequence_emb(identifiers)
            pad = self.sequence_emb_len * self.config.hidden_size - seq_emb.shape[-1]
            if pad > 0:
                seq_emb = F.pad(seq_emb, (0, pad))
            seq_emb = seq_emb.view(-1, self.sequence_emb_len, self.config.hidden_size)
            emb = torch.cat((seq_emb, emb), dim=1)

        if hasattr(self, "pos_embed"):
            emb = 0.707106781 * (emb + self.pos_embed.weight[: emb.shape[1]].unsqueeze(0))

        return emb * (self.config.hidden_size**0.5)

    def empty_carry(self, batch_size: int):
        total = self.config.seq_len + self.sequence_emb_len
        dtype = self.forward_dtype
        device = self.high_init.device
        return AthenaTRMInnerCarry(
            z_high=torch.zeros(batch_size, total, self.config.hidden_size, dtype=dtype, device=device),
            z_low=torch.zeros(batch_size, total, self.config.hidden_size, dtype=dtype, device=device),
        )

    def reset_carry(self, reset: torch.Tensor, carry: AthenaTRMInnerCarry):
        device = self.high_init.device
        reset = reset.to(device)
        high = torch.where(reset.view(-1, 1, 1), self.high_init, carry.z_high)
        low = torch.where(reset.view(-1, 1, 1), self.low_init, carry.z_low)
        return AthenaTRMInnerCarry(z_high=high, z_low=low)

    def _maybe_reasoning(self, z_low: torch.Tensor):
        if not self.config.enable_reasoning_stream:
            return None
        slice_len = min(self.config.reasoning_seq_len, z_low.shape[1])
        return self.reasoning_head(z_low[:, :slice_len])

    def forward(self, carry: AthenaTRMInnerCarry, batch: Dict[str, torch.Tensor]):
        seq_info = dict(cos_sin=self.rotary() if hasattr(self, "rotary") else None)

        embedding = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_high, z_low = carry.z_high, carry.z_low
        reasoning_logits = None

        # No-grad cycles for early H iterations
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_low = self.reasoning(z_low, z_high + embedding, **seq_info)
                z_high = self.reasoning(z_high, z_low, **seq_info)

        # Final cycle with gradients (optionally checkpointed)
        if self.training and self.config.use_gradient_checkpointing:
            # Checkpoint each L-cycle iteration to save memory
            for _ in range(self.config.L_cycles):
                z_low = checkpoint(self.reasoning, z_low, z_high + embedding, **seq_info, use_reentrant=False)
            z_high = checkpoint(self.reasoning, z_high, z_low, **seq_info, use_reentrant=False)
        else:
            for _ in range(self.config.L_cycles):
                z_low = self.reasoning(z_low, z_high + embedding, **seq_info)
            z_high = self.reasoning(z_high, z_low, **seq_info)

        reasoning_logits = self._maybe_reasoning(z_low)

        new_carry = AthenaTRMInnerCarry(z_high=z_high.detach(), z_low=z_low.detach())
        posterior = z_high
        if not self.training and self.config.post_refinement_steps > 0:
            refine = posterior
            zero = torch.zeros_like(embedding)
            for _ in range(self.config.post_refinement_steps):
                refine = self.reasoning(refine, zero, **seq_info)
            posterior = refine

        logits = self.lm_head(posterior)[:, self.sequence_emb_len :]
        q_logits = self.q_head(posterior[:, 0]).to(torch.float32)
        outputs = dict(logits=logits, q_halt_logits=q_logits[..., 0], q_continue_logits=q_logits[..., 1])
        if reasoning_logits is not None:
            outputs["reasoning_logits"] = reasoning_logits

        return new_carry, outputs


class AthenaTRM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = AthenaTRMConfig(**config_dict)
        self.inner = AthenaTRMInner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        bs = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return AthenaTRMCarry(
            inner=self.inner.empty_carry(bs),
            steps=torch.zeros(bs, dtype=torch.int32, device=device),
            halted=torch.ones(bs, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: AthenaTRMCarry, batch: Dict[str, torch.Tensor]):
        new_inner = self.inner.reset_carry(carry.halted, carry.inner)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        updated_data = {}
        for k, v in carry.current_data.items():
            updated_data[k] = torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
        updated_data["iteration_targets"] = batch.get("iteration_targets")

        new_inner, outputs = self.inner(new_inner, updated_data)

        with torch.no_grad():
            new_steps = new_steps + 1
            maxed = new_steps >= self.config.halt_max_steps
            halted = maxed

            if self.training and self.config.halt_max_steps > 1:
                logits = outputs["q_halt_logits"]
                halted = halted | (logits > 0)
                if self.config.force_min_steps > 0:
                    halted = halted & (new_steps >= self.config.force_min_steps)

                if updated_data.get("iteration_targets") is not None:
                    halted = halted | (new_steps >= updated_data["iteration_targets"])

                explore_mask = (torch.rand_like(logits) < self.config.halt_exploration_prob).bool()
                random_steps = torch.randint_like(new_steps, 2, self.config.halt_max_steps + 1)
                halted = torch.where(explore_mask, new_steps >= random_steps, halted)

                if not self.config.no_ACT_continue:
                    _, future = self.inner(new_inner, updated_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(maxed, future["q_halt_logits"], torch.maximum(future["q_halt_logits"], future["q_continue_logits"]))
                    )

        return AthenaTRMCarry(new_inner, new_steps, halted, updated_data), outputs
