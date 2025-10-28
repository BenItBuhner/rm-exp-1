from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import os
import math
import copy
import yaml

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# TPU support (optional)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    pl = None
    xmp = None

import tqdm
import wandb
import hydra
import coolname
import pydantic
from omegaconf import DictConfig
from torch.optim import AdamW

from sequence_dataset import SequenceDataset, SequenceDatasetConfig, SequenceDatasetMetadata
from utils.functions import load_model_class, get_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float
    grad_clip_norm: float = 1.0  # Gradient clipping max norm

    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False
    
    device_type: str = "auto"  # "auto", "cuda", "tpu", "cpu"


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


def detect_device(device_type: str = "auto") -> tuple[Any, str]:
    """
    Detect and return the appropriate device based on availability.
    
    Returns:
        device: torch.device or TPU device
        backend_type: "tpu", "cuda", or "cpu"
    """
    if device_type == "tpu" or (device_type == "auto" and TPU_AVAILABLE):
        if TPU_AVAILABLE:
            device = xm.xla_device()
            return device, "tpu"
        elif device_type == "tpu":
            raise RuntimeError("TPU requested but torch_xla not available")
    
    if device_type == "cuda" or (device_type == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        elif device_type == "cuda":
            raise RuntimeError("CUDA requested but not available")
    
    return torch.device("cpu"), "cpu"


def get_world_size(backend_type: str) -> int:
    """Get the number of devices based on backend."""
    if backend_type == "tpu":
        return xm.xrt_world_size()
    elif backend_type == "cuda" and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank(backend_type: str) -> int:
    """Get the device rank based on backend."""
    if backend_type == "tpu":
        return xm.get_ordinal()
    elif backend_type == "cuda" and dist.is_initialized():
        return dist.get_rank()
    return 0


def reduce_gradients(model: nn.Module, backend_type: str):
    """Reduce gradients across devices."""
    if backend_type == "tpu":
        # TPU uses xm.optimizer_step which handles gradient reduction
        pass
    elif backend_type == "cuda":
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)


def optimizer_step(optimizer: torch.optim.Optimizer, backend_type: str):
    """Step optimizer with appropriate backend."""
    if backend_type == "tpu":
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()


def mark_step(backend_type: str):
    """Mark step for TPU lazy execution."""
    if backend_type == "tpu":
        xm.mark_step()


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset_paths = config.data_paths_test if (config.data_paths_test and split == "test") else config.data_paths
    dataset = SequenceDataset(
        SequenceDatasetConfig(
            seed=config.seed,
            dataset_paths=dataset_paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    # Use multiple workers for data loading if not in distributed mode
    # Note: SequenceDataset currently doesn't support multi-worker due to mmap
    num_workers = 0  # Keep at 0 for now due to mmap limitations
    pin_memory = torch.cuda.is_available()  # Only pin memory for GPU training
    
    loader = DataLoader(
        dataset, 
        batch_size=None, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, dataset.metadata


def create_model(
    config: PretrainConfig,
    metadata: SequenceDatasetMetadata,
    rank: int,
    world_size: int,
    device: torch.device,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_sequence_identifiers=metadata.num_sequence_identifiers,
    )

    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    model = model_cls(model_cfg)
    model = loss_cls(model, **config.arch.loss.__pydantic_extra__)
    if device.type == "cuda" and "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model)
    model = model.to(device)
    print(f"[Athena] Model moved to device: {next(model.parameters()).device}")

    checkpoint_data = None
    if rank == 0 and config.load_checkpoint is not None:
        print(f"[Athena] Loading checkpoint from {config.load_checkpoint}")
        checkpoint_data = torch.load(config.load_checkpoint, map_location=device)
        # Handle both old (just state_dict) and new (dict with metadata) formats
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
            print(f"[Athena] Loaded model from step {checkpoint_data.get('step', 'unknown')}")
        else:
            model.load_state_dict(checkpoint_data, strict=False)
            print(f"[Athena] Loaded model (legacy format)")

    if world_size > 1:
        for param in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(param, src=0)

    if getattr(model.model, "inner", None) and getattr(model.model.inner, "sequence_emb", None):
        sparse_params = [
            model.model.inner.sequence_emb.local_weights,
            model.model.inner.sequence_emb.local_ids,
            model.model.inner.sequence_emb.weights,
        ]
    else:
        sparse_params = []

    optimizers = []
    lrs = []

    if sparse_params and not config.freeze_weights:
        optimizers.append(
            CastedSparseEmbeddingSignSGD(
                sparse_params,
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        )
        lrs.append(config.puzzle_emb_lr)

    if not config.freeze_weights:
        optimizers.append(
            AdamW(
                model.parameters(),
                lr=0,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
                fused=torch.cuda.is_available(),  # Use fused AdamW for better performance
            )
        )
        lrs.append(config.lr)

    # Load optimizer states if available
    if checkpoint_data is not None and "optimizer_states" in checkpoint_data:
        for opt, opt_state in zip(optimizers, checkpoint_data["optimizer_states"]):
            opt.load_state_dict(opt_state)
        print("[Athena] Loaded optimizer states from checkpoint")
    
    return model, optimizers, lrs, checkpoint_data


def cosine_schedule(step: int, *, base_lr: float, warmup: int, total: int, min_ratio: float):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def init_train_state(
    config: PretrainConfig,
    metadata: SequenceDatasetMetadata,
    rank: int,
    world_size: int,
    device: torch.device,
):
    total_steps = int(config.epochs * metadata.total_groups * metadata.mean_examples_per_group / config.global_batch_size)
    model, optimizers, lrs, checkpoint_data = create_model(config, metadata, rank, world_size, device=device)
    
    # Resume from checkpoint step if available
    start_step = 0
    if checkpoint_data is not None and "step" in checkpoint_data:
        start_step = checkpoint_data["step"]
        print(f"[Athena] Resuming training from step {start_step}")
    
    return TrainState(model=model, optimizers=optimizers, optimizer_lrs=lrs, carry=None, step=start_step, total_steps=total_steps), checkpoint_data


def train_batch(
    config: PretrainConfig,
    state: TrainState,
    batch: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    device: torch.device,
    backend_type: str = "cuda",
):
    state.step += 1
    if state.step > state.total_steps:
        return None

    batch = {k: v.to(device) for k, v in batch.items()}
    if state.carry is None:
        state.carry = state.model.initial_carry(batch)

    state.carry, loss, metrics, _, _ = state.model(carry=state.carry, batch=batch, return_keys=[])
    (loss / config.global_batch_size).backward()

    # Reduce gradients across devices (backend-specific)
    if world_size > 1:
        reduce_gradients(state.model, backend_type)

    # Gradient clipping for stability
    if config.grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), 
            max_norm=config.grad_clip_norm
        )
    else:
        grad_norm = None

    lr_this_step = None
    for optim, base_lr in zip(state.optimizers, state.optimizer_lrs):
        lr_this_step = cosine_schedule(
            state.step, base_lr=base_lr, warmup=config.lr_warmup_steps, total=state.total_steps, min_ratio=config.lr_min_ratio
        )
        for group in optim.param_groups:
            group["lr"] = lr_this_step
        optimizer_step(optim, backend_type)  # Backend-specific optimizer step
        optim.zero_grad(set_to_none=True)
    
    # Mark step for TPU lazy execution
    mark_step(backend_type)

    if not metrics:
        return None

    keys = sorted(metrics.keys())
    values = torch.stack([metrics[k] for k in keys])
    if world_size > 1:
        dist.reduce(values, dst=0)

    if rank == 0:
        reduced = {k: values[i].item() for i, k in enumerate(keys)}
        count = max(reduced.get("count", 1), 1)
        for k in list(reduced.keys()):
            if k.endswith("loss"):
                reduced[f"train/{k}"] = reduced[k] / config.global_batch_size
            else:
                reduced[f"train/{k}"] = reduced[k] / count
            del reduced[k]
        reduced["train/lr"] = lr_this_step
        if grad_norm is not None:
            reduced["train/grad_norm"] = grad_norm.item()
        return reduced
    return None


def evaluate(
    config: PretrainConfig,
    state: TrainState,
    loader: DataLoader,
    metadata: SequenceDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    device: torch.device,
):
    if loader is None:
        return None

    reduced_metrics = None
    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(getattr(evaluator, "required_outputs", []))

        carry = None
        metric_accum = None
        metric_keys: List[str] = []
        set_ids = {k: idx for idx, k in enumerate(metadata.sets)}

        for set_name, batch, global_bs in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = state.model.initial_carry(batch)

            while True:
                carry, loss, metrics, preds, finished = state.model(carry=carry, batch=batch, return_keys=return_keys)
                if finished:
                    break

            for evaluator in evaluators:
                evaluator.update_batch(batch, {**batch, **preds})

            if metrics:
                if metric_accum is None:
                    metric_keys = sorted(metrics.keys())
                    metric_accum = torch.zeros((len(set_ids), len(metric_keys)), device=device)
                metric_accum[set_ids[set_name]] += torch.stack([metrics[k] for k in metric_keys])

        if metric_accum is not None:
            if world_size > 1:
                dist.reduce(metric_accum, dst=0)
            if rank == 0:
                reduced_metrics = {}
                values = metric_accum.cpu().numpy()
                for set_name, idx in set_ids.items():
                    reduced_metrics[set_name] = {metric_keys[j]: float(values[idx, j]) for j in range(len(metric_keys))}

        for evaluator in evaluators:
            metrics = evaluator.result(None, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    import shutil

    files = [get_source_path(config.arch.name), get_source_path(config.arch.loss.name)]
    for file in files:
        if file:
            dest = os.path.join(config.checkpoint_path, os.path.basename(file))
            if os.path.abspath(file) != os.path.abspath(dest):
                shutil.copy(file, dest)
    with open(os.path.join(config.checkpoint_path, "config.yaml"), "w") as f:
        yaml.safe_dump(config.model_dump(), f)
    wandb.run.log_code(config.checkpoint_path)


def load_config(hydra_cfg: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        cfg = PretrainConfig(**hydra_cfg)
        if cfg.project_name is None:
            cfg.project_name = "Athena-v1"
        if cfg.run_name is None:
            cfg.run_name = f"{cfg.arch.name.split('@')[-1]}-{coolname.generate_slug(2)}"
        if cfg.checkpoint_path is None:
            cfg.checkpoint_path = os.path.join("checkpoints", cfg.project_name, cfg.run_name)
        objects[0] = cfg
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


def run_with_config(cfg: DictConfig, tpu_index: Optional[int] = None):
    """
    Main training function supporting CUDA GPU and TPU.
    
    Args:
        cfg: Hydra config
        tpu_index: TPU core index (for multi-TPU training)
    """
    # Get initial config to determine device type
    temp_config = PretrainConfig(**cfg)
    device, backend_type = detect_device(temp_config.device_type)
    
    # Initialize distributed training based on backend
    rank = 0
    world_size = 1
    cpu_group = None
    
    if backend_type == "tpu":
        rank = get_rank(backend_type)
        world_size = get_world_size(backend_type)
        print(f"[Athena] TPU detected. Core {rank}/{world_size}, Device: {device}")
        
    elif backend_type == "cuda":
        if "LOCAL_RANK" in os.environ:
            backend = "nccl"
            dist.init_process_group(backend=backend)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            cpu_group = dist.new_group(backend="gloo")
            print(f"[Athena] Multi-GPU training. Rank {rank}/{world_size}")
        else:
            print(f"[Athena] CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[Athena] Using CPU (no GPU/TPU available)")

    config = load_config(cfg, rank, world_size)
    torch.random.manual_seed(config.seed + rank)

    train_loader, train_meta = create_dataloader(
        config, "train", rank=rank, world_size=world_size, test_set_mode=False, epochs_per_iter=config.eval_interval or config.epochs, global_batch_size=config.global_batch_size
    )
    try:
        eval_loader, eval_meta = create_dataloader(
            config, "test", rank=rank, world_size=world_size, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size
        )
    except FileNotFoundError:
        eval_loader = None
        eval_meta = None
    
    # Wrap dataloaders for TPU
    if backend_type == "tpu":
        train_loader = pl.MpDeviceLoader(train_loader, device)
        if eval_loader is not None:
            eval_loader = pl.MpDeviceLoader(eval_loader, device)

    evaluators = []
    for evaluator in config.evaluators:
        cls = load_model_class(evaluator.name, prefix="evaluators.")
        evaluators.append(cls(data_path=config.data_paths[0], eval_metadata=eval_meta))

    state, checkpoint_data = init_train_state(config, train_meta, rank, world_size, device=device)

    progress = None
    ema = None
    if rank == 0:
        progress = tqdm.tqdm(total=state.total_steps, initial=state.step)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
        wandb.log({"num_params": sum(p.numel() for p in state.model.parameters())}, step=state.step)
        save_code_and_config(config)
    if config.ema:
        ema = EMAHelper(mu=config.ema_rate)
        if checkpoint_data is not None and "ema_shadow" in checkpoint_data:
            ema.shadow = checkpoint_data["ema_shadow"]
            print("[Athena] Loaded EMA shadow from checkpoint")
        else:
            ema.register(state.model)

    total_iters = config.epochs // (config.eval_interval or config.epochs)
    for iter_idx in range(total_iters):
        state.model.train()
        for set_name, batch, global_bs in train_loader:
            metrics = train_batch(config, state, batch, rank, world_size, device=device, backend_type=backend_type)
            if config.ema and ema is not None:
                ema.update(state.model)
            if rank == 0 and metrics is not None:
                wandb.log(metrics, step=state.step)
                progress.update(state.step - progress.n)

        if iter_idx >= config.min_eval_interval and eval_loader is not None:
            eval_state = state
            if config.ema and ema is not None:
                eval_state = copy.deepcopy(state)
                eval_state.model = ema.ema_copy(state.model)
            eval_state.model.eval()
            metrics = evaluate(
                config,
                eval_state,
                eval_loader,
                eval_meta,
                evaluators,
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
                device=device,
            )
            if rank == 0 and metrics is not None:
                wandb.log(metrics, step=state.step)
            if config.ema and eval_state is not state:
                del eval_state

            if rank == 0 and (config.checkpoint_every_eval or iter_idx == total_iters - 1):
                os.makedirs(config.checkpoint_path, exist_ok=True)
                checkpoint = {
                    "model_state_dict": state.model.state_dict(),
                    "optimizer_states": [opt.state_dict() for opt in state.optimizers],
                    "step": state.step,
                    "config": config.model_dump(),
                }
                if ema is not None:
                    checkpoint["ema_shadow"] = ema.shadow
                torch.save(checkpoint, os.path.join(config.checkpoint_path, f"step_{state.step}.pt"))
                print(f"[Athena] Checkpoint saved at step {state.step}")

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(cfg: DictConfig):
    """Main entry point that handles TPU multi-core launch if needed."""
    # Check if we should use TPU
    temp_config = PretrainConfig(**cfg)
    _, backend_type = detect_device(temp_config.device_type)
    
    if backend_type == "tpu" and TPU_AVAILABLE:
        # TPU multi-core training
        print(f"[Athena] Launching TPU training on {xm.xrt_world_size()} cores")
        xmp.spawn(run_with_config, args=(cfg,), nprocs=None)  # nprocs=None uses all available cores
    else:
        # Single-device or multi-GPU training
        run_with_config(cfg)


if __name__ == "__main__":
    launch()
