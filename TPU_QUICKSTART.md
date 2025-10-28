# Athena v2 - TPU Quick Start Guide

**Get training on TPU in 3 simple steps!**

---

## Prerequisites

- Google Cloud TPU pod allocated (v4-32 recommended)
- `gcloud` CLI configured
- This code repository on your local machine

---

## Step 1: Setup TPU Workers (One-Time)

This installs all dependencies and copies code to all TPU workers.

```bash
cd "Athena 1 Recursive Preparing/v2"
bash scripts/setup_tpu.sh
```

**What it does:**
- Installs Python packages (torch, torch_xla, datasets, etc.)
- Creates directories on all workers
- Copies your code to all workers
- Verifies TPU is accessible

**Time:** ~5 minutes

---

## Step 2: Prepare Dataset

This downloads a HuggingFace dataset and prepares it for training.

### Option A: Quick Test (Tiny Shakespeare - 1K examples)
```bash
bash scripts/prepare_dataset_tpu.sh
```

### Option B: Larger Dataset
```bash
DATASET=wikitext NUM_EXAMPLES=100000 bash scripts/prepare_dataset_tpu.sh
```

### Option C: Custom Dataset
```bash
DATASET=openwebtext NUM_EXAMPLES=500000 bash scripts/prepare_dataset_tpu.sh
```

**What it does:**
- Downloads dataset from HuggingFace
- Tokenizes with GPT-2 tokenizer
- Creates train/test splits (90/10)
- Copies to all workers

**Time:** ~2-10 minutes depending on size

---

## Step 3: Train!

### Option A: Quick Test (3 epochs, small batch)
```bash
bash scripts/test_train_tpu.sh
```

### Option B: Full Training
```bash
bash scripts/train_tpu.sh
```

### Option C: Custom Training
```bash
BATCH_SIZE=4096 LR=0.0008 EPOCHS=50000 bash scripts/train_tpu.sh
```

**What it does:**
- Runs training on all TPU cores simultaneously
- Saves checkpoints to `/mnt/disks/ssd/checkpoints/`
- Logs to `~/logs_*.txt` on each worker
- Uses WandB for experiment tracking

**Time:** Depends on your configuration

---

## Configuration Variables

All scripts support environment variables for easy customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `TPU_NAME` | `TPU` | Your TPU pod name |
| `ZONE` | `us-central2-b` | GCP zone |
| `DATASET` | `tiny_shakespeare` | HuggingFace dataset |
| `NUM_EXAMPLES` | `1000` | Number of examples |
| `BATCH_SIZE` | `2048` | Global batch size |
| `LR` | `0.00064` | Learning rate |
| `WARMUP` | `8000` | Warmup steps |
| `EPOCHS` | `100000` | Training epochs |
| `RUN_NAME` | `athena_v4_32` | Experiment name |

### Example:
```bash
TPU_NAME=my-tpu ZONE=us-central1-a BATCH_SIZE=4096 bash scripts/train_tpu.sh
```

---

## Monitoring Training

### View Logs
```bash
gcloud compute tpus tpu-vm ssh TPU --zone=us-central2-b --worker=0 --command="tail -f ~/logs_*.txt"
```

### Check WandB
Training metrics are automatically logged to [wandb.ai](https://wandb.ai)

### Download Checkpoints
```bash
gcloud compute tpus tpu-vm scp --zone=us-central2-b --worker=0 --recurse \
  testtm@TPU:/mnt/disks/ssd/checkpoints/athena_v4_32 ./
```

---

## Troubleshooting

### "No such file or directory: config/cfg_pretrain.yaml"
Run setup again: `bash scripts/setup_tpu.sh`

### "dataset.json not found"
Run dataset preparation: `bash scripts/prepare_dataset_tpu.sh`

### "CUDA out of memory" (on TPU?)
This shouldn't happen. If it does, the code is trying to use GPU. Check that `PJRT_DEVICE=TPU` is set.

### "World size: 1" instead of 32
This is expected with PJRT - each worker reports world_size=1 but they communicate automatically.

### Training stuck or slow
- Check logs: `gcloud compute tpus tpu-vm ssh TPU --zone=us-central2-b --worker=all --command="nvidia-smi"` (just kidding, use `top`)
- Verify all workers are running: check logs on all workers
- Increase batch size (TPUs like bigger batches)

---

## Complete Example Workflow

```bash
# Setup (one time)
cd "Athena 1 Recursive Preparing/v2"
bash scripts/setup_tpu.sh

# Prepare dataset
bash scripts/prepare_dataset_tpu.sh

# Quick test (verify everything works)
bash scripts/test_train_tpu.sh

# If test passes, run full training
EPOCHS=50000 bash scripts/train_tpu.sh

# Monitor
gcloud compute tpus tpu-vm ssh TPU --zone=us-central2-b --worker=0 --command="tail -f ~/logs_*.txt"
```

---

## Dataset Options

### Pre-configured Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| `tiny_shakespeare` | ~1MB | Shakespeare plays (quick test) |
| `wikitext` | ~100MB | Wikipedia articles |
| `openwebtext` | ~40GB | Web text corpus |
| `c4` | ~300GB | Colossal Clean Crawled Corpus |
| `pile` | ~800GB | The Pile dataset |

### Usage
```bash
# Tiny test
DATASET=tiny_shakespeare NUM_EXAMPLES=1000 bash scripts/prepare_dataset_tpu.sh

# Medium
DATASET=wikitext NUM_EXAMPLES=100000 bash scripts/prepare_dataset_tpu.sh

# Large
DATASET=openwebtext NUM_EXAMPLES=1000000 bash scripts/prepare_dataset_tpu.sh
```

---

## Advanced Configuration

### Modify Training Parameters

Edit the command directly in `scripts/train_tpu.sh` or set environment variables:

```bash
# High learning rate, long warmup
LR=0.001 WARMUP=16000 bash scripts/train_tpu.sh

# Huge batch size (for v4-128 or larger)
BATCH_SIZE=8192 bash scripts/train_tpu.sh

# Short run for testing
EPOCHS=100 bash scripts/train_tpu.sh
```

### Modify Model Architecture

Edit `config/arch/athena_trm_large.yaml` on your local machine, then:

```bash
# Re-copy config to TPU
gcloud compute tpus tpu-vm scp --zone=us-central2-b --worker=all \
  config/arch/athena_trm_large.yaml testtm@TPU:~/rm-exp-1/config/arch/

# Run training
bash scripts/train_tpu.sh
```

---

## Performance Tips

1. **Use larger batch sizes** - TPUs perform best with batch sizes ≥ 1024
2. **Enable gradient checkpointing** - Set `use_gradient_checkpointing: true` in arch config
3. **Use bfloat16** - Already enabled by default on TPU
4. **Increase sequence length** - TPUs can handle longer sequences efficiently
5. **Profile your code** - Use `torch_xla.debug.metrics.metrics_report()` to find bottlenecks

---

## Cost Optimization

### v4-32 (4 workers × 8 cores)
- **Cost:** ~$4-8/hour
- **Recommended for:** Development, medium-scale training
- **Batch size:** 1024-2048

### v4-128 (16 workers × 8 cores)
- **Cost:** ~$16-32/hour
- **Recommended for:** Large-scale training
- **Batch size:** 4096-8192

### Tips
- Use preemptible TPUs (70% cheaper, may be interrupted)
- Stop TPU when not training
- Use smaller datasets for development
- Run quick tests before long training runs

---

## Quick Reference

```bash
# Setup
bash scripts/setup_tpu.sh

# Prepare data
bash scripts/prepare_dataset_tpu.sh

# Test train
bash scripts/test_train_tpu.sh

# Full train
bash scripts/train_tpu.sh

# View logs
gcloud compute tpus tpu-vm ssh TPU --zone=us-central2-b --worker=0 --command="tail -f ~/logs_*.txt"

# Download checkpoints
gcloud compute tpus tpu-vm scp --zone=us-central2-b --worker=0 --recurse testtm@TPU:/mnt/disks/ssd/checkpoints/athena_v4_32 ./
```

---

**That's it! You're ready to train on TPU! 🚀**

