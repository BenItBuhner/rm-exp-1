#!/bin/bash
# Athena v2 - TPU Pod Training Script
# For multi-worker TPU pods (e.g., v4-32 with 4 workers × 8 cores)

set -e

# TPU pod training command with correct Hydra syntax
# Use this from your local machine via gcloud

TPU_NAME="${TPU_NAME:-TPU}"
ZONE="${ZONE:-us-central2-b}"
PYTHON="${PYTHON:-python3}"

# Build the training command
TRAIN_CMD="PJRT_DEVICE=TPU ${PYTHON} ~/rm-exp-1/pretrain.py \
  device_type=tpu \
  data_paths='[/mnt/disks/ssd/athena_data/lang_corpus]' \
  +checkpoint_path=/mnt/disks/ssd/checkpoints/athena_v4_32 \
  +run_name=athena_v4_32_local \
  global_batch_size=2048 \
  lr=6.4e-4 \
  lr_warmup_steps=8000 \
  epochs=3 \
  2>&1 | tee ~/logs_\$(hostname).txt"

echo "=========================================="
echo "Athena v2 - TPU Pod Training"
echo "=========================================="
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Workers: Running on all workers"
echo "=========================================="
echo ""
echo "Training command:"
echo "${TRAIN_CMD}"
echo ""
echo "=========================================="

# Run on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="${TRAIN_CMD}"

