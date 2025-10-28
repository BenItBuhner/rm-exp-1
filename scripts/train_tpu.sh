#!/bin/bash
# Athena v2 - TPU Training Script
# Full training run on TPU pod

set -e

TPU_NAME="${TPU_NAME:-TPU}"
ZONE="${ZONE:-us-central2-b}"
DATA_PATH="${DATA_PATH:-/mnt/disks/ssd/athena_data/lang_corpus}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/disks/ssd/checkpoints/athena_v4_32}"
RUN_NAME="${RUN_NAME:-athena_v4_32}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-0.00064}"
WARMUP="${WARMUP:-8000}"
EPOCHS="${EPOCHS:-100000}"

echo "=========================================="
echo "Athena v2 - TPU Training"
echo "=========================================="
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Data: ${DATA_PATH}"
echo "Checkpoints: ${CHECKPOINT_PATH}"
echo "Run Name: ${RUN_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Warmup Steps: ${WARMUP}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="
echo ""

echo "Starting training on all TPU workers..."
echo "Logs will be saved to ~/logs_<worker>.txt on each worker"
echo ""

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="cd ~/rm-exp-1 && \
    PJRT_DEVICE=TPU python3 pretrain.py \
      --config ~/rm-exp-1/config/cfg_pretrain.yaml \
      --device_type tpu \
      --data_paths ${DATA_PATH} \
      --checkpoint_path ${CHECKPOINT_PATH} \
      --run_name ${RUN_NAME} \
      --global_batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --lr_warmup_steps ${WARMUP} \
      --epochs ${EPOCHS} \
      2>&1 | tee ~/logs_\$(hostname).txt"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "View logs:"
echo "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 --command='tail -100 ~/logs_*.txt'"
echo ""
echo "Download checkpoints:"
echo "  gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=0 --recurse testtm@${TPU_NAME}:${CHECKPOINT_PATH} ./"

