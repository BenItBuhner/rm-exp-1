#!/bin/bash
# Athena v2 - TPU Test Training Script
# Quick test run with small dataset and few epochs

set -e

TPU_NAME="${TPU_NAME:-TPU}"
ZONE="${ZONE:-us-central2-b}"
DATA_PATH="${DATA_PATH:-/mnt/disks/ssd/athena_data/lang_corpus}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/disks/ssd/checkpoints/test_run}"
RUN_NAME="${RUN_NAME:-test_run}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.0001}"
WARMUP="${WARMUP:-100}"
EPOCHS="${EPOCHS:-3}"

echo "=========================================="
echo "Athena v2 - TPU Test Training"
echo "=========================================="
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Data: ${DATA_PATH}"
echo "Checkpoints: ${CHECKPOINT_PATH}"
echo "Run Name: ${RUN_NAME}"
echo "Batch Size: ${BATCH_SIZE} (small for testing)"
echo "Learning Rate: ${LR}"
echo "Warmup Steps: ${WARMUP}"
echo "Epochs: ${EPOCHS} (quick test)"
echo "=========================================="
echo ""
echo "This is a QUICK TEST with:"
echo "  - Small batch size (${BATCH_SIZE})"
echo "  - Few epochs (${EPOCHS})"
echo "  - Short warmup (${WARMUP})"
echo ""
echo "For full training, use train_tpu.sh"
echo ""

echo "Starting test training on all TPU workers..."
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
      --eval_interval 1 \
      2>&1 | tee ~/logs_test_\$(hostname).txt"

echo ""
echo "=========================================="
echo "Test training complete!"
echo "=========================================="
echo ""
echo "View logs:"
echo "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 --command='tail -100 ~/logs_test_*.txt'"
echo ""
echo "If successful, run full training with:"
echo "  ./scripts/train_tpu.sh"

