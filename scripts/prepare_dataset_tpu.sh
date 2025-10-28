#!/bin/bash
# Athena v2 - Prepare Dataset on TPU
# Downloads and prepares a HuggingFace dataset for training

set -e

TPU_NAME="${TPU_NAME:-TPU}"
ZONE="${ZONE:-us-central2-b}"
DATASET="${DATASET:-tiny_shakespeare}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/disks/ssd/athena_data/lang_corpus}"

echo "=========================================="
echo "Athena v2 - Dataset Preparation"
echo "=========================================="
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Dataset: ${DATASET}"
echo "Examples: ${NUM_EXAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

echo "[1/3] Preparing dataset on worker 0..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="python3 ~/rm-exp-1/scripts/prepare_tpu_dataset.py \
    --output_dir ${OUTPUT_DIR} \
    --dataset ${DATASET} \
    --num_examples ${NUM_EXAMPLES}"

echo ""
echo "[2/3] Creating archive..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="cd /mnt/disks/ssd/athena_data && tar -czf /tmp/dataset.tar.gz lang_corpus"

echo ""
echo "[3/3] Copying dataset to all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="mkdir -p /mnt/disks/ssd/athena_data && cd /mnt/disks/ssd/athena_data && tar -xzf /tmp/dataset.tar.gz 2>/dev/null || echo 'Already extracted on this worker'"

echo ""
echo "=========================================="
echo "✓ Dataset ready on all workers!"
echo "=========================================="
echo ""
echo "Dataset location: ${OUTPUT_DIR}"
echo ""
echo "Verify with:"
echo "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 --command='ls -lh ${OUTPUT_DIR}/train/'"
echo ""
echo "Next step: Run training with ./scripts/train_tpu.sh"

