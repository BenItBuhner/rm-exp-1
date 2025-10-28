#!/bin/bash
# Athena v2 - TPU Pod Setup Script
# Ensures all workers have required dependencies

set -e

TPU_NAME="${TPU_NAME:-TPU}"
ZONE="${ZONE:-us-central2-b}"

echo "=========================================="
echo "Athena v2 - TPU Pod Setup"
echo "=========================================="
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "=========================================="
echo ""

# Setup command to run on all workers
SETUP_CMD="
set -e
echo '===== Worker: \$(hostname) ====='
echo 'Installing dependencies...'

# Install Python packages
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip3 install einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core
pip3 install huggingface_hub tokenizers numba sentencepiece datasets

echo 'Dependencies installed on \$(hostname)'
echo ''
"

echo "Running setup on all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="${SETUP_CMD}"

echo ""
echo "=========================================="
echo "Setup complete on all workers!"
echo "=========================================="

