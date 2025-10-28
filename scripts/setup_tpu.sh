#!/bin/bash
# Athena v2 - TPU Setup Script
# Sets up all TPU workers with required dependencies and code

set -e

TPU_NAME="${TPU_NAME:-TPU}"
ZONE="${ZONE:-us-central2-b}"
LOCAL_CODE_DIR="${LOCAL_CODE_DIR:-./}"

echo "=========================================="
echo "Athena v2 - TPU Pod Setup"
echo "=========================================="
echo "TPU Name: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Local Code: ${LOCAL_CODE_DIR}"
echo "=========================================="
echo ""

echo "[1/4] Installing Python dependencies on all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="pip3 install -q torch torchvision torchaudio einops tqdm coolname pydantic argdantic wandb pyyaml huggingface_hub tokenizers numba sentencepiece datasets numpy torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html"

echo ""
echo "[2/4] Creating directories on all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="mkdir -p ~/rm-exp-1/config ~/rm-exp-1/models ~/rm-exp-1/utils ~/rm-exp-1/evaluators ~/rm-exp-1/scripts && mkdir -p /mnt/disks/ssd/athena_data /mnt/disks/ssd/checkpoints"

echo ""
echo "[3/4] Copying code to all workers..."
cd "${LOCAL_CODE_DIR}"

# Copy main files
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all pretrain.py testtm@${TPU_NAME}:~/rm-exp-1/
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all sequence_dataset.py testtm@${TPU_NAME}:~/rm-exp-1/

# Copy configs
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all --recurse config/ testtm@${TPU_NAME}:~/rm-exp-1/

# Copy models
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all --recurse models/ testtm@${TPU_NAME}:~/rm-exp-1/

# Copy utils
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all --recurse utils/ testtm@${TPU_NAME}:~/rm-exp-1/

# Copy evaluators
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all --recurse evaluators/ testtm@${TPU_NAME}:~/rm-exp-1/

# Copy scripts
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=all scripts/prepare_tpu_dataset.py testtm@${TPU_NAME}:~/rm-exp-1/scripts/

echo ""
echo "[4/4] Verifying installation..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="python3 -c 'import torch_xla; import torch_xla.core.xla_model as xm; print(f\"✓ TPU available: {xm.xla_device()}\")'"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare dataset: ./scripts/prepare_dataset_tpu.sh"
echo "2. Run training: ./scripts/train_tpu.sh"

