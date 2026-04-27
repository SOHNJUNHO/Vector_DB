#!/usr/bin/env bash
# =============================================================================
# deploy_gcp.sh — Deploy vLLM (Qwen3.5-VL-9B) on GCP with a persistent model cache
#
# Usage:
#   chmod +x deploy/deploy_gcp.sh
#   ./deploy/deploy_gcp.sh          # create VM + start vLLM
#   ./deploy/deploy_gcp.sh stop     # stop the VM
#   ./deploy/deploy_gcp.sh delete   # delete VM (keeps model disk)
#   ./deploy/deploy_gcp.sh ssh      # open SSH session
#
# Prerequisites:
#   gcloud CLI installed and authenticated (gcloud auth login)
#   GPU quota in your project for nvidia-l4 in the chosen region
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="us-central1"
ZONE="us-central1-a"

INSTANCE_NAME="vllm-qwen"
MACHINE_TYPE="g2-standard-8"          # 32 vCPU, 128 GB RAM — pairs with L4 GPU
GPU_TYPE="nvidia-l4"                   # 24 GB VRAM — fits Qwen3.5-VL-9B in fp16
GPU_COUNT=1

DISK_NAME="model-cache"                # persistent disk for HuggingFace model weights
DISK_SIZE="100GB"
DISK_MOUNT="/mnt/models"

MODEL_NAME="Qwen/Qwen3.5-VL-9B"
MAX_MODEL_LEN=8192
VLLM_PORT=8000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date +%H:%M:%S)] $*"; }

check_gcloud() {
    if ! command -v gcloud &>/dev/null; then
        echo "gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    if [[ "$PROJECT_ID" == "your-project-id" ]]; then
        echo "Set GCP_PROJECT_ID env var or edit PROJECT_ID in this script."
        exit 1
    fi
}

create_disk() {
    if gcloud compute disks describe "$DISK_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
        log "Persistent disk '$DISK_NAME' already exists — reusing."
    else
        log "Creating persistent disk '$DISK_NAME' (${DISK_SIZE})..."
        gcloud compute disks create "$DISK_NAME" \
            --zone="$ZONE" \
            --size="$DISK_SIZE" \
            --type="pd-balanced" \
            --project="$PROJECT_ID"
        log "Disk created."
    fi
}

create_instance() {
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
        log "Instance '$INSTANCE_NAME' already exists — starting if stopped..."
        gcloud compute instances start "$INSTANCE_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID"
    else
        log "Creating VM '$INSTANCE_NAME' (${MACHINE_TYPE} + ${GPU_COUNT}x ${GPU_TYPE})..."
        gcloud compute instances create "$INSTANCE_NAME" \
            --zone="$ZONE" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
            --maintenance-policy=TERMINATE \
            --restart-on-failure \
            --image-family=common-cu121 \
            --image-project=deeplearning-platform-release \
            --boot-disk-size=50GB \
            --boot-disk-type=pd-balanced \
            --disk="name=${DISK_NAME},mode=rw,auto-delete=no" \
            --tags=vllm-server \
            --project="$PROJECT_ID"
        log "VM created."
    fi
}

open_firewall() {
    RULE_NAME="allow-vllm"
    if gcloud compute firewall-rules describe "$RULE_NAME" --project="$PROJECT_ID" &>/dev/null; then
        log "Firewall rule '$RULE_NAME' already exists."
    else
        log "Creating firewall rule to allow port ${VLLM_PORT}..."
        gcloud compute firewall-rules create "$RULE_NAME" \
            --direction=INGRESS \
            --priority=1000 \
            --network=default \
            --action=ALLOW \
            --rules="tcp:${VLLM_PORT}" \
            --source-ranges=0.0.0.0/0 \
            --target-tags=vllm-server \
            --project="$PROJECT_ID"
    fi
}

wait_for_ssh() {
    log "Waiting for SSH to become available..."
    for i in $(seq 1 20); do
        if gcloud compute ssh "$INSTANCE_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" \
            --command="echo ok" --quiet 2>/dev/null; then
            log "SSH ready."
            return
        fi
        sleep 10
    done
    echo "SSH did not become available in time."
    exit 1
}

setup_vm() {
    log "Setting up VM (GPU drivers, Docker, model cache mount)..."
    gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" \
        --command="
set -e

# Mount persistent disk for model cache (skip if already mounted)
if ! mountpoint -q '${DISK_MOUNT}'; then
    DEVICE=\$(ls /dev/disk/by-id/google-${DISK_NAME} 2>/dev/null || echo '')
    if [ -n \"\$DEVICE\" ]; then
        sudo mkdir -p '${DISK_MOUNT}'
        # Format only if no filesystem present
        if ! sudo blkid \"\$DEVICE\" | grep -q TYPE; then
            sudo mkfs.ext4 -F \"\$DEVICE\"
        fi
        sudo mount \"\$DEVICE\" '${DISK_MOUNT}'
        sudo chmod 777 '${DISK_MOUNT}'
        # Persist across reboots
        echo \"\$DEVICE ${DISK_MOUNT} ext4 defaults 0 2\" | sudo tee -a /etc/fstab
    fi
fi

# Install Docker if missing
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker \$USER
    newgrp docker
fi

# Install NVIDIA Container Toolkit if missing
if ! docker info 2>/dev/null | grep -q 'Runtimes.*nvidia'; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -q
    sudo apt-get install -y -q nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

echo 'VM setup complete.'
"
}

start_vllm() {
    log "Starting vLLM server..."
    HF_TOKEN="${HF_TOKEN:-}"
    gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" \
        --command="
docker rm -f vllm 2>/dev/null || true

docker run -d \
    --name vllm \
    --runtime nvidia \
    --gpus all \
    -p ${VLLM_PORT}:${VLLM_PORT} \
    -e HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}' \
    -e HF_HOME='${DISK_MOUNT}' \
    -v '${DISK_MOUNT}:${DISK_MOUNT}' \
    vllm/vllm-openai:latest \
    --model '${MODEL_NAME}' \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype auto

echo 'vLLM container started. Model download may take 10-20 min on first run.'
echo 'Follow logs with:  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command=\"docker logs -f vllm\"'
"
    EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    log "Done."
    echo ""
    echo "  vLLM endpoint: http://${EXTERNAL_IP}:${VLLM_PORT}/v1"
    echo "  Set in .env:   VLLM_API_BASE=http://${EXTERNAL_IP}:${VLLM_PORT}/v1"
    echo ""
    echo "  Monitor:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='docker logs -f vllm'"
    echo "  Health:   curl http://${EXTERNAL_IP}:${VLLM_PORT}/health"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
COMMAND="${1:-deploy}"

case "$COMMAND" in
    deploy)
        check_gcloud
        create_disk
        create_instance
        open_firewall
        wait_for_ssh
        setup_vm
        start_vllm
        ;;
    stop)
        log "Stopping instance '$INSTANCE_NAME'..."
        gcloud compute instances stop "$INSTANCE_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID"
        log "Stopped. Model cache preserved on disk '$DISK_NAME'."
        ;;
    delete)
        log "Deleting instance '$INSTANCE_NAME' (model disk '$DISK_NAME' kept)..."
        gcloud compute instances delete "$INSTANCE_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet
        log "Instance deleted. Re-run without arguments to recreate."
        ;;
    ssh)
        gcloud compute ssh "$INSTANCE_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID"
        ;;
    *)
        echo "Usage: $0 [deploy|stop|delete|ssh]"
        exit 1
        ;;
esac
