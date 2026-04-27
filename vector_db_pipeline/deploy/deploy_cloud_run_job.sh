#!/usr/bin/env bash
# =============================================================================
# deploy_cloud_run_job.sh — Build the ingestion pipeline image and deploy it
#                           as a Cloud Run Job on GCP.
#
# Usage:
#   chmod +x deploy/deploy_cloud_run_job.sh
#   ./deploy/deploy_cloud_run_job.sh          # build + deploy
#   ./deploy/deploy_cloud_run_job.sh run      # trigger a job execution
#   ./deploy/deploy_cloud_run_job.sh logs     # tail logs from the last run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Artifact Registry API enabled in your GCP project
#   - .env file filled in (GCP_PROJECT_ID, QDRANT_URL, QDRANT_API_KEY,
#     VLLM_API_BASE, GCS_URI or GCS_BUCKET)
# =============================================================================

set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="us-central1"
JOB_NAME="ingestion-job"
REPO="vector-db"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/ingestion:latest"
GCS_INPUT="${GCS_URI:-${GCS_BUCKET:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/data/processed}"

if [[ "$PROJECT_ID" == "your-project-id" ]]; then
    echo "ERROR: Set GCP_PROJECT_ID in your .env file or as an environment variable."
    exit 1
fi

if [[ -n "$GCS_INPUT" && "$GCS_INPUT" != gs://* ]]; then
    GCS_INPUT="gs://${GCS_INPUT}"
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ensure_artifact_registry() {
    if ! gcloud artifacts repositories describe "$REPO" \
        --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
        log "Creating Artifact Registry repository '$REPO'..."
        gcloud artifacts repositories create "$REPO" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID"
    fi
}

build_and_push() {
    log "Building + pushing image to $IMAGE ..."
    gcloud builds submit \
        --tag "$IMAGE" \
        --dockerfile services/ingestion/Dockerfile \
        --project "$PROJECT_ID" \
        .
    log "Image pushed."
}

deploy_job() {
    log "Deploying Cloud Run Job '$JOB_NAME'..."
    gcloud run jobs deploy "$JOB_NAME" \
        --image "$IMAGE" \
        --region "$REGION" \
        --memory 16Gi \
        --cpu 4 \
        --task-timeout 86400 \
        --max-retries 0 \
        --set-env-vars "\
QDRANT_URL=${QDRANT_URL:-},\
QDRANT_API_KEY=${QDRANT_API_KEY:-},\
VLLM_API_BASE=${VLLM_API_BASE:-},\
GCS_URI=${GCS_INPUT},\
OUTPUT_DIR=${OUTPUT_DIR}" \
        --project "$PROJECT_ID"
    log "Job deployed."
    echo ""
    echo "  Trigger a run:  gcloud run jobs execute $JOB_NAME --region=$REGION --project=$PROJECT_ID"
    echo "  Watch logs:     ./deploy/deploy_cloud_run_job.sh logs"
    echo ""
}

run_job() {
    log "Triggering job execution..."
    gcloud run jobs execute "$JOB_NAME" \
        --region "$REGION" \
        --project "$PROJECT_ID"
    log "Job started. Monitor progress in the GCP Console → Cloud Run → Jobs."
}

tail_logs() {
    log "Fetching logs for the last execution of '$JOB_NAME'..."
    gcloud logging read \
        "resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME" \
        --project "$PROJECT_ID" \
        --limit 200 \
        --format "value(textPayload)" \
        --order asc
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
COMMAND="${1:-deploy}"

case "$COMMAND" in
    deploy)
        ensure_artifact_registry
        build_and_push
        deploy_job
        ;;
    run)
        run_job
        ;;
    logs)
        tail_logs
        ;;
    *)
        echo "Usage: $0 [deploy|run|logs]"
        exit 1
        ;;
esac
