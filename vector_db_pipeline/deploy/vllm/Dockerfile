# ============================================================
# vLLM Server — Dockerfile for cloud deployment (GCP)
# Used by deploy/deploy_gcp.sh
# ============================================================
FROM vllm/vllm-openai:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Default model can be overridden via env var
ENV MODEL_NAME=Qwen/Qwen3.5-VL-9B
ENV MAX_MODEL_LEN=8192

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "$MODEL_NAME", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "$MAX_MODEL_LEN", \
     "--dtype", "auto"]
