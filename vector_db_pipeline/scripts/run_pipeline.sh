#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — Quick launcher for the Vector DB pipeline
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found."
    exit 1
fi

echo "=========================================="
echo "  Vector DB Pipeline"
echo "=========================================="
echo ""

# --- Check prerequisites ---

# 1. vLLM server
echo "[1/4] Checking vLLM server..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    echo "  ✅ vLLM is running at http://localhost:8000"
else
    echo "  ⚠️  vLLM not detected at http://localhost:8000"
    echo "  Start it with:"
    echo "    uv run --python \"$VENV_PYTHON\" -m vllm.entrypoints.openai.api_server \\"
    echo "      --model Qwen/Qwen3.5-VL-9B \\"
    echo "      --port 8000"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Qdrant
echo ""
echo "[2/4] Checking Qdrant..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/collections | grep -q "200"; then
    echo "  ✅ Qdrant is running at http://localhost:6333"
else
    echo "  ⚠️  Qdrant not detected at http://localhost:6333"
    echo "  Start it with Docker:"
    echo "    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \\"
    echo "      qdrant/qdrant:latest"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. Python env
echo ""
echo "[3/4] Checking Python environment..."
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "  ❌ Python not found at $VENV_PYTHON"
    echo "  Create it first with:"
    echo "    uv venv \"$PROJECT_ROOT/.venv\" --python 3.10"
    echo "    uv pip install --python \"$PROJECT_ROOT/.venv/bin/python\" -e \"$PROJECT_ROOT[dev,streamlit]\""
    exit 1
fi
echo "  ✅ Python: $(uv run --python "$VENV_PYTHON" python --version)"

# 4. Dependencies
echo ""
echo "[4/4] Checking dependencies..."
if uv run --python "$VENV_PYTHON" python -c "import yaml, torch, qdrant_client" 2>/dev/null; then
    echo "  ✅ All dependencies installed"
else
    echo "  ❌ Dependencies are missing in $VENV_PYTHON"
    echo "  Install them with:"
    echo "    uv pip install --python \"$VENV_PYTHON\" -e \"$PROJECT_ROOT[dev,streamlit]\""
    exit 1
fi

echo ""
echo "=========================================="
echo "  Starting Pipeline"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"
uv run --python "$VENV_PYTHON" -m src.run_pipeline "$@"
