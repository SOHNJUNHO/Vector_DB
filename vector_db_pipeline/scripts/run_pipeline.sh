#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — Quick launcher for the Vector DB pipeline
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
ENV_FILE="$PROJECT_ROOT/.env"

# Load .env if present
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

RESOLVED_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
RESOLVED_QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

echo "=========================================="
echo "  Vector DB Pipeline"
echo "=========================================="
echo ""

# --- 1. VLM endpoint ---
echo "[1/4] Checking VLM endpoint..."
if echo "$RESOLVED_API_BASE" | grep -q "localhost"; then
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
        echo "  ✅ Local vLLM running at http://localhost:8000"
    else
        echo "  ⚠️  Local vLLM not detected at http://localhost:8000"
        echo "  Start it with:"
        echo "    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi
else
    if [[ -n "$FIREWORKS_API_KEY" ]]; then
        echo "  ✅ Hosted API: $RESOLVED_API_BASE"
    else
        echo "  ⚠️  FIREWORKS_API_KEY is not set — add it to .env"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi
fi

# --- 2. Qdrant ---
echo ""
echo "[2/4] Checking Qdrant..."
if echo "$RESOLVED_QDRANT_URL" | grep -q "localhost"; then
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/collections | grep -q "200"; then
        echo "  ✅ Qdrant running at http://localhost:6333"
    else
        echo "  ⚠️  Qdrant not detected at http://localhost:6333"
        echo "  Start it with Docker:"
        echo "    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi
else
    if [[ -n "$QDRANT_API_KEY" ]]; then
        echo "  ✅ Qdrant Cloud: $RESOLVED_QDRANT_URL"
    else
        echo "  ✅ Remote Qdrant: $RESOLVED_QDRANT_URL"
    fi
fi

# --- 3. Python env ---
echo ""
echo "[3/4] Checking Python environment..."
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "  ❌ Python not found at $VENV_PYTHON"
    echo "  Create it first:"
    echo "    uv venv \"$PROJECT_ROOT/.venv\" --python 3.10"
    echo "    uv pip install --python \"$PROJECT_ROOT/.venv/bin/python\" -e \"$PROJECT_ROOT[dev,streamlit]\""
    exit 1
fi
echo "  ✅ Python: $(uv run --python "$VENV_PYTHON" python --version)"

# --- 4. Dependencies ---
echo ""
echo "[4/4] Checking dependencies..."
if uv run --python "$VENV_PYTHON" python -c "import yaml, qdrant_client" 2>/dev/null; then
    echo "  ✅ Core dependencies present"
else
    echo "  ❌ Dependencies missing in $VENV_PYTHON"
    echo "  Install: uv pip install --python \"$VENV_PYTHON\" -e \"$PROJECT_ROOT[dev,streamlit]\""
    exit 1
fi

echo ""
echo "=========================================="
echo "  Starting Pipeline"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"
uv run --python "$VENV_PYTHON" -m src.run_pipeline "$@"
