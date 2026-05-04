# Korean Math Exam — Vector DB Pipeline

A local-first retrieval pipeline for Korean high school math exam images.
Each PNG is analyzed by a vision-language model, embedded with a text encoder,
and stored in a vector database for semantic search.

Designed to be **simple, clean, and fully offline** — no cloud APIs required after setup.

---

## How It Works

```
477 Korean math exam PNGs
          |
          v
  [ Document Discovery ]
    png/*.png  →  flat list of documents
          |
          v
  [ VLM Parser — Ollama ]
    Qwen2.5-VL:7b (GGUF, local)
    One call per image  →  structured JSON
    {
      "text":        "full transcription (Korean + LaTeX)",
      "concepts":    ["이차방정식", "판별식"],
      "description": "수식과 그래프가 포함된 문제"
    }
          |
          v
  [ Text Embedder — HuggingFace ]
    Qwen3-Embedding-0.6B (local, INT8 on CUDA)
    embed(text + description)  →  1024-dim vector
          |
          v
  [ Qdrant — local Docker ]
    Single collection: exam_questions
    INT8 scalar quantization
    Payload: document_id, text, concepts, description
          |
          v
  [ Query Agent ]
    Natural language  →  VLM parse  →  embed  →  search  →  answer
```

---

## Tech Stack

| Component | Tool | Notes |
|---|---|---|
| VLM inference | [Ollama](https://ollama.com) + Qwen2.5-VL:7b | GGUF, runs locally, OpenAI-compatible API |
| Text embedding | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace transformers, INT8 on CUDA |
| Vector database | [Qdrant](https://qdrant.tech) | Local Docker, INT8 scalar quantization |
| Language | Python 3.10+ | |
| Quality | pytest, Ruff, pre-commit | 48 tests, all mocked |

**Why two backends?**
Ollama handles vision inference (GGUF quantization makes 7B models viable on consumer GPUs).
HuggingFace transformers handles embedding (`Qwen3-VL-Embedding` isn't in Ollama yet, and INT8 via `bitsandbytes` gives fine-grained memory control).

---

## Project Layout

```
Vector_DB/
├── README.md
├── png/                              # 477 Korean math exam PNGs
└── vector_db_pipeline/
    ├── config/
    │   └── settings.yaml             # All config — VLM, embedder, Qdrant, prompts
    ├── src/
    │   ├── run_pipeline.py           # Orchestrator — discovery → parse → embed → store
    │   ├── document_parser.py        # Pure VLM parser (one call per PNG)
    │   ├── vlm_client.py             # Ollama wrapper (OpenAI-compatible)
    │   ├── vlm_generate.py           # System prompt + base64 encode
    │   ├── embed_text.py             # Qwen3-Embedding with INT8 quantization
    │   ├── qdrant_store.py           # Collection schema, insert, search
    │   ├── query_agent.py            # NL query → VLM parse → embed → search → answer
    │   ├── data_loader.py            # PNG discovery (flat mode)
    │   ├── health_check.py           # Ollama readiness check
    │   ├── artifacts.py              # Per-doc result.json + manifest (resumable runs)
    │   └── parse_result.py           # QuestionResult dataclass
    └── tests/                        # 48 mocked unit tests
```

---

## Quick Start

### 1. Install

```bash
cd vector_db_pipeline
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

### 2. Start local services

```bash
# Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Ollama — pull the VLM (downloads ~4GB GGUF once)
ollama pull qwen2.5-vl:7b
ollama serve
```

### 3. Configure

```bash
cp .env.example .env
# No changes needed for local setup — defaults point to localhost
```

All pipeline settings live in `config/settings.yaml`. Key values:

```yaml
vlm:
  api_base: "http://localhost:11434/v1"
  model_name: "qwen2.5-vl:7b"

embed_text:
  model_name: "Qwen/Qwen3-Embedding-0.6B"
  quantization: "int8"   # bitsandbytes INT8 on CUDA; auto-fallback on MPS/CPU

qdrant:
  url: "http://localhost:6333"
  collection_name: "exam_questions"
  text_dim: 1024
```

### 4. Run ingestion

```bash
# Dry run — verify Ollama health + count documents
python -m src.run_pipeline --dry-run

# Smoke test — process 3 PNGs end-to-end
python -m src.run_pipeline --limit 3

# Full run — all 477 PNGs
python -m src.run_pipeline
```

Progress is tracked in `data/artifacts/{document_id}/manifest.json`.
Re-running skips already-indexed documents automatically.

### 5. Query

```python
import yaml
from openai import OpenAI
from src.vlm_generate import init_vlm_client
from src.vlm_client import VlmClient
from src.embed_text import TextEmbedder
from src.qdrant_store import QdrantStore
from src.query_agent import QueryAgent

config = yaml.safe_load(open("config/settings.yaml"))

raw_client = init_vlm_client("http://localhost:11434/v1")
vlm        = VlmClient(raw_client, "qwen2.5-vl:7b")
embedder   = TextEmbedder()
store      = QdrantStore()
agent      = QueryAgent(raw_client, "qwen2.5-vl:7b", embedder, store, config)

results = agent.query("이차방정식 관련 어려운 문제")
answer  = agent.answer("이차방정식 관련 어려운 문제", results)
```

---

## Data Model

Each indexed document is one exam question:

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | PNG filename stem |
| `text` | `str` | Full transcription — Korean text + LaTeX math |
| `concepts` | `list[str]` | Mathematical concepts required (Korean terms) |
| `description` | `str` | Natural language description of the image |

The Qdrant collection stores a single `text_vector` (1024-dim, Cosine) per document
with INT8 scalar quantization, reducing index size ~4x with negligible recall loss.

---

## Artifacts

Each processed document writes two files under `data/artifacts/{document_id}/`:

```
result.json      # QuestionResult fields (text, concepts, description)
manifest.json    # Status: "stored" | "failed" (enables safe resume)
```

---

## Quality

```bash
# Linting
ruff check src tests

# Tests (no external services required — everything mocked)
pytest -v
```

---

## Why This Project

Korean math exam images are a genuinely hard retrieval problem:

- Dense mathematical notation (LaTeX, handwritten symbols)
- Korean text mixed with formulas
- Diagrams and graphs that carry semantic meaning
- No clean text layer — everything is image-only

This pipeline shows a practical approach: let a vision-language model do the heavy
lifting (transcription + concept extraction), then embed the resulting text for
fast approximate search. The architecture is intentionally local-first — no tokens
sent to third-party APIs, no cloud dependencies after the initial model download.
