# Multimodal Vector DB Pipeline

Production-minded ingestion and retrieval pipeline for scanned Korean math exam
questions. The system converts image-heavy documents into structured markdown,
generates text and visual embeddings, stores them in Qdrant, and exposes hybrid
semantic search through a Streamlit demo.

The project is designed around a practical document-AI pattern: use Docling for
layout-aware extraction, call a Qwen VLM only where visual understanding is
needed, and keep a pure-VLM backend available for fallback and benchmarking.

## Highlights

- **Hybrid document parsing**: Docling extracts structure, formulas, tables, and
  figure crops; Qwen VLM enriches diagrams, low-quality pages, metadata, and
  section boundaries.
- **Pure VLM backend**: Qwen-only page transcription can be enabled for fallback
  or parser quality benchmarks.
- **Multimodal retrieval**: Qwen text embeddings and Qwen VL embeddings are
  indexed as named vectors in Qdrant.
- **Reproducible ingestion**: deterministic IDs, per-document artifacts, and
  manifest status prevent accidental duplicate or partial indexing.
- **Operational shape**: batch processing, health checks, payload indexes,
  config-driven backends, tests, linting, and a small search UI.

## Architecture

```text
PNG pages or page folders
        |
        v
Document discovery
        |
        v
Parser backend
  - docling_vlm: Docling + selective Qwen VLM
  - pure_vlm: Qwen VLM direct transcription
        |
        v
Normalized parsed document
  - pages
  - chunks
  - figures
  - metadata
  - artifact paths
        |
        v
Embeddings
  - text_vector: Qwen3-Embedding
  - visual_vector: Qwen3-VL-Embedding
        |
        v
Qdrant
  - exam_questions
  - exam_questions_figures
        |
        v
Hybrid search + Streamlit demo
```

## Tech Stack

| Area | Tooling |
| --- | --- |
| Document parsing | Docling, Qwen VLM |
| VLM serving | Fireworks AI or local OpenAI-compatible vLLM |
| Text embedding | `Qwen/Qwen3-Embedding-0.6B` |
| Visual embedding | `Qwen/Qwen3-VL-Embedding-2B` |
| Vector database | Qdrant named vectors |
| App demo | Streamlit |
| Quality gates | pytest, Ruff, pre-commit |

## Repository Layout

```text
vector_db_pipeline/
├── config/settings.yaml       # Parser, model, path, and Qdrant config
├── scripts/run_pipeline.sh    # Convenience runner with checks
├── services/streamlit/app.py  # Search demo
├── src/
│   ├── run_pipeline.py        # Ingestion orchestration
│   ├── parser_backends.py     # docling_vlm and pure_vlm backends
│   ├── document_parser.py     # Docling-first parser implementation
│   ├── vlm_client.py          # Shared VLM task wrapper
│   ├── artifacts.py           # Per-document artifacts and manifests
│   ├── parse_result.py        # Normalized parser models
│   ├── embed_text.py          # Text embeddings
│   ├── embed_visual.py        # Visual/text multimodal embeddings
│   ├── qdrant_store.py        # Qdrant schema, indexing, retrieval
│   └── query_agent.py         # Query parsing and answer generation
└── tests/
```

## Quick Start

### 1. Install

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e ".[dev,streamlit]"
source .venv/bin/activate
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Required for hosted inference and Qdrant Cloud:

```text
FIREWORKS_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

For local VLM serving, set:

```text
VLLM_API_BASE=http://localhost:8000/v1
```

For local Qdrant:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### 3. Add Input Data

Flat mode, where each PNG is one document:

```text
png/
  question_001.png
  question_002.png
```

Document mode, where each folder is a multi-page document:

```text
png/
  exam_2024/
    page_001.png
    page_002.png
```

### 4. Run Ingestion

```bash
python -m src.run_pipeline --dry-run
python -m src.run_pipeline --limit 10
python -m src.run_pipeline
```

or:

```bash
./scripts/run_pipeline.sh --limit 10
```

### 5. Run Search UI

```bash
streamlit run services/streamlit/app.py
```

Example queries:

```text
Find a hard calculus problem with a diagram
삼각함수 다이어그램이 있는 문제
이차방정식 관련 어려운 문제 찾아줘
```

## Parser Configuration

`config/settings.yaml` controls the parser strategy:

```yaml
markdown_backend: "docling_vlm"

parser:
  default_backend: "docling_vlm"
  enable_pure_vlm_fallback: true
  benchmark_backends: []
  quality_threshold: 0.5
```

Recommended modes:

| Mode | Use case |
| --- | --- |
| `docling_vlm` | Default production path. Preserves structure and uses VLM selectively. |
| `pure_vlm` | Direct Qwen transcription for degraded scans or parser benchmarks. |
| `benchmark_backends: ["pure_vlm"]` | Save non-indexed comparison artifacts while indexing the default backend. |

Artifacts are saved under:

```text
data/artifacts/{document_id}/
  manifest.json
  parsed_document.json
  chunks.jsonl
  markdown.md
  figures/
  benchmark/
```

## Data Model

Main Qdrant collection: `exam_questions`

```text
text_vector
visual_vector
section_id
section_title
section_level
page_range_start
page_range_end
document_id
parser_backend
difficulty
topic
has_math
has_diagram
markdown
metadata_json
```

Figure collection: `exam_questions_figures`

```text
figure_visual_vector
figure_text_vector
figure_id
description
page_number
document_id
section_id
```

## Quality Checks

```bash
ruff check src services tests
pytest -q
```

Current test suite uses mocks for model calls, Docling, and Qdrant so it can run
quickly without external services.

## Why This Project Matters

Scanned educational documents combine OCR, layout understanding, math notation,
Korean text, and visual reasoning. This pipeline demonstrates how to build a
maintainable RAG ingestion system for that kind of data: structured where
possible, multimodal where necessary, and reproducible enough to operate and
debug.
