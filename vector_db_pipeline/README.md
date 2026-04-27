# Vector DB Pipeline — Korean Exam Questions

Extracts Korean exam pages from `png/`, parses them into structured sections,
builds text and visual embeddings, and stores them in Qdrant for semantic search.

## What This Repo Does

The pipeline is built for scanned exam materials, especially math-heavy pages
that contain formulas, diagrams, and worked solutions.

Main flow:

1. Discover PNG inputs from `png/`
2. Parse pages into structured markdown sections
3. Enrich difficult pages and figures with a VLM
4. Create text and visual embeddings
5. Store section and figure records in Qdrant
6. Query the stored data through a small Streamlit app

## Current Architecture

```
┌──────────────────────────────┐
│  Source PNGs / documents     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  run_pipeline.py             │
│  - discover documents        │
│  - choose parser backend     │
│  - load embedders            │
│  - batch insert into Qdrant  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Parsing layer               │
│  - Docling                   │
│  - Qwen VLM                  │
│  - LlamaParse (optional)     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Embedding layer             │
│  - Qwen3 text embeddings     │
│  - Qwen3 visual embeddings   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Qdrant                      │
│  - section collection        │
│  - figure collection         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Query layer                 │
│  - query_agent.py            │
│  - Streamlit demo            │
└──────────────────────────────┘
```

## Project Structure

```text
vector_db_pipeline/
├── config/
│   └── settings.yaml
├── scripts/
│   └── run_pipeline.sh
├── services/
│   ├── ingestion/
│   └── streamlit/
├── src/
│   ├── data_loader.py
│   ├── document_parser.py
│   ├── embed_text.py
│   ├── embed_visual.py
│   ├── health_check.py
│   ├── llamaparse_generate.py
│   ├── parse_result.py
│   ├── qdrant_store.py
│   ├── query_agent.py
│   ├── run_pipeline.py
│   ├── section_detector.py
│   └── vlm_generate.py
├── tests/
├── pyproject.toml
└── README.md
```

## Parser Backends

Configured by `markdown_backend` in `config/settings.yaml`.

- `docling_vlm`
  Best current path. Uses Docling for structure and Qwen VLM for selective enrichment.
- `vlm`
  Simple image-to-markdown path.
- `llamaparse`
  Alternative parser path when you want a non-VLM parsing backend.

All parser backends are expected to return normalized `SectionResult` objects.

## Storage Model

The pipeline stores two kinds of objects in Qdrant:

- Sections
  Main retrieval unit. Contains section markdown, metadata, text vector, and visual vector.
- Figures
  Child records for cropped figures. Contains figure description plus text and visual vectors.

This lets you retrieve by text meaning, image meaning, or both.

## Requirements

- Python 3.10+
- Qdrant
- Access to the configured VLM backend if you use `vlm` or `docling_vlm`

The repo is configured for Python 3.10 in `pyproject.toml`.

## Quick Start

### 1. Create a Local Environment

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e ".[dev,streamlit]"
```

Optional:

```bash
source .venv/bin/activate
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:latest
```

### 3. Configure the Pipeline

Edit `config/settings.yaml`.

Important fields:

```yaml
markdown_backend: "docling_vlm"

pipeline:
  batch_size: 50

vlm:
  api_base: "http://localhost:8000/v1"
  model_name: "Qwen/Qwen3.5-VL-9B"

qdrant:
  url: "http://localhost:6333"
  collection_name: "exam_questions"
```

### 4. Run a Dry Run

```bash
uv run --python .venv/bin/python -m src.run_pipeline --dry-run
```

### 5. Run the Pipeline

```bash
uv run --python .venv/bin/python -m src.run_pipeline --limit 10
uv run --python .venv/bin/python -m src.run_pipeline
```

Or use the helper:

```bash
./scripts/run_pipeline.sh --limit 10
```

## Query Demo

The Streamlit app uses the stored section vectors for retrieval.

Example:

```bash
streamlit run services/streamlit/app.py
```

Expected environment variables or secrets:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `VLLM_API_BASE`

## Testing

```bash
uv run --python .venv/bin/python pytest tests/ -v
```

Useful checks:

```bash
uv run --python .venv/bin/python ruff check src/ tests/
uv run --python .venv/bin/python ruff format src/ tests/
```

## Notes

- Multi-page documents are supported when `png/` contains subdirectories.
- Flat mode is supported when `png/` contains only `.png` files.
- For multi-page sections, the visual vector is now aggregated from all pages in the section instead of only the first page.
- If the VLM returns malformed JSON for section metadata, the parser retries once with a stricter prompt.
