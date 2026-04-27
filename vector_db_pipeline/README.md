# Multimodal RAG Pipeline — Korean Exam Question Retrieval

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that ingests scanned Korean exam question images, extracts structured content with a hybrid Docling + VLM parser, builds dual-modality embeddings, and stores them in Qdrant Cloud for semantic search.

---

## Overview

This system was built to make hundreds of scanned exam question images semantically searchable — by meaning, not just keywords. Given a natural-language query ("hard calculus problem with a diagram"), it retrieves the most relevant exam questions using both **textual** and **visual** similarity.

**Core problem:** raw exam pages are scanned images containing handwritten math, diagrams, and mixed Korean/LaTeX text. Standard text search fails entirely. This pipeline solves it with a three-layer parsing stack and a multimodal vector index.

---

## Architecture

```
Scanned PNGs (477 pages)
        │
        ▼
┌───────────────────────────────────────────────┐
│  Layer 1 — Docling                            │
│  Structure extraction: text, tables, math      │
│  (LaTeX-native, handles complex layouts)       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Layer 2 — Qwen VLM  (Fireworks AI / local)  │
│  Triggered selectively:                        │
│  · Figure semantic description                 │
│  · Low-quality page full-page fallback         │
│  · Section boundary detection (multi-page)     │
│  · Section metadata extraction (JSON)          │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Layer 3 — Embedding                          │
│  · Text:   Qwen3-Embedding-0.6B  (1024-dim)   │
│  · Visual: Qwen3-VL-Embedding-2B (2048-dim)   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Qdrant Cloud                                  │
│  exam_questions         — section records      │
│    ├─ text_vector   (1024-dim, COSINE)         │
│    └─ visual_vector (2048-dim, COSINE)         │
│  exam_questions_figures — figure records       │
│    ├─ figure_text_vector   (1024-dim)          │
│    └─ figure_visual_vector (2048-dim)          │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Hybrid Search + Streamlit Demo                │
│  · alpha-weighted score fusion (text + visual) │
│  · difficulty / topic payload filters          │
│  · query parsing via LLM                       │
└───────────────────────────────────────────────┘
```

---

## Technical Highlights

| Area | Choice | Reason |
|---|---|---|
| Document parsing | Docling + Qwen VLM | Docling handles structure natively; VLM enriches only where needed (cheaper, faster) |
| Text embedding | Qwen3-Embedding-0.6B | 1024-dim, strong multilingual + math performance |
| Visual embedding | Qwen3-VL-Embedding-2B | Shared image+text embedding space; 2048-dim output |
| Vector store | Qdrant Cloud | Named-vector collections, payload-filtered search, managed cloud |
| VLM inference | Fireworks AI | OpenAI-compatible API, no GPU required; swappable for local vLLM |
| Hybrid search | Alpha-weighted score fusion | Blends text and visual similarity; tunable `alpha` parameter |
| Idempotency | `document_id` scroll check | Safe to re-run; already-indexed documents are skipped |
| Batching | Configurable `batch_size` | Bounded memory usage across large corpora |

---

## Project Structure

```
vector_db_pipeline/
├── config/
│   └── settings.yaml          # All tunable parameters
├── scripts/
│   └── run_pipeline.sh        # Pre-flight checks + pipeline launcher
├── services/
│   └── streamlit/
│       └── app.py             # Search demo UI
├── src/
│   ├── data_loader.py         # PNG discovery (flat + folder modes)
│   ├── document_parser.py     # Docling + VLM hybrid parser
│   ├── embed_text.py          # Qwen3-Embedding text embedder
│   ├── embed_visual.py        # Qwen3-VL-Embedding multimodal embedder
│   ├── health_check.py        # VLM endpoint smoke tests
│   ├── llamaparse_generate.py # LlamaParse backend (optional)
│   ├── parse_result.py        # Shared SectionResult / FigureResult types
│   ├── qdrant_store.py        # Qdrant collection management + search
│   ├── query_agent.py         # LLM-assisted query parsing + answer synthesis
│   ├── run_pipeline.py        # End-to-end orchestrator
│   ├── section_detector.py    # VLM section boundary detector
│   └── vlm_generate.py        # VLM client + image-to-markdown helpers
├── tests/
├── pyproject.toml
└── .env.example
```

---

## Parser Backends

Controlled by `markdown_backend` in `config/settings.yaml`:

| Backend | Description |
|---|---|
| `docling_vlm` | **Recommended.** Docling for structure, VLM for targeted enrichment. |
| `vlm` | Simple image-to-markdown via VLM only. Slower, no section hierarchy. |
| `llamaparse` | Cloud parser via LlamaIndex. No GPU or VLM key required. |

All backends produce the same `SectionResult` objects — the rest of the pipeline is backend-agnostic.

---

## Data Model

**Section collection** (`exam_questions`) — primary retrieval unit:

```
text_vector    float[1024]   — Qwen3-Embedding of section markdown
visual_vector  float[2048]   — Qwen3-VL-Embedding of page image(s)
section_title  string
section_level  int (1/2/3)
page_range_*   int
document_id    string        — indexed for filtered search
difficulty     string        — "easy" | "medium" | "hard"
topic          string
has_math       bool
has_diagram    bool
markdown       string        — full extracted + enriched text
```

**Figure collection** (`exam_questions_figures`) — child records for cropped figures:

```
figure_visual_vector  float[2048]
figure_text_vector    float[1024]   — embedding of VLM description
description           string        — detailed VLM figure caption
page_number           int
document_id           string
section_id            string
```

---

## Setup

### 1. Install dependencies

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e ".[dev,streamlit]"
source .venv/bin/activate
```

### 2. Configure environment

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | From [fireworks.ai](https://fireworks.ai) — free tier available |
| `QDRANT_URL` | Qdrant Cloud cluster URL, or `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud API key (omit if running locally) |
| `VLLM_API_BASE` | Override VLM endpoint at runtime (optional) |

Local Qdrant:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### 3. Add input images

```
# Flat mode — each PNG is one document
png/
  question_001.png
  question_002.png

# Document mode — each subdirectory is one multi-page document
png/
  exam_2024/
    page_01.png
    page_02.png
```

### 4. Run the pipeline

```bash
# Dry run — discover documents only, no processing
python -m src.run_pipeline --dry-run

# Test on first 10 documents
python -m src.run_pipeline --limit 10

# Full run
python -m src.run_pipeline
```

Or via the launcher script (runs pre-flight checks first):

```bash
./scripts/run_pipeline.sh --limit 10
```

### 5. Search

```bash
streamlit run services/streamlit/app.py
```

Example queries:

- `Find a hard problem about quadratic equations`
- `Show me a trigonometry problem with a diagram`
- `미적분 관련 어려운 문제`

---

## Configuration Reference

`config/settings.yaml`:

```yaml
markdown_backend: "docling_vlm"    # vlm | llamaparse | docling_vlm

pipeline:
  batch_size: 50                   # Qdrant upsert batch size

vlm:
  api_base: "https://api.fireworks.ai/inference/v1"
  model_name: "accounts/<user>/deployments/<id>"
  max_tokens: 4096
  temperature: 0.1

embed_visual:
  model_name: "Qwen/Qwen3-VL-Embedding-2B"

embed_text:
  model_name: "Qwen/Qwen3-Embedding-0.6B"

qdrant:
  collection_name: "exam_questions"
  visual_dim: 2048
  text_dim: 1024

query:
  retrieval_mode: "hybrid"         # text | hybrid
  hybrid_alpha: 0.7                # 1.0 = text only, 0.0 = visual only
```

---

## Linting and Tests

```bash
ruff check src/ tests/
ruff format src/ tests/
pytest tests/ -v
```

---

## Key Design Decisions

**Selective VLM enrichment** — calling a VLM on every page is expensive. The parser only invokes the VLM when Docling signals that a page has complex figures, chemical formulas, or low quality score. This cuts VLM calls by ~60–70% on typical exam corpora while preserving quality where it matters.

**Dual-collection Qdrant schema** — sections and figures are stored separately so figure-level visual search is possible without polluting the section index. A figure's `section_id` enables joining back to the parent context.

**Hybrid score fusion** — raw scores from two named-vector searches are max-normalized, then combined as `alpha * text_score + (1 - alpha) * visual_score`. This makes the alpha parameter interpretable and avoids scale sensitivity between embedding spaces.

**Idempotent ingestion** — on each run, existing `document_id` values are fetched from Qdrant via scroll and filtered out before processing begins. Re-running the pipeline after a partial failure is safe.
