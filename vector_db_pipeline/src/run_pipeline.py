"""
run_pipeline.py — End-to-end pipeline orchestrator.

Usage:
  python -m src.run_pipeline [--config config/settings.yaml] [--dry-run] [--limit 10]

Steps:
  1. Discover documents (flat PNG mode or multi-page folder mode)
  2. Parse each document (Docling + VLM via configured backend)
  3. Embed section text + visual vectors
  4. Embed figure visual + text vectors
  5. Insert Section and Figure objects into Qdrant
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from .artifacts import ArtifactStore
    from .data_loader import Document, discover_documents
    from .embed_text import TextEmbedder
    from .embed_visual import VisualEmbedder
    from .parse_result import ParsedDocument, ParserBackend, SectionResult
    from .qdrant_store import QdrantStore
except ImportError:
    from artifacts import ArtifactStore
    from data_loader import Document, discover_documents
    from embed_text import TextEmbedder
    from embed_visual import VisualEmbedder
    from parse_result import ParsedDocument, ParserBackend, SectionResult
    from qdrant_store import QdrantStore

_DEFAULT_BATCH_SIZE = 50
_CLOUD_TMP_OUTPUT_DIR = "/tmp/data/processed"
_CLOUD_TMP_ARTIFACT_DIR = "/tmp/data/artifacts"


def _is_mock(value) -> bool:
    return isinstance(value, Mock)


def _has_real_method(obj, name: str) -> bool:
    method = getattr(obj, name, None)
    return callable(method) and not _is_mock(method)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_markdown(markdown_text: str, output_dir: str, section_id: str):
    """Save assembled section markdown for debugging/reuse."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{section_id}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)


def _build_parser(config: dict) -> ParserBackend:
    try:
        from .parser_backends import build_parser
    except ImportError:
        from parser_backends import build_parser

    return build_parser(config)


class _SimpleParser:
    """Backward-compatible markdown-only parser used by older tests/callers."""

    parser_backend = "pure_vlm"

    def __init__(self, generate_fn):
        self._generate = generate_fn

    def parse_single(self, image_path: str, document_id: str) -> SectionResult:
        try:
            from .parse_result import make_section_result
        except ImportError:
            from parse_result import make_section_result

        return make_section_result(
            document_id=document_id,
            section_title=document_id,
            section_level=0,
            page_range=(0, 0),
            markdown=self._generate(image_path),
            parser_backend=self.parser_backend,
        )

    def parse_document(self, page_paths: list[str], document_id: str) -> list[SectionResult]:
        sections = []
        for i, page_path in enumerate(page_paths):
            section = self.parse_single(page_path, document_id)
            section.section_title = f"{document_id} — page {i + 1}"
            section.page_range = (i, i)
            sections.append(section)
        return sections


def _get_batch_size(config: dict) -> int:
    return int(config.get("pipeline", {}).get("batch_size", _DEFAULT_BATCH_SIZE))


def _resolve_raw_dir(config: dict) -> str:
    data_cfg = config["data"]
    gcs_uri = os.environ.get("GCS_URI")
    if gcs_uri:
        return gcs_uri

    gcs_bucket = os.environ.get("GCS_BUCKET")
    if gcs_bucket:
        return gcs_bucket if gcs_bucket.startswith("gs://") else f"gs://{gcs_bucket}"

    return data_cfg["raw_dir"]


def _resolve_output_dir(config: dict, raw_dir: str) -> str:
    if os.environ.get("OUTPUT_DIR"):
        return os.environ["OUTPUT_DIR"]
    if raw_dir.startswith("gs://"):
        return _CLOUD_TMP_OUTPUT_DIR
    return config["data"]["output_dir"]


def _resolve_artifact_dir(config: dict, raw_dir: str) -> str:
    if os.environ.get("ARTIFACT_DIR"):
        return os.environ["ARTIFACT_DIR"]
    if raw_dir.startswith("gs://"):
        return _CLOUD_TMP_ARTIFACT_DIR
    return config.get("data", {}).get(
        "artifact_dir",
        str(Path(config["data"]["output_dir"]).parent / "artifacts"),
    )


def _discover_documents_for_run(raw_dir: str, limit: int | None = None) -> list[Document]:
    print("\n=== Step 0: Discovering documents ===")
    documents = discover_documents(raw_dir)
    if limit:
        documents = documents[:limit]
        print(f"  Limited to {limit} document(s).\n")
    else:
        print()
    return documents


def _connect_qdrant(config: dict) -> QdrantStore:
    print("\n=== Step 1: Connecting to Qdrant ===")
    q_cfg = config["qdrant"]
    qdrant_url = q_cfg.get("url") or os.environ.get("QDRANT_URL")
    qdrant_api_key = q_cfg.get("api_key") or os.environ.get("QDRANT_API_KEY") or None
    q_store = QdrantStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=q_cfg["collection_name"],
    )
    q_store.create_collections(
        visual_dim=q_cfg["visual_dim"],
        text_dim=q_cfg["text_dim"],
    )
    return q_store


def _skip_existing_documents(
    documents: list[Document],
    q_store: QdrantStore,
    artifacts: ArtifactStore,
) -> list[Document]:
    print("  Checking for already-processed documents...")
    existing_ids = q_store.get_existing_ids()
    if not existing_ids:
        return documents

    before = len(documents)
    remaining = [
        d for d in documents
        if not (d.doc_id in existing_ids and artifacts.is_stored(d.doc_id))
    ]
    print(f"  Skipping {before - len(remaining)} already-processed document(s).")
    return remaining


def _load_embedders(config: dict) -> tuple[VisualEmbedder, TextEmbedder]:
    embed_v_cfg = config["embed_visual"]
    embed_t_cfg = config["embed_text"]
    print("\n=== Step 3: Loading embedding models ===")
    visual_embedder = VisualEmbedder(model_name=embed_v_cfg["model_name"])
    text_embedder = TextEmbedder(model_name=embed_t_cfg["model_name"])
    return visual_embedder, text_embedder


def _select_section_page_paths(doc: Document, section: SectionResult) -> list[str]:
    start, end = section.page_range
    page_paths = doc.page_paths[start : end + 1]
    if not page_paths:
        raise ValueError(
            f"Section page_range {section.page_range} does not map to any pages for {doc.doc_id}"
        )
    return page_paths


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _mean_pool_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot mean-pool an empty vector list")
    size = len(vectors[0])
    if any(len(vec) != size for vec in vectors):
        raise ValueError("All vectors must have the same length for pooling")
    totals = [0.0] * size
    for vec in vectors:
        for i, value in enumerate(vec):
            totals[i] += value
    return [value / len(vectors) for value in totals]


def _embed_section_visual(
    visual_embedder: VisualEmbedder,
    section_page_paths: list[str],
) -> list[float]:
    if len(section_page_paths) == 1:
        return visual_embedder.embed_image(section_page_paths[0])

    if hasattr(visual_embedder, "embed_batch"):
        vectors = visual_embedder.embed_batch(image_paths=section_page_paths)
    else:
        vectors = [visual_embedder.embed_image(path) for path in section_page_paths]

    return _normalize_vector(_mean_pool_vectors(vectors))


def _section_record(
    section: SectionResult,
    text_vector: list[float],
    visual_vector: list[float],
) -> dict:
    return {
        "point_id": section.section_id,
        "section_id": section.section_id,
        "text_vector": text_vector,
        "visual_vector": visual_vector,
        "section_title": section.section_title,
        "section_level": section.section_level,
        "page_range_start": section.page_range[0],
        "page_range_end": section.page_range[1],
        "document_id": section.document_id,
        "markdown": section.markdown,
        "parser_backend": section.parser_backend,
        "metadata_json": json.dumps(section.metadata),
        **{
            key: section.metadata[key]
            for key in ("difficulty", "topic", "has_math", "has_diagram")
            if key in section.metadata
        },
    }


def _figure_record(
    section: SectionResult,
    fig,
    fig_visual_vector: list[float],
    fig_text_vector: list[float],
) -> dict:
    return {
        "point_id": fig.figure_id,
        "figure_id": fig.figure_id,
        "figure_visual_vector": fig_visual_vector,
        "figure_text_vector": fig_text_vector,
        "description": fig.description,
        "page_number": fig.page_number,
        "document_id": section.document_id,
        "section_id": section.section_id,
    }


def _parse_document(parser: ParserBackend, doc: Document) -> ParsedDocument:
    if _has_real_method(parser, "parse_parsed_document"):
        return parser.parse_parsed_document(doc.page_paths, doc.doc_id)

    sections = (
        [parser.parse_single(doc.page_paths[0], doc.doc_id)]
        if doc.is_single_page
        else parser.parse_document(doc.page_paths, doc.doc_id)
    )
    try:
        from .parse_result import sections_to_parsed_document, validate_section_result
    except ImportError:
        from parse_result import sections_to_parsed_document, validate_section_result

    parser_backend = getattr(parser, "parser_backend", "legacy")
    if _is_mock(parser_backend):
        parser_backend = "legacy"

    return sections_to_parsed_document(
        document_id=doc.doc_id,
        page_paths=doc.page_paths,
        sections=[validate_section_result(section, doc.doc_id) for section in sections],
        parser_backend=parser_backend,
    )


def _process_document(
    doc: Document,
    parser: ParserBackend,
    artifacts: ArtifactStore,
    visual_embedder: VisualEmbedder,
    text_embedder: TextEmbedder,
) -> tuple[list[dict], list[dict]]:
    parsed = _parse_document(parser, doc)
    artifacts.save_parsed_document(parsed)

    sections = parsed.chunks
    section_records: list[dict] = []
    figure_records: list[dict] = []

    section_texts = [section.markdown for section in sections]
    text_vectors = (
        text_embedder.embed_texts(section_texts)
        if section_texts and _has_real_method(text_embedder, "embed_texts")
        else [text_embedder.embed_text(section.markdown) for section in sections]
    )

    for section in sections:
        section_page_paths = _select_section_page_paths(doc, section)
        text_vec = text_vectors[len(section_records)]
        visual_vec = _embed_section_visual(visual_embedder, section_page_paths)
        section_records.append(_section_record(section, text_vec, visual_vec))

    figure_pairs = [
        (section, fig)
        for section in sections
        for fig in section.figures
    ]
    figure_texts = [fig.description or "" for _, fig in figure_pairs]
    figure_text_vectors = (
        text_embedder.embed_texts(figure_texts)
        if figure_texts and _has_real_method(text_embedder, "embed_texts")
        else [text_embedder.embed_text(text) for text in figure_texts]
    )
    for i, (section, fig) in enumerate(figure_pairs):
        fig_visual_vec = visual_embedder.embed_image(fig.cropped_image_path)
        fig_text_vec = figure_text_vectors[i]
        figure_records.append(_figure_record(section, fig, fig_visual_vec, fig_text_vec))

    return section_records, figure_records


def _flush_batches(
    q_store: QdrantStore,
    section_records: list[dict],
    figure_records: list[dict],
    *,
    final: bool = False,
) -> None:
    if not section_records:
        return

    prefix = "Flushing final batch" if final else "  Flushing batch"
    print(
        f"{prefix} ({len(section_records)} sections, "
        f"{len(figure_records)} figures)..."
    )
    q_store.insert_section_batch(list(section_records))
    q_store.insert_figure_batch(list(figure_records))
    section_records.clear()
    figure_records.clear()


def run_pipeline(config: dict, dry_run: bool = False, limit: int | None = None):
    raw_dir = _resolve_raw_dir(config)
    _resolve_output_dir(config, raw_dir)
    artifact_dir = _resolve_artifact_dir(config, raw_dir)
    batch_size = _get_batch_size(config)
    artifacts = ArtifactStore(artifact_dir)

    documents = _discover_documents_for_run(raw_dir, limit=limit)

    if dry_run:
        print("[DRY RUN] Would process:")
        for doc in documents[:5]:
            mode = "single" if doc.is_single_page else f"{len(doc.page_paths)} pages"
            print(f"  {doc.doc_id} ({mode})")
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more.")
        return

    q_store = _connect_qdrant(config)
    documents = _skip_existing_documents(documents, q_store, artifacts)
    if not documents:
        print("  All documents already processed. Nothing to do.")
        return

    parser = _build_parser(config)
    visual_embedder, text_embedder = _load_embedders(config)

    print(f"\n=== Step 4: Processing {len(documents)} document(s) ===\n")
    section_records: list[dict] = []
    figure_records: list[dict] = []
    failures: list[tuple[str, str]] = []

    try:
        for i, doc in enumerate(tqdm(documents, desc="Processing")):
            try:
                print(f"\n[{i + 1}/{len(documents)}] {doc.doc_id}")
                new_sections, new_figures = _process_document(
                    doc,
                    parser,
                    artifacts,
                    visual_embedder,
                    text_embedder,
                )
                section_records.extend(new_sections)
                figure_records.extend(new_figures)

                if len(section_records) >= batch_size:
                    _flush_batches(q_store, section_records, figure_records)

            except Exception as e:
                print(f"\n  [ERROR] Failed on {doc.doc_id}: {e}")
                artifacts.write_manifest(doc.doc_id, "failed", error=str(e))
                failures.append((doc.doc_id, str(e)))

        _flush_batches(q_store, section_records, figure_records, final=True)
        for doc in documents:
            if doc.doc_id not in {doc_id for doc_id, _ in failures}:
                artifacts.write_manifest(doc.doc_id, "stored")
    finally:
        print("\n=== Cleanup ===")
        visual_embedder.free()
        text_embedder.free()

    processed = len(documents) - len(failures)
    print(f"\nDone! {processed}/{len(documents)} document(s) processed successfully.")
    if failures:
        print(f"\n⚠ {len(failures)} document(s) failed:")
        for doc_id, err in failures:
            print(f"  {doc_id}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Vector DB Pipeline")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents and exit without processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents (for testing)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    run_pipeline(config, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
