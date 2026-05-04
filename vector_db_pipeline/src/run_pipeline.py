"""
run_pipeline.py — End-to-end pipeline orchestrator.

Usage:
  python -m src.run_pipeline [--config config/settings.yaml] [--dry-run] [--limit 10]

Steps:
  0. Discover PNG documents
  1. Connect Qdrant (local Docker)
  2. Health-check Ollama
  3. Load TextEmbedder (INT8 if CUDA)
  4. Parse → embed → batch → flush
  5. Cleanup
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from .artifacts import ArtifactStore
    from .data_loader import Document, discover_documents
    from .document_parser import DocumentParser
    from .embed_text import TextEmbedder
    from .health_check import run_all as health_check
    from .parse_result import QuestionResult, deterministic_id
    from .qdrant_store import QdrantStore
    from .vlm_client import VlmClient
    from .vlm_generate import init_vlm_client
except ImportError:
    from artifacts import ArtifactStore
    from data_loader import Document, discover_documents
    from document_parser import DocumentParser
    from embed_text import TextEmbedder
    from health_check import run_all as health_check
    from parse_result import QuestionResult, deterministic_id
    from qdrant_store import QdrantStore
    from vlm_client import VlmClient
    from vlm_generate import init_vlm_client

_DEFAULT_BATCH_SIZE = 50


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _connect_qdrant(config: dict) -> QdrantStore:
    print("\n=== Step 1: Connecting to Qdrant ===")
    q_cfg = config["qdrant"]
    url = q_cfg.get("url") or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = q_cfg.get("api_key") or os.environ.get("QDRANT_API_KEY") or None
    store = QdrantStore(url=url, api_key=api_key, collection_name=q_cfg["collection_name"])
    store.create_collection(text_dim=q_cfg.get("text_dim", 1024))
    return store


def _build_vlm(config: dict) -> VlmClient:
    vlm_cfg = config["vlm"]
    api_base = os.environ.get("OLLAMA_API_BASE") or vlm_cfg["api_base"]
    raw_client = init_vlm_client(api_base)
    return VlmClient(
        raw_client,
        vlm_cfg["model_name"],
        max_tokens=vlm_cfg.get("max_tokens", 2048),
        temperature=vlm_cfg.get("temperature", 0.1),
    )


def _load_embedder(config: dict) -> TextEmbedder:
    print("\n=== Step 3: Loading embedding model ===")
    e_cfg = config["embed_text"]
    return TextEmbedder(
        model_name=e_cfg["model_name"],
        quantization=e_cfg.get("quantization"),
    )


def _question_record(result: QuestionResult, text_vector: list[float]) -> dict:
    point_id = deterministic_id(result.document_id, "question", prefix="q_")
    return {
        "point_id": point_id,
        "text_vector": text_vector,
        "document_id": result.document_id,
        "text": result.text,
        "concepts": result.concepts,
        "description": result.description,
    }


def _flush(store: QdrantStore, records: list[dict], *, final: bool = False) -> None:
    if not records:
        return
    label = "Flushing final batch" if final else "Flushing batch"
    print(f"{label} ({len(records)} records)...")
    store.insert_batch(list(records))
    records.clear()


def run_pipeline(config: dict, dry_run: bool = False, limit: int | None = None):
    raw_dir = config["data"]["raw_dir"]
    artifact_dir = config["data"]["artifact_dir"]
    batch_size = int(config.get("pipeline", {}).get("batch_size", _DEFAULT_BATCH_SIZE))
    artifacts = ArtifactStore(artifact_dir)

    # Step 0: Discover
    print("\n=== Step 0: Discovering documents ===")
    documents = discover_documents(raw_dir)
    if limit:
        documents = documents[:limit]
        print(f"  Limited to {limit} document(s).")

    if dry_run:
        print("[DRY RUN] Would process:")
        for doc in documents[:5]:
            print(f"  {doc.doc_id}")
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more.")
        return

    # Step 1: Qdrant
    store = _connect_qdrant(config)
    existing_ids = store.get_existing_ids()
    documents = [
        d for d in documents
        if not (d.doc_id in existing_ids and artifacts.is_stored(d.doc_id))
    ]
    if not documents:
        print("  All documents already processed.")
        return

    # Step 2: VLM health check
    vlm_cfg = config["vlm"]
    api_base = os.environ.get("OLLAMA_API_BASE") or vlm_cfg["api_base"]
    print("\n=== Step 2: Health-checking Ollama ===")
    health_check(
        api_base,
        vlm_cfg["model_name"],
        retries=vlm_cfg.get("health_check_retries", 3),
        delay=vlm_cfg.get("health_check_delay", 5),
    )

    # Step 3: Build VLM client and parser
    vlm = _build_vlm(config)
    parser = DocumentParser(vlm)

    # Step 4: Load embedder
    embedder = _load_embedder(config)

    print(f"\n=== Step 4: Processing {len(documents)} document(s) ===\n")
    records: list[dict] = []
    failures: list[tuple[str, str]] = []

    try:
        for i, doc in enumerate(tqdm(documents, desc="Processing")):
            try:
                print(f"\n[{i + 1}/{len(documents)}] {doc.doc_id}")
                result = parser.parse_single(doc.page_paths[0], doc.doc_id)
                artifacts.save_result(result)

                embed_input = f"{result.text} {result.description}".strip()
                vector = embedder.embed_text(embed_input)
                records.append(_question_record(result, vector))

                if len(records) >= batch_size:
                    _flush(store, records)

            except Exception as e:
                print(f"\n  [ERROR] Failed on {doc.doc_id}: {e}")
                artifacts.write_manifest(doc.doc_id, "failed", error=str(e))
                failures.append((doc.doc_id, str(e)))

        _flush(store, records, final=True)
        failed_ids = {doc_id for doc_id, _ in failures}
        for doc in documents:
            if doc.doc_id not in failed_ids:
                artifacts.write_manifest(doc.doc_id, "stored")
    finally:
        print("\n=== Cleanup ===")
        embedder.free()

    processed = len(documents) - len(failures)
    print(f"\nDone! {processed}/{len(documents)} document(s) processed successfully.")
    if failures:
        print(f"\n{len(failures)} document(s) failed:")
        for doc_id, err in failures:
            print(f"  {doc_id}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Vector DB Pipeline")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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
