"""
tests/test_run_pipeline.py — Wiring tests for run_pipeline.py.

Mocks all external systems to validate config wiring and batch behavior
without model downloads or a live Qdrant or Ollama instance.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _fake_question_result(doc_id: str = "doc_001"):
    from parse_result import QuestionResult

    return QuestionResult(
        document_id=doc_id,
        text="이차방정식을 풀어라",
        concepts=["이차방정식", "판별식"],
        description="수식이 포함된 문제",
    )


def _fake_document(doc_id: str, page_path: str):
    from data_loader import Document

    return Document(doc_id=doc_id, page_paths=[page_path])


def _base_config(tmp_path) -> dict:
    return {
        "data": {
            "raw_dir": str(tmp_path),
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "pipeline": {"batch_size": 10},
        "vlm": {
            "api_base": "http://localhost:11434/v1",
            "model_name": "qwen2.5-vl:7b",
            "max_tokens": 2048,
            "temperature": 0.1,
        },
        "embed_text": {"model_name": "Qwen/Qwen3-Embedding-0.6B"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "exam_questions",
            "text_dim": 1024,
        },
    }


def test_run_pipeline_connects_to_qdrant_with_config(tmp_path):
    from run_pipeline import run_pipeline

    png_path = tmp_path / "png" / "doc_001.png"
    png_path.parent.mkdir(parents=True)
    png_path.write_bytes(b"fake")

    config = _base_config(tmp_path)
    config["qdrant"]["url"] = "http://qdrant.example"
    config["qdrant"]["api_key"] = "secret"

    fake_result = _fake_question_result()

    with (
        patch("run_pipeline.QdrantStore") as MockStore,
        patch("run_pipeline.init_vlm_client"),
        patch("run_pipeline.health_check"),
        patch("run_pipeline.DocumentParser") as MockParser,
        patch("run_pipeline.TextEmbedder") as MockEmbedder,
        patch("run_pipeline.ArtifactStore") as MockArtifacts,
    ):
        store = MockStore.return_value
        store.get_existing_ids.return_value = set()
        MockParser.return_value.parse_single.return_value = fake_result
        MockEmbedder.return_value.embed_text.return_value = [0.1] * 1024
        MockArtifacts.return_value.is_stored.return_value = False

        run_pipeline(config)

    MockStore.assert_called_once_with(
        url="http://qdrant.example",
        api_key="secret",
        collection_name="exam_questions",
    )
    store.create_collection.assert_called_once_with(text_dim=1024)


def test_run_pipeline_inserts_correct_payload(tmp_path):
    from run_pipeline import run_pipeline

    png_path = tmp_path / "png" / "doc_001.png"
    png_path.parent.mkdir(parents=True)
    png_path.write_bytes(b"fake")

    config = _base_config(tmp_path)
    fake_result = _fake_question_result("doc_001")

    with (
        patch("run_pipeline.QdrantStore") as MockStore,
        patch("run_pipeline.init_vlm_client"),
        patch("run_pipeline.health_check"),
        patch("run_pipeline.DocumentParser") as MockParser,
        patch("run_pipeline.TextEmbedder") as MockEmbedder,
        patch("run_pipeline.ArtifactStore") as MockArtifacts,
    ):
        store = MockStore.return_value
        store.get_existing_ids.return_value = set()
        MockParser.return_value.parse_single.return_value = fake_result
        MockEmbedder.return_value.embed_text.return_value = [0.2] * 1024
        MockArtifacts.return_value.is_stored.return_value = False

        run_pipeline(config)

    store.insert_batch.assert_called_once()
    records = store.insert_batch.call_args.args[0]
    assert len(records) == 1
    record = records[0]
    assert record["document_id"] == "doc_001"
    assert record["text"] == "이차방정식을 풀어라"
    assert record["concepts"] == ["이차방정식", "판별식"]
    assert record["description"] == "수식이 포함된 문제"
    assert "text_vector" in record  # passed to insert_batch which builds PointStruct


def test_run_pipeline_skips_already_processed_documents(tmp_path):
    from run_pipeline import run_pipeline

    for name in ("doc_001.png", "doc_002.png"):
        png = tmp_path / "png" / name
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"fake")

    config = _base_config(tmp_path)

    with (
        patch("run_pipeline.QdrantStore") as MockStore,
        patch("run_pipeline.init_vlm_client"),
        patch("run_pipeline.health_check"),
        patch("run_pipeline.DocumentParser") as MockParser,
        patch("run_pipeline.TextEmbedder") as MockEmbedder,
        patch("run_pipeline.ArtifactStore") as MockArtifacts,
    ):
        store = MockStore.return_value
        store.get_existing_ids.return_value = {"doc_001", "doc_002"}
        artifacts = MockArtifacts.return_value
        artifacts.is_stored.return_value = True

        run_pipeline(config)

    MockParser.return_value.parse_single.assert_not_called()


def test_run_pipeline_dry_run_does_not_process(tmp_path):
    from run_pipeline import run_pipeline

    png_path = tmp_path / "png" / "doc_001.png"
    png_path.parent.mkdir(parents=True)
    png_path.write_bytes(b"fake")

    config = _base_config(tmp_path)

    with (
        patch("run_pipeline.QdrantStore") as MockStore,
        patch("run_pipeline.DocumentParser") as MockParser,
    ):
        run_pipeline(config, dry_run=True)

    MockStore.assert_not_called()
    MockParser.assert_not_called()
