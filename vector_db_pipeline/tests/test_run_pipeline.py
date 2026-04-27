"""
tests/test_run_pipeline.py — High-level wiring tests for run_pipeline.py.

These tests mock all external systems so they can validate config wiring and
batch insertion behavior without model downloads or a live Qdrant instance.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _FakeSection:
    def __init__(self):
        from parse_result import SectionResult

        self.value = SectionResult(
            section_id="section-1",
            section_title="Section 1",
            section_level=1,
            page_range=(0, 0),
            markdown="## 문제\nExample",
            document_id="doc_001",
            metadata={
                "difficulty": "hard",
                "topic": "algebra",
                "has_math": True,
                "has_diagram": False,
            },
            figures=[],
        )


class _FakeDocument:
    def __init__(self, doc_id: str, page_path: str):
        self.doc_id = doc_id
        self.page_paths = [page_path]
        self.is_single_page = True


def test_run_pipeline_uses_qdrant_config(tmp_path):
    from run_pipeline import run_pipeline

    png_path = tmp_path / "doc_001.png"
    png_path.write_bytes(b"fake")

    config = {
        "data": {
            "raw_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        },
        "embed_visual": {"model_name": "visual-model"},
        "embed_text": {"model_name": "text-model"},
        "qdrant": {
            "url": "http://qdrant.example",
            "api_key": "secret",
            "collection_name": "exam_questions",
            "visual_dim": 1024,
            "text_dim": 1024,
        },
    }

    fake_doc = _FakeDocument("doc_001", str(png_path))
    fake_parser = MagicMock()
    fake_parser.parse_single.return_value = _FakeSection().value

    with (
        patch("run_pipeline.discover_documents", return_value=[fake_doc]),
        patch("run_pipeline.QdrantStore") as MockStore,
        patch("run_pipeline._build_parser", return_value=fake_parser),
        patch("run_pipeline.VisualEmbedder") as MockVisual,
        patch("run_pipeline.TextEmbedder") as MockText,
    ):
        visual = MockVisual.return_value
        visual.embed_image.return_value = [0.2] * 4
        text = MockText.return_value
        text.embed_text.return_value = [0.1] * 4

        store = MockStore.return_value
        store.get_existing_ids.return_value = set()

        run_pipeline(config)

    MockStore.assert_called_once_with(
        url="http://qdrant.example",
        api_key="secret",
        collection_name="exam_questions",
    )
    store.create_collections.assert_called_once_with(visual_dim=1024, text_dim=1024)
    store.insert_section_batch.assert_called_once()
    inserted_record = store.insert_section_batch.call_args.args[0][0]
    assert inserted_record["document_id"] == "doc_001"
    assert inserted_record["difficulty"] == "hard"
    assert inserted_record["topic"] == "algebra"


def test_run_pipeline_mean_pools_multi_page_visual_vectors(tmp_path):
    from run_pipeline import run_pipeline

    page_a = tmp_path / "doc_001_001.png"
    page_b = tmp_path / "doc_001_002.png"
    page_a.write_bytes(b"fake")
    page_b.write_bytes(b"fake")

    config = {
        "data": {"raw_dir": str(tmp_path), "output_dir": str(tmp_path / "out")},
        "embed_visual": {"model_name": "visual-model"},
        "embed_text": {"model_name": "text-model"},
        "pipeline": {"batch_size": 10},
        "qdrant": {
            "url": "http://qdrant.example",
            "api_key": "secret",
            "collection_name": "exam_questions",
            "visual_dim": 2,
            "text_dim": 2,
        },
    }

    fake_doc = MagicMock()
    fake_doc.doc_id = "doc_001"
    fake_doc.page_paths = [str(page_a), str(page_b)]
    fake_doc.is_single_page = False

    fake_section = _FakeSection().value
    fake_section.page_range = (0, 1)
    fake_parser = MagicMock()
    fake_parser.parse_document.return_value = [fake_section]

    with (
        patch("run_pipeline.discover_documents", return_value=[fake_doc]),
        patch("run_pipeline.QdrantStore") as MockStore,
        patch("run_pipeline._build_parser", return_value=fake_parser),
        patch("run_pipeline.VisualEmbedder") as MockVisual,
        patch("run_pipeline.TextEmbedder") as MockText,
    ):
        visual = MockVisual.return_value
        visual.embed_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]
        text = MockText.return_value
        text.embed_text.return_value = [0.1, 0.2]

        store = MockStore.return_value
        store.get_existing_ids.return_value = set()

        run_pipeline(config)

    inserted_record = store.insert_section_batch.call_args.args[0][0]
    assert inserted_record["visual_vector"] == pytest.approx([0.7071, 0.7071], rel=1e-3)


def test_simple_parser_preserves_document_id_for_multi_page_sections():
    from run_pipeline import _SimpleParser

    parser = _SimpleParser(lambda _path: "markdown")
    sections = parser.parse_document(["001.png", "002.png"], "doc_123")

    assert [section.document_id for section in sections] == ["doc_123", "doc_123"]
    assert [section.page_range for section in sections] == [(0, 0), (1, 1)]
    assert sections[0].section_title == "doc_123 — page 1"


def test_validate_section_result_rejects_mismatched_document_id():
    from parse_result import validate_section_result

    section = _FakeSection().value

    with pytest.raises(ValueError, match="mismatched document_id"):
        validate_section_result(section, "other_doc")
