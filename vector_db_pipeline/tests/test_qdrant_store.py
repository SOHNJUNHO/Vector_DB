"""
tests/test_qdrant_store.py — Tests for qdrant_store.py.
All tests are fully mocked — no live Qdrant connection required.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_store(collection_name: str = "exam_questions"):
    """Return a QdrantStore with a mocked QdrantClient."""
    with patch("qdrant_store.QdrantClient"):
        from qdrant_store import QdrantStore
        store = QdrantStore(url="http://mock", api_key="key", collection_name=collection_name)
        store.client = MagicMock()
    return store


class TestCreateCollection:
    def test_creates_collection_when_it_does_not_exist(self):
        store = _make_store()
        store.client.collection_exists.return_value = False

        store.create_collection(text_dim=1024)

        store.client.create_collection.assert_called_once()
        call_kwargs = store.client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "exam_questions"

    def test_skips_creation_when_collection_already_exists(self):
        store = _make_store()
        store.client.collection_exists.return_value = True

        store.create_collection()

        store.client.create_collection.assert_not_called()

    def test_scalar_quantization_is_applied(self):
        from qdrant_client.models import ScalarQuantization

        store = _make_store()
        store.client.collection_exists.return_value = False

        store.create_collection(text_dim=1024)

        call_kwargs = store.client.create_collection.call_args.kwargs
        assert isinstance(call_kwargs.get("quantization_config"), ScalarQuantization)


class TestGetExistingIds:
    def test_returns_document_ids_from_scroll(self):
        store = _make_store()

        point_a = MagicMock()
        point_a.payload = {"document_id": "doc_001"}
        point_b = MagicMock()
        point_b.payload = {"document_id": "doc_002"}

        store.client.scroll.return_value = ([point_a, point_b], None)

        ids = store.get_existing_ids()

        assert ids == {"doc_001", "doc_002"}

    def test_paginates_until_offset_is_none(self):
        store = _make_store()

        p1 = MagicMock()
        p1.payload = {"document_id": "doc_001"}
        p2 = MagicMock()
        p2.payload = {"document_id": "doc_002"}

        store.client.scroll.side_effect = [
            ([p1], "next_page_token"),
            ([p2], None),
        ]

        ids = store.get_existing_ids()

        assert ids == {"doc_001", "doc_002"}
        assert store.client.scroll.call_count == 2

    def test_ignores_points_without_document_id(self):
        store = _make_store()

        good = MagicMock()
        good.payload = {"document_id": "doc_001"}
        bad = MagicMock()
        bad.payload = {}

        store.client.scroll.return_value = ([good, bad], None)

        ids = store.get_existing_ids()

        assert ids == {"doc_001"}


class TestInsertBatch:
    def test_upserts_to_collection(self):
        store = _make_store()
        records = [
            {
                "text_vector": [0.1] * 1024,
                "document_id": "doc_001",
                "text": "이차방정식 문제",
                "concepts": ["이차방정식"],
                "description": "수식 문제",
            }
        ]

        store.insert_batch(records)

        store.client.upsert.assert_called_once()
        call_kwargs = store.client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "exam_questions"
        points = call_kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["document_id"] == "doc_001"
        assert "text_vector" not in points[0].payload

    def test_vector_stored_at_top_level(self):
        store = _make_store()
        vector = [0.5] * 1024
        records = [
            {
                "text_vector": vector,
                "document_id": "doc_001",
                "text": "test",
                "concepts": [],
                "description": "",
            }
        ]

        store.insert_batch(records)

        points = store.client.upsert.call_args.kwargs["points"]
        assert points[0].vector == vector

    def test_skips_empty_batch(self):
        store = _make_store()
        store.insert_batch([])
        store.client.upsert.assert_not_called()


class TestSearch:
    def test_returns_score_and_payload_fields(self):
        store = _make_store()
        hit = MagicMock()
        hit.score = 0.92
        hit.payload = {"document_id": "doc_001", "concepts": ["이차방정식"], "text": "문제"}
        store.client.search.return_value = [hit]

        results = store.search(vector=[0.1] * 1024, limit=5)

        assert len(results) == 1
        assert results[0]["score"] == 0.92
        assert results[0]["document_id"] == "doc_001"
        assert results[0]["concepts"] == ["이차방정식"]

    def test_passes_limit_to_client(self):
        store = _make_store()
        store.client.search.return_value = []

        store.search(vector=[0.1] * 1024, limit=3)

        call_kwargs = store.client.search.call_args.kwargs
        assert call_kwargs["limit"] == 3
