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
    with patch("qdrant_store.QdrantClient") as MockClient:
        from qdrant_store import QdrantStore
        store = QdrantStore(url="http://mock", api_key="key", collection_name=collection_name)
        store.client = MockClient.return_value
    return store


class TestCreateCollections:
    def test_creates_both_collections_when_neither_exists(self):
        store = _make_store()
        store.client.collection_exists.return_value = False

        store.create_collections(visual_dim=1024, text_dim=1024)

        assert store.client.create_collection.call_count == 2
        names = [c.kwargs["collection_name"] for c in store.client.create_collection.call_args_list]
        assert "exam_questions" in names
        assert "exam_questions_figures" in names

    def test_skips_creation_when_collections_already_exist(self):
        store = _make_store()
        store.client.collection_exists.return_value = True

        store.create_collections()

        store.client.create_collection.assert_not_called()

    def test_creates_only_missing_collection(self):
        store = _make_store()
        # Section exists, figures does not
        store.client.collection_exists.side_effect = lambda name: name == "exam_questions"

        store.create_collections()

        assert store.client.create_collection.call_count == 1
        created_name = store.client.create_collection.call_args.kwargs["collection_name"]
        assert created_name == "exam_questions_figures"


class TestGetExistingIds:
    def test_returns_document_ids_from_scroll(self):
        store = _make_store()

        point_a = MagicMock()
        point_a.payload = {"document_id": "doc_001"}
        point_b = MagicMock()
        point_b.payload = {"document_id": "doc_002"}

        # Single scroll page, then done (offset=None)
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


class TestInsertSectionBatch:
    def test_upserts_to_section_collection(self):
        store = _make_store()
        records = [
            {
                "text_vector": [0.1] * 1024,
                "visual_vector": [0.2] * 1024,
                "document_id": "doc_001",
                "markdown": "## Question",
                "section_title": "doc_001",
                "section_level": 0,
                "page_range_start": 0,
                "page_range_end": 0,
                "metadata_json": "{}",
            }
        ]

        store.insert_section_batch(records)

        store.client.upsert.assert_called_once()
        call_kwargs = store.client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "exam_questions"
        points = call_kwargs["points"]
        assert len(points) == 1
        assert "text_vector" in points[0].vector
        assert "visual_vector" in points[0].vector
        assert points[0].payload["document_id"] == "doc_001"
        assert "text_vector" not in points[0].payload

    def test_skips_empty_batch(self):
        store = _make_store()
        store.insert_section_batch([])
        store.client.upsert.assert_not_called()


class TestInsertFigureBatch:
    def test_upserts_to_figures_collection(self):
        store = _make_store()
        records = [
            {
                "figure_visual_vector": [0.3] * 1024,
                "figure_text_vector": [0.4] * 1024,
                "description": "A graph",
                "page_number": 0,
                "document_id": "doc_001",
                "section_id": "sec_abc",
            }
        ]

        store.insert_figure_batch(records)

        store.client.upsert.assert_called_once()
        call_kwargs = store.client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "exam_questions_figures"
        points = call_kwargs["points"]
        assert "figure_visual_vector" in points[0].vector
        assert "figure_text_vector" in points[0].vector

    def test_skips_empty_batch(self):
        store = _make_store()
        store.insert_figure_batch([])
        store.client.upsert.assert_not_called()


class TestSearchSections:
    def test_searches_with_difficulty_filter(self):
        store = _make_store()
        hit = MagicMock()
        hit.score = 0.95
        hit.payload = {"document_id": "doc_001", "markdown": "content", "difficulty": "hard"}
        store.client.search.return_value = [hit]

        results = store.search_sections(vector=[0.1] * 1024, difficulty="hard", limit=3)

        call_kwargs = store.client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is not None
        assert len(results) == 1
        assert results[0]["score"] == 0.95

    def test_searches_without_filter_when_difficulty_is_none(self):
        store = _make_store()
        store.client.search.return_value = []

        store.search_sections(vector=[0.1] * 1024, difficulty=None)

        call_kwargs = store.client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    def test_returns_score_and_payload_fields(self):
        store = _make_store()
        hit = MagicMock()
        hit.score = 0.88
        hit.payload = {"document_id": "doc_002", "topic": "algebra"}
        store.client.search.return_value = [hit]

        results = store.search_sections(vector=[0.0] * 1024)

        assert results[0]["score"] == 0.88
        assert results[0]["document_id"] == "doc_002"
        assert results[0]["topic"] == "algebra"


class TestSearchSectionsHybrid:
    def test_fuses_text_and_visual_scores(self):
        store = _make_store()
        text_only = {"document_id": "doc_text", "section_title": "Text", "score": 0.9}
        both = {"document_id": "doc_both", "section_title": "Both", "score": 0.8}
        visual_both = {"document_id": "doc_both", "section_title": "Both", "score": 0.7}
        visual_only = {"document_id": "doc_visual", "section_title": "Visual", "score": 1.0}

        store._search_named_vector = MagicMock(side_effect=[
            [text_only, both],
            [visual_both, visual_only],
        ])

        results = store.search_sections_hybrid(
            text_vector=[0.1] * 4,
            visual_text_vector=[0.2] * 4,
            limit=3,
            alpha=0.5,
        )

        assert results[0]["document_id"] == "doc_both"
        assert results[0]["retrieval_mode"] == "hybrid"
        assert {r["document_id"] for r in results} == {"doc_both", "doc_text", "doc_visual"}
