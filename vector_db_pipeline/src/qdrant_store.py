"""
qdrant_store.py — Qdrant Cloud vector store for Section and Figure collections.

Manages two collections:
  {name}         (Section parent) — text_vector + visual_vector
  {name}_figures (Figure child)   — figure_visual_vector + figure_text_vector
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


class QdrantStore:
    """Wraps Qdrant connection and operations for Section + Figure collections."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "exam_questions",
    ):
        self.collection_name = collection_name
        self.figure_collection_name = f"{collection_name}_figures"

        resolved_key = api_key or os.environ.get("QDRANT_API_KEY") or None
        self.client = QdrantClient(url=url, api_key=resolved_key)
        print(f"[Qdrant] Connected to {url}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_collections(self, visual_dim: int = 1024, text_dim: int = 1024):
        """Create Section and Figure collections if they don't already exist."""
        self._create_section_collection(visual_dim, text_dim)
        self._create_figure_collection(visual_dim, text_dim)
        self._ensure_payload_indexes()

    def _create_section_collection(self, visual_dim: int, text_dim: int):
        if self.client.collection_exists(self.collection_name):
            print(
                f"[Qdrant] Collection '{self.collection_name}' already exists. "
                "Skipping creation."
            )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_vector": VectorParams(size=text_dim, distance=Distance.COSINE),
                "visual_vector": VectorParams(size=visual_dim, distance=Distance.COSINE),
            },
        )
        print(f"[Qdrant] Collection '{self.collection_name}' created.")

    def _ensure_payload_indexes(self):
        """Create payload indexes required for filtered search. Idempotent."""
        for field in ("difficulty", "topic", "document_id"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def _create_figure_collection(self, visual_dim: int, text_dim: int):
        if self.client.collection_exists(self.figure_collection_name):
            print(
                f"[Qdrant] Collection '{self.figure_collection_name}' already exists. "
                "Skipping creation."
            )
            return
        self.client.create_collection(
            collection_name=self.figure_collection_name,
            vectors_config={
                "figure_visual_vector": VectorParams(size=visual_dim, distance=Distance.COSINE),
                "figure_text_vector": VectorParams(size=text_dim, distance=Distance.COSINE),
            },
        )
        print(f"[Qdrant] Collection '{self.figure_collection_name}' created.")

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def get_existing_ids(self) -> set[str]:
        """Return all document_id values already stored in the Section collection."""
        existing: set[str] = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["document_id"],
                limit=1000,
                offset=offset,
            )
            for point in results:
                did = (point.payload or {}).get("document_id")
                if did:
                    existing.add(did)
            if offset is None:
                break
        return existing

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert_section_batch(self, records: list[dict[str, Any]]):
        """
        Upsert Section records. Each record must contain:
          text_vector, visual_vector, section_title, section_level,
          page_range_start, page_range_end, document_id, markdown, metadata_json
          (+ any flattened metadata fields: difficulty, topic, has_math, has_diagram)
        """
        if not records:
            return
        _vector_keys = {"text_vector", "visual_vector"}
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={k: record[k] for k in _vector_keys if k in record},
                payload={k: v for k, v in record.items() if k not in _vector_keys},
            )
            for record in records
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Section batch upsert complete ({len(records)} records).")

    def insert_figure_batch(self, records: list[dict[str, Any]]):
        """
        Upsert Figure records. Each record must contain:
          figure_visual_vector, figure_text_vector,
          description, page_number, document_id, section_id
        """
        if not records:
            return
        _vector_keys = {"figure_visual_vector", "figure_text_vector"}
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={k: record[k] for k in _vector_keys if k in record},
                payload={k: v for k, v in record.items() if k not in _vector_keys},
            )
            for record in records
        ]
        self.client.upsert(collection_name=self.figure_collection_name, points=points)
        print(f"[Qdrant] Figure batch upsert complete ({len(records)} records).")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _difficulty_filter(self, difficulty: str | None):
        if not difficulty:
            return None
        return Filter(
            must=[FieldCondition(key="difficulty", match=MatchValue(value=difficulty))]
        )

    def _search_named_vector(
        self,
        vector_name: str,
        vector: list[float],
        *,
        difficulty: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        query_filter = self._difficulty_filter(difficulty)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "score": hit.score,
                **(hit.payload or {}),
            }
            for hit in response.points
        ]

    @staticmethod
    def _result_key(result: dict) -> tuple:
        return (
            result.get("document_id", ""),
            result.get("section_title", ""),
            result.get("page_range_start"),
            result.get("page_range_end"),
        )

    def search_sections(
        self,
        vector: list[float],
        difficulty: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Semantic search on text_vector with optional difficulty payload filter.

        Returns list of dicts with section payload fields + similarity score.
        """
        return self._search_named_vector(
            "text_vector",
            vector,
            difficulty=difficulty,
            limit=limit,
        )

    def search_sections_hybrid(
        self,
        *,
        text_vector: list[float],
        visual_text_vector: list[float],
        difficulty: str | None = None,
        limit: int = 5,
        alpha: float = 0.7,
    ) -> list[dict]:
        """
        Hybrid search over text_vector and visual_vector.

        `visual_text_vector` is produced by embedding the text query in the
        multimodal visual-text embedding space.
        """
        text_results = self._search_named_vector(
            "text_vector",
            text_vector,
            difficulty=difficulty,
            limit=limit,
        )
        visual_results = self._search_named_vector(
            "visual_vector",
            visual_text_vector,
            difficulty=difficulty,
            limit=limit,
        )

        max_text_score = max((r["score"] for r in text_results), default=1.0)
        max_visual_score = max((r["score"] for r in visual_results), default=1.0)

        merged: dict[tuple, dict] = {}

        for result in text_results:
            key = self._result_key(result)
            merged[key] = {
                **result,
                "text_score": result["score"],
                "visual_score": 0.0,
            }

        for result in visual_results:
            key = self._result_key(result)
            if key not in merged:
                merged[key] = {
                    **result,
                    "text_score": 0.0,
                    "visual_score": result["score"],
                }
            else:
                merged[key]["visual_score"] = result["score"]

        fused = []
        for result in merged.values():
            text_score = result["text_score"] / max_text_score if max_text_score else 0.0
            visual_score = (
                result["visual_score"] / max_visual_score if max_visual_score else 0.0
            )
            fused_score = alpha * text_score + (1.0 - alpha) * visual_score
            fused.append({
                **result,
                "score": fused_score,
                "retrieval_mode": "hybrid",
            })

        fused.sort(key=lambda item: item["score"], reverse=True)
        return fused[:limit]
