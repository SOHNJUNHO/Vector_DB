"""
qdrant_store.py — Qdrant vector store for exam questions.

Single collection with one text_vector per question (Cosine, 1024-dim)
and INT8 scalar quantization for storage efficiency.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)


class QdrantStore:
    """Wraps Qdrant connection for the exam_questions collection."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "exam_questions",
    ):
        self.collection_name = collection_name
        resolved_key = api_key or os.environ.get("QDRANT_API_KEY") or None
        self.client = QdrantClient(url=url, api_key=resolved_key)
        print(f"[Qdrant] Connected to {url}")

    def create_collection(self, text_dim: int = 1024):
        """Create collection with scalar quantization if it doesn't already exist."""
        if self.client.collection_exists(self.collection_name):
            print(f"[Qdrant] Collection '{self.collection_name}' already exists. Skipping creation.")
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=text_dim, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            ),
        )
        self._ensure_payload_indexes()
        print(f"[Qdrant] Collection '{self.collection_name}' created (INT8 scalar quantization).")

    def _ensure_payload_indexes(self):
        for field in ("document_id", "concepts"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def get_existing_ids(self) -> set[str]:
        """Return all document_id values already stored."""
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

    def insert_batch(self, records: list[dict[str, Any]]):
        """
        Upsert question records. Each record must contain:
          text_vector, document_id, text, concepts, description
        """
        if not records:
            return
        _internal_keys = {"text_vector", "point_id"}
        points = [
            PointStruct(
                id=record.get("point_id") or str(uuid.uuid4()),
                vector=record["text_vector"],
                payload={k: v for k, v in record.items() if k not in _internal_keys},
            )
            for record in records
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Batch upsert complete ({len(records)} records).")

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search on text_vector. Returns list of dicts with score + payload."""
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [{"score": hit.score, **(hit.payload or {})} for hit in hits]
