"""
artifacts.py — Per-document ingestion status tracking.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .parse_result import QuestionResult
except ImportError:
    from parse_result import QuestionResult


class ArtifactStore:
    """Writes question results and manifest state under data/artifacts/{document_id}/."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def document_dir(self, document_id: str) -> Path:
        return self.root_dir / document_id

    def manifest_path(self, document_id: str) -> Path:
        return self.document_dir(document_id) / "manifest.json"

    def read_manifest(self, document_id: str) -> dict[str, Any]:
        path = self.manifest_path(document_id)
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def write_manifest(
        self,
        document_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        doc_dir = self.document_dir(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"document_id": document_id, "status": status}
        if error:
            payload["error"] = error
        with self.manifest_path(document_id).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    def is_stored(self, document_id: str) -> bool:
        return self.read_manifest(document_id).get("status") == "stored"

    def save_result(self, result: QuestionResult) -> None:
        """Save a QuestionResult to disk for debugging and inspection."""
        doc_dir = self.document_dir(result.document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        with (doc_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2, sort_keys=True)
