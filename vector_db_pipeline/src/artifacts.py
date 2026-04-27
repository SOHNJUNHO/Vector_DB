"""
artifacts.py — Stable per-document ingestion artifacts and manifest state.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .parse_result import ParsedDocument
except ImportError:
    from parse_result import ParsedDocument


class ArtifactStore:
    """Writes parser outputs under data/artifacts/{document_id}/."""

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
        extra: dict[str, Any] | None = None,
    ) -> None:
        doc_dir = self.document_dir(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"document_id": document_id, "status": status}
        if error:
            payload["error"] = error
        if extra:
            payload.update(extra)
        with self.manifest_path(document_id).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    def is_stored(self, document_id: str) -> bool:
        return self.read_manifest(document_id).get("status") == "stored"

    def save_parsed_document(self, parsed: ParsedDocument) -> None:
        doc_dir = self.document_dir(parsed.document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        with (doc_dir / "parsed_document.json").open("w", encoding="utf-8") as f:
            json.dump(parsed.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

        chunks_path = doc_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in parsed.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True))
                f.write("\n")

        markdown = "\n\n".join(chunk.markdown for chunk in parsed.chunks)
        with (doc_dir / "markdown.md").open("w", encoding="utf-8") as f:
            f.write(markdown)

        for benchmark in parsed.artifacts.get("benchmark_documents", []):
            backend = benchmark.get("metadata", {}).get("parser_backend", "benchmark")
            backend_dir = doc_dir / "benchmark" / backend
            backend_dir.mkdir(parents=True, exist_ok=True)
            with (backend_dir / "parsed_document.json").open("w", encoding="utf-8") as f:
                json.dump(benchmark, f, ensure_ascii=False, indent=2, sort_keys=True)
            benchmark_markdown = "\n\n".join(
                chunk.get("markdown", "") for chunk in benchmark.get("chunks", [])
            )
            with (backend_dir / "markdown.md").open("w", encoding="utf-8") as f:
                f.write(benchmark_markdown)

        figures_dir = doc_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        for fig in parsed.figures:
            if not fig.cropped_image_path or not os.path.exists(fig.cropped_image_path):
                continue
            suffix = Path(fig.cropped_image_path).suffix or ".png"
            target = figures_dir / f"{fig.figure_id}{suffix}"
            if Path(fig.cropped_image_path) != target:
                shutil.copyfile(fig.cropped_image_path, target)
                fig.cropped_image_path = str(target)
