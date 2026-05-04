"""
document_parser.py — Pure VLM document parser (Qwen via Ollama).

One VLM call per PNG → QuestionResult with text, concepts, description.
"""

from __future__ import annotations

import logging

try:
    from .parse_result import QuestionResult
    from .vlm_client import VlmClient
except ImportError:
    from parse_result import QuestionResult
    from vlm_client import VlmClient

logger = logging.getLogger(__name__)


class DocumentParser:
    """Pure VLM parser: one image call per PNG → QuestionResult."""

    def __init__(self, vlm_client: VlmClient):
        self._vlm = vlm_client

    def parse_single(self, image_path: str, document_id: str) -> QuestionResult:
        """Parse a single PNG via one VLM call. Returns a QuestionResult."""
        data = self._vlm.transcribe_page(image_path)
        if not data:
            logger.warning("VLM returned empty response for %s", document_id)
        return QuestionResult(
            document_id=document_id,
            text=data.get("text", ""),
            concepts=data.get("concepts", []),
            description=data.get("description", ""),
        )
