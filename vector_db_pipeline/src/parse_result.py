"""
parse_result.py — Data model for the VLM-based exam question parser.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def deterministic_id(*parts: object, prefix: str = "") -> str:
    """Return a stable short ID for Qdrant point IDs."""
    raw = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{digest}" if prefix else digest


@dataclass
class QuestionResult:
    document_id: str
    text: str           # full transcription (Korean + LaTeX)
    concepts: list[str] # e.g. ["이차방정식", "판별식"]
    description: str    # natural language description of image contents
