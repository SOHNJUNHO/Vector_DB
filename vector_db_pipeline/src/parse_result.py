"""
parse_result.py — Shared data models for the document parser pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class FigureResult:
    figure_id: str
    page_number: int
    description: str          # VLM-generated; empty string if not enriched
    cropped_image_path: str   # path to Docling-cropped figure PNG


@dataclass
class SectionResult:
    section_id: str
    section_title: str
    section_level: int        # 0 = flat (single-image), 1 = H1, 2 = H2, 3 = H3
    page_range: tuple[int, int]
    markdown: str
    document_id: str
    metadata: dict = field(default_factory=dict)
    figures: list[FigureResult] = field(default_factory=list)


class ParserBackend(Protocol):
    """Contract that every parser backend in this repo must satisfy."""

    def parse_single(self, image_path: str, document_id: str) -> SectionResult:
        ...

    def parse_document(self, page_paths: list[str], document_id: str) -> list[SectionResult]:
        ...


def validate_section_result(section: SectionResult, expected_document_id: str) -> SectionResult:
    """
    Validate a parser result before it enters the embedding/storage pipeline.

    The contract is intentionally strict:
      - document_id must match the document currently being processed
      - page ranges must be non-negative and ordered
      - metadata must be a dict
      - figures must be FigureResult objects
    """
    if not isinstance(section, SectionResult):
        raise TypeError(f"Expected SectionResult, got {type(section).__name__}")

    if section.document_id != expected_document_id:
        raise ValueError(
            "Parser returned a section with mismatched document_id: "
            f"{section.document_id!r} != {expected_document_id!r}"
        )

    if not isinstance(section.section_title, str) or not section.section_title.strip():
        raise ValueError("SectionResult.section_title must be a non-empty string")

    if not isinstance(section.section_level, int) or section.section_level < 0:
        raise ValueError("SectionResult.section_level must be a non-negative int")

    if not isinstance(section.page_range, tuple) or len(section.page_range) != 2:
        raise ValueError("SectionResult.page_range must be a tuple(start, end)")
    start, end = section.page_range
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("SectionResult.page_range values must be ints")
    if start < 0 or end < 0 or start > end:
        raise ValueError("SectionResult.page_range must satisfy 0 <= start <= end")

    if not isinstance(section.markdown, str):
        raise ValueError("SectionResult.markdown must be a string")
    if not isinstance(section.metadata, dict):
        raise ValueError("SectionResult.metadata must be a dict")
    if not isinstance(section.figures, list):
        raise ValueError("SectionResult.figures must be a list")

    for fig in section.figures:
        if not isinstance(fig, FigureResult):
            raise ValueError("SectionResult.figures must contain FigureResult objects")

    return section
