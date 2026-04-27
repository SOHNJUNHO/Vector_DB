"""
parse_result.py — Shared data models for the document parser pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Protocol

_PARSER_VERSION = "v1"


def deterministic_id(*parts: object, prefix: str = "") -> str:
    """Return a stable short ID for parser artifacts and Qdrant point IDs."""
    raw = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{digest}" if prefix else digest


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
    parser_backend: str = ""


@dataclass
class ParsedPage:
    page_index: int
    image_path: str
    markdown: str
    parser_backend: str
    quality: float = 1.0


@dataclass
class ParsedDocument:
    document_id: str
    pages: list[ParsedPage]
    chunks: list[SectionResult]
    figures: list[FigureResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ParserBackend(Protocol):
    """Contract that every parser backend in this repo must satisfy."""

    def parse_single(self, image_path: str, document_id: str) -> SectionResult:
        ...

    def parse_document(self, page_paths: list[str], document_id: str) -> list[SectionResult]:
        ...

    def parse_parsed_document(
        self,
        page_paths: list[str],
        document_id: str,
    ) -> ParsedDocument:
        ...


def make_section_result(
    *,
    document_id: str,
    section_title: str,
    section_level: int,
    page_range: tuple[int, int],
    markdown: str,
    metadata: dict | None = None,
    figures: list[FigureResult] | None = None,
    parser_backend: str = "",
    chunk_index: int = 0,
) -> SectionResult:
    section_id = deterministic_id(
        document_id,
        page_range[0],
        page_range[1],
        chunk_index,
        parser_backend,
        _PARSER_VERSION,
        prefix="sec_",
    )
    return SectionResult(
        section_id=section_id,
        section_title=section_title,
        section_level=section_level,
        page_range=page_range,
        markdown=markdown,
        document_id=document_id,
        metadata=metadata or {},
        figures=figures or [],
        parser_backend=parser_backend,
    )


def sections_to_parsed_document(
    *,
    document_id: str,
    page_paths: list[str],
    sections: list[SectionResult],
    parser_backend: str,
) -> ParsedDocument:
    pages = [
        ParsedPage(
            page_index=i,
            image_path=path,
            markdown="",
            parser_backend=parser_backend,
        )
        for i, path in enumerate(page_paths)
    ]
    figures = [fig for section in sections for fig in section.figures]
    return ParsedDocument(
        document_id=document_id,
        pages=pages,
        chunks=sections,
        figures=figures,
        metadata={"parser_backend": parser_backend},
    )


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
