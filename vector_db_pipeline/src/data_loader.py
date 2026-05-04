"""
data_loader.py — Discover PNG images for processing.

Flat mode only: png/*.png → each PNG is one Document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    page_paths: list[str]


def discover_documents(raw_dir: str) -> list[Document]:
    """Find all PNGs in {raw_dir}/png/ and return them as Documents."""
    png_dir = Path(raw_dir) / "png"

    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")

    png_files = sorted(png_dir.glob("*.png"))
    documents = [
        Document(doc_id=p.stem, page_paths=[str(p)])
        for p in png_files
    ]
    print(f"  Found {len(documents)} PNG file(s) in {png_dir}")
    return documents
