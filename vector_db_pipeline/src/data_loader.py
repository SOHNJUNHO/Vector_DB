"""
data_loader.py — Discover PNG images / documents for processing.

Two modes, auto-detected from the png/ directory structure:
  Flat mode     : png/*.png          → each PNG is one Document (single page)
  Document mode : png/<name>/*.png   → each subdirectory is one Document (multi-page)

GCS support: pass raw_dir as "gs://bucket-name[/optional/prefix]" to automatically
download the png/ folder to /tmp before discovery. Used by Cloud Run Job.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    page_paths: list[str]   # ordered; sorted alphabetically within each doc
    is_single_page: bool


def _download_from_gcs(gcs_uri: str) -> str:
    """Download png/ folder from a GCS URI to a local temp directory. Returns local path."""
    from google.cloud import storage

    # Parse gs://bucket-name[/optional/prefix]
    path = gcs_uri[5:]  # strip "gs://"
    bucket_name, _, object_prefix = path.partition("/")

    local_dir = tempfile.mkdtemp(prefix="gcs_data_")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    png_gcs_prefix = f"{object_prefix}/png/" if object_prefix else "png/"
    blobs = list(bucket.list_blobs(prefix=png_gcs_prefix))

    if not blobs:
        raise FileNotFoundError(f"No files found at {gcs_uri}/png/")

    print(f"  Downloading {len(blobs)} file(s) from {gcs_uri}/png/ ...")
    for blob in blobs:
        relative = blob.name[len(object_prefix):].lstrip("/") if object_prefix else blob.name
        local_path = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

    return local_dir


def discover_documents(raw_dir: str) -> list[Document]:
    """
    Auto-detect flat vs. document mode and return an ordered list of Documents.

    Flat mode    : png/ contains only .png files
    Document mode: png/ contains subdirectories (each subdir = one multi-page doc,
                   pages sorted alphabetically — name files 001.png, 002.png, ...)

    If raw_dir starts with "gs://", the png/ folder is downloaded from GCS first.
    """
    if raw_dir.startswith("gs://"):
        raw_dir = _download_from_gcs(raw_dir)

    raw_dir = Path(raw_dir)
    png_dir = raw_dir / "png"

    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")

    subdirs = sorted(p for p in png_dir.iterdir() if p.is_dir())

    if subdirs:
        # Document mode: each subdir is one document
        documents = []
        for subdir in subdirs:
            pages = sorted(subdir.glob("*.png"))
            if not pages:
                continue
            documents.append(
                Document(
                    doc_id=subdir.name,
                    page_paths=[str(p) for p in pages],
                    is_single_page=len(pages) == 1,
                )
            )
        print(f"  Found {len(documents)} document(s) in {png_dir} [document mode]")
        return documents

    # Flat mode: each PNG is its own document
    png_files = sorted(png_dir.glob("*.png"))
    documents = [
        Document(doc_id=p.stem, page_paths=[str(p)], is_single_page=True)
        for p in png_files
    ]
    print(f"  Found {len(documents)} PNG file(s) in {png_dir} [flat mode]")
    return documents


# ---------------------------------------------------------------------------
# Legacy helper — kept for backward compatibility with existing callers
# ---------------------------------------------------------------------------

def discover_pngs(raw_dir: str):
    """
    Walk through the `png/` directory and yield paths of all PNG files.
    Yields (png_path, image_id) tuples.
    image_id is the filename stem (e.g. "S3_고등_1_006421").
    """
    raw_dir = Path(raw_dir)
    png_dir = raw_dir / "png"

    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")

    png_files = sorted(png_dir.glob("*.png"))
    print(f"  Found {len(png_files)} PNG files in {png_dir}")

    for png_path in png_files:
        yield str(png_path), png_path.stem


# ---------------------------------------------------------------------------
# PDF stitching utility
# ---------------------------------------------------------------------------

def pngs_to_pdf(page_paths: list[str], output_path: str) -> str:
    """Stitch ordered PNGs into a multi-page PDF using Pillow."""
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in page_paths]
    images[0].save(output_path, save_all=True, append_images=images[1:])
    return output_path
