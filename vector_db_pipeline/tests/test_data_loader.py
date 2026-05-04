"""
tests/test_data_loader.py — Tests for data_loader.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import discover_documents


class TestDiscoverDocuments:
    def test_missing_png_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PNG directory not found"):
            discover_documents(str(tmp_path))

    def test_discovers_png_files(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "S3_고등_1_000001.png").write_bytes(b"fake-png")
        (png_dir / "S3_고등_1_000002.png").write_bytes(b"fake-png")

        docs = discover_documents(str(tmp_path))

        assert len(docs) == 2
        ids = [d.doc_id for d in docs]
        assert "S3_고등_1_000001" in ids
        assert "S3_고등_1_000002" in ids

    def test_sorted_order(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "img_c.png").write_bytes(b"fake")
        (png_dir / "img_a.png").write_bytes(b"fake")
        (png_dir / "img_b.png").write_bytes(b"fake")

        docs = discover_documents(str(tmp_path))
        ids = [d.doc_id for d in docs]

        assert ids == ["img_a", "img_b", "img_c"]

    def test_ignores_non_png(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "valid.png").write_bytes(b"fake")
        (png_dir / "not-a-png.json").write_text("{}")

        docs = discover_documents(str(tmp_path))

        assert len(docs) == 1
        assert docs[0].doc_id == "valid"

    def test_each_document_has_one_page(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "q001.png").write_bytes(b"fake")
        (png_dir / "q002.png").write_bytes(b"fake")

        docs = discover_documents(str(tmp_path))

        for doc in docs:
            assert len(doc.page_paths) == 1
            assert doc.page_paths[0].endswith(".png")
