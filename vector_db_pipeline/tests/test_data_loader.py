"""
tests/test_data_loader.py — Tests for data_loader.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import discover_pngs


class TestDiscoverPngs:
    def test_missing_png_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PNG directory not found"):
            list(discover_pngs(str(tmp_path)))

    def test_discovers_png_files(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "S3_고등_1_000001.png").write_bytes(b"fake-png")
        (png_dir / "S3_고등_1_000002.png").write_bytes(b"fake-png")

        results = list(discover_pngs(str(tmp_path)))
        assert len(results) == 2

        paths = [p for p, _ in results]
        ids = [i for _, i in results]

        assert any("000001.png" in p for p in paths)
        assert any("000002.png" in p for p in paths)
        assert "S3_고등_1_000001" in ids
        assert "S3_고등_1_000002" in ids

    def test_sorted_order(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "img_c.png").write_bytes(b"fake")
        (png_dir / "img_a.png").write_bytes(b"fake")
        (png_dir / "img_b.png").write_bytes(b"fake")

        results = list(discover_pngs(str(tmp_path)))
        ids = [i for _, i in results]
        assert ids == ["img_a", "img_b", "img_c"]

    def test_ignores_non_png(self, tmp_path):
        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "valid.png").write_bytes(b"fake")
        (png_dir / "not-a-json.json").write_text("{}")

        results = list(discover_pngs(str(tmp_path)))
        assert len(results) == 1
        assert results[0][1] == "valid"
