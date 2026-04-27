"""
tests/test_document_parser.py — Tests for document_parser.py and section_detector.py.
All tests are fully mocked — no model downloads or Docling installations required.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vlm_client(response_text: str = "{}") -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
    return client


def _make_docling_result(
    markdown: str = "## Text\nSome content.",
    n_pictures: int = 0,
    quality: float = 1.0,
) -> MagicMock:
    result = MagicMock()
    result.document.export_to_markdown.return_value = markdown
    result.document.pictures = [MagicMock() for _ in range(n_pictures)]
    result.document.pages = [MagicMock()]
    result.confidence = None
    return result


# ---------------------------------------------------------------------------
# section_detector tests
# ---------------------------------------------------------------------------

class TestSectionDetector:
    def test_new_section_detected(self):
        from section_detector import SectionDetector

        client = _make_vlm_client(
            '{"is_new_section": true, "section_title": "Chapter 1", "section_level": 1}'
        )
        detector = SectionDetector(client, "model", {})

        with patch("section_detector.encode_image_to_base64", return_value="b64"):
            result = detector.detect("page.png", "some text")

        assert result.is_new_section is True
        assert result.title == "Chapter 1"
        assert result.level == 1

    def test_continuation_detected(self):
        from section_detector import SectionDetector

        client = _make_vlm_client(
            '{"is_new_section": false, "section_title": null, "section_level": null}'
        )
        detector = SectionDetector(client, "model", {})

        with patch("section_detector.encode_image_to_base64", return_value="b64"):
            result = detector.detect("page.png", "continued text")

        assert result.is_new_section is False
        assert result.title is None

    def test_malformed_json_falls_back(self):
        from section_detector import SectionDetector

        client = _make_vlm_client("not json at all")
        detector = SectionDetector(client, "model", {})

        with patch("section_detector.encode_image_to_base64", return_value="b64"):
            result = detector.detect("page.png", "text")

        assert result.is_new_section is False

    def test_uses_configured_prompt(self):
        from section_detector import SectionDetector

        client = _make_vlm_client('{"is_new_section": false}')
        custom_prompt = "Custom prompt for this project."
        detector = SectionDetector(client, "model", {"section_detection_prompt": custom_prompt})

        with patch("section_detector.encode_image_to_base64", return_value="b64"):
            detector.detect("page.png", "text")

        call_args = client.chat.completions.create.call_args
        user_content = call_args[1]["messages"][0]["content"]
        text_parts = [c["text"] for c in user_content if c.get("type") == "text"]
        assert any(custom_prompt in t for t in text_parts)


# ---------------------------------------------------------------------------
# document_parser._parse_json tests
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_extracts_fenced_json(self):
        from document_parser import _parse_json

        text = 'Some text\n```json\n{"key": "value"}\n```'
        assert _parse_json(text) == {"key": "value"}

    def test_extracts_bare_json(self):
        from document_parser import _parse_json

        text = 'Here is the result: {"subject": "math", "difficulty": "hard"}'
        result = _parse_json(text)
        assert result["subject"] == "math"

    def test_returns_empty_on_failure(self):
        from document_parser import _parse_json

        assert _parse_json("no json here") == {}
        assert _parse_json("") == {}


# ---------------------------------------------------------------------------
# document_parser._is_meaningful_figure tests
# ---------------------------------------------------------------------------

def _mock_pil(size: tuple[int, int]):
    """Return a sys.modules patch dict that stubs PIL.Image.open with the given size."""
    mock_img = MagicMock()
    mock_img.__enter__ = MagicMock(return_value=mock_img)
    mock_img.__exit__ = MagicMock(return_value=False)
    mock_img.size = size
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = mock_img
    return {"PIL": mock_pil, "PIL.Image": mock_pil.Image}


class TestIsMeaningfulFigure:
    def test_rejects_tiny_figure(self, tmp_path):
        from document_parser import _is_meaningful_figure

        tiny = tmp_path / "tiny.png"
        tiny.write_bytes(b"fake")
        with patch.dict(sys.modules, _mock_pil((30, 30))):
            assert _is_meaningful_figure(str(tiny)) is False

    def test_rejects_extreme_aspect_ratio(self, tmp_path):
        from document_parser import _is_meaningful_figure

        thin = tmp_path / "thin.png"
        thin.write_bytes(b"fake")
        with patch.dict(sys.modules, _mock_pil((1000, 10))):
            assert _is_meaningful_figure(str(thin)) is False

    def test_accepts_normal_figure(self, tmp_path):
        from document_parser import _is_meaningful_figure

        normal = tmp_path / "figure.png"
        normal.write_bytes(b"fake")
        with patch.dict(sys.modules, _mock_pil((400, 300))):
            assert _is_meaningful_figure(str(normal)) is True


# ---------------------------------------------------------------------------
# DocumentParser.parse_single tests
# ---------------------------------------------------------------------------

class TestParseSingle:
    def _make_parser(self, vlm_response: str = '{"topic": "math"}', n_pictures: int = 0):
        from document_parser import DocumentParser

        client = _make_vlm_client(vlm_response)
        cfg = {
            "complexity_triggers": ["figures"],
            "quality_threshold": 0.5,
            "temperature": 0.1,
            "max_tokens": 512,
        }
        parser = DocumentParser(client, "model", cfg)
        docling_result = _make_docling_result(
            markdown="## Text\nContent here.", n_pictures=n_pictures
        )
        parser._converter = MagicMock()
        parser._converter.convert.return_value = docling_result
        return parser, client

    def test_text_only_no_vlm_enrichment(self, tmp_path):
        """No figures → _enrich should return Docling markdown without a VLM image call."""
        parser, client = self._make_parser(n_pictures=0)
        png = tmp_path / "test.png"
        png.write_bytes(b"fake")

        with patch("document_parser._save_figure_crops", return_value=[]):
            result = parser.parse_single(str(png), "doc_001")

        assert result.section_level == 0
        assert result.document_id == "doc_001"
        # VLM called once only — for metadata (text-only call)
        assert client.chat.completions.create.call_count == 1
        text_call = client.chat.completions.create.call_args
        messages = text_call[1]["messages"]
        # Text-only metadata call has no image_url content
        assert isinstance(messages[0]["content"], str)

    def test_with_figure_triggers_vlm_enrichment(self, tmp_path):
        """Figure detected → VLM enrichment call (image call) + metadata call."""
        parser, client = self._make_parser(n_pictures=1)
        png = tmp_path / "test.png"
        png.write_bytes(b"fake")
        fig_path = tmp_path / "fig.png"
        fig_path.write_bytes(b"fake")

        with (
            patch("document_parser._save_figure_crops", return_value=[str(fig_path)]),
            patch("document_parser._is_meaningful_figure", return_value=True),
            patch("document_parser.encode_image_to_base64", return_value="b64"),
        ):
            result = parser.parse_single(str(png), "doc_001")

        assert len(result.figures) == 1
        # At minimum 2 VLM calls: enrichment (image) + figure description (image) + metadata (text)
        assert client.chat.completions.create.call_count >= 2

    def test_returns_section_result_fields(self, tmp_path):
        parser, _ = self._make_parser()
        png = tmp_path / "test.png"
        png.write_bytes(b"fake")

        with patch("document_parser._save_figure_crops", return_value=[]):
            result = parser.parse_single(str(png), "my_doc")

        assert result.section_title == "my_doc"
        assert result.section_level == 0
        assert result.page_range == (0, 0)
        assert isinstance(result.section_id, str)
        assert isinstance(result.metadata, dict)

    def test_metadata_retries_after_invalid_json(self, tmp_path):
        parser, client = self._make_parser(vlm_response="not json")
        png = tmp_path / "test.png"
        png.write_bytes(b"fake")

        retry_msg = MagicMock()
        retry_msg.content = '{"topic": "algebra"}'
        client.chat.completions.create.return_value.choices = [MagicMock(message=retry_msg)]
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="not json"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"topic": "algebra"}'))]),
        ]

        with patch("document_parser._save_figure_crops", return_value=[]):
            result = parser.parse_single(str(png), "doc_001")

        assert result.metadata == {"topic": "algebra"}
        assert client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# DocumentParser.parse_document tests
# ---------------------------------------------------------------------------

class TestParseDocument:
    def _make_multi_parser(self, boundaries):
        """boundaries: list of (is_new_section, title, level) tuples."""
        from document_parser import DocumentParser
        from section_detector import SectionBoundary

        boundary_objects = [
            SectionBoundary(is_new_section=b[0], title=b[1], level=b[2])
            for b in boundaries
        ]

        client = _make_vlm_client('{"topic": "test"}')
        cfg = {
            "complexity_triggers": ["figures"],
            "quality_threshold": 0.5,
            "temperature": 0.1,
            "max_tokens": 512,
        }
        parser = DocumentParser(client, "model", cfg)

        docling_result = _make_docling_result(markdown="Page content.", n_pictures=0)
        docling_result.document.pages = [MagicMock() for _ in boundary_objects]
        docling_result.document.export_to_markdown.return_value = "Page content."
        parser._converter = MagicMock()
        parser._converter.convert.return_value = docling_result

        return parser, boundary_objects

    def test_four_pages_two_sections(self, tmp_path):
        """Pages 0-1 → Section A, pages 2-3 → Section B."""
        parser, boundaries = self._make_multi_parser([
            (True, "Section A", 1),
            (False, None, None),
            (True, "Section B", 1),
            (False, None, None),
        ])
        page_paths = [str(tmp_path / f"p{i}.png") for i in range(4)]
        for p in page_paths:
            open(p, "wb").close()

        with (
            patch("document_parser._pngs_to_pdf"),
            patch("document_parser._save_figure_crops", return_value=[]),
            patch.object(parser, "_detect_boundaries", return_value=boundaries),
            patch("document_parser.encode_image_to_base64", return_value="b64"),
        ):
            sections = parser.parse_document(page_paths, "finance_report")

        assert len(sections) == 2
        assert sections[0].section_title == "Section A"
        assert sections[0].page_range == (0, 1)
        assert sections[1].section_title == "Section B"
        assert sections[1].page_range == (2, 3)

    def test_single_section_when_no_new_section_detected(self, tmp_path):
        """All pages → continuation → one section."""
        parser, boundaries = self._make_multi_parser([
            (True, "Only Section", 1),
            (False, None, None),
            (False, None, None),
        ])
        page_paths = [str(tmp_path / f"p{i}.png") for i in range(3)]
        for p in page_paths:
            open(p, "wb").close()

        with (
            patch("document_parser._pngs_to_pdf"),
            patch("document_parser._save_figure_crops", return_value=[]),
            patch.object(parser, "_detect_boundaries", return_value=boundaries),
            patch("document_parser.encode_image_to_base64", return_value="b64"),
        ):
            sections = parser.parse_document(page_paths, "doc")

        assert len(sections) == 1
        assert sections[0].page_range == (0, 2)


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------

class TestDiscoverDocuments:
    def test_flat_mode(self, tmp_path):
        from data_loader import discover_documents

        png_dir = tmp_path / "png"
        png_dir.mkdir()
        (png_dir / "img_a.png").write_bytes(b"fake")
        (png_dir / "img_b.png").write_bytes(b"fake")

        docs = discover_documents(str(tmp_path))
        assert len(docs) == 2
        assert all(d.is_single_page for d in docs)
        assert [d.doc_id for d in docs] == ["img_a", "img_b"]

    def test_document_mode(self, tmp_path):
        from data_loader import discover_documents

        png_dir = tmp_path / "png"
        doc_dir = png_dir / "report_2024"
        doc_dir.mkdir(parents=True)
        (doc_dir / "001.png").write_bytes(b"fake")
        (doc_dir / "002.png").write_bytes(b"fake")

        docs = discover_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].doc_id == "report_2024"
        assert len(docs[0].page_paths) == 2
        assert not docs[0].is_single_page

    def test_missing_png_dir_raises(self, tmp_path):
        from data_loader import discover_documents

        with pytest.raises(FileNotFoundError):
            discover_documents(str(tmp_path))
