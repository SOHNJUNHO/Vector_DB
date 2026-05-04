"""
tests/test_document_parser.py — Tests for document_parser.py and vlm_client.py.
All tests are fully mocked — no model downloads or Ollama required.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_vlm_client(response_text: str = "{}") -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
    return client


# ---------------------------------------------------------------------------
# vlm_client.parse_json_object tests
# ---------------------------------------------------------------------------

class TestParseJsonObject:
    def test_extracts_fenced_json(self):
        from vlm_client import parse_json_object

        text = 'Some text\n```json\n{"key": "value"}\n```'
        assert parse_json_object(text) == {"key": "value"}

    def test_extracts_bare_json(self):
        from vlm_client import parse_json_object

        text = 'Here is the result: {"text": "문제", "concepts": ["이차방정식"]}'
        result = parse_json_object(text)
        assert result["text"] == "문제"
        assert result["concepts"] == ["이차방정식"]

    def test_returns_empty_on_failure(self):
        from vlm_client import parse_json_object

        assert parse_json_object("no json here") == {}
        assert parse_json_object("") == {}


# ---------------------------------------------------------------------------
# VlmClient tests
# ---------------------------------------------------------------------------

class TestVlmClient:
    def _make_client(self, response: str) -> tuple:
        from vlm_client import VlmClient

        raw = _make_vlm_client(response)
        vlm = VlmClient(raw, "model")
        return vlm, raw

    def test_transcribe_page_returns_parsed_dict(self, tmp_path):
        from unittest.mock import patch

        json_response = '{"text": "문제입니다", "concepts": ["이차방정식"], "description": "수식이 포함된 문제"}'
        vlm, _ = self._make_client(json_response)

        png = tmp_path / "test.png"
        png.write_bytes(b"fake")

        with patch("vlm_client.encode_image_to_base64", return_value="b64"):
            result = vlm.transcribe_page(str(png))

        assert result["text"] == "문제입니다"
        assert result["concepts"] == ["이차방정식"]
        assert result["description"] == "수식이 포함된 문제"

    def test_transcribe_page_returns_empty_dict_on_bad_json(self, tmp_path):
        from unittest.mock import patch

        vlm, _ = self._make_client("not valid json")

        png = tmp_path / "test.png"
        png.write_bytes(b"fake")

        with patch("vlm_client.encode_image_to_base64", return_value="b64"):
            result = vlm.transcribe_page(str(png))

        assert result == {}

    def test_text_call_returns_string(self):
        vlm, raw = self._make_client("OK")
        result = vlm.text_call("Say OK")
        assert result == "OK"


# ---------------------------------------------------------------------------
# DocumentParser tests
# ---------------------------------------------------------------------------

class TestDocumentParser:
    def _make_parser(self, vlm_response: str):
        from document_parser import DocumentParser
        from vlm_client import VlmClient
        from unittest.mock import patch

        raw = _make_vlm_client(vlm_response)
        vlm = VlmClient(raw, "model")
        return DocumentParser(vlm), raw

    def test_parse_single_returns_question_result(self, tmp_path):
        from unittest.mock import patch

        json_resp = '{"text": "이차방정식을 풀어라", "concepts": ["이차방정식", "판별식"], "description": "수식 문제"}'
        parser, _ = self._make_parser(json_resp)

        png = tmp_path / "q001.png"
        png.write_bytes(b"fake")

        with patch("vlm_client.encode_image_to_base64", return_value="b64"):
            result = parser.parse_single(str(png), "q001")

        assert result.document_id == "q001"
        assert result.text == "이차방정식을 풀어라"
        assert result.concepts == ["이차방정식", "판별식"]
        assert result.description == "수식 문제"

    def test_parse_single_handles_empty_vlm_response(self, tmp_path):
        from unittest.mock import patch

        parser, _ = self._make_parser("not json")

        png = tmp_path / "q002.png"
        png.write_bytes(b"fake")

        with patch("vlm_client.encode_image_to_base64", return_value="b64"):
            result = parser.parse_single(str(png), "q002")

        assert result.document_id == "q002"
        assert result.text == ""
        assert result.concepts == []
        assert result.description == ""

    def test_parse_single_makes_exactly_one_vlm_call(self, tmp_path):
        from unittest.mock import patch

        json_resp = '{"text": "test", "concepts": [], "description": "test"}'
        parser, raw = self._make_parser(json_resp)

        png = tmp_path / "q003.png"
        png.write_bytes(b"fake")

        with patch("vlm_client.encode_image_to_base64", return_value="b64"):
            parser.parse_single(str(png), "q003")

        assert raw.chat.completions.create.call_count == 1
