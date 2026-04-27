"""
tests/test_vlm_generate.py — Tests for vlm_generate.py
"""

import base64
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlm_generate import SYSTEM_PROMPT, encode_image_to_base64


class TestEncodeImageToBase64:
    def test_encodes_png_file(self, tmp_path):
        # Create a fake PNG (just bytes, doesn't need to be valid)
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-image-data")

        result = encode_image_to_base64(str(png_path))
        decoded = base64.b64decode(result)

        assert decoded == b"\x89PNG\r\n\x1a\n" + b"fake-image-data"

    def test_returns_string(self, tmp_path):
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"data")

        result = encode_image_to_base64(str(png_path))
        assert isinstance(result, str)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/file.png")


class TestSystemPrompt:
    def test_contains_required_sections(self):
        """System prompt must mention the Korean section headers."""
        assert "## 문제" in SYSTEM_PROMPT
        assert "## 보기" in SYSTEM_PROMPT
        assert "## 정답" in SYSTEM_PROMPT
        assert "## 해설" in SYSTEM_PROMPT

    def test_mentions_latex(self):
        assert "$" in SYSTEM_PROMPT
        assert "LaTeX" in SYSTEM_PROMPT

    def test_mentions_korean(self):
        assert "Korean" in SYSTEM_PROMPT


class TestInitVlmClient:
    def test_client_created(self):
        from vlm_generate import init_vlm_client

        client = init_vlm_client("http://localhost:8000/v1")
        assert client is not None
        # Verify the base URL is set (URL object comparison)
        assert str(client.base_url) == "http://localhost:8000/v1/"

    def test_api_key_is_dummy(self):
        from vlm_generate import init_vlm_client

        client = init_vlm_client("http://example.com/v1")
        assert client.api_key == "not-needed"


class TestGenerateMarkdown:
    def test_calls_completion_with_image(self):
        """Verify the correct payload is sent to the API."""
        from vlm_generate import generate_markdown

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "## 문제\nTest question"
        mock_client.chat.completions.create.return_value = mock_response

        # Create a fake PNG
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"fake")
            tmp_path = f.name

        try:
            result = generate_markdown(
                tmp_path,
                client=mock_client,
                model_name="Qwen/Qwen3.5-VL-9B",
                max_tokens=4096,
                temperature=0.1,
            )

            assert result == "## 문제\nTest question"

            # Verify the API was called correctly
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "Qwen/Qwen3.5-VL-9B"
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["max_tokens"] == 4096

            # Verify message structure
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            user_content = messages[1]["content"]
            assert any(c["type"] == "image_url" for c in user_content)
            assert any(c["type"] == "text" for c in user_content)
        finally:
            os.unlink(tmp_path)
