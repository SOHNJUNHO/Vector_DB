"""
section_detector.py — VLM-based section boundary detection for multi-page scanned docs.

For each page, sends the page image + Docling-extracted text to Qwen VLM
and asks whether a new section begins. Returns a lightweight JSON result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

try:
    from .vlm_generate import encode_image_to_base64
except ImportError:
    from vlm_generate import encode_image_to_base64

_DEFAULT_PROMPT = (
    "Given this scanned page image and its extracted text, return JSON only:\n"
    '{"is_new_section": true|false, "section_title": "string or null", '
    '"section_level": 1|2|3|null}\n'
    "Mark a new section only when a clear heading or title begins a new major topic."
)


@dataclass
class SectionBoundary:
    is_new_section: bool
    title: str | None
    level: int | None   # 1 = H1, 2 = H2, 3 = H3; None if not a new section


class SectionDetector:
    """
    Lightweight VLM call per page to detect section boundaries.
    Used only in multi-page document mode.
    """

    def __init__(self, vlm_client, model_name: str, config: dict):
        self._client = vlm_client
        self._model_name = model_name
        self._prompt = config.get("section_detection_prompt", _DEFAULT_PROMPT)

    def detect(
        self,
        image_path: str,
        docling_markdown: str,
        prev_section_title: str | None = None,
    ) -> SectionBoundary:
        context = f"Extracted text from this page:\n\n{docling_markdown}\n\n"
        full_prompt = context + self._prompt
        b64 = encode_image_to_base64(image_path)

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=128,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_boundary(raw)


def _parse_boundary(raw: str) -> SectionBoundary:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return SectionBoundary(
                is_new_section=bool(data.get("is_new_section", False)),
                title=data.get("section_title") or None,
                level=data.get("section_level") or None,
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return SectionBoundary(is_new_section=False, title=None, level=None)
