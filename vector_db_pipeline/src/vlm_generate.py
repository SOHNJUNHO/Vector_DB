"""
vlm_generate.py — VLM prompt and utilities for exam image parsing.
"""

import base64

from openai import OpenAI

SYSTEM_PROMPT = """You are an expert Korean math exam question analyzer.
Given an exam question image, respond with ONLY a valid JSON object:

{
  "text": "full transcription of all text and math in the image",
  "concepts": ["mathematical concept 1", "mathematical concept 2"],
  "description": "brief description of what is visually in the image"
}

Rules:
1. "text": Transcribe all Korean text exactly as it appears. Write math in LaTeX \
($...$ inline, $$...$$ block). Include the question number, all options (① ② ③ ④ ⑤), \
and any given information.
2. "concepts": List the mathematical concepts or abilities required to solve this problem \
(e.g. ["이차방정식", "판별식", "근의 공식"]). Use Korean terms.
3. "description": One or two sentences describing what is visually present \
(e.g. "수식과 그래프가 포함된 이차방정식 문제입니다.").
4. Output ONLY the JSON. No preamble, no extra commentary."""


def encode_image_to_base64(image_path: str) -> str:
    """Read a PNG file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def init_vlm_client(api_base: str, api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client pointing at Ollama or any compatible endpoint."""
    return OpenAI(api_key=api_key or "ollama", base_url=api_base)
