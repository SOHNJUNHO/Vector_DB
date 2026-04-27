"""
vlm_client.py — Shared OpenAI-compatible VLM operations.
"""

from __future__ import annotations

import json
import re

try:
    from .vlm_generate import SYSTEM_PROMPT, encode_image_to_base64
except ImportError:
    from vlm_generate import SYSTEM_PROMPT, encode_image_to_base64


class VlmClient:
    """Small task-oriented wrapper around an OpenAI-compatible multimodal client."""

    def __init__(
        self,
        client,
        model_name: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def transcribe_page(self, image_path: str, prompt: str | None = None) -> str:
        return self.image_call(
            image_path,
            prompt or "Transcribe this exam image into structured markdown.",
            system_prompt=SYSTEM_PROMPT,
        )

    def describe_figure(self, image_path: str, prompt: str) -> str:
        return self.image_call(image_path, prompt)

    def extract_metadata(self, markdown: str, prompt: str) -> dict:
        response = self.text_call(f"{markdown}\n\n{prompt}", temperature=0.0, max_tokens=512)
        parsed = parse_json_object(response)
        if parsed:
            return parsed
        retry = self.text_call(
            f"{prompt}\n\nContent:\n{markdown}\n\nReturn JSON only.",
            temperature=0.0,
            max_tokens=512,
        )
        return parse_json_object(retry)

    def image_call(
        self,
        image_path: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        b64 = encode_image_to_base64(image_path)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def text_call(
        self,
        text: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


def parse_json_object(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}
