"""
vlm_generate.py — Generate structured markdown from exam images.

Uses Qwen3.5-VL via vLLM (OpenAI-compatible API) to transcribe
the image content into structured markdown with sections:
  ## 문제 (Question)
  ## 보기 (Options)
  ## 정답 (Answer)
  ## 해설 (Explanation)
"""

import base64

from openai import OpenAI

SYSTEM_PROMPT = """You are an expert Korean exam question transcriber.
Your task is to read the given image and produce a clean, structured markdown
transcription.

Rules:
1. Use the following exact section headers:
   ## 문제
   ## 보기
   ## 정답
   ## 해설

2. Write all mathematical formulas in LaTeX using $...$ for inline and
   $$...$$ for block equations.

3. Preserve the original Korean text exactly as it appears.

4. If the image contains only part of the above sections
   (e.g., just the question and options, no explanation),
   include only the sections that are present.
   Do NOT fabricate content that isn't in the image.

5. For multiple-choice options, number them as ① ② ③ ④ ⑤.

6. If there is a solution/explanation section, transcribe it step by step.

7. Output ONLY the markdown. No preamble, no extra commentary."""


def encode_image_to_base64(image_path: str) -> str:
    """Read a PNG file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_markdown(
    image_path: str,
    client: OpenAI,
    model_name: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """
    Send the image to Qwen3.5-VL and return structured markdown.
    """
    b64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this exam image into structured markdown.",
                    },
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


def init_vlm_client(api_base: str, api_key: str | None = None) -> OpenAI:
    """
    Create an OpenAI-compatible client.

    Uses an explicit API key when provided. Falls back to "not-needed" for
    local vLLM and tests which require no auth.
    """
    return OpenAI(api_key=api_key or "not-needed", base_url=api_base)
