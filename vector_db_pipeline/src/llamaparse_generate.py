"""
llamaparse_generate.py — Generate structured markdown from exam images.

Uses LlamaParse API to transcribe image content into structured markdown.
Since LlamaParse doesn't accept PNG directly, images are converted to PDF first.
"""

import os
import tempfile

from PIL import Image


def _png_to_pdf(png_path: str, pdf_path: str) -> str:
    """
    Convert a PNG image to a single-page PDF.

    Args:
        png_path: Path to the input PNG file
        pdf_path: Path where the PDF will be saved

    Returns:
        Path to the created PDF file.
    """
    image = Image.open(png_path).convert("RGB")
    image.save(pdf_path, "PDF", resolution=100)
    return pdf_path


def generate_markdown(
    image_path: str,
    api_key: str | None = None,
    language: str = "ko",
    result_type: str = "markdown",
) -> str:
    """
    Send the image to LlamaParse and return structured markdown.

    The image is first converted to PDF, then parsed by LlamaParse.

    Args:
        image_path: Path to the input PNG file
        api_key: LlamaParse API key (or set LLAMA_CLOUD_API_KEY env var)
        language: Language hint (default: "ko" for Korean)
        result_type: Output format — "markdown" or "text"

    Returns:
        Parsed markdown text.

    Raises:
        ImportError: If llama_parse is not installed
        ValueError: If the API key is not provided
    """
    try:
        from llama_parse import LlamaParse
    except ImportError as e:
        raise ImportError(
            "llama_parse is required. Install with: uv pip install llama-parse"
        ) from e

    if not api_key:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")

    if not api_key:
        raise ValueError(
            "LlamaParse API key is required. "
            "Set LLAMA_CLOUD_API_KEY env var or pass api_key parameter."
        )

    # Convert PNG → PDF (LlamaParse doesn't accept PNG directly)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name

    try:
        _png_to_pdf(image_path, pdf_path)

        parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,
            language=language,
            verbose=False,
        )

        documents = parser.load_data(pdf_path)

        if not documents:
            return ""

        # Return the text content from the parsed document
        return documents[0].text.strip()

    finally:
        # Clean up temporary PDF
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
