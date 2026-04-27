"""
document_parser.py — Three-layer hybrid document parser (Docling + Qwen VLM).

Layer 1: Docling   — structure, text, tables, math (LaTeX natively)
Layer 2: Qwen VLM  — figure description, low-quality page fallback,
                     section boundary detection (multi-page), metadata
Layer 3: Assembly  — returns SectionResult list ready for embedding + storage

Supports two input modes:
  parse_single(image_path)          → one flat SectionResult (single PNG)
  parse_document(page_paths)        → list of SectionResult with hierarchy
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .parse_result import FigureResult, SectionResult
    from .section_detector import SectionBoundary, SectionDetector
    from .vlm_generate import encode_image_to_base64
except ImportError:
    from parse_result import FigureResult, SectionResult
    from section_detector import SectionBoundary, SectionDetector
    from vlm_generate import encode_image_to_base64

logger = logging.getLogger(__name__)


@dataclass
class _PageResult:
    """Intermediate per-page result from Docling."""

    page_number: int
    markdown: str
    has_figures: bool
    figure_image_paths: list[str] = field(default_factory=list)
    quality_grade: float = 1.0


class DocumentParser:
    """
    Hybrid parser combining Docling (Layer 1) and Qwen VLM (Layer 2).

    VLM is triggered selectively:
      - Figure description: when semantically complex figures are detected
      - Low-quality fallback: when Docling page quality grade < threshold
      - Section detection: once per page in multi-page mode (lightweight call)
      - Metadata: once per section (text-only call, no image)
    """

    def __init__(self, vlm_client, vlm_model_name: str, config: dict):
        self._client = vlm_client
        self._model_name = vlm_model_name
        self._cfg = config
        self._triggers = set(
            config.get("complexity_triggers", ["figures", "chemical_formulas"])
        )
        self._quality_threshold = float(config.get("quality_threshold", 0.5))
        self._fig_prompt = config.get(
            "figure_description_prompt",
            "Describe this figure in full detail: all visual elements, labels, "
            "axes, values, and trends. Write math in LaTeX ($...$ inline, $$...$$ block).",
        )
        self._meta_prompt = config.get(
            "metadata_prompt",
            'Analyze this content. Return ONLY valid JSON:\n{"topic": "", "has_diagram": false}',
        )
        self._converter = None  # lazy init — Docling is slow to import

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_single(self, image_path: str, document_id: str) -> SectionResult:
        """Single PNG → one flat SectionResult (no section hierarchy)."""
        page = self._docling_page(image_path, page_number=0)
        markdown = self._enrich(image_path, page)
        return SectionResult(
            section_id=str(uuid.uuid4()),
            section_title=document_id,
            section_level=0,
            page_range=(0, 0),
            markdown=markdown,
            metadata=self._metadata(markdown),
            figures=self._process_figures(page, image_path),
            document_id=document_id,
        )

    def parse_document(self, page_paths: list[str], document_id: str) -> list[SectionResult]:
        """
        Ordered PNGs → list of SectionResult with section hierarchy.

        Steps:
          1. Stitch PNGs → temp PDF so Docling processes the full document at once
          2. VLM detects section boundaries page-by-page
          3. Group pages into sections
          4. Enrich + finalize each section
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name
        try:
            _pngs_to_pdf(page_paths, pdf_path)
            pages = self._docling_pdf(pdf_path, page_paths)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

        detector = SectionDetector(self._client, self._model_name, self._cfg)
        boundaries = self._detect_boundaries(page_paths, pages, detector)
        raw_sections = _group_pages(page_paths, pages, boundaries, document_id)
        return [self._finalize_section(raw) for raw in raw_sections]

    # ------------------------------------------------------------------
    # Docling layer
    # ------------------------------------------------------------------

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            try:
                from docling.datamodel.pipeline_options import PdfPipelineOptions as _Opts
            except ImportError:
                from docling.pipeline_options import PipelineOptions as _Opts  # type: ignore[no-redef]

            opts = _Opts()
            opts.generate_picture_images = True
            converter_kwargs = {}
            if "pipeline_options" in inspect.signature(DocumentConverter).parameters:
                converter_kwargs["pipeline_options"] = opts
            self._converter = DocumentConverter(**converter_kwargs)
        return self._converter

    def _docling_page(self, image_path: str, page_number: int) -> _PageResult:
        result = self._get_converter().convert(image_path)
        doc = result.document
        markdown = doc.export_to_markdown()
        fig_paths = _save_figure_crops(doc.pictures, page_no=None, stem=Path(image_path).stem)
        return _PageResult(
            page_number=page_number,
            markdown=markdown,
            has_figures=len(doc.pictures) > 0,
            figure_image_paths=fig_paths,
            quality_grade=_page_quality(result, 0),
        )

    def _docling_pdf(self, pdf_path: str, original_pngs: list[str]) -> list[_PageResult]:
        result = self._get_converter().convert(pdf_path)
        doc = result.document
        pages = []
        for i, _ in enumerate(doc.pages):
            page_no = i + 1  # Docling uses 1-based page numbers
            markdown = doc.export_to_markdown(page_no=page_no)
            pics_on_page = [
                p for p in doc.pictures
                if p.prov and any(pr.page_no == page_no for pr in p.prov)
            ]
            stem = Path(original_pngs[i]).stem if i < len(original_pngs) else f"page_{i}"
            fig_paths = _save_figure_crops(pics_on_page, page_no=page_no, stem=stem)
            pages.append(
                _PageResult(
                    page_number=i,
                    markdown=markdown,
                    has_figures=len(pics_on_page) > 0,
                    figure_image_paths=fig_paths,
                    quality_grade=_page_quality(result, i),
                )
            )
        return pages

    # ------------------------------------------------------------------
    # VLM layer
    # ------------------------------------------------------------------

    def _should_enrich(self, page: _PageResult) -> bool:
        if page.has_figures and (
            "figures" in self._triggers or "chemical_formulas" in self._triggers
        ):
            return True
        return page.quality_grade < self._quality_threshold

    def _enrich(self, image_path: str, page: _PageResult) -> str:
        """Return VLM-enriched markdown if triggered; otherwise return Docling output."""
        if not self._should_enrich(page):
            return page.markdown
        prompt = (
            f"Docling extracted the following structure from this page:\n\n"
            f"{page.markdown}\n\n"
            "Now transcribe the full page into structured markdown. "
            "Describe all figures and diagrams in detail. "
            "Preserve all tables and text exactly."
        )
        return self._vlm_image_call(image_path, prompt)

    def _describe_figure(self, fig_path: str) -> str:
        return self._vlm_image_call(fig_path, self._fig_prompt)

    def _metadata(self, markdown: str) -> dict:
        response = self._vlm_text_call(f"{markdown}\n\n{self._meta_prompt}")
        parsed = _parse_json(response, context="section metadata")
        if parsed:
            return parsed

        if response.strip():
            logger.warning("Metadata response was not valid JSON. Retrying with stricter prompt.")
        retry_response = self._vlm_text_call(
            f"{self._meta_prompt}\n\nContent:\n{markdown}\n\nReturn JSON only."
        )
        return _parse_json(retry_response, context="section metadata retry")

    def _vlm_image_call(self, image_path: str, prompt: str) -> str:
        b64 = encode_image_to_base64(image_path)
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
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
            ],
            temperature=self._cfg.get("temperature", 0.1),
            max_tokens=self._cfg.get("max_tokens", 4096),
        )
        return resp.choices[0].message.content.strip()

    def _vlm_text_call(self, text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Section detection + grouping
    # ------------------------------------------------------------------

    def _detect_boundaries(
        self,
        page_paths: list[str],
        pages: list[_PageResult],
        detector: SectionDetector,
    ) -> list[SectionBoundary]:
        boundaries = []
        prev_title: str | None = None
        for path, page in zip(page_paths, pages, strict=True):
            b = detector.detect(path, page.markdown, prev_title)
            boundaries.append(b)
            if b.is_new_section and b.title:
                prev_title = b.title
        return boundaries

    # ------------------------------------------------------------------
    # Figure processing
    # ------------------------------------------------------------------

    def _process_figures(self, page: _PageResult, image_path: str) -> list[FigureResult]:
        results = []
        for i, fig_path in enumerate(page.figure_image_paths):
            if not os.path.exists(fig_path) or not _is_meaningful_figure(fig_path):
                continue
            description = self._describe_figure(fig_path)
            results.append(
                FigureResult(
                    figure_id=f"{Path(image_path).stem}_p{page.page_number}_f{i}",
                    page_number=page.page_number,
                    description=description,
                    cropped_image_path=fig_path,
                )
            )
        return results

    def _finalize_section(self, raw: dict) -> SectionResult:
        parts = []
        all_figures: list[FigureResult] = []
        for path, page in zip(raw["page_paths"], raw["pages"], strict=True):
            parts.append(self._enrich(path, page))
            all_figures.extend(self._process_figures(page, path))
        markdown = "\n\n".join(parts)
        return SectionResult(
            section_id=str(uuid.uuid4()),
            section_title=raw["title"],
            section_level=raw["level"],
            page_range=(raw["start"], raw["end"]),
            markdown=markdown,
            metadata=self._metadata(markdown),
            figures=all_figures,
            document_id=raw["document_id"],
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _pngs_to_pdf(page_paths: list[str], output_path: str) -> None:
    """Stitch ordered PNGs into a multi-page PDF using Pillow."""
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in page_paths]
    images[0].save(output_path, save_all=True, append_images=images[1:])


def _save_figure_crops(pictures, page_no: int | None, stem: str) -> list[str]:
    """Save Docling picture crops to temp PNGs. Returns list of saved paths."""
    saved = []
    tmp_dir = tempfile.gettempdir()
    for i, pic in enumerate(pictures):
        try:
            if pic.image and pic.image.pil_image:
                tag = f"{stem}_pn{page_no or 0}_f{i}"
                out = os.path.join(tmp_dir, f"docling_fig_{tag}.png")
                pic.image.pil_image.save(out)
                saved.append(out)
        except Exception as exc:
            logger.warning("Failed to save Docling figure crop %s for %s: %s", i, stem, exc)
    return saved


def _page_quality(result, page_index: int) -> float:
    """Extract Docling page quality grade (0.0-1.0). Returns 1.0 on failure."""
    try:
        conf = getattr(result, "confidence", None)
        if conf is None:
            return 1.0
        pages = getattr(conf, "pages", None)
        if pages and len(pages) > page_index:
            grade = getattr(pages[page_index], "mean_grade", None)
            if grade is not None:
                return float(grade)
    except Exception:
        pass
    return 1.0


def _group_pages(
    page_paths: list[str],
    pages: list[_PageResult],
    boundaries: list[SectionBoundary],
    document_id: str,
) -> list[dict]:
    sections: list[dict] = []
    current: dict | None = None
    for i, (path, page, boundary) in enumerate(
        zip(page_paths, pages, boundaries, strict=True)
    ):
        if boundary.is_new_section or current is None:
            if current is not None:
                sections.append(current)
            current = {
                "title": boundary.title or f"Section {len(sections) + 1}",
                "level": boundary.level or 1,
                "start": i,
                "end": i,
                "page_paths": [path],
                "pages": [page],
                "document_id": document_id,
            }
        else:
            current["end"] = i
            current["page_paths"].append(path)
            current["pages"].append(page)
    if current is not None:
        sections.append(current)
    return sections


def _is_meaningful_figure(fig_path: str) -> bool:
    """Return False for tiny or decorative figures (too small or extreme aspect ratio)."""
    try:
        from PIL import Image

        with Image.open(fig_path) as img:
            w, h = img.size
        if w < 50 or h < 50:
            return False
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 20:
            return False
    except Exception as exc:
        logger.warning("Could not inspect figure %s, treating it as meaningful: %s", fig_path, exc)
    return True


def _parse_json(text: str, *, context: str = "VLM response") -> dict:
    """Extract JSON object from VLM response text. Returns {} on failure."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to decode fenced JSON for %s.", context)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to decode inline JSON for %s.", context)
            return {}
    if text.strip():
        logger.warning("No JSON object found in %s.", context)
    return {}
