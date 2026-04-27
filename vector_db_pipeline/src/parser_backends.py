"""
parser_backends.py — Parser backend factory and normalized parser wrappers.
"""

from __future__ import annotations

import os

try:
    from .document_parser import DocumentParser
    from .health_check import run_all as health_check
    from .parse_result import (
        ParsedDocument,
        make_section_result,
        sections_to_parsed_document,
        validate_section_result,
    )
    from .vlm_client import VlmClient
    from .vlm_generate import init_vlm_client
except ImportError:
    from document_parser import DocumentParser
    from health_check import run_all as health_check
    from parse_result import (
        ParsedDocument,
        make_section_result,
        sections_to_parsed_document,
        validate_section_result,
    )
    from vlm_client import VlmClient
    from vlm_generate import init_vlm_client


class PureVlmParser:
    """Qwen-only parser: each page is transcribed directly into markdown."""

    parser_backend = "pure_vlm"

    def __init__(self, vlm: VlmClient):
        self._vlm = vlm

    def parse_single(self, image_path: str, document_id: str):
        markdown = self._vlm.transcribe_page(image_path)
        return make_section_result(
            document_id=document_id,
            section_title=document_id,
            section_level=0,
            page_range=(0, 0),
            markdown=markdown,
            parser_backend=self.parser_backend,
            chunk_index=0,
        )

    def parse_document(self, page_paths: list[str], document_id: str):
        sections = []
        for i, page_path in enumerate(page_paths):
            markdown = self._vlm.transcribe_page(page_path)
            sections.append(
                make_section_result(
                    document_id=document_id,
                    section_title=f"{document_id} — page {i + 1}",
                    section_level=0,
                    page_range=(i, i),
                    markdown=markdown,
                    parser_backend=self.parser_backend,
                    chunk_index=i,
                )
            )
        return sections

    def parse_parsed_document(
        self,
        page_paths: list[str],
        document_id: str,
    ) -> ParsedDocument:
        sections = self.parse_document(page_paths, document_id)
        return sections_to_parsed_document(
            document_id=document_id,
            page_paths=page_paths,
            sections=sections,
            parser_backend=self.parser_backend,
        )


class DoclingVlmParser(DocumentParser):
    """Docling-first parser with selective VLM enrichment."""

    parser_backend = "docling_vlm"

    def parse_parsed_document(
        self,
        page_paths: list[str],
        document_id: str,
    ) -> ParsedDocument:
        sections = (
            [self.parse_single(page_paths[0], document_id)]
            if len(page_paths) == 1
            else self.parse_document(page_paths, document_id)
        )
        sections = [validate_section_result(section, document_id) for section in sections]
        return sections_to_parsed_document(
            document_id=document_id,
            page_paths=page_paths,
            sections=sections,
            parser_backend=self.parser_backend,
        )


class BenchmarkingParser:
    """Runs benchmark parsers for artifacts while returning primary chunks."""

    def __init__(self, primary, benchmarks: list):
        self.primary = primary
        self.benchmarks = benchmarks
        self.parser_backend = getattr(primary, "parser_backend", "primary")

    def parse_single(self, image_path: str, document_id: str):
        return self.primary.parse_single(image_path, document_id)

    def parse_document(self, page_paths: list[str], document_id: str):
        return self.primary.parse_document(page_paths, document_id)

    def parse_parsed_document(
        self,
        page_paths: list[str],
        document_id: str,
    ) -> ParsedDocument:
        parsed = self.primary.parse_parsed_document(page_paths, document_id)
        benchmark_docs = []
        for parser in self.benchmarks:
            benchmark_docs.append(
                parser.parse_parsed_document(page_paths, document_id).to_dict()
            )
        if benchmark_docs:
            parsed.artifacts["benchmark_documents"] = benchmark_docs
        return parsed


def build_parser(config: dict):
    backend = (
        config.get("parser", {}).get("default_backend")
        or config.get("markdown_backend")
        or "docling_vlm"
    )
    if backend == "vlm":
        backend = "pure_vlm"

    vlm_cfg = dict(config["vlm"])
    if os.environ.get("VLLM_API_BASE"):
        vlm_cfg["api_base"] = os.environ["VLLM_API_BASE"]

    api_key = os.environ.get("FIREWORKS_API_KEY") or "not-needed"
    print("\n=== Step 2: Checking VLM endpoint ===")
    health_check(
        api_base=vlm_cfg["api_base"],
        model_name=vlm_cfg["model_name"],
        retries=vlm_cfg["health_check_retries"],
        delay=vlm_cfg["health_check_delay"],
        api_key=api_key,
    )
    client = init_vlm_client(vlm_cfg["api_base"], api_key=api_key)
    vlm = VlmClient(
        client,
        vlm_cfg["model_name"],
        max_tokens=vlm_cfg.get("max_tokens", 4096),
        temperature=vlm_cfg.get("temperature", 0.1),
    )

    dp_cfg = dict(config.get("document_parser", {}))
    parser_cfg = config.get("parser", {})
    if "quality_threshold" in parser_cfg:
        dp_cfg["quality_threshold"] = parser_cfg["quality_threshold"]
    dp_cfg["enable_pure_vlm_fallback"] = parser_cfg.get(
        "enable_pure_vlm_fallback",
        True,
    )
    dp_cfg.setdefault("temperature", vlm_cfg.get("temperature", 0.1))
    dp_cfg.setdefault("max_tokens", vlm_cfg.get("max_tokens", 4096))

    primary = _make_backend(backend, client, vlm_cfg, dp_cfg, vlm)
    benchmarks = [
        _make_backend(name, client, vlm_cfg, dp_cfg, vlm)
        for name in parser_cfg.get("benchmark_backends", [])
        if name and name != backend
    ]
    if benchmarks:
        names = ", ".join(parser.parser_backend for parser in benchmarks)
        print(f"[Benchmark] Saving non-indexed parser outputs for: {names}")
        return BenchmarkingParser(primary, benchmarks)
    return primary


def _make_backend(backend: str, client, vlm_cfg: dict, dp_cfg: dict, vlm: VlmClient):
    if backend == "vlm":
        backend = "pure_vlm"
    if backend == "pure_vlm":
        print(f"[Backend] pure_vlm — {vlm_cfg['api_base']} / {vlm_cfg['model_name']}")
        return PureVlmParser(vlm)
    if backend != "docling_vlm":
        raise ValueError(f"Unsupported parser backend: {backend}")
    print(f"[Backend] docling_vlm — {vlm_cfg['api_base']} / {vlm_cfg['model_name']}")
    return DoclingVlmParser(client, vlm_cfg["model_name"], dp_cfg, vlm_service=vlm)
