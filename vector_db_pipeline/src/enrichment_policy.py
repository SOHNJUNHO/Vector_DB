"""
enrichment_policy.py — Rules for routing pages to VLM enrichment/fallback.
"""

from __future__ import annotations


class EnrichmentPolicy:
    def __init__(
        self,
        *,
        triggers: list[str] | None = None,
        quality_threshold: float = 0.5,
        enable_pure_vlm_fallback: bool = True,
    ):
        self.triggers = set(triggers or ["figures", "chemical_formulas"])
        self.quality_threshold = quality_threshold
        self.enable_pure_vlm_fallback = enable_pure_vlm_fallback

    def should_enrich(self, *, has_figures: bool, quality: float) -> bool:
        return self.should_describe_figures(has_figures=has_figures) or self.should_fallback(
            quality=quality
        )

    def should_describe_figures(self, *, has_figures: bool) -> bool:
        return has_figures and bool({"figures", "chemical_formulas"} & self.triggers)

    def should_fallback(self, *, quality: float) -> bool:
        return self.enable_pure_vlm_fallback and quality < self.quality_threshold
