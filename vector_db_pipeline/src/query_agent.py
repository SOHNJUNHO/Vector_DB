"""
query_agent.py — Single-shot natural language query agent for exam question retrieval.

Flow:
  1. VLM call: free-text → {topic, difficulty, limit}
  2. TextEmbedder: topic → vector
  3. QdrantStore.search_sections: near_vector + optional difficulty filter
  4. Format and return results
"""

from __future__ import annotations

import json
import re

_DEFAULT_PARSE_PROMPT = (
    'Given this query: "{query}"\n'
    "Return ONLY valid JSON:\n"
    '{{"topic": "...", "difficulty": "easy|medium|hard|null", "limit": 5}}\n'
    "- topic: the mathematical concept or subject area being asked about\n"
    "- difficulty: null if the user did not specify a difficulty level\n"
    "- limit: number of results requested (default 5)"
)

_MARKDOWN_PREVIEW_CHARS = 500
_DEFAULT_ANSWER_PROMPT = (
    'Answer the user query using only the retrieved context.\n'
    'User query: "{query}"\n\n'
    "Retrieved context:\n{context}\n\n"
    "Write a concise answer. If the context is not enough, say what is missing."
)


class QueryAgent:
    """
    Translates a natural language query into a Qdrant filtered vector search.
    Uses one VLM text call to extract structured parameters, then searches.
    """

    def __init__(
        self,
        vlm_client,
        vlm_model_name: str,
        text_embedder,
        qdrant_store,
        config: dict,
        visual_embedder=None,
    ):
        self._client = vlm_client
        self._model_name = vlm_model_name
        self._embedder = text_embedder
        self._visual_embedder = visual_embedder
        self._store = qdrant_store
        query_cfg = config.get("query", {})
        self._parse_prompt = query_cfg.get("parse_prompt", _DEFAULT_PARSE_PROMPT)
        self._answer_prompt = query_cfg.get("answer_prompt", _DEFAULT_ANSWER_PROMPT)
        self._default_limit = int(query_cfg.get("default_limit", 5))
        self._retrieval_mode = query_cfg.get("retrieval_mode", "hybrid")
        self._hybrid_alpha = float(query_cfg.get("hybrid_alpha", 0.7))
        self._answer_context_limit = int(query_cfg.get("answer_context_limit", 3))

    def query(self, natural_language: str) -> list[dict]:
        """
        Accept a free-text query and return matching exam question sections.

        Example:
            agent.query("Find me a hard question about quadratic equations")
        """
        parsed = self._parse(natural_language)
        results = self._retrieve(parsed)
        return self._format(results)

    def answer(self, natural_language: str, results: list[dict]) -> str:
        """Generate a short answer grounded in retrieved results."""
        if not results:
            return "No relevant results were found."

        context_chunks = []
        for i, result in enumerate(results[: self._answer_context_limit], start=1):
            title = result.get("section_title") or result.get("document_id", f"Result {i}")
            markdown = result.get("markdown", "")
            context_chunks.append(f"[{i}] {title}\n{markdown}")

        prompt = self._answer_prompt.format(
            query=natural_language,
            context="\n\n".join(context_chunks),
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            top = results[0]
            title = top.get("section_title") or top.get("document_id", "Top result")
            preview = top.get("markdown_preview") or top.get("markdown", "")
            return f"Top retrieved section: {title}\n\n{preview}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, natural_language: str) -> dict:
        """Call VLM to extract {topic, difficulty, limit} from free-text query."""
        prompt = self._parse_prompt.format(query=natural_language)
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_json(
                raw,
                fallback_topic=natural_language,
                default_limit=self._default_limit,
            )
        except Exception:
            return {
                "topic": natural_language,
                "difficulty": None,
                "limit": self._default_limit,
            }

    def _retrieve(self, parsed: dict) -> list[dict]:
        topic = parsed["topic"]
        difficulty = parsed.get("difficulty")
        limit = parsed.get("limit") or self._default_limit
        text_vector = self._embedder.embed_text(topic)

        if self._retrieval_mode == "hybrid" and self._visual_embedder is not None:
            visual_text_vector = self._visual_embedder.embed_text(topic)
            return self._store.search_sections_hybrid(
                text_vector=text_vector,
                visual_text_vector=visual_text_vector,
                difficulty=difficulty,
                limit=limit,
                alpha=self._hybrid_alpha,
            )

        return self._store.search_sections(
            vector=text_vector,
            difficulty=difficulty,
            limit=limit,
        )

    def _format(self, results: list[dict]) -> list[dict]:
        """Trim markdown previews and return clean result dicts."""
        formatted = []
        for r in results:
            markdown = r.get("markdown", "")
            formatted.append({
                "score": round(r.get("score", 0.0), 4),
                "document_id": r.get("document_id", ""),
                "section_title": r.get("section_title", ""),
                "difficulty": r.get("difficulty"),
                "topic": r.get("topic"),
                "retrieval_mode": r.get("retrieval_mode", "text"),
                "markdown": markdown,
                "markdown_preview": (
                    markdown[:_MARKDOWN_PREVIEW_CHARS] + "..."
                    if len(markdown) > _MARKDOWN_PREVIEW_CHARS
                    else markdown
                ),
            })
        return formatted


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_json(text: str, fallback_topic: str, default_limit: int) -> dict:
    """Extract JSON from VLM response. Falls back to safe defaults on failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "topic": data.get("topic") or fallback_topic,
                "difficulty": data.get("difficulty") or None,
                "limit": int(data.get("limit") or default_limit),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"topic": fallback_topic, "difficulty": None, "limit": default_limit}
