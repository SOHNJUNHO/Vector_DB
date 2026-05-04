"""
query_agent.py — Natural language query agent for exam question retrieval.

Flow:
  1. VLM call: free-text → {topic, limit}
  2. TextEmbedder: topic → vector
  3. QdrantStore.search: text vector search
  4. Format and return results
"""

from __future__ import annotations

import json
import re

_DEFAULT_PARSE_PROMPT = (
    'Given this query: "{query}"\n'
    "Return ONLY valid JSON:\n"
    '{{"topic": "...", "limit": 5}}\n'
    "- topic: the mathematical concept or subject area (in Korean if possible)\n"
    "- limit: number of results requested (default 5)"
)

_TEXT_PREVIEW_CHARS = 500
_DEFAULT_ANSWER_PROMPT = (
    "Answer the user query using only the retrieved context.\n"
    'User query: "{query}"\n\n'
    "Retrieved context:\n{context}\n\n"
    "Write a concise answer in Korean. If the context is not enough, say what is missing."
)


class QueryAgent:
    """Translates a natural language query into a Qdrant text vector search."""

    def __init__(
        self,
        vlm_client,
        vlm_model_name: str,
        text_embedder,
        qdrant_store,
        config: dict,
    ):
        self._client = vlm_client
        self._model_name = vlm_model_name
        self._embedder = text_embedder
        self._store = qdrant_store
        query_cfg = config.get("query", {})
        self._parse_prompt = query_cfg.get("parse_prompt", _DEFAULT_PARSE_PROMPT)
        self._answer_prompt = query_cfg.get("answer_prompt", _DEFAULT_ANSWER_PROMPT)
        self._default_limit = int(query_cfg.get("default_limit", 5))
        self._answer_context_limit = int(query_cfg.get("answer_context_limit", 3))

    def query(self, natural_language: str) -> list[dict]:
        """Accept a free-text query and return matching exam questions."""
        parsed = self._parse(natural_language)
        results = self._retrieve(parsed)
        return self._format(results)

    def answer(self, natural_language: str, results: list[dict]) -> str:
        """Generate a short answer grounded in retrieved results."""
        if not results:
            return "No relevant results were found."

        context_chunks = []
        for i, result in enumerate(results[: self._answer_context_limit], start=1):
            doc_id = result.get("document_id", f"Result {i}")
            text = result.get("text", "")
            concepts = result.get("concepts", [])
            context_chunks.append(f"[{i}] {doc_id} ({', '.join(concepts)})\n{text}")

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
            return (
                f"Top result: {top.get('document_id', '')}\n\n"
                f"{top.get('text', '')[:_TEXT_PREVIEW_CHARS]}"
            )

    def _parse(self, natural_language: str) -> dict:
        """Call VLM to extract {topic, limit} from free-text query."""
        prompt = self._parse_prompt.format(query=natural_language)
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_json(raw, fallback_topic=natural_language, default_limit=self._default_limit)
        except Exception:
            return {"topic": natural_language, "limit": self._default_limit}

    def _retrieve(self, parsed: dict) -> list[dict]:
        topic = parsed["topic"]
        limit = parsed.get("limit") or self._default_limit
        vector = self._embedder.embed_text(topic)
        return self._store.search(vector=vector, limit=limit)

    def _format(self, results: list[dict]) -> list[dict]:
        formatted = []
        for r in results:
            text = r.get("text", "")
            formatted.append({
                "score": round(r.get("score", 0.0), 4),
                "document_id": r.get("document_id", ""),
                "concepts": r.get("concepts", []),
                "description": r.get("description", ""),
                "text_preview": (
                    text[:_TEXT_PREVIEW_CHARS] + "..."
                    if len(text) > _TEXT_PREVIEW_CHARS
                    else text
                ),
            })
        return formatted


def _parse_json(text: str, fallback_topic: str, default_limit: int) -> dict:
    """Extract JSON from VLM response. Falls back to safe defaults on failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "topic": data.get("topic") or fallback_topic,
                "limit": int(data.get("limit") or default_limit),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"topic": fallback_topic, "limit": default_limit}
