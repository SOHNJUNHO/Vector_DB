"""
tests/test_query_agent.py — Tests for query_agent.py.
All tests are fully mocked — no VLM or Qdrant connection required.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_vlm_client(response_text: str = "{}") -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
    return client


def _make_agent(vlm_response: str = '{"topic": "quadratic equations", "difficulty": "hard", "limit": 5}'):
    from query_agent import QueryAgent

    vlm_client = _make_vlm_client(vlm_response)
    text_embedder = MagicMock()
    text_embedder.embed_text.return_value = [0.1] * 1024
    visual_embedder = MagicMock()
    visual_embedder.embed_text.return_value = [0.2] * 1024

    qdrant_store = MagicMock()
    qdrant_store.search_sections.return_value = []
    qdrant_store.search_sections_hybrid.return_value = []

    config = {"query": {"default_limit": 5, "retrieval_mode": "hybrid", "hybrid_alpha": 0.7}}
    agent = QueryAgent(
        vlm_client,
        "model",
        text_embedder,
        qdrant_store,
        config,
        visual_embedder=visual_embedder,
    )
    return agent, vlm_client, text_embedder, visual_embedder, qdrant_store


class TestParseQuery:
    def test_extracts_topic_and_difficulty(self):
        agent, *_ = _make_agent('{"topic": "quadratic equations", "difficulty": "hard", "limit": 5}')
        result = agent._parse("Find me a hard question about quadratic equations")
        assert result["topic"] == "quadratic equations"
        assert result["difficulty"] == "hard"
        assert result["limit"] == 5

    def test_null_difficulty_when_not_specified(self):
        agent, *_ = _make_agent('{"topic": "logarithms", "difficulty": null, "limit": 5}')
        result = agent._parse("Find me a question about logarithms")
        assert result["difficulty"] is None

    def test_falls_back_on_bad_json(self):
        agent, *_ = _make_agent("not valid json at all")
        result = agent._parse("some query text")
        assert result["topic"] == "some query text"
        assert result["difficulty"] is None
        assert result["limit"] == 5

    def test_falls_back_on_empty_response(self):
        agent, *_ = _make_agent("")
        result = agent._parse("my query")
        assert result["topic"] == "my query"

    def test_falls_back_when_vlm_parse_call_fails(self):
        agent, client, *_ = _make_agent()
        client.chat.completions.create.side_effect = RuntimeError("network down")
        result = agent._parse("my query")
        assert result == {"topic": "my query", "difficulty": None, "limit": 5}


class TestQuery:
    def test_calls_search_with_difficulty_filter(self):
        agent, _, _, visual_embedder, store = _make_agent(
            '{"topic": "quadratic equations", "difficulty": "hard", "limit": 3}'
        )
        agent.query("Find a hard question about quadratic equations")

        store.search_sections_hybrid.assert_called_once()
        call_kwargs = store.search_sections_hybrid.call_args.kwargs
        assert call_kwargs["difficulty"] == "hard"
        assert call_kwargs["limit"] == 3
        visual_embedder.embed_text.assert_called_once_with("quadratic equations")

    def test_calls_search_without_filter_when_no_difficulty(self):
        agent, _, _, _, store = _make_agent(
            '{"topic": "trigonometry", "difficulty": null, "limit": 5}'
        )
        agent.query("Find a question about trigonometry")

        call_kwargs = store.search_sections_hybrid.call_args.kwargs
        assert call_kwargs["difficulty"] is None

    def test_embeds_topic_not_full_query(self):
        agent, _, embedder, _, _ = _make_agent(
            '{"topic": "logarithms", "difficulty": "easy", "limit": 5}'
        )
        agent.query("Give me an easy question about logarithms please")

        embedder.embed_text.assert_called_once_with("logarithms")

    def test_uses_text_only_search_when_visual_embedder_missing(self):
        from query_agent import QueryAgent

        vlm_client = _make_vlm_client('{"topic": "geometry", "difficulty": null, "limit": 5}')
        text_embedder = MagicMock()
        text_embedder.embed_text.return_value = [0.1] * 1024
        store = MagicMock()
        store.search_sections.return_value = []
        config = {"query": {"default_limit": 5, "retrieval_mode": "hybrid"}}
        agent = QueryAgent(vlm_client, "model", text_embedder, store, config)

        agent.query("Find a geometry problem")

        store.search_sections.assert_called_once()


class TestFormat:
    def test_truncates_long_markdown(self):
        agent, _, _, _, store = _make_agent()
        long_markdown = "x" * 1000
        store.search_sections_hybrid.return_value = [
            {"score": 0.9, "document_id": "doc_1", "section_title": "Q1",
             "markdown": long_markdown, "difficulty": "hard", "topic": "algebra"}
        ]

        results = agent.query("find a question")

        assert results[0]["markdown_preview"].endswith("...")
        assert len(results[0]["markdown_preview"]) <= 503  # 500 chars + "..."

    def test_does_not_truncate_short_markdown(self):
        agent, _, _, _, store = _make_agent()
        store.search_sections_hybrid.return_value = [
            {"score": 0.9, "document_id": "doc_1", "section_title": "Q1",
             "markdown": "Short content", "difficulty": None, "topic": "algebra"}
        ]

        results = agent.query("find a question")

        assert results[0]["markdown_preview"] == "Short content"

    def test_score_is_rounded(self):
        agent, _, _, _, store = _make_agent()
        store.search_sections_hybrid.return_value = [
            {"score": 0.912345678, "document_id": "d", "section_title": "t",
             "markdown": "m", "difficulty": None, "topic": "t"}
        ]

        results = agent.query("q")

        assert results[0]["score"] == 0.9123


class TestAnswer:
    def test_answer_uses_retrieved_context(self):
        agent, client, *_ = _make_agent()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Grounded answer"))
        ]

        answer = agent.answer(
            "Explain this question",
            [{"section_title": "Q1", "markdown": "Context body", "markdown_preview": "Context body"}],
        )

        assert answer == "Grounded answer"

    def test_answer_falls_back_to_preview_on_vlm_failure(self):
        agent, client, *_ = _make_agent()
        client.chat.completions.create.side_effect = RuntimeError("api error")

        answer = agent.answer(
            "Explain this question",
            [{"section_title": "Q1", "markdown": "Context body", "markdown_preview": "Preview"}],
        )

        assert "Top retrieved section: Q1" in answer
