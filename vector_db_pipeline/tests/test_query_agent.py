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


def _make_agent(vlm_response: str = '{"topic": "이차방정식", "limit": 5}'):
    from query_agent import QueryAgent

    vlm_client = _make_vlm_client(vlm_response)
    text_embedder = MagicMock()
    text_embedder.embed_text.return_value = [0.1] * 1024

    qdrant_store = MagicMock()
    qdrant_store.search.return_value = []

    config = {"query": {"default_limit": 5, "answer_context_limit": 3}}
    agent = QueryAgent(vlm_client, "model", text_embedder, qdrant_store, config)
    return agent, vlm_client, text_embedder, qdrant_store


class TestParseQuery:
    def test_extracts_topic_and_limit(self):
        agent, *_ = _make_agent('{"topic": "이차방정식", "limit": 5}')
        result = agent._parse("이차방정식 문제 찾아줘")
        assert result["topic"] == "이차방정식"
        assert result["limit"] == 5

    def test_falls_back_on_bad_json(self):
        agent, *_ = _make_agent("not valid json at all")
        result = agent._parse("some query text")
        assert result["topic"] == "some query text"
        assert result["limit"] == 5

    def test_falls_back_on_empty_response(self):
        agent, *_ = _make_agent("")
        result = agent._parse("my query")
        assert result["topic"] == "my query"

    def test_falls_back_when_vlm_call_fails(self):
        agent, client, *_ = _make_agent()
        client.chat.completions.create.side_effect = RuntimeError("network down")
        result = agent._parse("my query")
        assert result == {"topic": "my query", "limit": 5}


class TestRetrieve:
    def test_embeds_topic_and_calls_search(self):
        agent, _, embedder, store = _make_agent('{"topic": "로그", "limit": 3}')
        agent.query("로그 관련 문제")

        embedder.embed_text.assert_called_once_with("로그")
        store.search.assert_called_once()
        call_kwargs = store.search.call_args.kwargs
        assert call_kwargs["limit"] == 3

    def test_uses_default_limit_when_not_specified(self):
        agent, _, _, store = _make_agent('{"topic": "삼각함수"}')
        agent.query("삼각함수 문제")

        call_kwargs = store.search.call_args.kwargs
        assert call_kwargs["limit"] == 5


class TestFormat:
    def test_truncates_long_text(self):
        agent, _, _, store = _make_agent()
        store.search.return_value = [
            {
                "score": 0.9,
                "document_id": "doc_001",
                "concepts": ["이차방정식"],
                "description": "수식 문제",
                "text": "x" * 1000,
            }
        ]

        results = agent.query("find a question")

        assert results[0]["text_preview"].endswith("...")
        assert len(results[0]["text_preview"]) <= 503

    def test_score_is_rounded(self):
        agent, _, _, store = _make_agent()
        store.search.return_value = [
            {
                "score": 0.912345678,
                "document_id": "d",
                "concepts": [],
                "description": "",
                "text": "t",
            }
        ]

        results = agent.query("q")

        assert results[0]["score"] == 0.9123

    def test_format_includes_required_fields(self):
        agent, _, _, store = _make_agent()
        store.search.return_value = [
            {
                "score": 0.8,
                "document_id": "doc_001",
                "concepts": ["미분"],
                "description": "미분 문제",
                "text": "미분을 풀어라",
            }
        ]

        results = agent.query("q")

        assert "score" in results[0]
        assert "document_id" in results[0]
        assert "concepts" in results[0]
        assert "description" in results[0]
        assert "text_preview" in results[0]


class TestAnswer:
    def test_answer_uses_retrieved_context(self):
        agent, client, *_ = _make_agent()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="이차방정식의 해는 판별식으로 구합니다."))
        ]

        answer = agent.answer(
            "이차방정식 풀이 방법은?",
            [{"document_id": "doc_001", "text": "판별식을 이용한다", "concepts": ["이차방정식"]}],
        )

        assert answer == "이차방정식의 해는 판별식으로 구합니다."

    def test_answer_falls_back_on_vlm_failure(self):
        agent, client, *_ = _make_agent()
        client.chat.completions.create.side_effect = RuntimeError("api error")

        answer = agent.answer(
            "질문",
            [{"document_id": "doc_001", "text": "내용", "concepts": []}],
        )

        assert "doc_001" in answer

    def test_answer_returns_no_results_message_when_empty(self):
        agent, *_ = _make_agent()
        answer = agent.answer("질문", [])
        assert answer == "No relevant results were found."
