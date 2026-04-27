"""
Streamlit demo — Exam Question Finder

Type a natural language query (e.g. "Find me a hard question about quadratic equations")
and the agent retrieves matching questions from the Qdrant vector database.

Secrets (set in Streamlit Cloud dashboard or local .env):
  QDRANT_URL, QDRANT_API_KEY, VLLM_API_BASE
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

# Add project root to path so `src` can be imported as a package.
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")


def _secret(key: str, fallback: str = "") -> str:
    """Read from Streamlit secrets (cloud) or env var (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, fallback)


@st.cache_resource(show_spinner="Loading model (first time ~30 seconds)...")
def _load_agent():
    from src.embed_text import TextEmbedder
    from src.embed_visual import VisualEmbedder
    from src.qdrant_store import QdrantStore
    from src.query_agent import QueryAgent
    from src.vlm_generate import init_vlm_client

    config_path = _ROOT / "config" / "settings.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    q_cfg = config["qdrant"]
    store = QdrantStore(
        url=_secret("QDRANT_URL") or q_cfg["url"],
        api_key=_secret("QDRANT_API_KEY") or q_cfg.get("api_key") or None,
        collection_name=q_cfg["collection_name"],
    )

    vllm_base = _secret("VLLM_API_BASE") or config["vlm"]["api_base"]
    vlm = init_vlm_client(vllm_base, api_key=_secret("FIREWORKS_API_KEY") or None)

    embedder = TextEmbedder(model_name=config["embed_text"]["model_name"])
    visual_embedder = None
    if config.get("query", {}).get("retrieval_mode", "hybrid") == "hybrid":
        visual_embedder = VisualEmbedder(model_name=config["embed_visual"]["model_name"])

    return QueryAgent(
        vlm,
        config["vlm"]["model_name"],
        embedder,
        store,
        config,
        visual_embedder=visual_embedder,
    )


# ── Page layout ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Exam Question Finder", layout="wide")
st.title("Exam Question Finder")
st.caption("Ask in plain language — the agent finds matching questions from the database.")

agent = _load_agent()

query = st.text_input(
    "Your query",
    placeholder="Find me a hard question about quadratic equations",
)
search_clicked = st.button("Search", type="primary", disabled=not query)

# ── Search & display results ─────────────────────────────────────────────────

if search_clicked and query:
    with st.spinner("Searching..."):
        results = agent.query(query)

    if not results:
        st.info("No results found. Try rephrasing your query.")
    else:
        with st.spinner("Generating answer..."):
            answer = agent.answer(query, results)

        st.subheader("Answer")
        st.markdown(answer)
        st.success(f"Found {len(results)} result(s)")
        for i, r in enumerate(results):
            title = r.get("section_title") or r.get("document_id", "Untitled")
            score = r.get("score", 0.0)
            with st.expander(f"#{i + 1} — {title}  (score: {score:.4f})", expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                col1.metric("Difficulty", r.get("difficulty") or "—")
                col2.metric("Topic", r.get("topic") or "—")
                col3.metric("Mode", r.get("retrieval_mode", "text"))
                st.caption(f"Document: {r.get('document_id', '—')}")
                st.markdown(r.get("markdown_preview", "_No preview available._"))
