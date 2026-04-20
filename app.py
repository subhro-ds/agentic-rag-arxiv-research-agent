"""
app.py — Streamlit frontend for the Agentic RAG research assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG — arXiv Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .paper-card {
        background: #1e1e2e; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
        border-left: 3px solid #6366f1;
    }
    .step-box {
        background: #0f172a; border-radius: 6px;
        padding: 10px; margin: 4px 0; font-size: 0.85rem;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "agent": None,
        "rag_chain": None,
        "conv_chain": None,
        "vsm": None,
        "fetched_papers": [],
        "ingested_ids": set(),
        "chat_history": [],
        "mode": "agent",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Lazy Imports (avoid slow startup) ────────────────────────────────────────
@st.cache_resource
def get_agent():
    from research_agent import ResearchAgent
    return ResearchAgent()


@st.cache_resource
def get_vsm():
    from vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    try:
        vsm.load_or_create()
    except Exception:
        pass  # No existing store yet
    return vsm


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Mode",
        ["🤖 Autonomous Agent", "📚 RAG Chat", "🔍 Search & Ingest"],
        index=0,
    )
    st.session_state["mode"] = mode

    st.divider()
    st.markdown("### 📄 Ingested Papers")
    if st.session_state["ingested_ids"]:
        for pid in sorted(st.session_state["ingested_ids"]):
            st.success(f"✅ {pid}")
    else:
        st.info("No papers ingested yet.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="main-header">🤖 Agentic RAG — arXiv Research Assistant</h1>',
    unsafe_allow_html=True,
)
st.caption(
    "Powered by LangChain · arXiv · FAISS · Claude / GPT"
)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Autonomous Agent
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🤖 Autonomous Agent":
    st.markdown(
        "### 🧠 Autonomous Agent Mode\n"
        "Ask a research question and the agent will **autonomously search arXiv, "
        "ingest relevant papers, and synthesize an answer**."
    )

    query = st.text_area(
        "Research Question",
        placeholder="e.g. What are the main architectural patterns in agentic AI systems?",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True)

    if run_btn and query.strip():
        agent = get_agent()
        with st.spinner("🤔 Agent is thinking…"):
            result = agent.run(query)

        # ── Final Answer
        st.markdown("#### ✅ Final Answer")
        st.markdown(result["output"])

        # ── Reasoning Steps
        if result["steps"]:
            with st.expander(f"🔍 Agent Reasoning ({len(result['steps'])} steps)", expanded=False):
                for i, step in enumerate(result["steps"], 1):
                    st.markdown(f"**Step {i} — `{step['action']}`**")
                    st.markdown(
                        f'<div class="step-box">'
                        f"<b>Input:</b> {json.dumps(step['action_input'], indent=2)}<br><br>"
                        f"<b>Observation:</b> {step['observation']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: RAG Chat
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📚 RAG Chat":
    st.markdown(
        "### 💬 RAG Chat Mode\n"
        "Ask questions about **already-ingested papers** (use Search & Ingest first)."
    )

    # Render chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about the ingested papers…"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        vsm = get_vsm()
        if not st.session_state["ingested_ids"] and not (Path("./data/vector_store/index.faiss").exists()):
            with st.chat_message("assistant"):
                st.warning(
                    "No papers ingested yet. Switch to **Search & Ingest** mode first."
                )
        else:
            from rag_chain import ConversationalRAGChain

            if st.session_state["conv_chain"] is None:
                vsm.load_or_create()
                st.session_state["conv_chain"] = ConversationalRAGChain(vsm)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving…"):
                    answer = st.session_state["conv_chain"].chat(prompt, session_id="streamlit")
                st.markdown(answer)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: Search & Ingest
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔍 Search & Ingest":
    st.markdown(
        "### 🔍 Search & Ingest Mode\n"
        "Search arXiv and selectively ingest papers into the knowledge base."
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input(
            "Search Query", placeholder="agentic AI autonomous agents LLM"
        )
    with col2:
        max_r = st.number_input("Max Results", min_value=1, max_value=20, value=5)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔎 Search", type="primary", use_container_width=True)

    if search_btn and search_query:
        from arxiv_fetcher import ArxivFetcher

        with st.spinner(f"Searching arXiv for '{search_query}'…"):
            fetcher = ArxivFetcher()
            papers = fetcher.search(search_query, max_results=int(max_r))
            st.session_state["fetched_papers"] = papers

    # Display fetched papers
    if st.session_state["fetched_papers"]:
        st.markdown(f"#### Found {len(st.session_state['fetched_papers'])} Papers")

        selected_ids = []
        for paper in st.session_state["fetched_papers"]:
            already = paper.paper_id in st.session_state["ingested_ids"]
            label = f"{'✅ ' if already else ''}**{paper.title}**"
            checked = st.checkbox(
                label,
                key=f"chk_{paper.paper_id}",
                value=already,
                disabled=already,
            )
            if checked and not already:
                selected_ids.append(paper.paper_id)

            st.markdown(
                f'<div class="paper-card">'
                f"👤 {', '.join(paper.authors[:3])} &nbsp;|&nbsp; 📅 {paper.published}<br>"
                f"🏷️ {', '.join(paper.categories[:3])}<br>"
                f"📝 {paper.abstract[:250]}…<br>"
                f"🔗 <a href='{paper.url}' target='_blank'>View on arXiv</a>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if selected_ids:
            if st.button(
                f"⬇️ Ingest {len(selected_ids)} Selected Paper(s)", type="primary"
            ):
                from arxiv_fetcher import ArxivFetcher
                from document_processor import DocumentProcessor
                from vector_store import VectorStoreManager

                fetcher = ArxivFetcher()
                processor = DocumentProcessor()
                vsm = get_vsm()

                papers_to_ingest = [
                    p
                    for p in st.session_state["fetched_papers"]
                    if p.paper_id in selected_ids
                ]

                progress = st.progress(0, text="Downloading PDFs…")
                papers_to_ingest = fetcher.download_pdfs(papers_to_ingest)

                progress.progress(50, text="Processing & embedding…")
                docs = processor.process_papers(papers_to_ingest)

                try:
                    vsm.load_or_create(docs)
                except Exception:
                    vsm.add_documents(docs)

                for p in papers_to_ingest:
                    st.session_state["ingested_ids"].add(p.paper_id)

                progress.progress(100, text="Done!")
                st.success(
                    f"✅ Ingested {len(papers_to_ingest)} paper(s) "
                    f"({len(docs)} chunks). Ready for RAG Chat!"
                )

                # Reset conv chain so it picks up new docs
                st.session_state["conv_chain"] = None