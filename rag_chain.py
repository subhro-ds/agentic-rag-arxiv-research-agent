"""
rag_chain.py — Builds the Retrieval-Augmented Generation pipeline using LangChain.

Architecture
------------
User question
    │
    ▼
[Retriever]  ─── top-k relevant chunks from vector store
    │
    ▼
[Prompt]     ─── system + context + question
    │
    ▼
[LLM]        ─── Claude / GPT
    │
    ▼
[StrOutputParser] → final answer (with citations)

Also exposes a ConversationalRAGChain that maintains chat history.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from loguru import logger

from config import settings
from vector_store import VectorStoreManager


# ── LLM Factory ──────────────────────────────────────────────────────────────

def get_llm(streaming: bool = False):
    """Return the configured LLM (Anthropic Claude or OpenAI GPT)."""
    if settings.PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            streaming=streaming,
            temperature=0.2,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=streaming,
        temperature=0.2,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert research assistant specializing in Agentic AI and machine learning.
Your role is to help researchers navigate complex arXiv papers efficiently.

You have been given a set of relevant excerpts from research papers (the "context").
Use ONLY this context to answer the user's question. If the answer is not contained
in the context, say so clearly — do NOT hallucinate.

Guidelines:
- Cite paper titles and authors when referencing specific claims.
- Highlight key contributions, methods, and findings.
- Explain technical jargon in plain language when helpful.
- Be concise but thorough; prioritize the most important information.
- When multiple papers discuss the same concept, synthesize their perspectives.

Context from retrieved papers:
{context}
"""

CONDENSE_QUESTION_PROMPT = """\
Given the following conversation history and a follow-up question, rephrase the
follow-up question so it is self-contained (includes all necessary context).

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (
            f"[{i}] {meta.get('title', 'Unknown Title')} "
            f"({meta.get('authors', '')}, {meta.get('published', '')})"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Simple RAG Chain ──────────────────────────────────────────────────────────

class RAGChain:
    """
    Single-turn RAG chain.

    Usage
    -----
    >>> chain = RAGChain(vsm)
    >>> answer = chain.invoke("What is the ReAct framework?")
    >>> for token in chain.stream("Explain tool use in LLM agents"):
    ...     print(token, end="", flush=True)
    """

    def __init__(self, vsm: VectorStoreManager, streaming: bool = False) -> None:
        self.vsm = vsm
        self.llm = get_llm(streaming=streaming)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        retriever = vsm.as_retriever()

        self.chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAGChain initialized.")

    def invoke(self, question: str) -> str:
        """Return the full answer string."""
        logger.info(f"RAG query: {question[:80]}")
        return self.chain.invoke(question)

    def stream(self, question: str) -> Iterator[str]:
        """Stream the answer token by token."""
        logger.info(f"RAG stream query: {question[:80]}")
        yield from self.chain.stream(question)

    def invoke_with_sources(self, question: str) -> Dict[str, Any]:
        """Return answer + source documents."""
        retriever = self.vsm.as_retriever()
        source_docs = retriever.invoke(question)
        answer = self.invoke(question)
        return {
            "answer": answer,
            "sources": [
                {
                    "title": d.metadata.get("title", ""),
                    "authors": d.metadata.get("authors", ""),
                    "published": d.metadata.get("published", ""),
                    "url": d.metadata.get("url", ""),
                    "snippet": d.page_content[:300],
                }
                for d in source_docs
            ],
        }


# ── Conversational RAG Chain ──────────────────────────────────────────────────

class ConversationalRAGChain:
    """
    Multi-turn RAG chain with chat history.

    Each session is keyed by `session_id`.

    Usage
    -----
    >>> conv = ConversationalRAGChain(vsm)
    >>> conv.chat("What is AutoGPT?", session_id="user-1")
    >>> conv.chat("How does it compare to BabyAGI?", session_id="user-1")
    """

    def __init__(self, vsm: VectorStoreManager) -> None:
        self.vsm = vsm
        self.llm = get_llm()
        self._sessions: Dict[str, ChatMessageHistory] = {}

        retriever = vsm.as_retriever()

        # Step 1: Condense follow-up → standalone question
        condense_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
                (
                    "human",
                    "Given the conversation above, generate a standalone question "
                    "that contains all necessary context.",
                ),
            ]
        )
        condense_chain = condense_prompt | self.llm | StrOutputParser()

        # Step 2: RAG answer with context
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        def retrieve_and_format(inputs: dict) -> dict:
            question = inputs.get("standalone_question", inputs["question"])
            docs = retriever.invoke(question)
            return {
                "context": _format_docs(docs),
                "question": inputs["question"],
                "chat_history": inputs.get("chat_history", []),
            }

        self.chain = (
            RunnablePassthrough.assign(
                standalone_question=condense_chain
            )
            | retrieve_and_format
            | answer_prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("ConversationalRAGChain initialized.")

    def chat(self, question: str, session_id: str = "default") -> str:
        """Send a message and get a response (history-aware)."""
        history = self._sessions.setdefault(session_id, ChatMessageHistory())

        answer = self.chain.invoke(
            {"question": question, "chat_history": history.messages}
        )

        history.add_user_message(question)
        history.add_ai_message(answer)
        logger.info(f"[session={session_id}] Q: {question[:50]} | A: {answer[:80]}")
        return answer

    def get_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        history = self._sessions.get(session_id)
        if not history:
            return []
        return [
            {"role": "human" if msg.type == "human" else "ai", "content": msg.content}
            for msg in history.messages
        ]

    def clear_history(self, session_id: str = "default") -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info(f"Cleared history for session '{session_id}'")