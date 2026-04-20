"""
vector_store.py — Manages the embedding and retrieval layer.

Supports FAISS (local, fast) and ChromaDB (persistent, filterable).
Embeddings are generated via sentence-transformers (local, free)
or OpenAI text-embedding-3-small (cloud).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from loguru import logger

from config import settings


def _get_embeddings():
    """
    Return the embedding model.
    Uses HuggingFace sentence-transformers by default (no API key needed).
    Switches to OpenAI embeddings when OPENAI_API_KEY is set and the model name
    starts with 'text-embedding'.
    """
    model_name = settings.EMBEDDING_MODEL

    if model_name.startswith("text-embedding") and settings.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings: {model_name}")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    logger.info(f"Using HuggingFace embeddings: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStoreManager:
    """
    Wraps FAISS or ChromaDB with a consistent interface.

    Usage
    -----
    >>> vsm = VectorStoreManager()
    >>> vsm.add_documents(chunks)
    >>> results = vsm.similarity_search("agentic planning loop", k=5)
    >>> retriever = vsm.as_retriever()
    """

    def __init__(
        self,
        store_type: str = settings.VECTOR_STORE_TYPE,
        persist_path: str = settings.VECTOR_STORE_PATH,
    ) -> None:
        self.store_type = store_type.lower()
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = _get_embeddings()
        self._store: Optional[VectorStore] = None
        logger.info(
            f"VectorStoreManager ready (type={self.store_type}, path={self.persist_path})"
        )

    # ── Store Lifecycle ───────────────────────────────────────────────────────

    def load_or_create(self, documents: Optional[List[Document]] = None) -> VectorStore:
        """
        Load an existing store from disk, or create a fresh one from `documents`.
        If both an existing store and new documents are provided, the documents
        are added to the existing store.
        """
        existing = self._try_load()

        if existing and documents:
            logger.info("Loaded existing store. Adding new documents…")
            existing.add_documents(documents)
            self._save(existing)
            self._store = existing
            return existing

        if existing:
            self._store = existing
            return existing

        if documents:
            logger.info(f"Creating new {self.store_type} vector store…")
            store = self._create(documents)
            self._save(store)
            self._store = store
            return store

        raise ValueError("No existing store found and no documents provided to create one.")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the current store and persist."""
        if self._store is None:
            self.load_or_create(documents)
            return
        self._store.add_documents(documents)
        self._save(self._store)
        logger.success(f"Added {len(documents)} documents to the vector store.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = settings.TOP_K_DOCS) -> List[Document]:
        self._require_store()
        return self._store.similarity_search(query, k=k)

    def as_retriever(
        self,
        search_type: str = settings.RETRIEVAL_TYPE,
        k: int = settings.TOP_K_DOCS,
    ):
        """
        Return a LangChain retriever.

        search_type : "similarity" | "mmr"
          MMR (Maximal Marginal Relevance) reduces redundancy in results.
        """
        self._require_store()
        kwargs = {"k": k}
        if search_type == "mmr":
            kwargs["fetch_k"] = k * 3
        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _try_load(self) -> Optional[VectorStore]:
        try:
            if self.store_type == "faiss":
                return self._load_faiss()
            elif self.store_type == "chroma":
                return self._load_chroma()
        except Exception as e:
            logger.warning(f"Could not load existing store: {e}")
        return None

    def _create(self, documents: List[Document]) -> VectorStore:
        if self.store_type == "faiss":
            return self._create_faiss(documents)
        elif self.store_type == "chroma":
            return self._create_chroma(documents)
        raise ValueError(f"Unknown store type: {self.store_type}")

    def _save(self, store: VectorStore) -> None:
        if self.store_type == "faiss":
            store.save_local(str(self.persist_path))
            logger.debug(f"FAISS store saved → {self.persist_path}")
        # ChromaDB auto-persists

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _create_faiss(self, documents: List[Document]) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        logger.info("Building FAISS index…")
        return FAISS.from_documents(documents, self.embeddings)

    def _load_faiss(self) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        index_file = self.persist_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"No FAISS index at {index_file}")
        logger.info(f"Loading FAISS index from {self.persist_path}")
        return FAISS.load_local(
            str(self.persist_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def _create_chroma(self, documents: List[Document]) -> VectorStore:
        from langchain_community.vectorstores import Chroma

        logger.info("Building Chroma collection…")
        return Chroma.from_documents(
            documents,
            self.embeddings,
            collection_name=settings.CHROMA_COLLECTION,
            persist_directory=str(self.persist_path),
        )

    def _load_chroma(self) -> VectorStore:
        from langchain_community.vectorstores import Chroma

        logger.info(f"Loading Chroma from {self.persist_path}")
        return Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def _require_store(self):
        if self._store is None:
            raise RuntimeError(
                "Vector store not initialized. Call load_or_create() first."
            )