"""
config.py — Central configuration for the Agentic RAG system.
All environment variables and tunable parameters live here.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Set PROVIDER to "openai" or "anthropic"
    PROVIDER: str = os.getenv("PROVIDER", "anthropic")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Vector Store ──────────────────────────────────────────────────────────
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")  # "faiss" | "chroma"
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "arxiv_papers")

    # ── Arxiv Fetcher ─────────────────────────────────────────────────────────
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "10"))
    ARXIV_DOWNLOAD_DIR: str = os.getenv("ARXIV_DOWNLOAD_DIR", "./data/papers")
    ARXIV_SORT_BY: str = os.getenv("ARXIV_SORT_BY", "relevance")  # "relevance" | "lastUpdatedDate" | "submittedDate"

    # ── Text Splitting ────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_DOCS: int = int(os.getenv("TOP_K_DOCS", "5"))
    RETRIEVAL_TYPE: str = os.getenv("RETRIEVAL_TYPE", "mmr")  # "similarity" | "mmr"

    # ── Agent ─────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

    class Config:
        env_file = ".env"


settings = Settings()