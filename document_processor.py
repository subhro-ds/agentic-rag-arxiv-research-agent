"""
document_processor.py — Loads, cleans, and chunks arXiv PDFs for the vector store.

Pipeline
--------
PDF → PyMuPDF/PyPDF → clean text → RecursiveCharacterTextSplitter → Document chunks
with rich metadata attached to every chunk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from arxiv_fetcher import ArxivPaper
from config import settings


class DocumentProcessor:
    """
    Converts ArxivPaper objects (with local PDFs) into chunked LangChain Documents
    ready for embedding.

    Usage
    -----
    >>> processor = DocumentProcessor()
    >>> docs = processor.process_papers(papers)
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"DocumentProcessor ready (chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_papers(self, papers: List[ArxivPaper]) -> List[Document]:
        """
        Process a list of ArxivPaper objects.

        For papers with a downloaded PDF the full text is extracted.
        For papers without a PDF the abstract is used as a fallback.

        Returns a flat list of LangChain Document chunks.
        """
        all_docs: List[Document] = []

        for paper in papers:
            try:
                if paper.local_pdf_path and Path(paper.local_pdf_path).exists():
                    docs = self._process_pdf(paper)
                else:
                    logger.warning(
                        f"No PDF for [{paper.paper_id}], using abstract as fallback."
                    )
                    docs = self._process_abstract(paper)

                all_docs.extend(docs)
                logger.info(
                    f"[{paper.paper_id}] → {len(docs)} chunks "
                    f"({'PDF' if paper.local_pdf_path else 'abstract'})"
                )

            except Exception as e:
                logger.error(f"Failed to process [{paper.paper_id}]: {e}")
                # Graceful fallback to abstract
                try:
                    all_docs.extend(self._process_abstract(paper))
                except Exception:
                    pass

        logger.success(f"Total chunks produced: {len(all_docs)}")
        return all_docs

    def process_raw_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split arbitrary text into chunks (useful for testing)."""
        chunks = self.splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _process_pdf(self, paper: ArxivPaper) -> List[Document]:
        """Extract text from PDF using PyMuPDF (fitz) with PyPDF fallback."""
        text = self._extract_text_pymupdf(paper.local_pdf_path)

        if not text or len(text.strip()) < 200:
            logger.warning(f"PyMuPDF extraction thin, trying PyPDF for {paper.paper_id}")
            text = self._extract_text_pypdf(paper.local_pdf_path)

        text = self._clean_text(text)
        return self._chunk(text, paper)

    def _extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fast and accurate)."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to PyPDF.")
            return ""
        except Exception as e:
            logger.error(f"PyMuPDF error: {e}")
            return ""

    def _extract_text_pypdf(self, pdf_path: str) -> str:
        """Extract text using PyPDF (reliable fallback)."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as e:
            logger.error(f"PyPDF error: {e}")
            return ""

    def _process_abstract(self, paper: ArxivPaper) -> List[Document]:
        """Use the abstract when no PDF is available."""
        text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        return self._chunk(text, paper)

    def _chunk(self, text: str, paper: ArxivPaper) -> List[Document]:
        """Split text into chunks and attach paper metadata."""
        if not text.strip():
            return []

        metadata = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": ", ".join(paper.authors[:3])
            + (" et al." if len(paper.authors) > 3 else ""),
            "published": paper.published,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "categories": ", ".join(paper.categories),
            "source": "arxiv",
        }

        chunks = self.splitter.split_text(text)
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
            )
            docs.append(doc)
        return docs

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove common PDF artifacts while preserving content structure."""
        if not text:
            return ""

        # Remove form feeds and carriage returns
        text = text.replace("\f", "\n").replace("\r", "\n")

        # Collapse excessive whitespace / blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove lines that are clearly page numbers or headers (very short, numeric)
        lines = text.split("\n")
        cleaned = [
            line for line in lines
            if not re.match(r"^\s*\d{1,3}\s*$", line)
        ]
        return "\n".join(cleaned).strip()