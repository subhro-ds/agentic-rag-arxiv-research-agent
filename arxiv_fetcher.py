"""
arxiv_fetcher.py — Searches and downloads research papers from arXiv.

Responsibilities:
  - Query the arXiv API using the `arxiv` Python package.
  - Download PDFs locally for ingestion.
  - Return rich metadata (title, authors, abstract, URL, categories).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import arxiv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


@dataclass
class ArxivPaper:
    """A single arXiv paper with its metadata and (optionally) its local PDF path."""

    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: str
    url: str
    pdf_url: str
    categories: List[str]
    local_pdf_path: Optional[str] = None
    summary: str = ""

    # Keep a reference to the raw arxiv.Result for downstream use
    _raw: Optional[object] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "categories": self.categories,
            "local_pdf_path": self.local_pdf_path,
        }


class ArxivFetcher:
    """
    Fetches papers from arXiv and optionally downloads their PDFs.

    Usage
    -----
    >>> fetcher = ArxivFetcher()
    >>> papers = fetcher.search("agentic AI autonomous agents", max_results=5)
    >>> papers = fetcher.download_pdfs(papers)
    """

    SORT_CRITERIA = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }

    def __init__(
        self,
        download_dir: str = settings.ARXIV_DOWNLOAD_DIR,
        max_results: int = settings.ARXIV_MAX_RESULTS,
        sort_by: str = settings.ARXIV_SORT_BY,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self.sort_criterion = self.SORT_CRITERIA.get(
            sort_by, arxiv.SortCriterion.Relevance
        )
        self.client = arxiv.Client(
            page_size=50,
            delay_seconds=3,   # be polite to arXiv servers
            num_retries=5,
        )
        logger.info(f"ArxivFetcher ready. Download dir: {self.download_dir}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> List[ArxivPaper]:
        """
        Search arXiv and return a list of ArxivPaper objects.

        Parameters
        ----------
        query : str
            Search terms, e.g. "agentic AI multi-agent systems".
        max_results : int, optional
            Overrides the instance-level default.
        categories : list[str], optional
            Filter to specific arXiv categories, e.g. ["cs.AI", "cs.LG"].

        Returns
        -------
        List[ArxivPaper]
        """
        n = max_results or self.max_results

        # Optionally narrow to specific categories
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            full_query = f"({query}) AND ({cat_filter})"
        else:
            full_query = query

        logger.info(f"Searching arXiv for: '{full_query}' (max {n})")

        search = arxiv.Search(
            query=full_query,
            max_results=n,
            sort_by=self.sort_criterion,
        )

        papers: List[ArxivPaper] = []
        for result in self.client.results(search):
            paper = ArxivPaper(
                paper_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.strftime("%Y-%m-%d"),
                updated=result.updated.strftime("%Y-%m-%d"),
                url=result.entry_id,
                pdf_url=result.pdf_url,
                categories=result.categories,
                _raw=result,
            )
            papers.append(paper)
            logger.debug(f"  Found: [{paper.paper_id}] {paper.title[:70]}")

        logger.info(f"Found {len(papers)} papers.")
        return papers

    def download_pdfs(
        self, papers: List[ArxivPaper], delay: float = 2.0
    ) -> List[ArxivPaper]:
        """
        Download PDFs for a list of ArxivPaper objects.
        Files are saved as <download_dir>/<paper_id>.pdf.
        Already-downloaded files are skipped.

        Parameters
        ----------
        papers : list[ArxivPaper]
        delay : float
            Seconds to sleep between downloads (arXiv rate limiting).

        Returns
        -------
        List[ArxivPaper] with `local_pdf_path` populated.
        """
        for i, paper in enumerate(papers):
            pdf_path = self.download_dir / f"{paper.paper_id}.pdf"

            if pdf_path.exists():
                logger.info(f"[{i+1}/{len(papers)}] Already downloaded: {pdf_path.name}")
                paper.local_pdf_path = str(pdf_path)
                continue

            try:
                logger.info(f"[{i+1}/{len(papers)}] Downloading: {paper.title[:60]}…")
                if paper._raw:
                    paper._raw.download_pdf(dirpath=str(self.download_dir), filename=f"{paper.paper_id}.pdf")
                else:
                    # Fallback: use arxiv client to re-fetch
                    search = arxiv.Search(id_list=[paper.paper_id])
                    result = next(self.client.results(search))
                    result.download_pdf(dirpath=str(self.download_dir), filename=f"{paper.paper_id}.pdf")

                paper.local_pdf_path = str(pdf_path)
                logger.success(f"  Saved → {pdf_path}")

                if i < len(papers) - 1:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"  Failed to download [{paper.paper_id}]: {e}")

        return papers

    def fetch_by_ids(self, paper_ids: List[str]) -> List[ArxivPaper]:
        """Fetch specific papers by their arXiv IDs (e.g. '2310.06824')."""
        logger.info(f"Fetching {len(paper_ids)} papers by ID…")
        search = arxiv.Search(id_list=paper_ids)
        papers = []
        for result in self.client.results(search):
            papers.append(
                ArxivPaper(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    published=result.published.strftime("%Y-%m-%d"),
                    updated=result.updated.strftime("%Y-%m-%d"),
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    categories=result.categories,
                    _raw=result,
                )
            )
        return papers