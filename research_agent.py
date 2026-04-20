"""
research_agent.py — An autonomous research agent built with LangChain.

The agent has access to four tools:
  1. search_arxiv          — Search arXiv for papers on a topic.
  2. ingest_papers         — Download & index fetched papers into the vector store.
  3. query_knowledge_base  — RAG query over already-ingested papers.
  4. summarize_paper       — Summarise a specific paper by its arXiv ID.

It uses the ReAct (Reason + Act) framework: the LLM reasons about which tool
to use, calls it, observes the result, and repeats until it can answer the
original question.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool, Tool
from loguru import logger
from pydantic import BaseModel, Field

from arxiv_fetcher import ArxivFetcher
from config import settings
from document_processor import DocumentProcessor
from rag_chain import RAGChain, get_llm
from vector_store import VectorStoreManager


# ── Shared state accessible by all tools ─────────────────────────────────────

class AgentState:
    """Mutable state shared across all tool invocations in a single agent session."""

    def __init__(self) -> None:
        self.fetcher = ArxivFetcher()
        self.processor = DocumentProcessor()
        self.vsm = VectorStoreManager()
        self.rag_chain: Optional[RAGChain] = None
        self.ingested_ids: set[str] = set()
        self._store_ready = False

    def ensure_rag_chain(self) -> RAGChain:
        """Lazily initialise the RAG chain after documents are ingested."""
        if self.rag_chain is None:
            if not self._store_ready:
                raise RuntimeError(
                    "No papers ingested yet. Use 'search_arxiv' then 'ingest_papers' first."
                )
            self.rag_chain = RAGChain(self.vsm)
        return self.rag_chain


# ── Tool Input Schemas ────────────────────────────────────────────────────────

class SearchArxivInput(BaseModel):
    query: str = Field(description="Search query for arXiv, e.g. 'agentic AI planning'")
    max_results: int = Field(default=5, description="Number of papers to fetch (1-20)")
    categories: Optional[List[str]] = Field(
        default=None,
        description="Optional arXiv categories to filter, e.g. ['cs.AI', 'cs.LG']",
    )


class IngestPapersInput(BaseModel):
    paper_ids: List[str] = Field(
        description="List of arXiv paper IDs returned by search_arxiv, e.g. ['2310.06824']"
    )


class QueryKBInput(BaseModel):
    question: str = Field(description="Question to answer using the knowledge base")


class SummarizePaperInput(BaseModel):
    paper_id: str = Field(description="arXiv paper ID, e.g. '2310.06824'")


# ── Tool Factory ──────────────────────────────────────────────────────────────

def build_tools(state: AgentState) -> List:
    """Return the list of LangChain tools, all sharing `state`."""

    # ── Tool 1: Search arXiv ─────────────────────────────────────────────────

    def search_arxiv(
        query: str,
        max_results: int = 5,
        categories: Optional[List[str]] = None,
    ) -> str:
        papers = state.fetcher.search(query, max_results=max_results, categories=categories)
        if not papers:
            return "No papers found for this query."

        lines = [f"Found {len(papers)} papers:\n"]
        for p in papers:
            lines.append(
                f"  ID: {p.paper_id}\n"
                f"  Title: {p.title}\n"
                f"  Authors: {', '.join(p.authors[:3])}\n"
                f"  Published: {p.published}\n"
                f"  Abstract snippet: {p.abstract[:200]}…\n"
            )
            # Cache for ingest_papers
            state._paper_cache = {p.paper_id: p for p in papers}

        return "\n".join(lines)

    # ── Tool 2: Ingest Papers ────────────────────────────────────────────────

    def ingest_papers(paper_ids: List[str]) -> str:
        cached = getattr(state, "_paper_cache", {})
        to_ingest = []

        for pid in paper_ids:
            if pid in state.ingested_ids:
                logger.info(f"Already ingested: {pid}")
                continue
            paper = cached.get(pid)
            if not paper:
                # Re-fetch from arXiv
                fetched = state.fetcher.fetch_by_ids([pid])
                if fetched:
                    paper = fetched[0]
                else:
                    return f"Paper {pid} not found on arXiv."
            to_ingest.append(paper)

        if not to_ingest:
            return "All specified papers are already in the knowledge base."

        # Download PDFs
        to_ingest = state.fetcher.download_pdfs(to_ingest)

        # Process and embed
        docs = state.processor.process_papers(to_ingest)
        if not docs:
            return "No content could be extracted from the selected papers."

        try:
            state.vsm.load_or_create(docs)
        except Exception:
            state.vsm.add_documents(docs)

        state._store_ready = True
        state.rag_chain = None  # reset so it's recreated with new docs

        for p in to_ingest:
            state.ingested_ids.add(p.paper_id)

        return (
            f"Successfully ingested {len(to_ingest)} paper(s) "
            f"({sum(1 for p in to_ingest if p.local_pdf_path)} with full PDF). "
            f"Total chunks in knowledge base: {len(docs)}. "
            f"Knowledge base is now ready for queries."
        )

    # ── Tool 3: Query Knowledge Base ─────────────────────────────────────────

    def query_knowledge_base(question: str) -> str:
        rag = state.ensure_rag_chain()
        result = rag.invoke_with_sources(question)
        answer = result["answer"]
        sources = result["sources"]

        source_lines = "\n\nSources consulted:"
        for s in sources[:3]:
            source_lines += (
                f"\n  • {s['title']} ({s['authors']}, {s['published']}) — {s['url']}"
            )

        return answer + source_lines

    # ── Tool 4: Summarize Paper ──────────────────────────────────────────────

    def summarize_paper(paper_id: str) -> str:
        cached = getattr(state, "_paper_cache", {})
        paper = cached.get(paper_id)

        if not paper:
            fetched = state.fetcher.fetch_by_ids([paper_id])
            if not fetched:
                return f"Paper {paper_id} not found."
            paper = fetched[0]

        llm = get_llm()
        from langchain_core.messages import HumanMessage

        prompt = (
            f"Summarize the following research paper in 5-7 bullet points. "
            f"Focus on: (1) problem addressed, (2) key method/approach, "
            f"(3) main contributions, (4) results/findings, (5) limitations.\n\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Abstract:\n{paper.abstract}"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        return f"Summary of '{paper.title}':\n\n{response.content}"

    # ── Assemble tools ───────────────────────────────────────────────────────

    return [
        StructuredTool(
            name="search_arxiv",
            func=search_arxiv,
            description=(
                "Search arXiv for research papers on a given topic. "
                "Returns paper IDs, titles, authors, and abstract snippets. "
                "Use this FIRST to discover relevant papers."
            ),
            args_schema=SearchArxivInput,
        ),
        StructuredTool(
            name="ingest_papers",
            func=ingest_papers,
            description=(
                "Download and index selected arXiv papers into the knowledge base. "
                "Takes a list of paper IDs from search_arxiv. "
                "Must be called before query_knowledge_base."
            ),
            args_schema=IngestPapersInput,
        ),
        StructuredTool(
            name="query_knowledge_base",
            func=query_knowledge_base,
            description=(
                "Answer a question using the ingested papers (RAG). "
                "Returns an answer with citations. "
                "Only works after ingest_papers has been called."
            ),
            args_schema=QueryKBInput,
        ),
        StructuredTool(
            name="summarize_paper",
            func=summarize_paper,
            description=(
                "Generate a structured bullet-point summary of a specific paper "
                "given its arXiv ID. Does not require ingest_papers."
            ),
            args_schema=SummarizePaperInput,
        ),
    ]


# ── Agent Prompt ──────────────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant specializing in Agentic AI.
Your goal is to help researchers efficiently explore arXiv papers.

You have access to these tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: reason about what to do next
Action: the action to take (one of [{tool_names}])
Action Input: the input to the action (as a JSON object matching the tool's schema)
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the complete answer to the original question

Strategy:
1. Always use search_arxiv first to find relevant papers.
2. Use ingest_papers to index the most relevant ones (pick 3-5).
3. Use query_knowledge_base for detailed questions.
4. Use summarize_paper when the user asks about a specific paper.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


# ── Agent Builder ─────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    An autonomous agent that searches, ingests, and answers questions
    about arXiv papers on Agentic AI.

    Usage
    -----
    >>> agent = ResearchAgent()
    >>> result = agent.run("What are the main approaches to tool use in LLM agents?")
    >>> print(result)
    """

    def __init__(self) -> None:
        self.state = AgentState()
        self.llm = get_llm()
        self.tools = build_tools(self.state)

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=REACT_PROMPT,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=settings.AGENT_VERBOSE,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        logger.info("ResearchAgent ready.")

    def run(self, query: str) -> dict[str, Any]:
        """
        Run the agent on a query.

        Returns
        -------
        dict with keys:
          - 'output'  : final answer string
          - 'steps'   : list of (action, observation) intermediate steps
        """
        logger.info(f"Agent query: {query}")
        result = self.executor.invoke({"input": query})
        return {
            "output": result.get("output", ""),
            "steps": [
                {
                    "action": step[0].tool,
                    "action_input": step[0].tool_input,
                    "observation": str(step[1])[:500],
                }
                for step in result.get("intermediate_steps", [])
            ],
        }