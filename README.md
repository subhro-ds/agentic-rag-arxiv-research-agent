# 🤖 Agentic RAG — arXiv Research Assistant

A **Retrieval-Augmented Generation (RAG) system** built with **LangChain** to help researchers explore **Agentic AI** papers from arXiv. The system autonomously searches, ingests, and synthesizes knowledge from complex research papers.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                     │
│    ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│    │  Agent Mode │   │   RAG Chat   │   │  Search & Ingest  │   │
│    └──────┬──────┘   └──────┬───────┘   └────────┬──────────┘   │
└───────────┼─────────────────┼────────────────────┼──────────────┘
            │                 │                    │
            ▼                 ▼                    ▼
┌───────────────────────────────────────────────────────────────┐
│                    Research Agent (ReAct)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │search_arxiv │  │ingest_papers │  │ query_knowledge_base  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬───────────┘ │
└─────────┼────────────────┼─────────────────────┼─────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  ArxivFetcher   │  │DocumentProcessor │  │  VectorStore     │
│  (arxiv API)    │  │ (PyMuPDF/PyPDF)  │  │  (FAISS/Chroma)  │
└─────────────────┘  └──────────────────┘  └──────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │    RAG Chain     │
                                          │  (LangChain LCEL)│
                                          └──────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Claude / GPT-4o │
                                          └──────────────────┘
```

---

## Quick Start

### 1. Clone / Set up the project

```bash
git clone <your-repo>
cd agentic-rag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# Choose your LLM provider: "anthropic" or "openai"
PROVIDER=anthropic

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI (GPT-4o) — only needed if PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Embeddings (local, no API key needed)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector store: "faiss" (default) or "chroma"
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=./data/vector_store

# arXiv settings
ARXIV_MAX_RESULTS=10
ARXIV_DOWNLOAD_DIR=./data/papers
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The UI opens at **http://localhost:8501**.

---

## Modes of Operation

### 🤖 Autonomous Agent Mode
Ask any research question. The agent will:
1. **Search** arXiv for relevant papers.
2. **Download & index** the top papers into FAISS.
3. **Synthesize** an answer with citations using RAG.
4. Show its **full reasoning chain** (ReAct steps).

**Example prompts:**
- *"What are the main architectural patterns in agentic AI systems?"*
- *"Compare ReAct and Chain-of-Thought prompting for tool-using agents."*
- *"Summarize recent work on multi-agent coordination."*

### 📚 RAG Chat Mode
Multi-turn conversational Q&A over already-ingested papers.
Maintains chat history and resolves follow-up questions.

### 🔍 Search & Ingest Mode
Manually search arXiv, browse results, and selectively ingest papers.
Gives you fine-grained control over what enters the knowledge base.

---

## Project Structure

```
agentic-rag/
├── app.py                 # Streamlit UI (three modes)
├── config.py              # Settings from .env
├── arxiv_fetcher.py       # arXiv search + PDF download
├── document_processor.py  # PDF text extraction + chunking
├── vector_store.py        # FAISS / ChromaDB wrapper
├── rag_chain.py           # LangChain RAG pipeline (LCEL)
├── research_agent.py      # ReAct agent + tools
├── requirements.txt
├── .env                   # (you create this)
└── data/
    ├── papers/            # Downloaded PDFs
    └── vector_store/      # FAISS index files
```

---

## Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Agent Framework** | LangChain ReAct | Transparent reasoning, easy tool integration |
| **Embeddings** | sentence-transformers (local) | Free, no API key, good quality |
| **Vector Store** | FAISS (default) | Fast, local, no server required |
| **Retrieval** | MMR (Maximal Marginal Relevance) | Reduces redundant chunks |
| **LLM** | Claude Sonnet / GPT-4o | Strong reasoning + long context |
| **PDF Parsing** | PyMuPDF → PyPDF fallback | Best coverage across paper formats |

---

## Extending the System

### Add a new tool to the agent
```python
# In research_agent.py → build_tools()
def my_new_tool(input: str) -> str:
    ...

tools.append(
    Tool(name="my_tool", func=my_new_tool, description="...")
)
```

### Switch to OpenAI embeddings
```env
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

### Use ChromaDB for persistent filtering
```env
VECTOR_STORE_TYPE=chroma
CHROMA_COLLECTION=my_papers
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'fitz'` | `pip install pymupdf` |
| Slow first embedding | First run downloads the sentence-transformer model (~90MB) |
| arXiv rate limit errors | Increase `delay` in `download_pdfs()` to 5s |
| Empty PDF text | Some papers are scanned; the system falls back to the abstract |
