# Agentic Deep Research System

An end-to-end **agentic RAG pipeline** that answers complex research questions about LLM agents by decomposing them into sub-questions, retrieving relevant academic papers from arXiv, synthesizing grounded answers with inline citations, and verifying claims against source material.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Command Reference](#command-reference)
7. [Technical Deep Dive](#technical-deep-dive)
8. [Evaluation & Ablation Studies](#evaluation--ablation-studies)
9. [Interactive Demo](#interactive-demo)
10. [Reproducing the Project](#reproducing-the-project)
11. [Troubleshooting](#troubleshooting)

---

## What This Project Does

This project implements an **agentic deep research pipeline** that combines classical information retrieval with large language models to answer research questions with cited evidence from arXiv papers (2024–2026).

### Core Use Case

Given a research question like *"What are recent advances in multi-agent reasoning?"*, the system:

1. **Plans** — Decomposes the question into 2–4 focused, non-overlapping sub-questions
2. **Retrieves** — Fetches relevant passages from ~700 academic papers using hybrid BM25 + semantic search with RRF fusion
3. **Reflects** — Evaluates whether retrieved evidence is sufficient; if not, generates refined queries and retrieves again (up to 3 rounds)
4. **Synthesizes** — Writes a comprehensive answer grounded **only** in retrieved passages, with inline `[arxiv:ID]` citations and a three-tier claim discipline (supported / background / uncertain)
5. **Verifies** — Checks each citation via lexical word-overlap between citing sentences and source passages (no extra LLM call; fast and deterministic)

### Key Design Principles

- **Interpretable** — Every claim is cited with `[arxiv:ID]`; you can trace which paper supports which statement
- **Zero-cost capable** — Runs entirely locally with Ollama, or uses free APIs (Groq, OpenRouter). No paid APIs required
- **Measurable** — 7 ablation configurations to quantify each component's contribution across 30 evaluation questions
- **Modular** — Swap LLM providers, toggle individual pipeline stages, or change retrieval parameters via config flags

---

## System Architecture

### High-Level Pipeline

```
User Question
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              Agent Pipeline (5 Stages)                │
├──────────────────────────────────────────────────────┤
│ 1. PLANNER     → Decompose into 2–4 sub-questions    │
│ 2. RETRIEVER   → Hybrid BM25 + semantic + RRF fusion │
│ 3. REFLECTOR   → Loop until evidence is sufficient   │
│ 4. SYNTHESIZER → Generate cited answer (3-tier)      │
│ 5. VERIFIER    → Lexical overlap faithfulness check  │
└──────────────────────────────────────────────────────┘
    │
    ▼
Answer + [arxiv:ID] citations + verification stats
    │
    ├──→ predictions/*.jsonl     (saved results)
    ├──→ eval/evaluate.py        (LLM-as-judge + metrics)
    └──→ demo/app.py             (Streamlit UI)
```

### Data Collection & Indexing Pipeline (One-time)

```
arXiv API (cs.CL, cs.AI, cs.LG)
    │
    ▼
scraper/collect.py          → data/raw/metadata.jsonl + data/raw/pdfs/
    │
    ▼
indexer/chunk.py            → data/chunks/*_chunks.jsonl
    │                         (section-aware sliding window, 1800 chars, 200 overlap)
    ▼
indexer/build_index.py      → data/index/bm25.pkl
                            → data/index/chroma/ (ChromaDB persistent store)
```

---

## Project Structure

```
DEEP_RESEARCH/
├── agent/
│   └── Agent.py                 # 5-stage agentic pipeline + 7 ablation configs
├── indexer/
│   ├── chunk.py                 # Section-aware PDF chunking (PyMuPDF)
│   ├── build_index.py           # BM25 + ChromaDB index builder
│   └── retriever.py             # Hybrid retriever (BM25 + semantic + RRF + reranker)
├── scraper/
│   └── collect.py               # arXiv API collector with keyword filtering
├── eval/
│   ├── evaluate.py              # LLM-as-judge + citation precision/recall
│   ├── questions.jsonl          # 30 evaluation questions (factoid/comparative/survey)
│   └── results.json             # Cached ablation results
├── demo/
│   └── app.py                   # Streamlit interactive research UI (dark/light mode)
├── data/
│   ├── raw/                     # metadata.jsonl + pdfs/
│   ├── chunks/                  # Per-paper chunk JSONL files
│   └── index/                   # bm25.pkl + chroma/ vector store
├── predictions/                 # Agent outputs per config (*.jsonl)
├── logs/                        # Execution logs (run.log)
├── cache/                       # Runtime cache directory
├── llm_client.py                # Multi-provider LLM client (Groq/Ollama/OpenRouter/OpenAI)
├── run_pipeline.py              # Master end-to-end pipeline orchestrator
├── run.py                       # Run specific agent configs on eval questions
├── download_pdfs.py             # Standalone PDF downloader for existing metadata
├── chroma_db.py                 # Direct ChromaDB query utility
├── requirements.txt             # Python dependencies
├── .env                         # API keys (gitignored)
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** (3.10+ recommended)
- **~10 GB disk space** (PDFs + indexes for full 700-paper corpus)
- **4+ GB RAM** (embedding models + agent inference)
- **Internet connection** (arXiv API + LLM API calls)
- **macOS, Linux, or Windows** (supports MPS on Apple Silicon, CUDA on NVIDIA, or CPU)

### Step 1: Clone & Install Dependencies

```bash
git clone <repo-url>
cd DEEP_RESEARCH

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| `pymupdf` | ≥1.24.0 | PDF text extraction via `fitz` |
| `sentence-transformers` | ≥2.7.0 | Embeddings (`bge-small-en-v1.5`) + reranker (`ms-marco-MiniLM`) |
| `rank-bm25` | ≥0.2.2 | BM25Okapi lexical search |
| `chromadb` | ≥0.5.0 | Persistent vector database |
| `requests` | ≥2.31.0 | HTTP client |
| `tqdm` | ≥4.66.0 | Progress bars |
| `python-dotenv` | ≥1.0.0 | `.env` config loading |
| `numpy` | ≥1.26.0 | Numerical operations |
| `streamlit` | ≥1.35.0 | Interactive demo UI |
| `pandas` | ≥2.2.0 | Data tables in demo sidebar |

### Step 2: Configure LLM Provider

The system auto-detects available providers in priority order. Create a `.env` file with at least one key:

```bash
# .env — add one or more keys

# Option 1: Groq (recommended — free, fast, highest priority)
GROQ_API_KEY=gsk_your_key_here

# Option 2: Ollama (fully local, no API key needed)
# Just run: ollama serve && ollama pull llama3.2

# Option 3: OpenRouter (free tier available)
OPENROUTER_API_KEY=sk-or-your_key_here

# Option 4: OpenAI (paid, reliable fallback)
OPENAI_API_KEY=sk-your_key_here
```

**Provider priority order** (configured in `llm_client.py`):

| Priority | Provider | Model | Cost |
|----------|----------|-------|------|
| 1 | **Groq** | `llama-3.3-70b-versatile` | Free |
| 2 | **Ollama** | `llama3.2:latest` (local) | Free |
| 3 | **OpenRouter** | `mistral-7b-instruct:free` | Free |
| 4 | **OpenAI** | `gpt-4o-mini` | Paid |

All providers are accessed via the OpenAI-compatible API format. The client tries each available provider in order and falls back automatically on failure.

### Step 3: Verify Installation

```bash
# Test core imports
python -c "
from sentence_transformers import SentenceTransformer
from chromadb import Client
from rank_bm25 import BM25Okapi
print('✅ All dependencies installed')
"

# Test LLM connectivity
python -c "from llm_client import call_llm; print(call_llm('Say hello in one sentence'))"
```

---

## Quick Start

### Option 1: Full Pipeline (Recommended First Run)

```bash
# Quick test — 5 papers, no PDFs (fast, ~2 minutes)
python run_pipeline.py --max 5 --skip_pdfs

# Medium run — 50 papers with PDFs
python run_pipeline.py --max 50

# Full corpus — 700 papers (production run)
python run_pipeline.py --max 700
```

This runs the complete pipeline:
1. **Collect** papers from arXiv API (categories: cs.CL, cs.AI, cs.LG)
2. **Chunk** PDFs into section-aware passages
3. **Build** BM25 + ChromaDB indexes
4. **Run** all 7 agent configs on 30 evaluation questions
5. **Evaluate** with LLM-as-judge scoring
6. **Print** ablation comparison table

### Option 2: Skip Completed Steps

```bash
# Skip collection (data already exists)
python run_pipeline.py --skip_collect

# Skip index build (index already exists)
python run_pipeline.py --skip_index

# Only run evaluation (everything else done)
python run_pipeline.py --only_eval

# Skip agent, just re-evaluate existing predictions
python run_pipeline.py --skip_agent
```

### Option 3: Run Specific Agent Configs

```bash
# Single config
python run.py --config full_agent

# Multiple configs
python run.py --config full_agent baseline no_planner

# All 7 ablation configs
python run.py --config all
```

### Option 4: Interactive Demo

```bash
streamlit run demo/app.py
# Opens at http://localhost:8501
```

---

## Command Reference

### Master Pipeline (`run_pipeline.py`)

```bash
python run_pipeline.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max N` | 700 | Max papers to collect |
| `--start YYYY-MM-DD` | 2024-01-01 | Collection start date |
| `--end YYYY-MM-DD` | 2026-04-30 | Collection end date |
| `--skip_pdfs` | — | Skip PDF download (metadata only) |
| `--skip_collect` | — | Skip corpus collection step |
| `--skip_index` | — | Skip index building step |
| `--skip_agent` | — | Skip running agent configs |
| `--only_eval` | — | Only run evaluation (sets all skip flags) |
| `--no_llm_judge` | — | Skip LLM-as-judge scoring (fast mode) |
| `--configs ...` | all 7 | Which agent configs to run |

### Individual Pipeline Steps

| Command | Purpose |
|---------|---------|
| `python scraper/collect.py --max 50 --start 2025-01-01 --end 2026-04-30` | Collect papers from arXiv |
| `python scraper/collect.py --max 20 --skip_pdfs` | Metadata only, no PDFs |
| `python indexer/chunk.py --raw data/raw --out data/chunks` | Chunk PDFs into passages |
| `python indexer/build_index.py --chunks data/chunks --index data/index` | Build BM25 + ChromaDB index |
| `python download_pdfs.py` | Download PDFs for existing metadata |

### Agent Execution (`run.py`)

```bash
python run.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config NAME [NAME ...]` | `full_agent` | Config name(s) or `all` |
| `--questions PATH` | `eval/questions.jsonl` | Questions file |
| `--index_dir PATH` | `data/index` | Index directory |
| `--chunks_dir PATH` | `data/chunks` | Chunks directory |

**Available configs** (7 ablation variants):

| Config | Planner | Hybrid | Reranker | Reflector | Verifier |
|--------|---------|--------|----------|-----------|----------|
| `full_agent` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `baseline` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `no_planner` | ❌ | ✅ | ✅ | ✅ | ✅ |
| `no_reranker` | ✅ | ✅ | ❌ | ✅ | ✅ |
| `no_reflector` | ✅ | ✅ | ✅ | ❌ | ✅ |
| `no_hybrid` | ✅ | ❌ | ✅ | ✅ | ✅ |
| `no_citation_verifier` | ✅ | ✅ | ✅ | ✅ | ❌ |

### Evaluation (`eval/evaluate.py`)

```bash
python eval/evaluate.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--preds PATH` | `predictions/` | Predictions directory |
| `--output PATH` | `eval/results.json` | Output results file |
| `--gt PATH` | None | Ground truth JSONL (optional) |
| `--no_llm_judge` | — | Skip LLM judge (fast mode) |

---

## Technical Deep Dive

### Stage 1: Planner (`agent/Agent.py → plan()`)

Decomposes a complex research question into 2–4 focused, non-overlapping sub-questions using an LLM call.

```
Input:  "What recent advances have been made in multi-agent systems?"
    ↓
Output: [
  "What frameworks enable multi-agent coordination?",
  "How do agents communicate in multi-agent systems?",
  "What benchmarks evaluate multi-agent performance?"
]
```

- Uses `call_llm_json()` with structured JSON output
- Falls back to `[original_question]` on any failure (pipeline never blocks)
- Cap: max 4 sub-questions, all sanitized and deduplicated

**Ablation**: `planner=False` → uses original question as-is

---

### Stage 2: Retriever (`indexer/retriever.py → Retriever`)

Hybrid retrieval combining BM25 lexical search with dense semantic search, fused via Reciprocal Rank Fusion (RRF).

```
Query
  ├──→ BM25Okapi (keyword matching)
  │      └── Score threshold: ≥1.0 (hard cutoff)
  │
  ├──→ ChromaDB semantic search (BAAI/bge-small-en-v1.5)
  │      └── Cosine similarity threshold: ≥0.40 (hard cutoff)
  │
  └──→ RRF Fusion (k=60)
         └── score(doc) = Σ 1/(60 + rank_i) across both lists
              │
              ▼
       [Optional] Cross-encoder reranking
       (cross-encoder/ms-marco-MiniLM-L-6-v2)
              │
              ▼
       Top-5 chunks returned
```

**Key parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `top_k` | 5 | Chunks returned per query |
| `candidate_k` | 30 | Candidates before reranking |
| `RRF_K` | 60 | RRF smoothing constant |
| `SEMANTIC_SIM_THRESHOLD` | 0.40 | Minimum cosine similarity |
| `BM25_SCORE_THRESHOLD` | 1.0 | Minimum BM25 score |

**Hard cutoff filters** prevent irrelevant passages from entering the pipeline — a critical defense against hallucination.

**Multi-query**: `retrieve_multi()` retrieves for all sub-questions and deduplicates by `chunk_id`, keeping the highest-scoring instance.

---

### Stage 3: Reflector (`agent/Agent.py → reflect()`)

Iterative evidence quality loop. An LLM evaluates whether the accumulated passages are sufficient to answer the question. If not, it generates refined queries for another retrieval round.

```python
# Pseudocode
for round in range(1, MAX_ROUNDS+1):  # MAX_ROUNDS = 3
    chunks = retriever.retrieve_multi(queries)
    sufficient, refined = reflect(question, chunks, round)
    if sufficient or not refined:
        break
    queries = refined  # up to 3 refined queries
```

- Builds a compact evidence summary (deduplicated by `arxiv_id`, max 6 chunks)
- Returns `(sufficient: bool, refined_queries: list[str])`
- Hard stop at round 3 regardless of sufficiency

**Ablation**: `reflector=False` → marks sufficient immediately (single-pass retrieval)

---

### Stage 4: Synthesizer (`agent/Agent.py → synthesize()`)

Generates a grounded, cited answer using a detailed prompt with three-tier claim discipline:

| Tier | Description | Action |
|------|-------------|--------|
| **(A) Supported** | Directly backed by a retrieved passage | Cite with `[arxiv:ID]` |
| **(B) Background** | General knowledge, not paper-specific | No citation, optionally note "(general background)" |
| **(C) Uncertain** | Not found in evidence | Flag in "Uncertainty notes" section |

**Evidence block construction**:
- Each chunk is tagged `[PRIMARY]` or `[CONTEXT]` based on lexical word-overlap (≥10% overlap = PRIMARY)
- Includes paper title, section header, and first 600 chars of text
- Max 10 chunks passed to the synthesizer

**Post-processing** (`_postprocess_answer()`):
- Strips preamble phrases ("Sure!", "Certainly!")
- Replaces banned refusal phrases with approved alternatives
- Ensures "Uncertainty notes:" section exists
- Normalizes whitespace

---

### Stage 5: Citation Verifier (`agent/Agent.py → verify_citations()`)

Fast, deterministic citation verification using **lexical word-overlap** — no LLM calls.

```
For each cited arxiv_id:
  1. Check if ID exists in retrieved chunks (else → hallucinated)
  2. Find all sentences in the answer that cite this ID
  3. Compute word-overlap between citing sentences and source passage
  4. If avg overlap ≥ 20% → verified; else → unverified
```

**Output metrics**:
- `verified_ids` — Citations grounded in source passages
- `hallucinated_ids` — Citations referencing papers not in the retrieved corpus
- `unverified_ids` — Citations in corpus but with low lexical overlap
- `faithfulness` — `len(verified) / len(total_cited)`

**Ablation**: `verifier=False` → treats all grounded citations as verified (skips overlap check)

---

### Chunking Strategy (`indexer/chunk.py`)

PDFs are processed with section-aware sliding-window chunking:

1. **Extract text** from PDF via PyMuPDF (`fitz`)
2. **Clean** — de-hyphenate, collapse blanks, remove page numbers
3. **Split by section headers** — regex matches standard academic sections (Abstract, Introduction, Methods, etc.)
4. **Sliding window** — 1800 chars per chunk, 200 char overlap, break at sentence boundaries
5. **Context injection** — Each chunk text is prefixed with:
   ```
   [DOCUMENT SOURCE PAPER: <title> (arXiv ID: <id>)]
   [SECTION: <section_name>]
   ```
   This prevents the LLM from confusing sources.

**Parameters**: `MAX_CHARS=1800` (~450 tokens), `OVERLAP_CHARS=200`, `MIN_CHARS=80`

---

### LLM Client (`llm_client.py`)

Unified multi-provider client using the OpenAI-compatible API format:

- **Auto-detection**: Checks each provider in priority order (Groq → Ollama → OpenRouter → OpenAI)
- **Fallback**: If one provider fails, tries the next available
- **JSON mode**: `call_llm_json()` appends a JSON instruction and extracts JSON from the response via regex
- **Temperature**: 0.1 for deterministic outputs
- **`.env` auto-loading**: Reads `.env` file on import

---

## Evaluation & Ablation Studies

### Evaluation Metrics

The evaluation system (`eval/evaluate.py`) computes metrics across three categories:

#### 1. LLM-as-Judge (3 dimensions, 1–5 scale)
- **Accuracy** — Do claims match the source material?
- **Completeness** — Does it fully answer the question?
- **Coherence** — Is the explanation clear and well-structured?

#### 2. Faithfulness (from Citation Verifier)
- Lexical overlap-based score from `verify_citations()`
- `faithfulness = len(verified_ids) / len(cited_ids)`

#### 3. Citation Precision
- `precision = (cited - hallucinated) / cited`
- Measures what fraction of cited papers are actually in the retrieved corpus

### Ablation Results (30 questions × 7 configs)

Results from `eval/results.json`:

| Config | n | Accuracy | Completeness | Coherence | Faithfulness | Cite-P | Avg Latency | Tool Calls |
|--------|---|----------|-------------|-----------|-------------|--------|-------------|------------|
| **full_agent** | 30 | 3.07 | 3.53 | 3.70 | 0.959 | 0.959 | 49.5s | 5.5 |
| **baseline** | 30 | 3.30 | 3.53 | 3.87 | 0.960 | 0.960 | 19.6s | 1.0 |
| **no_planner** | 30 | 3.03 | 3.53 | 3.87 | 0.958 | 0.958 | 30.2s | 4.9 |
| **no_reranker** | 30 | 3.07 | 3.60 | 3.53 | 0.981 | 0.981 | 38.5s | 5.7 |
| **no_reflector** | 30 | 3.00 | 3.47 | 3.77 | 0.971 | 0.971 | 28.8s | 1.8 |
| **no_hybrid** | 30 | 2.93 | 3.73 | 3.40 | 0.959 | 0.959 | 43.4s | 5.8 |
| **no_citation_verifier** | 30 | 3.00 | 3.40 | 3.60 | 0.977 | 0.977 | 41.7s | 5.6 |

### Key Insights

- **Hybrid retrieval matters** — `no_hybrid` has the lowest accuracy (2.93), showing BM25+semantic fusion improves answer quality
- **Reflector adds depth** — `no_reflector` reduces tool calls from 5.5→1.8 but reduces completeness
- **Planner reduces latency** — `no_planner` is faster (30s vs 49s) with marginal quality impact
- **Citation verifier is cheap** — Lexical overlap verification adds negligible latency
- **Full agent trades speed for thoroughness** — ~2.5× slower than baseline but retrieves more diverse evidence

### Evaluation Questions

The 30 evaluation questions in `eval/questions.jsonl` cover three types:

| Type | Count | IDs | Description |
|------|-------|-----|-------------|
| **Factoid** | 10 | f01–f10 | Specific facts (e.g., "What is ReAct?") |
| **Comparative** | 10 | c01–c10 | Compare methods (e.g., "Self-RAG vs standard RAG") |
| **Survey** | 10 | s01–s10 | Broad surveys (e.g., "Agent memory architectures in 2024–2025") |

---

## Interactive Demo

The Streamlit demo (`demo/app.py`) provides a polished research interface:

```bash
streamlit run demo/app.py
```

### Features

- **Live agent execution** — Watch the pipeline run step-by-step (plan → retrieve → reflect → synthesize → verify)
- **Component toggles** — Enable/disable each agent stage via sidebar switches (planner, hybrid, reranker, reflector, verifier)
- **Dark / Light mode** — Full dual-theme support with curated color palettes
- **Source cards** — Clickable cards for each retrieved paper with snippets
- **Inline citations** — `[arxiv:ID]` rendered as interactive superscript badges linking to arXiv
- **Reference list** — Full reference section with verification status badges (Verified ✓ / Not in corpus ⚠)
- **Research trace** — Expandable section showing the full pipeline execution trace
- **Stats row** — Passages retrieved, citations, faithfulness %, latency, rounds
- **Suggested questions** — Pre-loaded example questions for quick testing
- **Ablation table** — Sidebar displays `eval/results.json` if available

---

## Reproducing the Project

### Full Reproduction

```bash
# 1. Fresh environment
cd DEEP_RESEARCH
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure LLM
echo 'GROQ_API_KEY=gsk_your_key_here' > .env

# 4. Quick test (2 minutes)
python run_pipeline.py --max 5 --skip_pdfs

# 5. Full run (30–60 minutes)
python run_pipeline.py --max 700

# 6. Launch demo
streamlit run demo/app.py
```

### Resume Support

All pipeline steps support resumption:
- **Collection** — `metadata.jsonl` is appended incrementally; existing papers are skipped
- **Chunking** — Existing `*_chunks.jsonl` files are skipped
- **Indexing** — ChromaDB checks existing IDs; only new chunks are embedded
- **Agent runs** — `predictions/*.jsonl` tracks completed `question_id`s; done questions are skipped

### Reproducibility Notes

- **Deterministic retrieval** — BM25 scores and embedding cosine distances are deterministic for the same index
- **RRF fusion is stateless** — Same inputs always produce the same ranking
- **LLM outputs vary** — Temperature is set to 0.1 but outputs still vary; average across runs for stable metrics
- **arXiv is a growing corpus** — Freeze `data/raw/metadata.jsonl` for exact reproducibility

---

## Troubleshooting

### No LLM provider available

```
RuntimeError: No LLM provider available!
```

**Solution**: Configure at least one provider in `.env`:
```bash
# Fastest free option:
echo 'GROQ_API_KEY=gsk_...' >> .env

# Or run Ollama locally:
ollama serve
ollama pull llama3.2
```

### ChromaDB index corrupt

```
Error: Collection already exists with different metadata
```

**Solution**: Delete and rebuild:
```bash
rm -rf data/index/chroma/
python indexer/build_index.py --chunks data/chunks --index data/index
```

### Out of memory during embedding

**Solution**: The batch size adapts automatically (`BATCH=64` on GPU/MPS, `32` on CPU). If still OOM, reduce the batch size in `indexer/build_index.py`.

### Streamlit port in use

```bash
# Kill existing process
lsof -i :8501 | awk 'NR>1 {print $2}' | xargs kill -9

# Or use a different port
streamlit run demo/app.py --server.port 8502
```

### Slow inference

Switch to a faster provider. Groq is typically fastest:
```bash
echo 'GROQ_API_KEY=gsk_...' >> .env
python run.py --config full_agent
```

### Debugging Commands

```bash
# Test LLM connection
python -c "from llm_client import call_llm; print(call_llm('Say hello'))"

# Check chunk count
python -c "
from pathlib import Path
n = sum(1 for f in Path('data/chunks').glob('*.jsonl') for _ in open(f))
print(f'{n} chunks in data/chunks/')
"

# Check ChromaDB
python -c "
import chromadb
c = chromadb.PersistentClient('data/index/chroma')
col = c.get_collection('arxiv_chunks')
print(f'{col.count()} docs in ChromaDB')
"

# Quick retriever test
python chroma_db.py
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| arXiv categories searched | cs.CL, cs.AI, cs.LG |
| Date range | 2024-01-01 → 2026-04-30 |
| Max papers in corpus | ~700 |
| Avg chunk length | ~1,800 characters (~450 tokens) |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Retrieval methods | BM25 + Semantic + RRF Fusion |
| Vector database | ChromaDB (persistent, cosine space) |
| Evaluation questions | 30 (factoid / comparative / survey) |
| Agent configurations | 7 (full + baseline + 5 ablations) |
| LLM providers supported | 4 (Groq, Ollama, OpenRouter, OpenAI) |

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Llama 3.3 70B (Groq) / Llama 3.2 (Ollama) | Fast, free, capable |
| **Embeddings** | BAAI/bge-small-en-v1.5 | SOTA for size, 130MB, runs on CPU |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder, 80MB |
| **Lexical search** | BM25Okapi | Pure Python, deterministic, no dependencies |
| **Vector DB** | ChromaDB | Persistent, simple, embedded |
| **PDF parsing** | PyMuPDF (fitz) | Fast, handles complex academic PDFs |
| **UI framework** | Streamlit | Minimal code for rich interactive apps |
| **API format** | OpenAI-compatible | All 4 providers use the same API shape |

---

**Last Updated**: June 2026
**Status**: Active Development