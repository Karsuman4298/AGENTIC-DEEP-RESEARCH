

# Agentic Deep Research System

An end-to-end research assistant pipeline for answering questions about LLM agents using a closed-world arXiv corpus.

This repository contains tools to:
- collect and filter recent arXiv papers,
- chunk PDF text into searchable passages,
- build hybrid retrieval indexes (BM25 + ChromaDB),
- run an agent pipeline with planning, retrieval, reflection, synthesis, and citation verification,
- evaluate outputs using automated LLM judging and citation metrics,
- run an interactive Streamlit demo.

---

# Demo
A demonstration video of the Agentic Deep Research System is available in the GitHub Releases section.

Release: v1
Demo Video:
https://github.com/Karsuman4298/AGENTIC-DEEP-RESEARCH/releases/tag/v1

<img width="2940" height="1668" alt="image" src="https://github.com/user-attachments/assets/0db6eda7-1b9e-4abe-a47e-885118d89523" />


## Project Overview

The system is designed for evidence-grounded research Q&A on papers about agentic systems and language model agents.
It works as a closed-world retrieval-augmented generation pipeline: the final answer is produced from passages retrieved from the indexed paper corpus.

The main pipeline stages are:
1. **Planner** — decomposes the input question into targeted search queries.
2. **Retriever** — searches paper chunks with BM25, semantic embeddings, and optional reranking.
3. **Reflector** — checks whether the retrieved evidence is sufficient and optionally performs another retrieval round.
4. **Synthesizer** — generates a grounded answer with inline `[arxiv:ID]` citations.
5. **Verifier** — evaluates citation grounding using lexical overlap between answer text and source passages.

---

## Repository Layout

```
DEEP_RESEARCH/
├── agent/
│   └── Agent.py
├── indexer/
│   ├── build_index.py
│   ├── chunk.py
│   └── retriever.py
├── scraper/
│   └── collect.py
├── eval/
│   ├── evaluate.py
│   ├── questions.jsonl
│   └── results.json
├── demo/
│   └── app.py
├── data/
│   ├── raw/
│   ├── chunks/
│   └── index/
├── predictions/
├── logs/
├── cache/
├── llm_client.py
├── run_pipeline.py
├── run.py
├── download_pdfs.py
├── chroma_db.py
├── requirements.txt
└── README.md
```

---

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure LLM provider credentials in `.env` if needed.

Example `.env`:

```bash
GROQ_API_KEY=gsk_your_key_here
# or
# OPENROUTER_API_KEY=or_your_key_here
# or
# OPENAI_API_KEY=sk-your_key_here
```

Local Ollama is also supported if you run `ollama serve` on `localhost:11434`.

---

## Dependency List

- `pymupdf`
- `sentence-transformers`
- `rank-bm25`
- `chromadb`
- `requests`
- `tqdm`
- `python-dotenv`
- `numpy`
- `streamlit`
- `pandas`

---

## Core Scripts

### `run_pipeline.py`

A master script for the full workflow:
- collect papers from arXiv,
- chunk PDFs,
- build retrieval indexes,
- run agent configurations,
- evaluate outputs.

Example:

```bash
python run_pipeline.py --max 5 --skip_pdfs
```

Run the full pipeline with the full corpus:

```bash
python run_pipeline.py --max 700
```

Useful flags:
- `--skip_collect`
- `--skip_index`
- `--skip_agent`
- `--only_eval`
- `--no_llm_judge`
- `--configs full_agent baseline no_planner`

### `run.py`

Runs agent configurations against evaluation questions in `eval/questions.jsonl` and writes predictions to `predictions/*.jsonl`.

Example:

```bash
python run.py --config full_agent
python run.py --config all
```

### `eval/evaluate.py`

Scores predictions and prints ablation results.

Example:

```bash
python eval/evaluate.py
```

### `demo/app.py`

Launches the Streamlit demo UI. The interface includes a "Research trace" expander showing planner queries, retrieval rounds, reflector decisions, citation counts, and verifier results.

```bash
streamlit run demo/app.py
```

---

## Pipeline Components

### Data Collection

- `scraper/collect.py` fetches arXiv metadata for categories `cs.CL`, `cs.AI`, and `cs.LG`.
- It filters papers by agent-related keywords in title and abstract.
- Metadata is stored in `data/raw/metadata.jsonl`.
- PDFs are optionally downloaded to `data/raw/pdfs/`.
- Downloading is polite with retry/backoff and delay to respect arXiv rate limits.

### Chunking

- `indexer/chunk.py` extracts PDF text with `PyMuPDF`.
- It cleans text, removes page numbers, and dehyphenates broken lines.
- It splits text by section headers and then applies a sliding window of 1800 characters with 200-character overlap.
- Each chunk includes explicit source markers so the model can preserve document provenance.
- Chunk outputs are saved as JSONL files in `data/chunks/`.

### Indexing

- `indexer/build_index.py` builds a BM25 index using `rank_bm25`.
- It also builds dense embeddings with `sentence-transformers` and stores them in ChromaDB.
- The BM25 index is saved to `data/index/bm25.pkl`.
- The dense index is stored in `data/index/chroma/`.

### Retrieval

- `indexer/retriever.py` performs hybrid retrieval.
- BM25 lexical search and ChromaDB semantic search results are fused with Reciprocal Rank Fusion (RRF).
- Optional reranking uses a cross-encoder model.
- The retriever returns top chunks for each query.

### Agent Pipeline

- `agent/Agent.py` implements the 5-stage agentic pipeline.
- `plan()` optionally generates multiple keyword queries from the question.
- `reflect()` decides whether more retrieval is needed.
- `synthesize()` generates the final answer with permitted citations.
- `verify_citations()` checks grounding using lexical overlap.
- The pipeline supports seven configurations for ablation.

### Evaluation

- `eval/evaluate.py` computes accuracy, completeness, coherence, faithfulness, and citation precision.
- It can optionally use an LLM judge with an automated scoring prompt.
- Results are saved to `eval/results.json`.

---

## Agent Configurations

The repository defines seven configs in `agent/Agent.py`:

- `full_agent`
- `baseline`
- `no_planner`
- `no_reranker`
- `no_reflector`
- `no_hybrid`
- `no_citation_verifier`

Each config enables or disables specific pipeline stages so the system can be evaluated by component.

---

## LLM Provider Selection

`llm_client.py` uses automatic provider selection and failover.
It checks availability in this order:
1. Groq
2. Ollama
3. OpenRouter
4. OpenAI

If one provider fails, it retries and moves to the next available provider.

---

## Notes and Limitations

- The system is designed to answer questions using the indexed arXiv corpus only.
- Citation verification is heuristic and based on lexical overlap.
- Evaluation is automated and not human-reviewed.
- The pipeline focuses on LLM agents and related papers; it is not a general domain QA system.

---

## Recommended Workflow

1. Collect paper's metadata: `python scraper/collect.py --max 500 --skip_pdfs`
2. Download the paper's PDF: `python run download_pdfs.py`
3. Chunk PDFs: `python indexer/chunk.py --raw data/raw --out data/chunks`
4. Build indexes: `python indexer/build_index.py --chunks data/chunks --index data/index`
5. Run the agent: `python run.py --config full_agent`
6. Evaluate: `python eval/evaluate.py`
7. Demo: `streamlit run demo/app.py`
