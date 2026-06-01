# Agentic Deep Research System

**AIMS DTU Research Intern 2026**

An end-to-end system that answers complex research questions about LLM agents by decomposing them into sub-questions, retrieving relevant academic papers, synthesizing grounded answers with citations, and verifying claims against source material.

🎯 **100% free** • ⚡ **CPU-only** • 🔓 **No paid APIs** • 🔄 **Fully reproducible**

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Quick Start](#quick-start)
5. [Command Reference](#command-reference)
6. [Technical Deep Dive](#technical-deep-dive)
7. [Evaluation & Ablation Studies](#evaluation--ablation-studies)
8. [Reproducing the Project](#reproducing-the-project)
9. [Troubleshooting](#troubleshooting)

---

## What This Project Does

This project implements an **agentic deep research pipeline** that combines classical information retrieval with large language models to answer research questions with cited evidence.

### Core Use Case
Given a research question like *"What are recent advances in multi-agent reasoning?"*, the system:

1. **Decomposes** the question into 2–4 focused sub-questions
2. **Retrieves** relevant sections from ~700 academic papers on LLM agents (using hybrid BM25 + semantic search)
3. **Reflects** on whether retrieved evidence is sufficient; if not, refines queries and retrieves again
4. **Synthesizes** a comprehensive answer using ONLY the retrieved passages, with inline citations like `[arxiv:2403.14281]`
5. **Verifies** each claim by checking it against the cited passage (detects hallucinations)
6. **Evaluates** answer quality using an LLM-as-judge and citation precision/recall metrics

### Why This Architecture?

- **Interpretable**: Every answer is fully cited; you can trace which paper supports which claim
- **Reproducible**: No closed APIs; runs 100% locally on CPU with free services
- **Measurable**: 7 ablation configs to quantify the contribution of each component
- **Modular**: Swap LLM providers, retrievers, or synthesis strategies easily

---

## System Architecture

### High-Level Data Flow

```
User Question
    ↓
┌─────────────────────────────────────────────────────┐
│            Agent Pipeline (5 Stages)                │
├─────────────────────────────────────────────────────┤
│ 1. PLANNER           → decompose into sub-questions │
│ 2. RETRIEVER         → hybrid BM25 + semantic search│
│ 3. REFLECTOR         → loop until enough evidence   │
│ 4. SYNTHESIZER       → generate cited answer        │
│ 5. VERIFIER          → check claims vs passages     │
└─────────────────────────────────────────────────────┘
    ↓
Answer (with citations + verification stats)
    ↓
    ├─→ [eval/evaluate.py] → LLM-as-judge scores
    ├─→ [demo/app.py] → Streamlit UI
    └─→ [predictions/*.jsonl] → saved results
```

### Component Diagram

```
Data Collection & Indexing (One-time setup)
────────────────────────────────────────────────
  arXiv API
      ↓
  [scraper/collect.py]
      ↓
  PDFs + metadata.jsonl
      ↓
  [indexer/chunk.py]
      ↓
  [data/chunks/*.jsonl] (chunked docs)
      ↓
  [indexer/build_index.py]
      ├─→ BM25 index (cached on disk)
      ├─→ embeddings (BAAI/bge-small-en-v1.5)
      └─→ ChromaDB vector store
────────────────────────────────────────────────
Agent Inference (Per question)
────────────────────────────────────────────────
  Question
      ↓
  [agent/Agent.py]
      ├─→ planner.plan()
      ├─→ retriever.retrieve_multi()
      ├─→ reflector.is_sufficient()
      ├─→ synthesizer.generate()
      └─→ verifier.verify_citations()
      ↓
  Answer + citations + stats
      ↓
  [predictions/full_agent.jsonl]
      ↓
  [eval/evaluate.py] → metrics
```

### Directory Structure

```
DEEP_RESEARCH/
├── agent/                          # Core agent pipeline
│   └── Agent.py                    # 5-stage agentic orchestration
├── indexer/                        # Retrieval index building
│   ├── build_index.py              # Creates BM25 + ChromaDB indexes
│   ├── chunk.py                    # Chunks PDFs with smart headers
│   └── retriever.py                # Hybrid retrieval + RRF fusion
├── scraper/                        # Data collection
│   └── collect.py                  # Downloads papers from arXiv
├── eval/                           # Evaluation & metrics
│   ├── evaluate.py                 # LLM-as-judge + citation metrics
│   ├── questions.jsonl             # Test questions
│   └── results.json                # Cached results
├── demo/                           # Interactive UI
│   └── app.py                      # Streamlit application
├── data/                           # Data storage
│   ├── raw/                        # Original PDFs + metadata
│   ├── chunks/                     # Parsed, chunked documents
│   └── index/chroma/               # Persistent ChromaDB index
├── cache/                          # Cache directories
├── logs/                           # Execution logs
├── predictions/                    # Agent outputs (per config)
├── chroma_db.py                    # Direct ChromaDB query utility
├── llm_client.py                   # Unified LLM provider wrapper
├── retriever_t.py                  # Quick retriever test script
├── run_pipeline.py                 # Master orchestrator
├── run.py                          # Run specific configs
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** (3.10+ recommended)
- **~10 GB disk space** (PDFs + indexes)
- **4+ GB RAM** (embeddings + agent inference)
- **Internet connection** (arXiv API + LLM inference)
- **macOS, Linux, or Windows** (CPU-only, no GPU required)

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
cd ~/Desktop/DEEP_RESEARCH

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies breakdown** (see [requirements.txt](requirements.txt)):
- `pymupdf>=1.24.0` — PDF text extraction
- `sentence-transformers>=2.7.0` — Embeddings (BAAI/bge-small) + reranker
- `rank-bm25>=0.2.2` — BM25Okapi lexical search
- `chromadb>=0.5.0` — Persistent vector database
- `requests>=2.31.0` — HTTP client for arXiv API
- `numpy`, `pandas`, `tqdm` — Utilities
- `streamlit>=1.35.0` — Web UI
- `python-dotenv>=1.0.0` — `.env` config loading

### Step 2: Configure LLM Provider

The system automatically detects available LLM providers in this order:
1. **Local Ollama** (if running on localhost:11434)
2. **Groq API** (free, requires API key)
3. **OpenRouter API** (free tier available)
4. **OpenAI API** (fallback, paid)
5. **HuggingFace Inference API** (free with READ token)

**To use Groq (recommended for free cloud inference)**:
```bash
# Get free API key from: https://groq.com
# Then set:
cp .env.example .env
# Edit .env:
export GROQ_API_KEY="your_groq_api_key_here"
```

**To use HuggingFace API**:
```bash
# Get free READ token from: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"
```

**To use Ollama (fully local, no APIs)**:
```bash
# Install: https://ollama.ai
ollama pull mistral  # ~4.1 GB
# Then run: ollama serve
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "
from sentence_transformers import SentenceTransformer
from chromadb import Client
from rank_bm25 import BM25Okapi
print('All dependencies installed')
"

# List available LLM providers
python llm_client.py
```

---

## Quick Start

### Option 1: Full Pipeline (Recommended for first run)

```bash
# Run everything end-to-end
python run_pipeline.py
```

This will:
1. **Collect** papers from arXiv (if not already done)
2. **Chunk** PDFs into passages
3. **Index** with BM25 + ChromaDB (resumable if interrupted)
4. **Run agent** on 10 sample questions
5. **Evaluate** results with LLM-as-judge
6. **Save predictions** to `predictions/` directory

### Option 2: Skip Data Collection (If already indexed)

```bash
# If you already have data/chunks/ and index/chroma/
python run_pipeline.py --skip_collect --skip_chunk --skip_index
```

### Option 3: Run Specific Configs Only

```bash
# Run only full_agent config
python run.py --config full_agent

# Run all 7 ablation configs
python run.py --config all

# Run with custom question
python run.py --config full_agent --question "What is multi-agent reasoning?"
```

### Option 4: Evaluate Existing Results

```bash
# Score all predictions in predictions/ directory
python eval/evaluate.py

# Fast mode (skip LLM judge)
python eval/evaluate.py --no_llm_judge
```

### Option 5: Interactive Demo

```bash
# Launch Streamlit UI
streamlit run demo/app.py

# Open browser: http://localhost:8501
# Use sidebar to toggle: planner, hybrid, reranker, reflector, verifier
```

---

## Command Reference

### Master Pipeline

| Command | Purpose | Flags |
|---------|---------|-------|
| `python run_pipeline.py` | Run full pipeline | `--skip_collect`, `--skip_chunk`, `--skip_index`, `--skip_eval`, `--max 10`, `--skip_pdfs` |
| `python run_pipeline.py --only_eval` | Skip to evaluation only | — |
| `python run_pipeline.py --skip_pdfs` | Collect metadata only, no PDFs | — |
| `python run_pipeline.py --max 5` | Limit to 5 papers | — |

### Individual Steps

| Command | Purpose | Flags |
|---------|---------|-------|
| `python scraper/collect.py` | Download papers from arXiv | `--max 5`, `--start_date 2024-01-01` |
| `python indexer/chunk.py` | Parse PDFs, create chunks | `--chunk_size 1800` |
| `python indexer/build_index.py` | Build BM25 + ChromaDB | `--resume`, `--clear` |
| `python chroma_db.py` | Query ChromaDB directly | — |
| `python retriever_t.py` | Test hybrid retriever | — |

### Agent Execution

| Command | Purpose | Flags |
|---------|---------|-------|
| `python run.py --config full_agent` | Run with all components | — |
| `python run.py --config all` | Run all 7 ablation configs | — |
| `python run.py --config baseline` | Run minimal config (no agent features) | — |
| `python run.py --question "Your question here"` | Custom question | `--config full_agent` |

**Config options**:
- `full_agent` — planner + hybrid + reranker + reflector + verifier
- `baseline` — none of the above
- `no_planner` — skip question decomposition
- `no_hybrid` — semantic search only (no BM25)
- `no_reranker` — skip cross-encoder reranking
- `no_reflector` — single-pass retrieval (no looping)
- `no_citation_verifier` — skip verification step

### Evaluation

| Command | Purpose | Flags |
|---------|---------|-------|
| `python eval/evaluate.py` | Score all predictions | `--no_llm_judge`, `--output results.json` |
| `python eval/evaluate.py --no_llm_judge` | Fast evaluation (skip LLM scoring) | — |

### Interactive Demo

| Command | Purpose |
|---------|---------|
| `streamlit run demo/app.py` | Launch Streamlit UI on localhost:8501 |

---

## Technical Deep Dive

### Stage 1: Planner

**Purpose**: Decompose complex questions into 2–4 focused sub-questions

**Prompt**: Instructs LLM to ask specific, answerable sub-questions
```
User question: "What recent advances have been made in multi-agent systems?"
↓
Sub-questions:
1. "What are agent frameworks used for multi-agent systems?"
2. "How do agents communicate and coordinate?"
3. "What evaluation benchmarks exist for multi-agent systems?"
```

**Ablation**: Set `use_planner=False` to skip (baseline searches the original question directly)

---

### Stage 2: Retriever

**Hybrid approach** combines lexical and semantic search:

```
Question
    ↓
┌─────────────────────────────────────────┐
│ BM25Okapi (lexical)                     │
│ - Keyword matching on chunks            │
│ - Returns top-k chunks by rank          │
│ - Fast, interpretable, good for         │
│   specific terms                        │
│ - Ranking: TF-IDF-like scoring          │
└─────────────────────────────────────────┘
    ↓
Ranks: [chunk_1 (rank 1), chunk_5 (rank 2), ...]
    ↓
┌─────────────────────────────────────────┐
│ Semantic (BAAI/bge-small-en-v1.5)       │
│ - Embed question + chunks (130M model)  │
│ - Similarity: cosine distance           │
│ - Returns top-k by similarity           │
│ - Good for paraphrases, nuance          │
└─────────────────────────────────────────┘
    ↓
Ranks: [chunk_3 (rank 1), chunk_1 (rank 3), ...]
    ↓
┌─────────────────────────────────────────┐
│ Reciprocal Rank Fusion (RRF)            │
│ - Combines both rankings: RRF(i) =      │
│   (1/(60 + rank_bm25)) +                │
│   (1/(60 + rank_semantic))              │
│ - De-duplicates, merges by score        │
│ - Final top-k chunks                    │
└─────────────────────────────────────────┘
    ↓
Final retrieved chunks
    ↓
[Optional] Rerank with cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2)
    ↓
Final ranked chunks → Synthesizer
```

**Parameters**:
- `top_k=5` — retrieve top 5 chunks per query
- `use_hybrid=True` — blend BM25 + semantic
- `use_reranker=True` — apply cross-encoder reranking

**Output**: List of (chunk_text, arxiv_id, score) tuples

---

### Stage 3: Reflector

**Purpose**: Assess if retrieved evidence is sufficient; loop if needed

**Logic**:
```python
for round in range(1, 4):  # max 3 rounds
    retrieved = retriever.retrieve(question)
    
    if reflector.is_sufficient(question, retrieved):
        break  # enough evidence
    else:
        # refine question or add new query
        question = reflector.refine(question, retrieved)
        # loop to Stage 2 with refined question
```

**Prompt**: LLM evaluates if retrieved docs answer the question sufficiently
```
Question: "How do agents handle tool use?"
Retrieved: [chunk_1, chunk_2, chunk_3]
↓
LLM judges: "These chunks mention tools, but don't explain 
how agents choose tools. Need to search for 'tool selection'."
↓
Refined question: "How do agents select and use tools?"
↓
Retrieve again with refined query
```

**Ablation**: Set `use_reflector=False` to skip (single-pass retrieval)

---

### Stage 4: Synthesizer

**Purpose**: Generate answer using ONLY retrieved passages, with inline citations

**Key constraint**: Answer must be grounded in retrieval results
- Every claim must reference a retrieved chunk
- If a fact isn't in retrieved chunks, it can't be stated
- Cites as `[arxiv:2403.14281]` (inline)

**Prompt template**:
```
Question: {question}

Retrieved passages:
{retrieved_chunks}

Using ONLY the above passages, write a comprehensive answer.
Include inline citations like [arxiv:ID] for each claim.
Do NOT add information not in the passages.
```

**Output format**:
```json
{
  "question": "How do agents handle tool use?",
  "answer": "Recent work shows that agents use planning to select tools [arxiv:2403.14281]. Tool selection can be done via prompting [arxiv:2404.05129] or learned policies [arxiv:2405.01017].",
  "citations": ["arxiv:2403.14281", "arxiv:2404.05129", "arxiv:2405.01017"]
}
```

---

### Stage 5: Verifier

**Purpose**: Check each claim against its cited passage (detect hallucinations)

**Process**:
```
Answer: "Agents use planning to select tools [arxiv:2403.14281]."
    ↓
Extract claim: "Agents use planning to select tools"
Citation: "arxiv:2403.14281"
    ↓
Retrieve cited passage text
    ↓
LLM prompt: "Does this passage support the claim?"
    ↓
Judgment: SUPPORTED / HALLUCINATED
```

**Output**:
```json
{
  "claim": "Agents use planning to select tools",
  "cited_paper": "arxiv:2403.14281",
  "passage": "Planning-based selection of tools enables...",
  "verification": "supported",
  "confidence": 0.95
}
```

**Metrics**:
- **Citation Precision**: (# verified claims) / (total claims)
- **Citation Recall**: (# cited papers matching ground truth) / (total ground truth papers)

**Ablation**: Set `use_verifier=False` to skip verification

---

### Architecture: How Configs Are Organized

Each config is defined by a tuple of 5 boolean flags:

```python
configs = {
    "full_agent": {"planner": True, "hybrid": True, "reranker": True, "reflector": True, "verifier": True},
    "baseline": {"planner": False, "hybrid": False, "reranker": False, "reflector": False, "verifier": False},
    "no_planner": {"planner": False, "hybrid": True, "reranker": True, "reflector": True, "verifier": True},
    # ... etc
}
```

When you run `python run.py --config no_planner`, the system:
1. Loads the config dict
2. Instantiates Agent with those flags
3. Runs on all test questions
4. Saves predictions to `predictions/no_planner.jsonl`

---

## Evaluation & Ablation Studies

### Evaluation Metrics

The system computes three categories of metrics:

#### 1. **LLM-as-Judge Scores** (requires LLM call per answer)

An LLM evaluates answer quality on three dimensions (1–5 scale):
- **Accuracy**: Do claims match the source material?
- **Completeness**: Does it fully answer the question?
- **Coherence**: Is the explanation clear and well-structured?

**Prompt**:
```
Question: {question}
Answer: {answer}
Retrieved passages: {passages}

Rate accuracy, completeness, and coherence (1-5 each).
Justify your rating.
```

#### 2. **Citation Precision**

Percentage of citations that actually support the claim:
$$\text{Citation Precision} = \frac{\text{# verified claims}}{\text{# total claims}}$$

- Detected via Stage 5 (Verifier)
- Perfect score: 1.0 (no hallucinations)

#### 3. **Citation Recall** (optional)

If you have ground-truth relevant papers:
$$\text{Citation Recall} = \frac{\text{# cited papers in ground truth}}{\text{# total ground truth papers}}$$

### Running Evaluation

```bash
# Full evaluation (slow, but complete)
python eval/evaluate.py

# Fast mode (skip LLM judge, use heuristics)
python eval/evaluate.py --no_llm_judge

# Evaluate specific config
python eval/evaluate.py --config full_agent

# Save to custom file
python eval/evaluate.py --output my_results.json
```

### Interpreting Results

Output: `results.json`
```json
{
  "config": "full_agent",
  "num_questions": 10,
  "metrics": {
    "avg_accuracy": 4.2,
    "avg_completeness": 4.1,
    "avg_coherence": 4.3,
    "citation_precision": 0.94,
    "citation_recall": 0.87
  }
}
```

---

## Reproducing the Project

### Complete Reproduction from Scratch

```bash
# 1. Fresh environment
cd ~/Desktop/DEEP_RESEARCH
python -m venv venv_fresh
source venv_fresh/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure LLM (choose one)
export GROQ_API_KEY="your_key_here"
# OR
export HF_TOKEN="hf_your_token_here"

# 4. Run full pipeline
python run_pipeline.py

# 5. Check results
python eval/evaluate.py
```

**Expected output**:
```
✅ Collect: Downloaded X papers from arXiv
✅ Chunk: Created Y chunks from Z PDFs
✅ Index: Built BM25 + ChromaDB indexes
✅ Agent: Ran 10 questions with full_agent config
✅ Evaluate: Computed metrics for all configs
```

### Reproducibility Checkpoints

1. **Deterministic retrieval** (same question → same chunks returned)
   - Confirmed: BM25 and embeddings are deterministic
   - Check: Run `python retriever_t.py` twice, compare chunk IDs

2. **Deterministic ranking** (RRF fusion always same order)
   - Confirmed: RRF math is stateless
   - Check: Compare `predictions/full_agent.jsonl` runs on same index

3. **Non-deterministic LLM outputs** (different generations across runs)
   - Expected: LLM sampling has temperature > 0
   - Mitigation: Set `temperature=0` in `llm_client.py` if needed
   - Use: Multiple runs → average metrics

4. **Data versioning**:
   - arXiv is a growing corpus; re-running `collect.py` may get newer papers
   - Solution: Freeze `data/raw/metadata.jsonl` if you need exact reproducibility

### Sharing Results

```bash
# Create reproducible archive
tar -czf deep_research_results.tar.gz \
  predictions/ \
  eval/results.json \
  data/chunks/ \
  data/index/chroma/ \
  requirements.txt \
  .env  # (sanitized)

# Share: other users can extract and run
tar -xzf deep_research_results.tar.gz
python run.py --config full_agent --question "..."
```

---

## Troubleshooting

### Issue: LLM provider not found

```
Error: No LLM provider available
```

**Solution**: Set at least one of:
```bash
export GROQ_API_KEY="..."  # Free option
# OR
export HF_TOKEN="..."       # Free HF inference
# OR
export OPENAI_API_KEY="..." # Paid, but reliable
# OR
ollama serve              # Local option
```

### Issue: ChromaDB index is corrupt

```
Error: Collection already exists with different metadata
```

**Solution**: Rebuild from scratch
```bash
rm -rf data/index/chroma/
python indexer/build_index.py
```

### Issue: Out of memory during embedding

```
CUDA out of memory / RuntimeError: Cannot allocate X bytes
```

**Solution**: Reduce batch size
```bash
# In indexer/build_index.py:
# Change: batch_size = 256 → batch_size = 32
python indexer/build_index.py --batch_size 32
```

### Issue: slow LLM inference

```
Waiting for LLM response... (>60 seconds)
```

**Solution**: Switch to faster provider
```bash
# Switch from HF API (slow) to Groq (fast)
export GROQ_API_KEY="..."
python run.py --config full_agent
```

### Issue: Streamlit app crashes

```
Error: Port 8501 already in use
```

**Solution**:
```bash
# Kill existing process
lsof -i :8501 | grep -v PID | awk '{print $2}' | xargs kill -9

# Or use different port
streamlit run demo/app.py --server.port 8502
```

### Issue: Questions not in eval/questions.jsonl

**Solution**: Add your own
```bash
# Edit eval/questions.jsonl
echo '{"id": 11, "question": "Your question here?"}' >> eval/questions.jsonl

# Re-run evaluation
python eval/evaluate.py
```

---

## Advanced Usage

### Custom Question Answering

```python
from agent.Agent import Agent
from indexer.retriever import Retriever

# Load retriever
retriever = Retriever(index_dir="data/index/")

# Create agent
agent = Agent(
    retriever=retriever,
    config={
        "use_planner": True,
        "use_hybrid": True,
        "use_reranker": True,
        "use_reflector": True,
        "use_verifier": True
    }
)

# Answer question
question = "What are recent advances in multi-agent reasoning?"
result = agent.answer(question)

print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
print(f"Verification: {result['verification']}")
```

### Swapping Components

**Use a different LLM**:
```python
# In agent/Agent.py
from llm_client import LLMClient

llm = LLMClient(provider="openai", model="gpt-4")  # instead of default
```

**Use only semantic search** (no BM25):
```python
retriever.retrieve_multi(question, use_hybrid=False)
```

**Disable reranking**:
```python
retriever.retrieve_multi(question, use_reranker=False)
```

---

---

## Free Stack Technology Choices

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Mistral-7B (via Groq/HF) | Fast, capable, free tier available |
| **Embeddings** | BAAI/bge-small-en-v1.5 | State-of-art, 130MB, local CPU |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder, 80MB, local |
| **Lexical** | BM25Okapi | Pure Python, no dependencies, deterministic |
| **Vector DB** | ChromaDB | Persistent, simple, no external service |
| **PDF parsing** | PyMuPDF (fitz) | Fast, accurate, handles complex PDFs |
| **Web framework** | Streamlit | Minimal code for interactive UI |

---

## Contributing & Extensions

### Add a new stage to the agent pipeline

```python
# In agent/Agent.py, add a new class
class MyNewStage:
    def process(self, input):
        # your logic here
        return output

# Then in Agent.answer(), call it:
result = self.my_stage.process(retrieved_chunks)
```

### Create a new retriever

```python
# In indexer/retriever.py
class MyRetriever:
    def retrieve(self, question, top_k=5):
        # your retrieval logic
        return [(chunk_text, source_id, score), ...]

# Update agent to use it
agent.retriever = MyRetriever(...)
```

### Add new evaluation metrics

```python
# In eval/evaluate.py
def compute_my_metric(answer, ground_truth):
    # your metric logic
    return metric_value
```

---

## Citation

If you use this project in research, please cite:

```bibtex
@misc{deep_research_2026,
  title={Agentic Deep Research System},
  author={AIMS DTU},
  year={2026},
  howpublished={\url{https://github.com/...}},
  note={100% free, CPU-only research pipeline}
}
```


## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check [Troubleshooting](#troubleshooting) section
- Review logs in `logs/` directory

---

**Last Updated**: June 2026  
**Status**: Active Development
| `no_reflector` | ✅ | ✅ | ✅ | ❌ | ✅ |
| `no_hybrid` | ✅ | ❌ | ✅ | ✅ | ✅ |
| `no_citation_verifier` | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## Project Structure

```
project/
├── scraper/collect.py          # arXiv paper collector
├── indexer/
│   ├── chunk.py                # PDF parser + chunker
│   ├── build_index.py          # BM25 + ChromaDB builder
│   └── retriever.py            # Hybrid retriever class
├── agent/agent.py              # All 5 agent components + configs
├── eval/
│   ├── questions.jsonl         # 30 evaluation questions
│   └── evaluate.py             # Scoring + ablation table
├── predictions/                # <config>.jsonl outputs
├── demo/app.py                 # Streamlit trace viewer
├── llm_client.py               # HF + Groq LLM client
├── run.py                      # Agent runner
├── run_pipeline.py             # Master pipeline script
├── requirements.txt
└── .env.example
```

---

## Debugging

```bash
# Test LLM connection
python -c "from llm_client import call_llm; print(call_llm('Say hello'))"

# Test retriever
python indexer/retriever.py --query "what is ReAct?"  # after building index

# Check how many chunks exist
ls data/chunks/ | wc -l
python -c "import json; print(sum(1 for f in __import__('pathlib').Path('data/chunks').glob('*.jsonl') for _ in open(f)))"

# Check ChromaDB
python -c "import chromadb; c=chromadb.PersistentClient('data/index/chroma'); col=c.get_collection('arxiv_chunks'); print(f'{col.count()} docs in ChromaDB')"
```