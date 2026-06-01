

import json
import sys
import os
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Agentic Deep Research",
    layout="wide",
)

st.title("Agentic Deep Research System")

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(" Configuration")
    use_planner  = st.toggle("Planner (question decomposition)", value=True)
    use_hybrid   = st.toggle("Hybrid retrieval (BM25 + semantic)", value=True)
    use_reranker = st.toggle("Cross-encoder reranker", value=True)
    use_reflector = st.toggle("Reflector (evidence loop)", value=True)
    use_verifier  = st.toggle("Citation verifier", value=True)

    st.divider()
    index_dir  = st.text_input("Index dir",  value="data/index")
    chunks_dir = st.text_input("Chunks dir", value="data/chunks")

    st.divider()
    st.markdown("**Stack**")
    st.markdown("- LLM: Mistral-7B via HuggingFace API")
    st.markdown("- Embedder: BAAI/bge-small-en-v1.5")
    st.markdown("- Reranker: ms-marco-MiniLM-L-6-v2")
    st.markdown("- Vector DB: ChromaDB (local)")
    st.markdown("- Lexical: BM25Okapi")


# ── Load retriever ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval index...")
def load_retriever(index_dir, chunks_dir, use_hybrid, use_reranker):
    from indexer.retriever import Retriever
    r = Retriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
    )
    r.load()
    return r


# ── Main UI ────────────────────────────────────────────────────────────────────
question = st.text_area(
    "Ask a research question about LLM agents:",
    placeholder="e.g. What is Self-RAG and how does it improve retrieval?",
    height=80,
)

col1, col2 = st.columns([1, 4])
run_btn = col1.button(" Run Agent", type="primary", use_container_width=True)
clr_btn = col2.button("Clear", use_container_width=True)

if clr_btn:
    st.rerun()

if run_btn and question.strip():
    try:
        retriever = load_retriever(index_dir, chunks_dir, use_hybrid, use_reranker)
    except Exception as e:
        st.error(f"Failed to load index: {e}\n\nRun the pipeline first: `python run_pipeline.py`")
        st.stop()

    from agent.Agent import plan, reflect, synthesize, verify_citations

    cfg = {
        "planner":  use_planner,
        "hybrid":   use_hybrid,
        "reranker": use_reranker,
        "reflector": use_reflector,
        "verifier": use_verifier,
    }

    import time
    t0 = time.time()

    # ── Step 1: Plan ───────────────────────────────────────────────────────────
    with st.expander("📋 Step 1: Planner", expanded=True):
        with st.spinner("Decomposing question..."):
            sub_qs = plan(question, use_planner=use_planner)
        if use_planner:
            st.success(f"Decomposed into {len(sub_qs)} sub-questions")
            for i, q in enumerate(sub_qs, 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.info("Planner disabled — using original question")

    # ── Step 2: Retrieve + Reflect ─────────────────────────────────────────────
    all_chunks = []
    current_queries = sub_qs
    round_results = []

    for round_num in range(1, 4):
        new_chunks = retriever.retrieve_multi(current_queries, top_k=5)
        existing_ids = {c["chunk_id"] for c in all_chunks}
        for c in new_chunks:
            if c["chunk_id"] not in existing_ids:
                all_chunks.append(c)

        with st.expander(f" Step 2: Retrieval Round {round_num}", expanded=(round_num==1)):
            st.write(f"Queries: `{current_queries}`")
            st.write(f"Retrieved {len(new_chunks)} new chunks (total: {len(all_chunks)})")
            for c in new_chunks[:3]:
                st.markdown(f"**{c.get('arxiv_id','?')}** — {c.get('title','')[:60]}")
                st.caption(f"Section: {c.get('section','')} | Score: {c.get('score',0):.3f}")
                st.text(c["text"][:250] + "...")
                st.divider()

        with st.expander(f" Step 3: Reflector Round {round_num}", expanded=(round_num==1)):
            with st.spinner("Evaluating evidence..."):
                done, refined = reflect(question, all_chunks, round_num, use_reflector)
            if done:
                st.success(" Evidence sufficient — proceeding to synthesis")
                break
            else:
                st.warning(f" Evidence insufficient — refined queries: {refined}")
                current_queries = refined

    # ── Step 3: Synthesize ─────────────────────────────────────────────────────
    with st.expander(" Step 4: Synthesizer", expanded=True):
        with st.spinner("Writing answer..."):
            answer, cited_ids = synthesize(question, all_chunks)
        st.success(f"Generated {len(answer.split())} words with {len(cited_ids)} citations")

    # ── Step 4: Verify ─────────────────────────────────────────────────────────
    with st.expander(" Step 5: Citation Verifier", expanded=True):
        with st.spinner("Verifying citations..."):
            verification = verify_citations(answer, cited_ids, all_chunks, use_verifier)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cited IDs",       len(cited_ids))
        col_b.metric("Verified",        len(verification.get("verified_ids", [])))
        col_c.metric("Hallucinated",    len(verification.get("hallucinated_ids", [])),
                     delta_color="inverse")

        if verification.get("hallucinated_ids"):
            st.error(f"Hallucinated citations: {verification['hallucinated_ids']}")

    # ── Final answer ───────────────────────────────────────────────────────────
    latency = round(time.time() - t0, 1)
    st.divider()
    st.subheader(" Final Answer")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency",     f"{latency}s")
    m2.metric("Chunks used", len(all_chunks))
    m3.metric("Citations",   len(cited_ids))
    m4.metric("Verified",    f"{verification.get('verification_rate',1.0)*100:.0f}%")

    st.markdown(answer)

    st.divider()
    st.subheader("Papers Cited")
    for cite in cited_ids:
        status = "found" if cite in verification.get("verified_ids", []) else "not found"
        st.markdown(f"{status} `arxiv:{cite}`")

elif run_btn and not question.strip():
    st.warning("Please enter a question.")

# ── Results browser ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Ablation Table")

results_path = Path("eval/results.json")
if results_path.exists():
    results = json.load(open(results_path))
    import pandas as pd
    df = pd.DataFrame([{
        "Config":       r["config"],
        "Accuracy":     r.get("accuracy", "—"),
        "Faithfulness": r.get("faithfulness", "—"),
        "Cite-P":       r.get("cite_precision", "—"),
        "Cite-R":       r.get("cite_recall", "—"),
        "Latency (s)":  r.get("avg_latency_s", "—"),
        "Tool Calls":   r.get("avg_tool_calls", "—"),
    } for r in results])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No results yet. Run `python eval/evaluate.py` after running all configs.")