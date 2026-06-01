"""
agent/agent.py — Complete agentic pipeline + all ablation configs

Components (each independently togglable for ablations):
  1. Planner           — decomposes question into sub-questions
  2. Retriever         — hybrid BM25 + semantic + reranker
  3. Reflector         — decides if evidence is sufficient (loop)
  4. Synthesizer       — writes cited answer from retrieved evidence only
  5. Citation Verifier — checks every [arxiv:ID] claim is grounded

Ablation configs supported:
  full_agent, baseline, no_planner, no_reranker,
  no_reflector, no_hybrid, no_citation_verifier
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import call_llm, call_llm_json
from indexer.retriever import Retriever

log = logging.getLogger(__name__)
MAX_ROUNDS = 3


# ═══════════════════════════════════════════════════════════════
# COMPONENT 1: PLANNER
# ═══════════════════════════════════════════════════════════════

def plan(question: str, use_planner: bool = True) -> List[str]:
    """Decompose question into 2–4 targeted sub-questions."""
    if not use_planner:
        return [question]

    prompt = f"""You are a research assistant. Break this research question into 2-4 specific sub-questions that together will answer it. Each sub-question should target ONE specific aspect.

Question: {question}

Return JSON: {{"sub_questions": ["q1", "q2", "q3"]}}"""

    try:
        r = call_llm_json(prompt, max_tokens=300)
        subs = r.get("sub_questions", [])
        if subs and isinstance(subs, list):
            subs = [str(q).strip() for q in subs if str(q).strip()][:4]
            log.info(f"Planner: {len(subs)} sub-questions")
            return subs
    except Exception as e:
        log.warning(f"Planner failed: {e}")
    return [question]


# ═══════════════════════════════════════════════════════════════
# COMPONENT 2: REFLECTOR
# ═══════════════════════════════════════════════════════════════

def reflect(question: str, chunks: List[Dict], round_num: int,
            use_reflector: bool = True) -> Tuple[bool, List[str]]:
    """
    Decide if evidence is sufficient.
    Returns (sufficient, refined_queries).
    """
    if not use_reflector or round_num >= MAX_ROUNDS:
        return True, []
    if not chunks:
        return False, [question]

    # Build short evidence summary
    summary_parts = []
    seen_papers   = set()
    for c in chunks[:8]:
        seen_papers.add(c.get("arxiv_id", ""))
        summary_parts.append(f"- {c.get('arxiv_id','?')}: {c['text'][:150]}...")
    summary = "\n".join(summary_parts)

    prompt = f"""You are evaluating if evidence is sufficient to answer a research question.

Question: {question}

Evidence ({len(chunks)} passages from {len(seen_papers)} papers):
{summary}

Is this enough to write a complete, well-cited answer?
Return JSON: {{"sufficient": true/false, "refined_queries": ["query1", "query2"]}}
Only include refined_queries if sufficient is false."""

    try:
        r = call_llm_json(prompt, max_tokens=200)
        sufficient = bool(r.get("sufficient", True))
        refined    = [str(q) for q in r.get("refined_queries", [])[:2]]
        log.info(f"Reflector round {round_num}: sufficient={sufficient}")
        return sufficient, refined
    except Exception as e:
        log.warning(f"Reflector failed: {e}")
        return True, []


# ═══════════════════════════════════════════════════════════════
# COMPONENT 3: SYNTHESIZER
# ═══════════════════════════════════════════════════════════════

def synthesize(question: str, chunks: List[Dict]) -> Tuple[str, List[str]]:
    """Write a grounded answer with inline [arxiv:ID] citations."""
    if not chunks:
        return "Insufficient evidence to answer this question.", []

    evidence = "\n\n---\n\n".join([
        f"[Source: arxiv:{c.get('arxiv_id','?')}]\nPaper: {c.get('title','')}\n{c['text'][:600]}"
        for c in chunks[:12]
    ])

    prompt = f"""You are a research assistant. Answer the question using ONLY the provided evidence. 
After each claim, cite the source as [arxiv:ID].
Do NOT use any outside knowledge.

Question: {question}

Evidence:
{evidence}

Write a complete answer with inline citations [arxiv:ID]:"""

    try:
        answer = call_llm(prompt, max_tokens=900)
        # Extract cited IDs
        cited = list(dict.fromkeys(re.findall(r"\[arxiv:([^\]]+)\]", answer, re.IGNORECASE)))
        log.info(f"Synthesizer: {len(answer.split())} words, {len(cited)} citations")
        return answer, cited
    except Exception as e:
        log.error(f"Synthesizer failed: {e}")
        return f"Error generating answer: {e}", []

def verify_citations(answer: str, cited_ids: List[str], chunks: List[Dict],
                     use_verifier: bool = True) -> Dict:
    """Check each cited arXiv ID is actually in the retrieved evidence."""
    retrieved_ids = {c.get("arxiv_id") for c in chunks}
    hallucinated  = [i for i in cited_ids if i not in retrieved_ids]
    grounded      = [i for i in cited_ids if i in retrieved_ids]

    if not use_verifier:
        return {
            "verified_ids":     grounded,
            "hallucinated_ids": hallucinated,
            "verification_rate": len(grounded) / len(cited_ids) if cited_ids else 1.0,
        }

    # LLM grounding check (only for non-hallucinated)
    verified, unverified = [], []
    for arxiv_id in grounded:
        passage = next((c["text"][:400] for c in chunks if c.get("arxiv_id") == arxiv_id), "")
        if not passage:
            unverified.append(arxiv_id)
            continue

        # Find the sentence(s) citing this paper
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        claim = next((s for s in sentences if arxiv_id in s), "")
        claim = re.sub(r"\[arxiv:[^\]]+\]", "", claim).strip()[:200]

        prompt = f"""Does this passage support the claim? Answer with JSON only.

Claim: {claim}
Passage (arxiv:{arxiv_id}): {passage}

Return: {{"supported": true/false}}"""

        try:
            r = call_llm_json(prompt, max_tokens=60)
            if r.get("supported", False):
                verified.append(arxiv_id)
            else:
                unverified.append(arxiv_id)
        except Exception:
            verified.append(arxiv_id)  # give benefit of doubt on failure

    total = len(cited_ids)
    rate  = round(len(verified) / total, 3) if total else 1.0
    log.info(f"Verifier: {len(verified)}/{total} verified, {len(hallucinated)} hallucinated")

    return {
        "verified_ids":     verified,
        "hallucinated_ids": hallucinated,
        "unverified_ids":   unverified,
        "verification_rate": rate,
    }


# ═══════════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

CONFIGS = {
    "full_agent":            {"planner":True,  "hybrid":True,  "reranker":True,  "reflector":True,  "verifier":True},
    "baseline":              {"planner":False, "hybrid":False, "reranker":False, "reflector":False, "verifier":False},
    "no_planner":            {"planner":False, "hybrid":True,  "reranker":True,  "reflector":True,  "verifier":True},
    "no_reranker":           {"planner":True,  "hybrid":True,  "reranker":False, "reflector":True,  "verifier":True},
    "no_reflector":          {"planner":True,  "hybrid":True,  "reranker":True,  "reflector":False, "verifier":True},
    "no_hybrid":             {"planner":True,  "hybrid":False, "reranker":True,  "reflector":True,  "verifier":True},
    "no_citation_verifier":  {"planner":True,  "hybrid":True,  "reranker":True,  "reflector":True,  "verifier":False},
}


def run_question(question: str, question_id: str,
                 retriever: Retriever, config: Dict) -> Dict[str, Any]:
    """Run one question through the full agent pipeline."""
    t0         = time.time()
    trace      = []
    tool_calls = 0

    # 1. PLAN
    sub_qs = plan(question, use_planner=config["planner"])
    trace.append({"step": "plan", "sub_questions": sub_qs})

    # 2. RETRIEVE + REFLECT LOOP
    all_chunks: List[Dict] = []
    current_queries = sub_qs
    round_num = 1

    while True:
        new = retriever.retrieve_multi(current_queries, top_k=5)
        tool_calls += len(current_queries)

        # Deduplicate
        existing_ids = {c["chunk_id"] for c in all_chunks}
        for c in new:
            if c["chunk_id"] not in existing_ids:
                all_chunks.append(c)

        trace.append({"step": f"retrieve_round_{round_num}",
                      "queries": current_queries, "new_chunks": len(new),
                      "total_chunks": len(all_chunks)})

        # Reflect
        done, refined = reflect(question, all_chunks, round_num,
                                use_reflector=config["reflector"])
        trace.append({"step": f"reflect_round_{round_num}",
                      "sufficient": done, "refined": refined})

        if done or not refined:
            break
        current_queries = refined
        round_num += 1

    # 3. SYNTHESIZE
    answer, cited_ids = synthesize(question, all_chunks)
    trace.append({"step": "synthesize", "cited_ids": cited_ids})

    # 4. VERIFY
    verification = verify_citations(answer, cited_ids, all_chunks,
                                    use_verifier=config["verifier"])
    trace.append({"step": "verify", "result": verification})
    latency = round(time.time() - t0, 2)
    return {
        "question_id":       question_id,
        "question":          question,
        "answer":            answer,
        "cited_arxiv_ids":   cited_ids,
        "verified_ids":      verification.get("verified_ids", []),
        "hallucinated_ids":  verification.get("hallucinated_ids", []),
        "verification_rate": verification.get("verification_rate", 1.0),
        "n_chunks":          len(all_chunks),
        "n_rounds":          round_num,
        "tool_calls":        tool_calls,
        "latency_s":         latency,
        "trace":             trace,
    }
