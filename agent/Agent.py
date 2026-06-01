"""
agent/agent.py
--------------
Full RAG agent pipeline:
  plan → retrieve (+ reflect loop) → synthesize → verify
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import call_llm, call_llm_json
from indexer.retriever import Retriever


log = logging.getLogger(__name__)

MAX_ROUNDS: int = 3          # Maximum reflection / retrieval rounds
MAX_CHUNKS_SYNTH: int = 10   # Max chunks passed to synthesizer
MAX_CHUNKS_REFLECT: int = 6  # Max chunks summarised for reflector




CONFIGS: Dict[str, Dict[str, bool]] = {
    "full_agent":           {"planner": True,  "hybrid": True,  "reranker": True,  "reflector": True,  "verifier": True},
    "baseline":             {"planner": False, "hybrid": False, "reranker": False, "reflector": False, "verifier": False},
    "no_planner":           {"planner": False, "hybrid": True,  "reranker": True,  "reflector": True,  "verifier": True},
    "no_reranker":          {"planner": True,  "hybrid": True,  "reranker": False, "reflector": True,  "verifier": True},
    "no_reflector":         {"planner": True,  "hybrid": True,  "reranker": True,  "reflector": False, "verifier": True},
    "no_hybrid":            {"planner": True,  "hybrid": False, "reranker": True,  "reflector": True,  "verifier": True},
    "no_citation_verifier": {"planner": True,  "hybrid": True,  "reranker": True,  "reflector": True,  "verifier": False},
}




def plan(question: str, use_planner: bool = True) -> List[str]:
    """
    Decompose *question* into 2–4 focused sub-questions.

    Returns a list of strings.  Falls back to ``[question]`` on any failure so
    the pipeline is never blocked by this step.
    """
    if not use_planner:
        log.debug("Planner disabled — using original question.")
        return [question]

    prompt = (
        """
        Convert the question into atomic factual claims that can each be independently verified from academic text.
        Rules:
            - 3–7 claims max
            - each claim must be a single factual statement
            - no reasoning chains inside a claim
            - no background explanations

        Return JSON:
        { "claims": [...] }
        """
    )

    try:
        raw = call_llm_json(prompt, max_tokens=400)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict, got {type(raw).__name__}")

        subs: List[str] = raw.get("sub_questions", [])

        if not isinstance(subs, list) or not subs:
            raise ValueError("'sub_questions' missing or empty")

        # Sanitise: stringify, strip whitespace, drop blanks, cap at 4
        subs = [str(q).strip() for q in subs if str(q).strip()][:4]

        if not subs:
            raise ValueError("All sub-questions were blank after sanitisation")

        log.info("Planner produced %d sub-question(s).", len(subs))
        return subs

    except Exception as exc:
        log.warning("Planner failed (%s) — falling back to original question.", exc)
        return [question]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b — REFLECTOR
# ══════════════════════════════════════════════════════════════════════════════

def reflect(
    question: str,
    chunks: List[Dict],
    round_num: int,
    use_reflector: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Decide whether the accumulated evidence is sufficient to answer *question*.

    Returns
    -------
    (sufficient, refined_queries)
        * sufficient=True  → stop retrieval loop
        * sufficient=False → continue with *refined_queries*
    """
    # Hard stops: reflector disabled, round cap reached, or no chunks at all
    if not use_reflector:
        log.debug("Reflector disabled — marking sufficient.")
        return True, []

    if round_num >= MAX_ROUNDS:
        log.info("Reflector: MAX_ROUNDS (%d) reached — forcing sufficient.", MAX_ROUNDS)
        return True, []

    if not chunks:
        log.info("Reflector round %d: no chunks retrieved — requesting retry.", round_num)
        return False, [question]

    # Build a compact evidence summary (deduplicated by arxiv_id)
    seen_ids: set = set()
    summary_lines: List[str] = []
    for chunk in chunks[:MAX_CHUNKS_REFLECT]:
        arxiv_id = chunk.get("arxiv_id", "unknown")
        snippet  = chunk.get("text", "")[:150].replace("\n", " ")
        if arxiv_id not in seen_ids:
            seen_ids.add(arxiv_id)
        summary_lines.append(f"  [{arxiv_id}] {snippet}…")

    summary = "\n".join(summary_lines)
    n_papers = len(seen_ids)

    prompt = (
        "You are a research quality evaluator.\n\n"
        f"Question: {question}\n\n"
        f"Evidence — {len(chunks)} passage(s) from {n_papers} unique paper(s):\n"
        f"{summary}\n\n"
        "Is this evidence sufficient to write a complete, well-cited answer?\n"
        "If YES return: {\"sufficient\": true, \"refined_queries\": []}\n"
        "If NO  return: {\"sufficient\": false, \"refined_queries\": [\"specific query 1\", \"specific query 2\"]}\n"
        "Return ONLY valid JSON — no extra text."
    )

    try:
        raw = call_llm_json(prompt, max_tokens=300)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict, got {type(raw).__name__}")

        sufficient: bool = bool(raw.get("sufficient", True))
        refined: List[str] = [
            str(q).strip()
            for q in raw.get("refined_queries", [])
            if str(q).strip()
        ][:3]  # cap refined queries

        log.info(
            "Reflector round %d: sufficient=%s, refined_queries=%d.",
            round_num, sufficient, len(refined),
        )
        return sufficient, refined

    except Exception as exc:
        log.warning(
            "Reflector round %d failed (%s) — assuming sufficient.", round_num, exc
        )
        return True, []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SYNTHESIZER
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Phrases the LLM must never output — enforced in post-processing.
# ---------------------------------------------------------------------------
_BANNED_PHRASES: List[str] = [
    "Evidence does not address this",
    "the evidence does not address",
    "no evidence was provided",
    "the provided evidence does not cover",
    "cannot be answered from the evidence",
    "I cannot answer",
    "I don't know",
]

# Minimum word-overlap fraction for a chunk to be considered PRIMARY evidence.
_PRIMARY_OVERLAP_THRESHOLD: float = 0.10


def _score_chunk_relevance(question: str, chunk_text: str) -> float:
    """
    Lightweight lexical relevance score: fraction of question content-words
    (len >= 4) that appear in *chunk_text*.  No LLM call needed.
    """
    q_words   = set(re.findall(r"\b[a-z]{4,}\b", question.lower()))
    c_words   = set(re.findall(r"\b[a-z]{4,}\b", chunk_text.lower()))
    if not q_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


def _build_evidence_block(
    question:   str,
    chunks:     List[Dict],
    max_chunks: int = MAX_CHUNKS_SYNTH,
) -> str:
    """
    Format the top-N chunks into a labelled evidence block.

    Each source receives a support-level tag:
      • [PRIMARY]  — chunk directly overlaps with question vocabulary (≥ 10 %)
      • [CONTEXT]  — chunk is topically related but less directly relevant

    This tag is part of the prompt text so the LLM can calibrate how
    confidently to cite each source without any extra LLM call.
    """
    parts: List[str] = []
    for i, chunk in enumerate(chunks[:max_chunks], start=1):
        arxiv_id = chunk.get("arxiv_id", "unknown")
        title    = (chunk.get("title")   or "").strip()
        section  = (chunk.get("section") or "").strip()
        text     = (chunk.get("text")    or "")[:600].strip()

        score   = _score_chunk_relevance(question, text)
        support = "PRIMARY" if score >= _PRIMARY_OVERLAP_THRESHOLD else "CONTEXT"

        header = f"[Source {i} | arxiv:{arxiv_id} | {support}]"
        if title:
            header += f"\nPaper   : {title}"
        if section:
            header += f"\nSection : {section}"

        parts.append(f"{header}\nText    : {text}")

    return "\n\n---\n\n".join(parts)


def _extract_cited_ids(text: str) -> List[str]:
    """Return deduplicated list of arxiv IDs found in *text* (order-preserving)."""
    ids = re.findall(r"\[arxiv:([^\]\s]+)\]", text, re.IGNORECASE)
    return list(dict.fromkeys(ids))


def _postprocess_answer(raw: str) -> str:
    """
    Clean the raw LLM output:
      1. Strip common preamble phrases.
      2. Replace every banned phrase with an approved alternative.
      3. Ensure an 'Uncertainty notes:' section exists (add a stub if absent).
      4. Normalise whitespace.
    """
    # 1. Strip preamble
    answer = re.sub(
        r"^(Sure[,!]?|Of course[,!]?|Certainly[,!]?|"
        r"Here(?:'s| is)(?: the| your| an?)?(?: answer)?[:\.]?)\s*",
        "",
        raw.strip(),
        flags=re.IGNORECASE,
    ).strip()

    # 2. Replace every banned phrase with the approved wording
    for phrase in _BANNED_PHRASES:
        answer = re.sub(
            re.escape(phrase),
            "The provided passages do not fully cover this aspect",
            answer,
            flags=re.IGNORECASE,
        )

    # 3. Ensure 'Uncertainty notes:' section is present
    if not re.search(r"uncertainty\s+notes\s*:", answer, re.IGNORECASE):
        answer = answer.rstrip() + (
            "\n\nUncertainty notes:\n"
            "- No additional gaps identified beyond what is stated above."
        )

    # 4. Collapse runs of 3+ blank lines to 2
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer.strip()


# ---------------------------------------------------------------------------
# Synthesizer prompt template  (single source of truth)
# ---------------------------------------------------------------------------
_SYNTHESIZER_SYSTEM = """\
You are a precise academic research assistant answering questions grounded in \
retrieved arXiv paper passages.\
"""

_SYNTHESIZER_PROMPT = """\

{system}

MOST IMPORTANT : You are strictly forbidden from using any knowledge not explicitly present in the provided evidence.
Even if you are confident, you must ignore pretraining knowledge.Do NOT mention or use any dataset, benchmark, or paper unless it appears in the provided evidence block.

════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════

1. EVIDENCE-GROUNDED CLAIMS
   Use the retrieved passages as the primary source for all specific factual
   claims (methods, results, numbers, model names, paper-specific statements).
   Cite every such claim inline as [arxiv:ID] using the exact arxiv ID shown
   in the source header.

2. PARTIAL EVIDENCE IS ENOUGH — DO NOT REFUSE
   If evidence only partially addresses the question, answer what you CAN from
   the evidence and clearly flag the rest as uncertain.
   NEVER output the phrase "Evidence does not address this."
   Instead write: "The provided passages do not fully cover this aspect."

3. BACKGROUND KNOWLEDGE (use sparingly)
   You MAY use general, widely accepted scientific knowledge ONLY to define
   or contextualise well-known concepts (e.g. "RAG", "LLM", "attention
   mechanism").
   • Do NOT attribute background knowledge to any retrieved paper.
   • Do NOT introduce paper-specific claims, numbers, or methods from memory.

4. NO HALLUCINATION
   Never invent model names, benchmark scores, dataset sizes, or any
   paper-specific detail not explicitly present in the retrieved text.

5. CITATION DISCIPLINE
   • Cite [arxiv:ID] ONLY when the passage genuinely supports the claim.
   • Do NOT force a citation just to meet a quota.
   • Every paragraph must contain at least one [arxiv:ID] citation OR an
     explicit uncertainty marker.

6. THREE-TIER CLAIM LABELLING
   Mentally classify every sentence as one of:
     (A) Supported  — directly backed by a retrieved passage  → cite it
     (B) Background — general knowledge, not paper-specific   → no citation,
                      optionally note "(general background)"
     (C) Uncertain  — not found in evidence                   → flag it in
                      the "Uncertainty notes" section

7. OUTPUT FORMAT  (follow exactly)

   Answer:

   [Paragraph 1 — main findings from PRIMARY sources with citations]

   [Paragraph 2 — supporting context, extensions, or CONTEXT sources]

   [Paragraph 3+ — additional synthesis as needed]

   Uncertainty notes:
   - [aspect not covered by evidence]
   - [another gap, if any]
   (Write "None identified." if evidence is fully sufficient.)

8. STYLE
   • Begin immediately with "Answer:" — no preamble.
   • Be concise but complete; prefer correctness over verbosity.
   • Do not over-cite; do not cite the same source more than needed per claim.

════════════════════════════════════════
EVIDENCE SOURCES
════════════════════════════════════════

Each source is tagged [PRIMARY] (directly relevant) or [CONTEXT] (related).
Prefer PRIMARY sources for specific claims; use CONTEXT sources for background.

{evidence_block}

════════════════════════════════════════
QUESTION
════════════════════════════════════════

{question}

════════════════════════════════════════
YOUR ANSWER
════════════════════════════════════════\
"""


def synthesize(question: str, chunks: List[Dict]) -> Tuple[str, List[str]]:
    """
    Write a grounded answer with inline ``[arxiv:ID]`` citations.

    The prompt enforces a strict three-tier claim discipline:
      (A) Evidence-grounded claims  → must be cited with [arxiv:ID]
      (B) General background        → allowed without citation, not attributed
      (C) Uncertain / not in evidence → collected in 'Uncertainty notes' section

    Returns
    -------
    (answer_text, cited_arxiv_ids)
    """
    if not chunks:
        log.warning("Synthesizer called with no chunks — returning minimal answer.")
        return (
            "Answer:\n\nThe provided passages do not fully cover this aspect "
            "as no evidence was retrieved for this question.\n\n"
            "Uncertainty notes:\n- All aspects of the question are uncovered.",
            [],
        )

    evidence_block = _build_evidence_block(question, chunks)

    prompt = _SYNTHESIZER_PROMPT.format(
        system        = _SYNTHESIZER_SYSTEM,
        evidence_block= evidence_block,
        question      = question,
    )

    try:
        raw_answer = call_llm(prompt, max_tokens=1400)
        answer     = _postprocess_answer(raw_answer)

        cited_ids  = _extract_cited_ids(answer)

        # Warn if the LLM produced no citations at all (may indicate prompt failure)
        if not cited_ids:
            log.warning(
                "Synthesizer produced 0 citations — check evidence quality "
                "or LLM compliance with the prompt."
            )

        log.info(
            "Synthesizer: %d words, %d unique citation(s).",
            len(answer.split()), len(cited_ids),
        )
        return answer, cited_ids

    except Exception as exc:
        log.error("Synthesizer failed: %s", exc)
        return f"Error generating answer: {exc}", []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CITATION VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

def _word_set(text: str, min_len: int = 4) -> set:
    """Return a set of lowercase alphabetic words longer than *min_len* chars."""
    return set(re.findall(rf"\b[a-z]{{{min_len},}}\b", text.lower()))


def _overlap(claim_words: set, passage_words: set) -> float:
    """Jaccard-style overlap: |claim ∩ passage| / |claim|."""
    if not claim_words:
        return 0.0
    return len(claim_words & passage_words) / len(claim_words)


def verify_citations(
    answer: str,
    cited_ids: List[str],
    chunks: List[Dict],
    use_verifier: bool = True,
) -> Dict[str, Any]:
    """
    Check each cited arXiv ID is lexically grounded in the retrieved evidence.

    Uses word-overlap between citing sentences and source passages — fast,
    deterministic, and no extra LLM calls.

    Returns
    -------
    dict with keys:
        verified_ids, hallucinated_ids, unverified_ids,
        verification_rate, faithfulness
    """
    if not cited_ids:
        return {
            "verified_ids":      [],
            "hallucinated_ids":  [],
            "unverified_ids":    [],
            "verification_rate": 1.0,
            "faithfulness":      1.0,
        }

    retrieved_ids: set = {c.get("arxiv_id") for c in chunks if c.get("arxiv_id")}
    hallucinated:  List[str] = [i for i in cited_ids if i not in retrieved_ids]
    grounded:      List[str] = [i for i in cited_ids if i in retrieved_ids]

    # Build a quick lookup: arxiv_id → concatenated passage text
    id_to_passage: Dict[str, str] = {}
    for chunk in chunks:
        aid = chunk.get("arxiv_id")
        if aid and aid in retrieved_ids:
            id_to_passage[aid] = id_to_passage.get(aid, "") + " " + chunk.get("text", "")

    if not use_verifier:
        # Ablation mode: skip overlap check, treat all grounded as verified
        rate = round(len(grounded) / len(cited_ids), 3) if cited_ids else 1.0
        log.info("Verifier disabled (ablation): faithfulness=%.3f", rate)
        return {
            "verified_ids":      grounded,
            "hallucinated_ids":  hallucinated,
            "unverified_ids":    [],
            "verification_rate": rate,
            "faithfulness":      rate,
        }

    # Split answer into sentences once
    sentences: List[str] = re.split(r"(?<=[.!?])\s+", answer)

    verified:   List[str] = []
    unverified: List[str] = []

    OVERLAP_THRESHOLD = 0.20  # at least 20 % of claim words must appear in passage

    for arxiv_id in grounded:
        passage = id_to_passage.get(arxiv_id, "").strip()

        if not passage:
            # No passage text — can't verify, be conservative
            unverified.append(arxiv_id)
            continue

        passage_words = _word_set(passage)

        # Sentences that explicitly cite this paper
        citing: List[str] = [
            re.sub(r"\[arxiv:[^\]]+\]", "", s).strip()
            for s in sentences
            if arxiv_id in s
        ]

        if not citing:
            # Cited but no clearly associated sentence — give benefit of the doubt
            verified.append(arxiv_id)
            continue

        # Average overlap across all citing sentences
        scores = [_overlap(_word_set(c), passage_words) for c in citing if c]
        avg_overlap = sum(scores) / len(scores) if scores else 0.0

        if avg_overlap >= OVERLAP_THRESHOLD:
            verified.append(arxiv_id)
        else:
            unverified.append(arxiv_id)

    total = len(cited_ids)
    # Faithfulness penalises both hallucinations and low-overlap citations
    faithfulness = round(len(verified) / total, 3) if total else 1.0

    log.info(
        "Verifier: %d/%d grounded verified, %d hallucinated, %d unverified, "
        "faithfulness=%.3f",
        len(verified), len(grounded), len(hallucinated), len(unverified), faithfulness,
    )

    return {
        "verified_ids":      verified,
        "hallucinated_ids":  hallucinated,
        "unverified_ids":    unverified,
        "verification_rate": faithfulness,
        "faithfulness":      faithfulness,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_question(
    question:    str,
    question_id: str,
    retriever:   Retriever,
    config:      Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run one question through the full agent pipeline.

    Pipeline
    --------
    1. Plan        — decompose question into sub-questions
    2. Retrieve    — fetch chunks for every sub-question
    3. Reflect     — decide if more retrieval is needed (repeat 2–3 up to MAX_ROUNDS)
    4. Synthesize  — write a grounded, cited answer
    5. Verify      — check citation faithfulness

    Returns a result dict suitable for downstream evaluation.
    """
    t0 = time.time()
    trace: List[Dict] = []
    tool_calls: int   = 0

    # ------------------------------------------------------------------
    # 1. Plan
    # ------------------------------------------------------------------
    try:
        sub_qs = plan(question, use_planner=config.get("planner", True))
    except Exception as exc:
        log.error("Plan step crashed: %s", exc)
        sub_qs = [question]

    trace.append({"step": "plan", "sub_questions": sub_qs})

    # ------------------------------------------------------------------
    # 2–3. Retrieve + Reflect loop
    # ------------------------------------------------------------------
    all_chunks:      List[Dict] = []
    seen_chunk_ids:  set        = set()
    current_queries: List[str]  = sub_qs
    round_num:       int        = 1

    while True:
        try:
            new_chunks: List[Dict] = retriever.retrieve_multi(
                current_queries, top_k=5
            )
            tool_calls += len(current_queries)
        except Exception as exc:
            log.error("Retriever failed on round %d: %s", round_num, exc)
            new_chunks = []

        # Deduplicate by chunk_id
        added = 0
        for chunk in new_chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                all_chunks.append(chunk)
                added += 1

        trace.append({
            "step":         f"retrieve_round_{round_num}",
            "queries":      current_queries,
            "new_chunks":   added,
            "total_chunks": len(all_chunks),
        })
        log.info(
            "Retrieve round %d: +%d new chunks (%d total).",
            round_num, added, len(all_chunks),
        )

        # Reflect
        try:
            sufficient, refined = reflect(
                question, all_chunks, round_num,
                use_reflector=config.get("reflector", True),
            )
        except Exception as exc:
            log.error("Reflect step crashed on round %d: %s", round_num, exc)
            sufficient, refined = True, []

        trace.append({
            "step":      f"reflect_round_{round_num}",
            "sufficient": sufficient,
            "refined":    refined,
        })

        if sufficient or not refined:
            break

        current_queries = refined
        round_num += 1

    # ------------------------------------------------------------------
    # 4. Synthesize
    # ------------------------------------------------------------------
    try:
        answer, cited_ids = synthesize(question, all_chunks)
    except Exception as exc:
        log.error("Synthesize step crashed: %s", exc)
        answer, cited_ids = f"Error during synthesis: {exc}", []

    trace.append({"step": "synthesize", "cited_ids": cited_ids})

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    try:
        verification = verify_citations(
            answer, cited_ids, all_chunks,
            use_verifier=config.get("verifier", True),
        )
    except Exception as exc:
        log.error("Verify step crashed: %s", exc)
        verification = {
            "verified_ids":      [],
            "hallucinated_ids":  cited_ids,
            "unverified_ids":    [],
            "verification_rate": 0.0,
            "faithfulness":      0.0,
        }

    trace.append({"step": "verify", "result": verification})

    latency = round(time.time() - t0, 2)
    log.info(
        "run_question done: question_id=%s latency=%.2fs rounds=%d chunks=%d cited=%d",
        question_id, latency, round_num, len(all_chunks), len(cited_ids),
    )

    return {
        # Core outputs
        "question_id":       question_id,
        "question":          question,
        "answer":            answer,
        # Citation info
        "cited_arxiv_ids":   cited_ids,
        "verified_ids":      verification.get("verified_ids",     []),
        "hallucinated_ids":  verification.get("hallucinated_ids", []),
        "unverified_ids":    verification.get("unverified_ids",   []),
        # Metrics
        "verification_rate": verification.get("faithfulness", 1.0),
        "faithfulness":      verification.get("faithfulness", 1.0),
        "n_chunks":          len(all_chunks),
        "n_rounds":          round_num,
        "tool_calls":        tool_calls,
        "latency_s":         latency,
        # Debug
        "trace":             trace,
    }
