"""
indexer/retriever.py — Hybrid retriever, tuned thresholds for small corpus

Threshold rationale (BGE-small on ~500 papers):
  SEMANTIC_SIM_THRESHOLD = 0.20  — BGE-small cosine similarities cluster
                                    around 0.15-0.45 on a domain corpus.
                                    0.35 was cutting ~80% of valid chunks.
  BM25_SCORE_THRESHOLD   = 0.5   — Raw BM25 on short keyword queries rarely
                                    exceeds 2.0; 1.0 was too strict.
  RERANKER_SCORE_GATE    = -5.0  — Cross-encoder logits: truly irrelevant
                                    chunks score below -5. -2.0 was cutting
                                    borderline-relevant chunks.

Citation safety is enforced in agent.py via the allowed_ids whitelist,
NOT here. Retriever's job is recall; agent's job is precision.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"

log.info(f"Retriever device: {DEVICE}")

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION   = "arxiv_chunks"
RRF_K        = 60

# ── Tuned thresholds ───────────────────────────────────────────
# Purpose: high recall so synthesizer has enough evidence to work with.
# Citation hallucination prevention is handled in agent.py, not here.
SEMANTIC_SIM_THRESHOLD = 0.20   # was 0.35 — too aggressive for BGE-small
BM25_SCORE_THRESHOLD   = 0.5    # was 1.0  — too strict for keyword queries
RERANKER_SCORE_GATE    = -5.0   # was -2.0 — cutting valid borderline chunks

_embed_model  = None
_rerank_model = None


def _get_embed():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedder: {EMBED_MODEL} (device={DEVICE})")
        _embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        _embed_model.max_seq_length = 512
    return _embed_model


def _get_reranker():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        log.info(f"Loading reranker: {RERANK_MODEL} (device={DEVICE})")
        _rerank_model = CrossEncoder(RERANK_MODEL, max_length=512, device=DEVICE)
    return _rerank_model


def _rrf(bm25_list: List[Tuple], sem_list: List[Tuple], k: int = RRF_K) -> List[Tuple]:
    scores: Dict[str, float] = {}
    for rank, (cid, *_) in enumerate(bm25_list):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, *_) in enumerate(sem_list):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    """
    Hybrid BM25 + semantic retriever.

    Ablation flags:
      use_hybrid=False   → semantic only  (ablation: no_hybrid)
      use_reranker=False → skip reranker  (ablation: no_reranker)
    """

    def __init__(
        self,
        index_dir:    str  = "data/index",
        chunks_dir:   str  = "data/chunks",
        use_hybrid:   bool = True,
        use_reranker: bool = True,
    ):
        self.use_hybrid   = use_hybrid
        self.use_reranker = use_reranker
        self._store:     Dict[str, Dict] = {}
        self._bm25       = None
        self._bm25_ids:  List[str] = []
        self._chroma_col = None
        self._index_dir  = index_dir
        self._chunks_dir = chunks_dir

    def load(self) -> None:
        # Chunk store
        chunks = []
        for f in sorted(Path(self._chunks_dir).glob("*_chunks.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        self._store = {c["chunk_id"]: c for c in chunks}
        log.info(f"Chunk store: {len(self._store)} chunks")

        # BM25
        bm25_path = Path(self._index_dir) / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                d = pickle.load(f)
            self._bm25, self._bm25_ids = d["bm25"], d["ids"]
            log.info(f"BM25: {len(self._bm25_ids)} docs")
        else:
            log.warning("BM25 not found — hybrid disabled")
            self.use_hybrid = False

        # ChromaDB
        import chromadb
        client = chromadb.PersistentClient(path=str(Path(self._index_dir) / "chroma"))
        self._chroma_col = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        log.info(f"ChromaDB: {self._chroma_col.count()} docs")

        _get_embed()
        if self.use_reranker:
            _get_reranker()

    def retrieve(self, query: str, top_k: int = 5,
                 candidate_k: int = 40) -> List[Dict[str, Any]]:
        """
        Retrieve top_k relevant chunks for query.
        Returns [] only if corpus is empty or completely unrelated.
        """
        if self._chroma_col is None:
            log.error("Retriever not loaded — call load() first")
            return []

        # Semantic search
        model  = _get_embed()
        q_pref = f"Represent this sentence for searching relevant passages: {query}"
        q_vec  = model.encode(q_pref, normalize_embeddings=True).tolist()

        n = min(candidate_k, self._chroma_col.count())
        if n == 0:
            return []

        res = self._chroma_col.query(
            query_embeddings=[q_vec], n_results=n,
            include=["distances", "metadatas", "documents"],
        )

        # Filter by semantic threshold
        sem_list: List[Tuple[str, float]] = []
        if res["ids"] and res["ids"][0]:
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                sim = 1.0 - float(dist)
                if sim >= SEMANTIC_SIM_THRESHOLD:
                    sem_list.append((cid, sim))

        n_raw = len(res["ids"][0]) if res["ids"] and res["ids"][0] else 0
        log.debug(f"Semantic: {n_raw} raw → {len(sem_list)} pass sim>={SEMANTIC_SIM_THRESHOLD}")

        # BM25 with threshold
        bm25_list: List[Tuple[str, float]] = []
        if self.use_hybrid and self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            filtered = [
                (cid, float(s))
                for cid, s in zip(self._bm25_ids, bm25_scores.tolist())
                if float(s) >= BM25_SCORE_THRESHOLD
            ]
            bm25_list = sorted(filtered, key=lambda x: x[1], reverse=True)[:candidate_k]
            log.debug(f"BM25: {len(bm25_scores)} raw → {len(bm25_list)} pass score>={BM25_SCORE_THRESHOLD}")

        # Fusion
        if sem_list or bm25_list:
            if self.use_hybrid and bm25_list:
                fused    = _rrf(bm25_list, sem_list)
                cand_ids = [cid for cid, _ in fused[:candidate_k]]
            else:
                cand_ids = [cid for cid, _ in sem_list[:candidate_k]]
        else:
            log.info(f"retrieve({query[:50]!r}): nothing above thresholds → []")
            return []

        # Hydrate
        candidates = [self._store[cid] for cid in cand_ids if cid in self._store]
        if not candidates:
            return []

        # Attach semantic score
        sem_score_map = {cid: sc for cid, sc in sem_list}
        for c in candidates:
            c["score"] = round(sem_score_map.get(c["chunk_id"], 0.0), 4)

        # Reranking
        if self.use_reranker and len(candidates) > 1:
            reranker = _get_reranker()
            pairs    = [(query, c["text"][:512]) for c in candidates]
            scores   = reranker.predict(pairs, batch_size=8).tolist()

            # Gate: discard only truly irrelevant (very low logit)
            ranked = [
                ({**c, "score": round(float(s), 4)}, s)
                for c, s in zip(candidates, scores)
                if float(s) >= RERANKER_SCORE_GATE
            ]
            ranked.sort(key=lambda x: x[1], reverse=True)

            if not ranked:
                # Reranker rejected everything — fall back to semantic ordering
                log.info(f"Reranker rejected all candidates — falling back to semantic order")
                candidates.sort(key=lambda x: x["score"], reverse=True)
                return candidates[:top_k]

            results = [item for item, _ in ranked[:top_k]]
            log.debug(f"Reranker: {len(candidates)} → {len(results)}")
            return results

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def retrieve_multi(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Retrieve across multiple queries, deduplicate by chunk_id."""
        seen: Dict[str, Dict] = {}
        for q in queries:
            for r in self.retrieve(q, top_k=top_k):
                cid = r.get("chunk_id")
                if cid and (cid not in seen or r["score"] > seen[cid]["score"]):
                    seen[cid] = r
        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        log.info(f"retrieve_multi({len(queries)} queries): {len(results)} unique chunks")
        return results