"""
indexer/retriever.py
--------------------
Hybrid retriever: BM25 + ChromaDB semantic search + cross-encoder reranking.

NO sys.path manipulation here — path is handled at project root level only.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Device detection ──────────────────────────────────────────────────────────
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"

log.info("Retriever device: %s", DEVICE)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION   = "arxiv_chunks"

RRF_K                    = 60
SEMANTIC_SIM_THRESHOLD   = 0.30   # lowered from 0.40
BM25_PERCENTILE_THRESHOLD = 60    # keep top-40% of BM25 scores

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embed_model  = None
_rerank_model = None


def _get_embed():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedder: %s on %s", EMBED_MODEL, DEVICE)
        _embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        _embed_model.max_seq_length = 512
    return _embed_model


def _get_reranker():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading reranker: %s on %s", RERANK_MODEL, DEVICE)
        _rerank_model = CrossEncoder(RERANK_MODEL, max_length=512, device=DEVICE)
    return _rerank_model


def _expand_query(query: str) -> str:
    """BGE instruction prefix for retrieval tasks."""
    return f"Represent this sentence for searching relevant passages: {query}"


def _rrf(
    bm25_list: List[Tuple[str, float]],
    sem_list:  List[Tuple[str, float]],
    k:         int = RRF_K,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: Dict[str, float] = {}
    for rank, (cid, _) in enumerate(bm25_list):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(sem_list):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
class Retriever:
    """
    Hybrid retriever: BM25 + ChromaDB dense search + cross-encoder reranking.

    Usage:
        r = Retriever()
        r.load()
        results = r.retrieve("what is RAG?", top_k=5)
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
        self._bm25                       = None
        self._bm25_ids:  List[str]       = []
        self._chroma_col                 = None

        self._index_dir  = Path(index_dir)
        self._chunks_dir = Path(chunks_dir)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all indexes. Call once before any retrieve()."""

        # 1. Chunk store
        chunks: List[Dict] = []
        for f in sorted(self._chunks_dir.glob("*_chunks.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            log.warning("Bad chunk line in %s: %s", f.name, e)

        self._store = {
            c["chunk_id"]: c
            for c in chunks
            if "chunk_id" in c
        }
        log.info("Chunk store: %d chunks", len(self._store))

        # 2. BM25
        bm25_path = self._index_dir / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                d = pickle.load(f)
            self._bm25     = d["bm25"]
            self._bm25_ids = d["ids"]
            log.info("BM25 index: %d docs", len(self._bm25_ids))
        else:
            log.warning("BM25 not found at %s — hybrid disabled", bm25_path)
            self.use_hybrid = False

        # 3. ChromaDB
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(self._index_dir / "chroma")
            )
            self._chroma_col = client.get_or_create_collection(
                COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            log.info("ChromaDB: %d docs", self._chroma_col.count())
        except Exception as e:
            log.error("ChromaDB load failed: %s", e, exc_info=True)
            raise

        # 4. Warm-up
        _get_embed()
        if self.use_reranker:
            _get_reranker()

        log.info(
            "Retriever ready | hybrid=%s | reranker=%s",
            self.use_hybrid, self.use_reranker,
        )

    # ── Single query ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:       str,
        top_k:       int = 5,
        candidate_k: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for a single query.

        Returns list of dicts with keys:
            chunk_id, arxiv_id, title, section, text, score
        """
        if not query.strip():
            log.warning("Empty query passed to retrieve() — returning []")
            return []

        # ── Embed ──────────────────────────────────────────────────────────
        model = _get_embed()
        q_vec = model.encode(
            _expand_query(query),
            normalize_embeddings=True,
        ).tolist()

        # ── Semantic search ────────────────────────────────────────────────
        n = min(candidate_k, self._chroma_col.count())
        if n == 0:
            log.warning("ChromaDB is empty")
            return []

        try:
            res = self._chroma_col.query(
                query_embeddings=[q_vec],
                n_results=n,
                include=["distances", "metadatas", "documents"],
            )
        except Exception as e:
            log.error("ChromaDB query error: %s", e, exc_info=True)
            return []

        sem_list: List[Tuple[str, float]] = []
        if res["ids"] and res["ids"][0]:
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                sim = 1.0 - float(dist)          # cosine dist → similarity
                if sim >= SEMANTIC_SIM_THRESHOLD:
                    sem_list.append((cid, sim))

        log.debug(
            "Semantic: %d/%d passed threshold %.2f",
            len(sem_list), n, SEMANTIC_SIM_THRESHOLD,
        )

        # ── BM25 ────────────────────────────────────────────────────────────
        bm25_list: List[Tuple[str, float]] = []
        if self.use_hybrid and self._bm25 is not None:
            tokens      = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokens)

            pos_scores = bm25_scores[bm25_scores > 0]
            threshold  = (
                float(np.percentile(pos_scores, BM25_PERCENTILE_THRESHOLD))
                if len(pos_scores) > 0
                else 0.0
            )

            bm25_list = sorted(
                [
                    (cid, float(sc))
                    for cid, sc in zip(self._bm25_ids, bm25_scores.tolist())
                    if float(sc) >= threshold
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:candidate_k]

            log.debug("BM25: %d candidates (threshold=%.3f)", len(bm25_list), threshold)

        # ── Fusion ──────────────────────────────────────────────────────────
        if self.use_hybrid and (bm25_list or sem_list):
            fused    = _rrf(bm25_list, sem_list)
            cand_ids = [cid for cid, _ in fused[:candidate_k]]
        else:
            cand_ids = [cid for cid, _ in sem_list[:candidate_k]]

        if not cand_ids:
            log.warning("No candidates after fusion for: '%s'", query[:60])
            return []

        candidates = [self._store[cid] for cid in cand_ids if cid in self._store]
        if not candidates:
            log.warning("Candidates not found in chunk store (index mismatch?)")
            return []

        # ── Rerank ──────────────────────────────────────────────────────────
        if self.use_reranker and len(candidates) > top_k:
            try:
                reranker = _get_reranker()
                pairs    = [(query, c.get("text", "")[:512]) for c in candidates]
                scores   = reranker.predict(pairs, batch_size=16).tolist()
                ranked   = sorted(
                    zip(candidates, scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                return [
                    {**c, "score": round(float(s), 4)}
                    for c, s in ranked[:top_k]
                ]
            except Exception as e:
                log.error("Reranker error: %s — using semantic scores", e)

        # Fallback: semantic scores
        sem_map = {cid: sc for cid, sc in sem_list}
        return [
            {**c, "score": round(sem_map.get(c["chunk_id"], 0.0), 4)}
            for c in candidates[:top_k]
        ]

    # ── Multi-query ───────────────────────────────────────────────────────────

    def retrieve_multi(
        self,
        queries: List[str],
        top_k:   int = 5,
    ) -> List[Dict]:
        """
        Retrieve across multiple queries, deduplicated by chunk_id.
        Keeps highest-score version of each chunk.
        Returns results sorted by score descending.
        """
        if not queries:
            return []

        seen: Dict[str, Dict] = {}

        for q in queries:
            q = q.strip()
            if not q:
                continue
            for chunk in self.retrieve(q, top_k=top_k):
                cid   = chunk.get("chunk_id")
                score = chunk.get("score", 0.0)
                if cid and score > seen.get(cid, {}).get("score", -1.0):
                    seen[cid] = chunk

        result = sorted(
            seen.values(),
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        log.debug(
            "retrieve_multi: %d queries → %d unique chunks",
            len(queries), len(result),
        )
        return result