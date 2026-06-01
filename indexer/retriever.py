import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

log = logging.getLogger(__name__)

import sys
import torch
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION   = "arxiv_chunks"
RRF_K        = 60

# --- HARD CUTOFF THRESHOLDS ---
SEMANTIC_SIM_THRESHOLD = 0.40  # Reject cosine similarity values below 40%
BM25_SCORE_THRESHOLD   = 1.0   # Reject documents with trivial keyword overlaps

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


def _rrf(bm25_list, sem_list, k=RRF_K) -> List:
    scores: Dict[str, float] = {}
    for rank, (cid, *_) in enumerate(bm25_list):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, *_) in enumerate(sem_list):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(
        self,
        index_dir:    str  = "data/index",
        chunks_dir:   str  = "data/chunks",
        use_hybrid:   bool = True,
        use_reranker: bool = True,
    ):
        self.use_hybrid   = use_hybrid
        self.use_reranker = use_reranker
        self._store: Dict[str, Dict] = {}  # chunk_id → chunk dict
        self._bm25        = None
        self._bm25_ids:   List[str] = []
        self._chroma_col  = None

        self._index_dir  = index_dir
        self._chunks_dir = chunks_dir

    def load(self) -> None:
        """Load all indexes into memory. Call once before any retrieve()."""
        # 1. Load chunk store
        chunks = []
        for f in sorted(Path(self._chunks_dir).glob("*_chunks.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        self._store = {c["chunk_id"]: c for c in chunks}
        log.info(f"Chunk store: {len(self._store)} chunks")

        # 2. Load BM25
        bm25_path = Path(self._index_dir) / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                d = pickle.load(f)
            self._bm25, self._bm25_ids = d["bm25"], d["ids"]
            log.info(f"BM25 loaded: {len(self._bm25_ids)} docs")
        else:
            log.warning("BM25 cache not found — hybrid disabled")
            self.use_hybrid = False

        # 3. Load ChromaDB
        import chromadb
        client = chromadb.PersistentClient(path=str(Path(self._index_dir) / "chroma"))
        self._chroma_col = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        log.info(f"ChromaDB loaded: {self._chroma_col.count()} docs")

        # 4. Warm up reranker
        if self.use_reranker:
            _get_reranker()

        # 5. Warm up embedder
        _get_embed()

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for query with strict similarity filtering to prevent hallucinations.
        Returns list of dicts with keys: chunk_id, arxiv_id, title, section, text, score
        """
        # Embed query
        model  = _get_embed()
        q_pref = f"Represent this sentence for searching relevant passages: {query}"
        q_vec  = model.encode(q_pref, normalize_embeddings=True).tolist()

        # Semantic search
        n = min(candidate_k, self._chroma_col.count())
        if n == 0:
            return []

        res = self._chroma_col.query(
            query_embeddings=[q_vec], n_results=n,
            include=["distances", "metadatas", "documents"],
        )
        
        # --- CRITICAL FIX 1: SEMANTIC CUTOFF FILTER ---
        sem_list = []
        if res["ids"] and len(res["ids"][0]) > 0:
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                similarity = 1.0 - dist # Map Cosine space to similarity
                if similarity >= SEMANTIC_SIM_THRESHOLD:
                    sem_list.append((cid, similarity))

        # --- CRITICAL FIX 2: BM25 CUTOFF FILTER ---
        bm25_list = []
        if self.use_hybrid and self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            raw_bm25 = zip(self._bm25_ids, bm25_scores.tolist())
            
            # Keep only chunks that actually hit clear keywords
            filtered_bm25 = [
                (cid, score) for cid, score in raw_bm25 
                if score >= BM25_SCORE_THRESHOLD
            ]
            bm25_list = sorted(filtered_bm25, key=lambda x: x[1], reverse=True)[:candidate_k]

        # --- FUSION EXECUTION BOUNDARIES ---
        if self.use_hybrid and (bm25_list or sem_list):
            fused = _rrf(bm25_list, sem_list)
            cand_ids = [cid for cid, _ in fused[:candidate_k]]
        else:
            # Fallback if hybrid is off or BM25 returned empty
            cand_ids = [cid for cid, _ in sem_list[:candidate_k]]

        # Hydrate chunks from store
        candidates = [self._store[cid] for cid in cand_ids if cid in self._store]
        if not candidates:
            return [] # Fail safely with an empty list if nothing matches

        # Reranking Layer
        if self.use_reranker and len(candidates) > top_k:
            reranker = _get_reranker()
            pairs = [(query, c["text"][:512]) for c in candidates]
            scores = reranker.predict(pairs, batch_size=8).tolist()
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [{**c, "score": round(float(s), 4)} for c, s in ranked[:top_k]]
        else:
            sem_map = {cid: sc for cid, sc in sem_list}
            return [{**c, "score": round(sem_map.get(c["chunk_id"], 0.0), 4)}
                    for c in candidates[:top_k]]

    def retrieve_multi(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Retrieve across multiple queries, deduplicate by chunk_id."""
        seen: Dict[str, Dict] = {}
        for q in queries:
            for r in self.retrieve(q, top_k=top_k):
                cid = r["chunk_id"]
                if cid not in seen or r["score"] > seen[cid]["score"]:
                    seen[cid] = r
        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)
