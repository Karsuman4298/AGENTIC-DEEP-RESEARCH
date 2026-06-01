import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION     = "arxiv_chunks"
# Larger batch on GPU/MPS, conservative on CPU
BATCH  = 64 if DEVICE in ("cuda", "mps") else 32


def load_all_chunks(chunks_dir: str) -> List[Dict]:
    chunks = []
    for f in sorted(Path(chunks_dir).glob("*_chunks.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    log.info(f"Loaded {len(chunks)} chunks")
    return chunks


def build_bm25(chunks: List[Dict], cache_path: str) -> None:
    from rank_bm25 import BM25Okapi
    log.info("Building BM25 index...")
    corpus = []
    ids    = []
    for c in tqdm(chunks, desc="BM25", unit="chunk"):
        text = f"{c.get('title','')} {c.get('section','')} {c['text']}"
        corpus.append(text.lower().split())
        ids.append(c["chunk_id"])
    bm25 = BM25Okapi(corpus)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    log.info(f"BM25 saved → {cache_path}")


def build_chroma(chunks: List[Dict], chroma_dir: str) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading embedding model: {EMBED_MODEL} (device={DEVICE})")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    model.max_seq_length = 512

    client = chromadb.PersistentClient(path=chroma_dir)
    col    = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    existing   = set(col.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        log.info("ChromaDB: all chunks already indexed")
        return

    log.info(f"ChromaDB: embedding {len(new_chunks)} new chunks...")

    for i in tqdm(range(0, len(new_chunks), BATCH), desc="Chroma", unit="batch"):
        batch = new_chunks[i:i+BATCH]
        texts = [
            f"Represent this passage for retrieval: {c.get('title','')}. {c.get('section','')}. {c['text']}"
            for c in batch
        ]
        embs = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=8).tolist()
        col.add(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = embs,
            documents  = [c["text"] for c in batch],
            metadatas  = [{
                "arxiv_id":    c["arxiv_id"],
                "title":       c.get("title","")[:200],
                "section":     c.get("section","")[:100],
                "chunk_index": int(c.get("chunk_index", 0)),
            } for c in batch],
        )
    log.info(f"ChromaDB done: {col.count()} total docs")


def build_index(chunks_dir: str = "data/chunks", index_dir: str = "data/index") -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    chunks = load_all_chunks(chunks_dir)
    if not chunks:
        raise RuntimeError("No chunks found — run chunk.py first")

    bm25_path   = str(Path(index_dir) / "bm25.pkl")
    chroma_path = str(Path(index_dir) / "chroma")

    build_bm25(chunks, bm25_path)
    build_chroma(chunks, chroma_path)
    log.info(f" Index built → {index_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks")
    ap.add_argument("--index",  default="data/index")
    args = ap.parse_args()
    build_index(args.chunks, args.index)