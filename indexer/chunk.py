import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MAX_CHARS    = 1800   # ~450 tokens — safe for CPU inference
OVERLAP_CHARS = 200
MIN_CHARS     = 80

SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(?:Abstract|Introduction|Related Work|Background|"
    r"Methodology|Method|Approach|Experiments?|Results?|Evaluation|"
    r"Discussion|Conclusion|Appendix|References|Acknowledgements?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate
    text = re.sub(r"\n{3,}", "\n\n", text)           # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)           # collapse spaces
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # page numbers
    return text.strip()


def _extract_text(pdf_path: str) -> Optional[str]:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return _clean("\n".join(pages))
    except Exception as e:
        log.warning(f"PDF parse failed {pdf_path}: {e}")
        return None


def _split_sections(text: str) -> List[Dict]:
    """Split text at section headers, return list of {title, text}."""
    positions = [(m.start(), m.group().strip()) for m in SECTION_RE.finditer(text)]
    if not positions:
        return [{"title": "Full Text", "text": text}]

    sections = []
    for i, (pos, header) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(text)
        body = text[pos:end].strip()
        if len(body) > MIN_CHARS:
            sections.append({"title": header, "text": body})
    return sections


def _windows(text: str) -> List[str]:
    """Sliding window over text with overlap."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            # Break at sentence boundary
            dot = text.rfind(". ", start + MAX_CHARS // 2, end)
            if dot > 0:
                end = dot + 1
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


def chunk_paper(pdf_path: str, arxiv_id: str, title: str) -> List[Dict]:
    text = _extract_text(pdf_path)
    if not text:
        return []

    sections = _split_sections(text)
    chunks, idx = [], 0

    for sec in sections:
        for window in _windows(sec["text"]):
            # --- CRITICAL FIX: HARD CONTEXT BOUNDARY INJECTION ---
            # This embeds the absolute source info directly inside the text string payload.
            # When the retriever pulls this text, the LLM cannot confuse the source.
            structured_text = (
                f"[DOCUMENT SOURCE PAPER: {title} (arXiv ID: {arxiv_id})]\n"
                f"[SECTION: {sec['title']}]\n\n"
                f"{window}"
            )

            chunks.append({
                "chunk_id":    f"{arxiv_id.replace('/','_')}_c{idx:04d}",
                "arxiv_id":    arxiv_id,
                "title":       title,
                "section":     sec["title"],
                "text":        structured_text, # Swapped with our bounded context
                "chunk_index": idx,
            })
            idx += 1
    return chunks


def chunk_corpus(raw_dir: str = "data/raw", out_dir: str = "data/chunks") -> int:
    raw  = Path(raw_dir)
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = raw / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found — run collect.py first")

    papers = [json.loads(l) for l in open(meta) if l.strip()]
    total, skipped = 0, 0

    for i, p in enumerate(papers):
        aid  = p["arxiv_id"]
        safe = aid.replace("/", "_")
        out_f = out / f"{safe}_chunks.jsonl"

        if out_f.exists():
            total += sum(1 for _ in open(out_f))
            continue

        safe = aid.replace("/", "_")
        pdf_file = p.get("pdf_path")
        if not pdf_file:
            pdf_file = raw / "pdfs" / f"{safe}.pdf"
        if not Path(pdf_file).exists():
            skipped += 1
            continue

        chunks = chunk_paper(str(pdf_file), aid, p["title"])
        if not chunks:
            skipped += 1
            continue

        with open(out_f, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        total += len(chunks)
        if (i+1) % 50 == 0:
            log.info(f"  {i+1}/{len(papers)} papers | {total} chunks so far")
    log.info(f" Chunking done: {total} chunks, {skipped} skipped")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/chunks")
    args = ap.parse_args()
    chunk_corpus(args.raw, args.out)
