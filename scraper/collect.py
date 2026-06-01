
import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
# arXiv API 
NS  = "http://www.w3.org/2005/Atom"
API = "http://export.arxiv.org/api/query"

# Agent-related keywords (title OR abstract must contain at least one)
KEYWORDS = [
    "llm agent",
    "language model agent",
    "agentic",
    "agentic rag",
    "tool use",
    "tool-use",
    "tool calling",
    "function calling",
    "multi-agent",
    "multiagent",
    "autonomous agent",
    "web agent",
    "code agent",
    "computer use agent",
    "computer-use agent",
    "react",
    "reflexion",
    "self-rag",
    "agent memory",
    "agent planning",
    "agent benchmark",
    "agent evaluation",
]

CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]


def _build_url(cat: str, start_date: str, end_date: str,
               offset: int, batch: int) -> str:
    """Build arXiv API query URL."""
    # Convert YYYY-MM-DD → YYYYMMDD for arXiv API
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    query = (
    f'cat:{cat} AND '
    f'submittedDate:[{s}0000 TO {e}2359] AND '
    f'(all:"agentic" OR '
    f'all:"llm agent" OR '
    f'all:"tool use" OR '
    f'all:"multi-agent" OR '
    f'all:"agent memory" OR '
    f'all:"computer use")'
)
    params = {
        "search_query": query,
        "start":        offset,
        "max_results":  batch,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    return f"{API}?{urllib.parse.urlencode(params)}"


def _parse_entries(root: ET.Element) -> List[Dict]:
    """Parse Atom XML into list of paper dicts."""
    papers = []
    for entry in root.findall(f"{{{NS}}}entry"):
        try:
            arxiv_id = entry.find(f"{{{NS}}}id").text.split("/abs/")[-1].strip()
            title    = entry.find(f"{{{NS}}}title").text.strip().replace("\n", " ")
            abstract = entry.find(f"{{{NS}}}summary").text.strip().replace("\n", " ")
            pub      = entry.find(f"{{{NS}}}published").text[:10]
            authors  = [
                a.find(f"{{{NS}}}name").text
                for a in entry.findall(f"{{{NS}}}author")
            ]
            papers.append({
                "arxiv_id":  arxiv_id,
                "title":     title,
                "abstract":  abstract,
                "published": pub,
                "authors":   authors,
                "pdf_url":   f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "pdf_path":  None,
            })
        except Exception as e:
            log.debug(f"Skipping malformed entry: {e}")
    return papers


def _is_relevant(paper: Dict) -> bool:
    """Return True if title or abstract contains an agent-related keyword."""
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return any(kw in text for kw in KEYWORDS)


def _download_pdf(paper: Dict, pdf_dir: Path) -> Optional[str]:
    """Download PDF. Returns local path or None on failure."""
    safe    = paper["arxiv_id"].replace("/", "_")
    out     = pdf_dir / f"{safe}.pdf"
    if out.exists():
        log.debug(f"PDF already exists: {safe}.pdf")
        return str(out)
    try:
        req = urllib.request.Request(
            paper["pdf_url"],
            headers={"User-Agent": "AgentResearchBot/1.0 (academic research)"}
        )
        with safe_urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) < 2000:  
            return None
        out.write_bytes(data)
        log.debug(f"Downloaded {safe}.pdf ({len(data)//1024} KB)")
        return str(out)
    except Exception as e:
        log.debug(f"PDF download failed {paper['arxiv_id']}: {e}")
        return None
    

def fetch_with_retry(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                    "SKResearchBot/1.0 (Academic Research)"
                }
            )
            with safe_urlopen(req, timeout=120) as r:
                return ET.fromstring(r.read())
        except Exception as e:
            wait = (attempt + 1) * 10
            log.warning(
                f"API error: {e} | retry {attempt+1}/{retries}"
            )
            time.sleep(wait)
    return None

def safe_urlopen(req, timeout=120, max_retries=5):
    """
    Opens URL with exponential backoff.
    Handles:
        429 Too Many Requests
        503 Service Unavailable
        Timeouts
    """

    for attempt in range(max_retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            wait = min(60, 5 * (2 ** attempt))
            log.warning(
                f"API error: {e} | retry {attempt+1}/{max_retries}"
            )
            log.warning(f"Sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("Maximum retries exceeded")

def collect(
    max_papers:  int  = 700,
    start_date:  str  = "2024-01-01",
    end_date:    str  = "2026-04-30",
    out_dir:     str  = "data/raw",
    skip_pdfs:   bool = False,
    batch_size:  int  = 100,
) -> List[Dict]:
    """
    Collect up to max_papers relevant arXiv papers between start_date and end_date.

    Parameters
    ----------
    max_papers  : Maximum number of papers to collect (e.g. 5, 50, 700)
    start_date  : Start date in YYYY-MM-DD format
    end_date    : End date in YYYY-MM-DD format
    out_dir     : Output directory (creates metadata.jsonl + pdfs/)
    skip_pdfs   : If True, skip PDF download (faster, metadata only)
    batch_size  : Papers per API call (max 100 for arXiv)

    Returns
    -------
    List of paper dicts
    """
    out     = Path(out_dir)
    pdf_dir = out / "pdfs"
    out.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)

    meta_path = out / "metadata.jsonl"
    seen:   set       = set()
    papers: List[Dict] = []

    # Resume from existing 
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    seen.add(p["arxiv_id"])
                    papers.append(p)
        log.info(f"Resuming — {len(papers)} papers already collected")

    if len(papers) >= max_papers:
        log.info(f"Already have {len(papers)} papers (max={max_papers}) — nothing to do")
        return papers

    # Fetch from arXiv
    log.info(f"Collecting up to {max_papers} papers | {start_date} → {end_date}")
    for cat in CATEGORIES:
        if len(papers) >= max_papers:
            break

        log.info(f"  Category: {cat}")
        cat_added = 0
        offset    = 0

        while len(papers) < max_papers:

            fetch_n = min(
                batch_size,
                max_papers - len(papers)
            )

            url  = _build_url(cat, start_date, end_date, offset, fetch_n)

            root = fetch_with_retry(url)
            if root is None:
                log.warning(f"Failed after retries: offset={offset}")
                break

            entries = _parse_entries(root)
            if not entries:
                log.info(f"  No more entries for {cat}")
                break

            for p in entries:
                if len(papers) >= max_papers:
                    break
                if p["arxiv_id"] in seen:
                    continue
                if not _is_relevant(p):
                    continue

                seen.add(p["arxiv_id"])

                # Download PDF
                if not skip_pdfs:
                    p["pdf_path"] = _download_pdf(p, pdf_dir)
                    time.sleep(3)   # polite delay

                papers.append(p)
                cat_added += 1

                # Write immediately (crash-safe)
                with open(meta_path, "a") as f:
                    f.write(json.dumps(p) + "\n")
                log.info(f"  [{len(papers)}/{max_papers}] {p['arxiv_id']} — {p['title'][:60]}")
            offset += batch_size
            time.sleep(5)   # arXiv rate limit: 1 req/5s
        log.info(f"  {cat}: added {cat_added} papers")
    log.info(f"\n Done — {len(papers)} papers collected → {meta_path}")
    if not skip_pdfs:
        n_pdfs = sum(1 for p in papers if p.get("pdf_path"))
        log.info(f"   PDFs downloaded: {n_pdfs}/{len(papers)}")
    return papers


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Collect LLM-agent papers from arXiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper/collect.py                              # 700 papers, full date range
  python scraper/collect.py --max 5                      # quick test: 5 papers
  python scraper/collect.py --max 50 --start 2025-06-01  # 50 recent papers
  python scraper/collect.py --max 700 --start 2024-01-01 --end 2026-04-30
  python scraper/collect.py --max 20 --skip_pdfs         # metadata only, no PDFs
        """
    )
    ap.add_argument(
        "--max",
        type=int,
        default=700,
        help="Max papers to collect (default: 700, minimum: 1)"
    )
    ap.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01)"
    )
    ap.add_argument(
        "--end",
        default="2026-04-30",
        help="End date YYYY-MM-DD (default: 2026-04-30)"
    )
    ap.add_argument(
        "--out",
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    ap.add_argument(
        "--skip_pdfs",
        action="store_true",
        help="Skip PDF download — collect metadata only (much faster)"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=25,
        help="API batch size (default: 50, max: 100)"
    )

    args = ap.parse_args()
    if args.max < 1:
        ap.error("--max must be at least 1")
    if args.max > 700:
        log.warning(f"--max={args.max} exceeds recommended 700. arXiv may throttle.")

    collect(
        max_papers = args.max,
        start_date = args.start,
        end_date   = args.end,
        out_dir    = args.out,
        skip_pdfs  = args.skip_pdfs,
        batch_size = args.batch,
    )