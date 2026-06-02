"""
run_pipeline.py — Master pipeline script

Usage:
  python run_pipeline.py                        # full run, 700 papers + PDFs
  python run_pipeline.py --max 5 --skip_pdfs   # quick test, 5 papers, no PDFs
  python run_pipeline.py --max 50              # 50 papers with PDFs
  python run_pipeline.py --skip_collect        # skip collection (already done)
  python run_pipeline.py --skip_index          # skip index build (already done)
  python run_pipeline.py --only_eval           # only run evaluation
  python run_pipeline.py --no_llm_judge        # skip LLM judge (fast eval)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(cmd: str, desc: str) -> None:
    log.info(f"\n{'='*55}\n  {desc}\n  $ {cmd}\n{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        log.error(f"FAILED: {cmd}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Agentic Deep Research — full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --max 5 --skip_pdfs     # quick test (2 min)
  python run_pipeline.py --max 50                # medium test
  python run_pipeline.py --max 700               # full corpus
  python run_pipeline.py --skip_collect          # index already built
  python run_pipeline.py --only_eval             # just re-score predictions
        """
    )

    # Collection args
    ap.add_argument("--max",        type=int, default=700,
                    help="Max papers to collect (default: 700)")
    ap.add_argument("--start",      default="2024-01-01",
                    help="Start date YYYY-MM-DD (default: 2024-01-01)")
    ap.add_argument("--end",        default="2026-04-30",
                    help="End date YYYY-MM-DD (default: 2026-04-30)")
    ap.add_argument("--skip_pdfs",  action="store_true",
                    help="Skip PDF download — metadata only (much faster)")

    # Skip flags
    ap.add_argument("--skip_collect", action="store_true",
                    help="Skip corpus collection (already done)")
    ap.add_argument("--skip_index",   action="store_true",
                    help="Skip index building (already done)")
    ap.add_argument("--skip_agent",   action="store_true",
                    help="Skip running agent configs")
    ap.add_argument("--only_eval",    action="store_true",
                    help="Only run evaluation (skip everything else)")

    # Eval args
    ap.add_argument("--no_llm_judge", action="store_true",
                    help="Skip LLM-as-judge scoring (fast mode)")
    ap.add_argument("--configs", nargs="+",
                    default=["full_agent","baseline","no_planner","no_reranker",
                             "no_reflector","no_hybrid","no_citation_verifier"],
                    help="Which agent configs to run")

    args = ap.parse_args()

    # --only_eval sets all skip flags
    if args.only_eval:
        args.skip_collect = True
        args.skip_index   = True
        args.skip_agent   = True

    # Create directories
    for d in ["data/raw","data/chunks","data/index","predictions","eval","logs","cache"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect ───────────────────────────────────────
    if not args.skip_collect:
        meta = Path("data/raw/metadata.jsonl")
        if meta.exists():
            n = sum(1 for _ in open(meta))
            if n >= args.max:
                log.info(f"Corpus already has {n} papers — skipping collection")
            else:
                pdf_flag = "--skip_pdfs" if args.skip_pdfs else ""
                run(
                    f"python scraper/collect.py --max {args.max} "
                    f"--start {args.start} --end {args.end} {pdf_flag}",
                    f"Step 1: Collecting up to {args.max} papers"
                )
        else:
            pdf_flag = "--skip_pdfs" if args.skip_pdfs else ""
            run(
                f"python scraper/collect.py --max {args.max} "
                f"--start {args.start} --end {args.end} {pdf_flag}",
                f"Step 1: Collecting up to {args.max} papers"
            )
    else:
        log.info("Skipping collection (--skip_collect)")

    # Step 2:─  Chunk 
    if not args.skip_index:
        chunk_files = list(Path("data/chunks").glob("*_chunks.jsonl"))
        if chunk_files:
            log.info(f"Chunks exist ({len(chunk_files)} files) — skipping chunking")
        else:
            run(
                "python indexer/chunk.py --raw data/raw --out data/chunks",
                "Step 2: Chunking PDFs"
            )

        # Step 3: Build index 
        bm25_ok   = Path("data/index/bm25.pkl").exists()
        chroma_ok = Path("data/index/chroma").exists()
        if bm25_ok and chroma_ok:
            log.info("Index already exists — skipping build")
        else:
            run(
                "python indexer/build_index.py --chunks data/chunks --index data/index",
                "Step 3: Building BM25 + ChromaDB index"
            )
    else:
        log.info("Skipping index build (--skip_index)")

    # Step 4: Run agent configs
    if not args.skip_agent:
        cfg_str = " ".join(args.configs)
        run(
            f"python run.py --config {cfg_str} --questions eval/questions.jsonl",
            f"Step 4: Running configs: {cfg_str}"
        )
    else:
        log.info("Skipping agent run (--skip_agent)")

    # Step 5: Evaluate
    judge_flag = "--no_llm_judge" if args.no_llm_judge else ""
    run(
        f"python eval/evaluate.py --preds predictions/ --output eval/results.json {judge_flag}",
        "Step 5: Evaluating + printing ablation table"
    )

    log.info("\n" + "="*55)
    log.info("   Pipeline complete!")
    log.info("    Predictions → predictions/")
    log.info("    Results     → eval/results.json")
    log.info("    Demo        → streamlit run demo/app.py")
    log.info("="*55)


if __name__ == "__main__":
    main()