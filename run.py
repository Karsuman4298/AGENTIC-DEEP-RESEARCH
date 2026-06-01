
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run.log"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from agent.Agent import CONFIGS, run_question
from indexer.retriever import Retriever


def run_config(config_name: str, questions: list, index_dir: str, chunks_dir: str) -> None:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Options: {list(CONFIGS)}")

    cfg     = CONFIGS[config_name]
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{config_name}.jsonl"

    log.info(f"\n{'='*50}")
    log.info(f"Config: {config_name}")
    log.info(f"Settings: {cfg}")
    log.info(f"Questions: {len(questions)}")
    log.info(f"Output: {out_path}")

    # Load retriever with ablation flags
    retriever = Retriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        use_hybrid=cfg["hybrid"],
        use_reranker=cfg["reranker"],
    )
    retriever.load()

    # Load existing results (resume support)
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj.get("question_id", ""))
        log.info(f"Resuming — {len(done_ids)} already done")

    with open(out_path, "a") as out_f:
        for i, q_obj in enumerate(questions):
            qid      = q_obj.get("question_id", str(i))
            question = q_obj.get("question", "")

            if qid in done_ids:
                continue

            log.info(f"[{i+1}/{len(questions)}] {qid}: {question[:70]}...")

            try:
                result = run_question(question, qid, retriever, cfg)
            except Exception as e:
                log.error(f"Error on {qid}: {e}")
                result = {
                    "question_id":     qid,
                    "question":        question,
                    "answer":          f"Error: {e}",
                    "cited_arxiv_ids": [],
                    "error":           str(e),
                    "latency_s":       0,
                    "tool_calls":      0,
                }

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            log.info(f"  ✓ {result.get('latency_s', '?')}s | "
                     f"{result.get('tool_calls', '?')} calls | "
                     f"{len(result.get('cited_arxiv_ids', []))} citations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     nargs="+", default=["full_agent"],
                    help="Config name(s) or 'all'")
    ap.add_argument("--questions",  default="eval/questions.jsonl")
    ap.add_argument("--index_dir",  default="data/index")
    ap.add_argument("--chunks_dir", default="data/chunks")
    args = ap.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("predictions").mkdir(exist_ok=True)

    # Load questions
    q_path = Path(args.questions)
    if not q_path.exists():
        log.error(f"Questions file not found: {q_path}")
        sys.exit(1)
    questions = [json.loads(l) for l in open(q_path) if l.strip()]
    log.info(f"Loaded {len(questions)} questions")

    # Which configs to run
    configs_to_run = list(CONFIGS.keys()) if "all" in args.config else args.config

    for cfg_name in configs_to_run:
        try:
            run_config(cfg_name, questions, args.index_dir, args.chunks_dir)
        except Exception as e:
            log.error(f"Config {cfg_name} failed: {e}")


if __name__ == "__main__":
    main()