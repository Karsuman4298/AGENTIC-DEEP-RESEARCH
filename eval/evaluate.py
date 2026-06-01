
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import call_llm_json

JUDGE_PROMPT = """Rate this research answer from 1 to 5 on each dimension.

Question: {question}

Answer: {answer}

Score each (1=poor, 5=excellent):
- accuracy: factually correct and relevant
- completeness: addresses all aspects  
- coherence: well-structured and clear

Return ONLY JSON: {{"accuracy": N, "completeness": N, "coherence": N}}"""


def judge_one(question: str, answer: str) -> Dict[str, float]:
    try:
        prompt = JUDGE_PROMPT.format(question=question, answer=answer[:1500])
        r = call_llm_json(prompt, max_tokens=100)
        return {
            "accuracy":     float(r.get("accuracy", 3)),
            "completeness": float(r.get("completeness", 3)),
            "coherence":    float(r.get("coherence", 3)),
        }
    except Exception as e:
        log.warning(f"Judge failed: {e}")
        return {"accuracy": 3.0, "completeness": 3.0, "coherence": 3.0}


def score_file(pred_path: str, ground_truth: dict = None,
               use_llm_judge: bool = True) -> dict:
    """Score one predictions JSONL file."""
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    if not preds:
        return {}

    config = Path(pred_path).stem

    # LLM-as-judge
    judge_scores = []
    if use_llm_judge:
        log.info(f"  LLM-as-judge on {len(preds)} answers...")
        for p in preds:
            s = judge_one(p["question"], p.get("answer", ""))
            judge_scores.append(s)
            time.sleep(1)  # rate limit
    else:
        judge_scores = [{"accuracy": 3.0, "completeness": 3.0, "coherence": 3.0}
                        for _ in preds]

    # Citation precision
    precisions = []
    for p in preds:
        cited    = p.get("cited_arxiv_ids", [])
        verified = p.get("verified_ids", cited)  # fallback: assume all verified
        halluc   = p.get("hallucinated_ids", [])
        if cited:
            prec = (len(cited) - len(halluc)) / len(cited)
        else:
            prec = 1.0
        precisions.append(max(0.0, prec))

    # Citation recall (needs ground truth)
    recalls = []
    if ground_truth:
        for p in preds:
            qid       = p.get("question_id", "")
            cited     = set(p.get("cited_arxiv_ids", []))
            must_cite = set(ground_truth.get(qid, {}).get("must_cite", []))
            if must_cite:
                recalls.append(len(cited & must_cite) / len(must_cite))

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "config":         config,
        "n":              len(preds),
        "accuracy":       avg([s["accuracy"]     for s in judge_scores]),
        "completeness":   avg([s["completeness"] for s in judge_scores]),
        "coherence":      avg([s["coherence"]    for s in judge_scores]),
        "faithfulness":   avg([p.get("verification_rate", 1.0) for p in preds]),
        "cite_precision": avg(precisions),
        "cite_recall":    avg(recalls) if recalls else "N/A",
        "avg_latency_s":  avg([p.get("latency_s", 0) for p in preds]),
        "avg_tool_calls": avg([p.get("tool_calls", 0) for p in preds]),
    }


def print_table(results: list) -> None:
    """Print clean ablation table."""
    cols = [
        ("Config",           "config",         22),
        ("Accuracy",         "accuracy",        9),
        ("Faithful",         "faithfulness",    9),
        ("Cite-P",           "cite_precision",  7),
        ("Cite-R",           "cite_recall",     7),
        ("Latency",          "avg_latency_s",   8),
        ("ToolCalls",        "avg_tool_calls",  10),
    ]

    header = "  ".join(f"{h:<{w}}" for h, _, w in cols)
    line   = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("ABLATION TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print(line)

    order = ["full_agent", "baseline", "no_planner", "no_reranker",
             "no_reflector", "no_hybrid", "no_citation_verifier"]
    sorted_r = sorted(results, key=lambda r: order.index(r["config"])
                      if r["config"] in order else 99)

    for r in sorted_r:
        row = "  ".join(f"{str(r.get(k,'N/A')):<{w}}" for _, k, w in cols)
        print(row)
    print(line)
    print()


# Fix missing import
from typing import Dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds",        default="predictions/",   help="Predictions directory")
    ap.add_argument("--output",       default="eval/results.json")
    ap.add_argument("--gt",           default=None,             help="Ground truth JSONL")
    ap.add_argument("--no_llm_judge", action="store_true",      help="Skip LLM judge (fast)")
    args = ap.parse_args()

    # Load ground truth if provided
    gt = None
    if args.gt and Path(args.gt).exists():
        gt = {}
        for line in open(args.gt):
            obj = json.loads(line)
            gt[obj["question_id"]] = obj
        log.info(f"Ground truth loaded: {len(gt)} entries")

    # Score all prediction files
    results = []
    pred_dir = Path(args.preds)
    files    = sorted(pred_dir.glob("*.jsonl"))

    if not files:
        log.error(f"No .jsonl files found in {pred_dir}")
        sys.exit(1)

    for f in files:
        log.info(f"Scoring: {f.name}")
        r = score_file(str(f), gt, use_llm_judge=not args.no_llm_judge)
        if r:
            results.append(r)
            log.info(f"  accuracy={r['accuracy']} cite_P={r['cite_precision']}")

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {args.output}")

    # Print ablation table
    print_table(results)


if __name__ == "__main__":
    main()