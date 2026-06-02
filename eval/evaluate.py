
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import call_llm_json

JUDGE_PROMPT = """You are evaluating a research answer. Score it from 1 to 5 on three dimensions.

Question: {question}

Answer: {answer}

Scoring guide:
- accuracy:     1=wrong/irrelevant, 3=partially correct, 5=fully correct and on-topic
- completeness: 1=misses key aspects, 3=covers main points, 5=thorough and comprehensive
- coherence:    1=incoherent, 3=readable but disorganised, 5=clear, well-structured

Be strict. Most answers should score 2-4. Only exceptional answers score 5.

Return ONLY valid JSON:
{{"accuracy": N, "completeness": N, "coherence": N}}"""


def judge_one(question: str, answer: str) -> Dict[str, float]:
    """Score one answer with LLM-as-judge. Returns scores 1-5."""
    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer[:1200]
        )
        r = call_llm_json(prompt, max_tokens=150)

        # Validate scores are in range
        scores = {}
        for key in ["accuracy", "completeness", "coherence"]:
            val = r.get(key, 3)
            try:
                val = float(val)
                val = max(1.0, min(5.0, val))  # clamp to 1-5
            except Exception:
                val = 3.0
            scores[key] = val
        return scores

    except Exception as e:
        log.warning(f"Judge failed: {e}")
        return {"accuracy": 3.0, "completeness": 3.0, "coherence": 3.0}


def score_file(pred_path: str, ground_truth: dict = None,
               use_llm_judge: bool = True) -> dict:
    """Score one predictions JSONL file. Returns metrics dict."""
    preds = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    if not preds:
        return {}

    config = Path(pred_path).stem

    # LLM-as-judge
    judge_scores = []
    if use_llm_judge:
        log.info(f"  Judging {len(preds)} answers for '{config}'...")
        for i, p in enumerate(preds):
            s = judge_one(p["question"], p.get("answer", ""))
            judge_scores.append(s)
            log.debug(f"    [{i+1}/{len(preds)}] accuracy={s['accuracy']}")
            time.sleep(0.5)  # avoid rate limits
    else:
        judge_scores = [
            {"accuracy": 3.0, "completeness": 3.0, "coherence": 3.0}
            for _ in preds
        ]

    # Faithfulness
    faithfulness_scores = []
    for p in preds:
        # New field written by fixed agent.py
        f = p.get("faithfulness", None)
        if f is None:
            # Fallback for old predictions: use citation precision as proxy
            cited   = p.get("cited_arxiv_ids", [])
            halluc  = p.get("hallucinated_ids", [])
            f = (len(cited) - len(halluc)) / len(cited) if cited else 1.0
            f = max(0.0, f)
        faithfulness_scores.append(float(f))

    # Citation precision
    # = (cited - hallucinated) / cited
    precisions = []
    for p in preds:
        cited  = p.get("cited_arxiv_ids", [])
        halluc = p.get("hallucinated_ids", [])
        if cited:
            prec = max(0.0, (len(cited) - len(halluc)) / len(cited))
        else:
            prec = 1.0
        precisions.append(prec)

    # Citation recall
    # Needs ground truth with "must_cite" field
    recalls = []
    if ground_truth:
        for p in preds:
            qid       = p.get("question_id", "")
            cited     = set(p.get("cited_arxiv_ids", []))
            must_cite = set(ground_truth.get(qid, {}).get("must_cite", []))
            if must_cite:
                recalls.append(len(cited & must_cite) / len(must_cite))

    def avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "config":         config,
        "n":              len(preds),
        "accuracy":       avg([s["accuracy"]     for s in judge_scores]),
        "completeness":   avg([s["completeness"] for s in judge_scores]),
        "coherence":      avg([s["coherence"]    for s in judge_scores]),
        "faithfulness":   avg(faithfulness_scores),   # FIXED: lexical overlap
        "cite_precision": avg(precisions),
        "cite_recall":    avg(recalls) if recalls else "N/A",
        "avg_latency_s":  avg([p.get("latency_s", 0)  for p in preds]),
        "avg_tool_calls": avg([p.get("tool_calls", 0) for p in preds]),
    }


def print_table(results: list) -> None:
    """Print formatted ablation table to terminal."""
    cols = [
        ("Config",      "config",         24),
        ("Accuracy",    "accuracy",        9),
        ("Complete",    "completeness",    9),
        ("Faithful",    "faithfulness",    9),
        ("Cite-P",      "cite_precision",  7),
        ("Cite-R",      "cite_recall",     7),
        ("Latency(s)",  "avg_latency_s",   10),
        ("ToolCalls",   "avg_tool_calls",  10),
    ]

    header = "  ".join(f"{h:<{w}}" for h, _, w in cols)
    sep    = "=" * len(header)

    print(f"\n{sep}")
    print("ABLATION TABLE")
    print(sep)
    print(header)
    print("-" * len(header))

    order = ["full_agent", "baseline", "no_planner", "no_reranker",
             "no_reflector", "no_hybrid", "no_citation_verifier"]

    sorted_r = sorted(
        results,
        key=lambda r: order.index(r["config"]) if r["config"] in order else 99
    )

    for r in sorted_r:
        row = "  ".join(f"{str(r.get(k, 'N/A')):<{w}}" for _, k, w in cols)
        print(row)

    print(sep)
    print()


def main():
    ap = argparse.ArgumentParser(description="Evaluate all prediction files")
    ap.add_argument("--preds",        default="predictions/", help="Predictions directory")
    ap.add_argument("--output",       default="eval/results.json")
    ap.add_argument("--gt",           default=None,           help="Ground truth JSONL path")
    ap.add_argument("--no_llm_judge", action="store_true",    help="Skip LLM judge (fast)")
    args = ap.parse_args()

    # Load ground truth
    gt = None
    if args.gt and Path(args.gt).exists():
        gt = {}
        with open(args.gt) as f:
            for line in f:
                obj = json.loads(line)
                gt[obj["question_id"]] = obj
        log.info(f"Ground truth: {len(gt)} entries")

    # Score all files
    pred_dir = Path(args.preds)
    files    = sorted(pred_dir.glob("*.jsonl"))
    if not files:
        log.error(f"No .jsonl files in {pred_dir}")
        sys.exit(1)

    results = []
    for f in files:
        log.info(f"Scoring: {f.name}")
        r = score_file(str(f), gt, use_llm_judge=not args.no_llm_judge)
        if r:
            results.append(r)
            log.info(
                f"  accuracy={r['accuracy']}  "
                f"faithfulness={r['faithfulness']}  "
                f"cite_P={r['cite_precision']}"
            )

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved → {args.output}")

    print_table(results)


if __name__ == "__main__":
    main()