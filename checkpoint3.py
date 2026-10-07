import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.embeddings import MODEL_NAME, EmbeddingModel
from src.evaluation import evaluate_retrievers
from src.load_data import load_qasper
from src.reranking import RERANKER_MODEL, CrossEncoderReranker


def arguments():
    parser = argparse.ArgumentParser(description="Checkpoint 3 retrieval comparison")
    parser.add_argument("--evaluation-papers", type=int, default=40)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--without-reranker", action="store_true")
    parser.add_argument("--results", default="reports/checkpoint3_results.json")
    return parser.parse_args()


def main():
    args = arguments()
    if min(args.evaluation_papers, args.questions, args.top_k, args.candidate_pool) <= 0:
        raise SystemExit("numeric arguments must be positive")
    if args.candidate_pool < args.top_k:
        raise SystemExit("--candidate-pool must be greater than or equal to --top-k")
    if not 0.0 <= args.dense_weight <= 1.0:
        raise SystemExit("--dense-weight must be between 0 and 1")

    dataset = load_qasper()
    papers = dataset["validation"].select(
        range(min(args.evaluation_papers, len(dataset["validation"])))
    )
    embedding_model = EmbeddingModel(args.model)
    reranker = None if args.without_reranker else CrossEncoderReranker(args.reranker_model)
    metrics = evaluate_retrievers(
        papers, embedding_model, args.top_k, args.questions, args.candidate_pool,
        args.dense_weight, reranker,
    )
    payload = {
        "experiment": "Checkpoint 3 controlled retrieval comparison on QASPER validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args),
        "metrics": metrics,
        "notes": {
            "scope": "Known-paper passage retrieval; each question is searched within its source paper.",
            "evidence_match": "A hit requires at least 50% token coverage of one annotated evidence span.",
            "latency": "Warm per-query retrieval latency; excludes paper chunking and dense embedding construction.",
            "test_split": "Not used; reserved for final evaluation.",
        },
    }
    result_path = Path(args.results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
