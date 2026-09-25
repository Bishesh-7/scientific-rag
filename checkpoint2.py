import argparse
import json
from pathlib import Path

from src.chunking import chunk_papers
from src.embeddings import MODEL_NAME, EmbeddingModel
from src.evaluation import evaluate_dense_retrieval
from src.generation import GENERATOR_MODEL, LocalRAGGenerator, generate_grounded_answer
from src.load_data import load_qasper
from src.vector_store import ChromaVectorStore


def arguments():
    parser = argparse.ArgumentParser(description="Checkpoint 2 dense RAG baseline")
    parser.add_argument("--papers", type=int, default=25)
    parser.add_argument("--evaluation-papers", type=int, default=10)
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", default="How is the annotation experiment evaluated?")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--generator-model", default=GENERATOR_MODEL)
    parser.add_argument("--database", default="data/checkpoint2_chroma")
    parser.add_argument("--results", default="reports/checkpoint2_results.json")
    parser.add_argument(
        "--extractive", action="store_true",
        help="Use the focused extractive fallback instead of the local LLM",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if min(args.papers, args.evaluation_papers, args.questions, args.top_k) <= 0:
        raise SystemExit("numeric arguments must be positive")

    dataset = load_qasper()
    model = EmbeddingModel(args.model)
    database_path = Path(args.database)
    store = ChromaVectorStore(database_path)
    if store.exists() and not args.rebuild:
        metadata = store.metadata()
        if metadata["model"] != args.model:
            raise SystemExit("vector database uses another model; pass --rebuild")
        print(f"Loaded {metadata['chunks']} vectors from {database_path}")
    else:
        selected = dataset["train"].select(range(min(args.papers, len(dataset["train"]))))
        chunks = chunk_papers(selected)
        embeddings = model.encode([item["embedding_text"] for item in chunks])
        metadata = store.create(embeddings, chunks, args.model)
        print(f"Built ChromaDB vector database with {len(chunks)} chunks at {database_path}")

    passages = store.retrieve(args.query, model, args.top_k)
    if args.extractive:
        response = generate_grounded_answer(args.query, passages)
        response["model"] = "focused extractive fallback"
    else:
        generator = LocalRAGGenerator(args.generator_model)
        response = generator.generate(args.query, passages)
    print(f"\nQuestion: {args.query}\nAnswer: {response['answer']}")
    for citation in response["citations"]:
        print(f"[{citation['number']}] {citation['title']} — {citation['section']}")

    validation = dataset["validation"].select(
        range(min(args.evaluation_papers, len(dataset["validation"])))
    )
    metrics = evaluate_dense_retrieval(validation, model, args.top_k, args.questions)
    payload = {
        "configuration": vars(args), "vector_database": metadata,
        "retrieval_metrics": metrics, "sample": response,
    }
    result_path = Path(args.results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nValidation metrics")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
