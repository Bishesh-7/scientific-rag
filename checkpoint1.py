import argparse

from src.chunking import chunk_papers
from src.embeddings import MODEL_NAME, EmbeddingModel, save_index
from src.load_data import load_qasper
from src.retrieval import retrieve


DEMO_QUESTION = "How is the annotation experiment evaluated?"


def get_arguments():
    parser = argparse.ArgumentParser(description="Checkpoint 1 retrieval demo")
    parser.add_argument("--papers", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output", default="data/checkpoint1_index")
    return parser.parse_args()


def main():
    args = get_arguments()
    if args.papers <= 0 or args.top_k <= 0:
        raise SystemExit("--papers and --top-k must be positive numbers")

    dataset = load_qasper()

    amount = min(args.papers, len(dataset["train"]))
    selected = dataset["train"].select(range(amount))
    papers = [selected[i] for i in range(len(selected))]

    print(f"\nChunking {len(papers)} papers...")
    chunks = chunk_papers(papers)
    print(f"Created {len(chunks)} chunks.")

    print(f"\nLoading embedding model: {args.model}")
    model = EmbeddingModel(args.model)
    texts = [chunk["embedding_text"] for chunk in chunks]
    embeddings = model.encode(texts)

    output = save_index(args.output, embeddings, chunks, model.model_name)
    print(f"Saved embedding matrix {embeddings.shape} to {output}")

    if args.query:
        question = args.query
    elif len(papers) == 1:
        question = papers[0]["qas"]["question"][0]
    else:
        question = DEMO_QUESTION

    print(f"\nQuestion: {question}")
    print("\nTop retrieved passages")
    print("=" * 80)

    results = retrieve(question, chunks, embeddings, model, args.top_k)
    for number, result in enumerate(results, start=1):
        print(f"\n{number}. Similarity score: {result['score']:.4f}")
        print(f"Paper: {result['title']}")
        print(f"Section: {result['section']}")
        print(result["text"][:500])


if __name__ == "__main__":
    main()
