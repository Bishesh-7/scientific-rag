import numpy as np


def retrieve(query, chunks, chunk_embeddings, model, top_k=5):
    if not query.strip():
        raise ValueError("The question cannot be empty")
    if len(chunks) != len(chunk_embeddings):
        raise ValueError("The number of chunks and embeddings must match")

    query_embedding = model.encode([query], show_progress=False)[0]

    # The embeddings are normalised, so this gives cosine similarity.
    scores = chunk_embeddings @ query_embedding
    best_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]

    results = []
    for index in best_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results
