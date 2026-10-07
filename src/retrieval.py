import re
from collections import Counter

import numpy as np


def tokenize(text):
    """Tokenize text for lexical retrieval using lowercase alphanumeric terms."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Small dependency-free Okapi BM25 index for a collection of chunks."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk["embedding_text"])) for chunk in chunks]
        self.lengths = np.asarray([sum(values.values()) for values in self.term_frequencies])
        self.average_length = float(np.mean(self.lengths)) or 1.0
        document_frequencies = Counter()
        for values in self.term_frequencies:
            document_frequencies.update(values.keys())
        size = len(chunks)
        self.idf = {
            term: np.log(1.0 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query):
        if not query.strip():
            raise ValueError("The question cannot be empty")
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            frequencies = np.asarray(
                [values.get(term, 0) for values in self.term_frequencies], dtype=np.float32
            )
            denominator = frequencies + self.k1 * (
                1.0 - self.b + self.b * self.lengths / self.average_length
            )
            scores += idf * frequencies * (self.k1 + 1.0) / denominator
        return scores

    def retrieve(self, query, top_k=5):
        scores = self.scores(query)
        return _rank_chunks(self.chunks, scores, top_k)


def _rank_chunks(chunks, scores, top_k):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    best_indices = np.argsort(scores, kind="stable")[::-1][: min(top_k, len(chunks))]
    results = []
    for index in best_indices:
        result = chunks[int(index)].copy()
        result["score"] = float(scores[index])
        results.append(result)
    return results


def retrieve(query, chunks, chunk_embeddings, model, top_k=5):
    if not query.strip():
        raise ValueError("The question cannot be empty")
    if len(chunks) != len(chunk_embeddings):
        raise ValueError("The number of chunks and embeddings must match")

    query_embedding = model.encode([query], show_progress=False)[0]

    # The embeddings are normalised, so this gives cosine similarity.
    scores = chunk_embeddings @ query_embedding
    return _rank_chunks(chunks, scores, top_k)


def reciprocal_rank_fusion(dense_results, bm25_results, top_k=5, dense_weight=0.5, k=60):
    """Fuse dense and BM25 rankings without assuming comparable score scales."""
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must be between 0 and 1")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    combined = {}
    lexical_weight = 1.0 - dense_weight
    for weight, results in ((dense_weight, dense_results), (lexical_weight, bm25_results)):
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            if chunk_id not in combined:
                combined[chunk_id] = {**item, "score": 0.0}
            combined[chunk_id]["score"] += weight / (k + rank)
    return sorted(combined.values(), key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]


def hybrid_retrieve(
    query, chunks, chunk_embeddings, model, bm25_index=None, top_k=5,
    candidate_pool=20, dense_weight=0.5,
):
    """Retrieve candidates with dense and BM25 search and combine them using RRF."""
    pool = min(max(top_k, candidate_pool), len(chunks))
    bm25_index = bm25_index or BM25Retriever(chunks)
    dense_results = retrieve(query, chunks, chunk_embeddings, model, pool)
    bm25_results = bm25_index.retrieve(query, pool)
    return reciprocal_rank_fusion(dense_results, bm25_results, top_k, dense_weight)
