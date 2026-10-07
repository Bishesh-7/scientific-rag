from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    """Rerank retrieved candidates by scoring each question-passage pair."""

    def __init__(self, model_name=RERANKER_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, question, candidates, top_k=5):
        if not question.strip():
            raise ValueError("The question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        pairs = [(question, item["embedding_text"]) for item in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        reranked = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["score"] = float(score)
            reranked.append(item)
        return sorted(reranked, key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]
