import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, show_progress=True):
        if not texts:
            raise ValueError("There is no text to embed")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)


def save_index(folder, embeddings, chunks, model_name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    np.save(folder / "embeddings.npy", embeddings)
    with open(folder / "chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    details = {
        "model": model_name,
        "documents": len({chunk["paper_id"] for chunk in chunks}),
        "chunks": len(chunks),
        "embedding_dimensions": embeddings.shape[1],
    }
    with open(folder / "index_metadata.json", "w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    return folder


def load_index(folder):
    """Load a persistent dense index and validate its stored dimensions."""
    folder = Path(folder)
    embeddings = np.load(folder / "embeddings.npy")
    with open(folder / "chunks.json", encoding="utf-8") as file:
        chunks = json.load(file)
    with open(folder / "index_metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
        raise ValueError("Saved index is inconsistent: chunks and vectors do not match")
    if metadata.get("embedding_dimensions") != embeddings.shape[1]:
        raise ValueError("Saved index is inconsistent: embedding dimension does not match")
    return embeddings.astype(np.float32), chunks, metadata
