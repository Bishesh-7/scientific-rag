from pathlib import Path

import chromadb
import numpy as np


class ChromaVectorStore:
    """Persistent ChromaDB collection using cosine-distance HNSW search."""

    COLLECTION = "qasper_chunks"

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))

    def exists(self):
        try:
            self.client.get_collection(self.COLLECTION)
            return True
        except Exception:
            return False

    def create(self, embeddings, chunks, model_name):
        if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have matching rows")
        if self.exists():
            self.client.delete_collection(self.COLLECTION)
        details = {
            "model": model_name,
            "documents": len({chunk["paper_id"] for chunk in chunks}),
            "chunks": len(chunks),
            "embedding_dimensions": int(embeddings.shape[1]),
            "storage": "ChromaDB persistent vector database (HNSW cosine search)",
        }
        collection = self.client.create_collection(
            self.COLLECTION,
            metadata={**details, "hnsw:space": "cosine"},
        )
        collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "paper_id": chunk["paper_id"],
                    "title": chunk["title"],
                    "section": chunk["section"],
                    "embedding_text": chunk["embedding_text"],
                }
                for chunk in chunks
            ],
        )
        return details

    def metadata(self):
        collection = self.client.get_collection(self.COLLECTION)
        values = dict(collection.metadata or {})
        values.pop("hnsw:space", None)
        values.pop("_type", None)
        return values

    def search(self, query_vector, top_k=5, paper_id=None):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = np.asarray(query_vector, dtype=np.float32)
        details = self.metadata()
        if query_vector.shape != (details["embedding_dimensions"],):
            raise ValueError("query vector dimension does not match the database")
        collection = self.client.get_collection(self.COLLECTION)
        amount = min(top_k, collection.count())
        if amount == 0:
            return []
        query = {
            "query_embeddings": [query_vector.tolist()],
            "n_results": amount,
            "include": ["documents", "metadatas", "distances"],
        }
        if paper_id is not None:
            query["where"] = {"paper_id": paper_id}
        response = collection.query(**query)
        results = []
        for chunk_id, text, metadata, distance in zip(
            response["ids"][0], response["documents"][0],
            response["metadatas"][0], response["distances"][0],
        ):
            results.append(
                {
                    "chunk_id": chunk_id, "paper_id": metadata["paper_id"],
                    "title": metadata["title"], "section": metadata["section"],
                    "text": text, "embedding_text": metadata["embedding_text"],
                    "score": 1.0 - float(distance),
                }
            )
        return results

    def retrieve(self, query, model, top_k=5, paper_id=None):
        if not query.strip():
            raise ValueError("the question cannot be empty")
        query_vector = model.encode([query], show_progress=False)[0]
        return self.search(query_vector, top_k, paper_id)
