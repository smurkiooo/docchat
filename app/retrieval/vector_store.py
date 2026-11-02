import contextlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models import Chunk


class VectorStore:
    def __init__(self, path: Path) -> None:
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )

    @staticmethod
    def _name(doc_id: str) -> str:
        return f"doc_{doc_id}"

    def add(self, doc_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        collection = self._client.get_or_create_collection(
            self._name(doc_id), metadata={"hnsw:space": "cosine"}
        )
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings[start : start + batch_size],
                documents=[c.text for c in batch],
                metadatas=[{"page_start": c.page_start, "page_end": c.page_end} for c in batch],
            )

    def search(self, doc_id: str, embedding: list[float], k: int) -> list[tuple[str, float]]:
        collection = self._client.get_collection(self._name(doc_id))
        k = min(k, collection.count())
        if k == 0:
            return []
        result = collection.query(query_embeddings=[embedding], n_results=k, include=["distances"])
        ids, distances = result["ids"][0], result["distances"][0]
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances, strict=True)]

    def delete(self, doc_id: str) -> None:
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._name(doc_id))
