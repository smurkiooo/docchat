import logging
import threading
from collections.abc import Callable, Sequence
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def embed_passages(
        self, texts: Sequence[str], on_progress: Callable[[float], None] | None = None
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._passage_prefix = "passage: " if is_e5 else ""

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("загрузка эмбеддинг модели %s", self.model_name)
                self._model = SentenceTransformer(
                    self.model_name, revision=self.revision, device=self.device
                )
            return self._model

    def warmup(self) -> None:
        self._get_model()

    def embed_passages(
        self, texts: Sequence[str], on_progress: Callable[[float], None] | None = None
    ) -> list[list[float]]:
        model = self._get_model()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self._passage_prefix + t for t in texts[start : start + self.batch_size]]
            vectors.extend(model.encode(batch, normalize_embeddings=True).tolist())
            if on_progress:
                on_progress(len(vectors) / len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        return model.encode(self._query_prefix + text, normalize_embeddings=True).tolist()
