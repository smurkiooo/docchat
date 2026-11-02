import logging
import threading

from app.models import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, revision: str | None = None, device: str = "cpu") -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker %s", self.model_name)
                self._model = CrossEncoder(
                    self.model_name, revision=self.revision, device=self.device
                )
            return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        scores = self._get_model().predict([(query, c.chunk.text) for c in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)
