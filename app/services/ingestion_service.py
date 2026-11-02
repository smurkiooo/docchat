import logging
import threading
import time
from pathlib import Path

from app.config import Settings
from app.db import Database
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_pages
from app.ingestion.pdf_parser import PdfError, parse_pdf
from app.models import DocumentStatus
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


PARSE_PROGRESS = 0.1


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ) -> None:
        self._settings = settings
        self._db = db
        self._embedder = embedder
        self._vectors = vector_store
        self._bm25 = bm25_store

        self._lock = threading.Lock()

    def ingest(self, doc_id: str, path: Path) -> None:
        with self._lock:
            started = time.perf_counter()
            try:
                self._ingest(doc_id, path)
            except PdfError as exc:
                self._fail(doc_id, str(exc))
            except Exception:
                logger.exception("Ingestion of %s failed", doc_id)
                self._fail(doc_id, "Внутренняя ошибка при индексации документа.")
            else:
                logger.info("Indexed %s in %.1fs", doc_id, time.perf_counter() - started)

    def _ingest(self, doc_id: str, path: Path) -> None:
        s = self._settings
        pages = clean_pages(parse_pdf(path))
        chunks = chunk_pages(pages, doc_id, s.chunk_size, s.chunk_overlap, s.min_chunk_chars)
        if not chunks:
            raise PdfError("Не удалось извлечь текст из документа.")
        self._db.update_document(doc_id, pages=len(pages), progress=PARSE_PROGRESS)

        def report(share: float) -> None:
            progress = PARSE_PROGRESS + (1 - PARSE_PROGRESS) * share * 0.99
            self._db.update_document(doc_id, progress=round(progress, 3))

        embeddings = self._embedder.embed_passages([c.text for c in chunks], on_progress=report)
        self._db.save_chunks(chunks)
        self._vectors.add(doc_id, chunks, embeddings)
        self._bm25.build(doc_id, chunks)
        self._db.update_document(
            doc_id, status=DocumentStatus.READY, progress=1.0, chunks=len(chunks)
        )

    def _fail(self, doc_id: str, message: str) -> None:
        self._vectors.delete(doc_id)
        self._bm25.delete(doc_id)
        self._db.delete_chunks(doc_id)
        self._db.update_document(
            doc_id, status=DocumentStatus.ERROR, error=message, progress=0.0, chunks=0
        )

    def delete(self, doc_id: str, path: Path) -> None:
        self._vectors.delete(doc_id)
        self._bm25.delete(doc_id)
        self._db.delete_document(doc_id)
        path.unlink(missing_ok=True)
