from dataclasses import dataclass

from app.config import Settings
from app.db import Database
from app.llm.base import LLMProvider
from app.llm.factory import create_llm
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder, SentenceTransformerEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RagService


@dataclass
class Container:
    settings: Settings
    db: Database
    embedder: Embedder
    llm: LLMProvider
    ingestion: IngestionService
    rag: RagService


def build_container(
    settings: Settings, embedder: Embedder | None = None, llm: LLMProvider | None = None
) -> Container:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    embedder = embedder or SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_batch_size,
        settings.embedding_revision,
        settings.model_device,
    )
    llm = llm or create_llm(settings)
    vector_store = VectorStore(settings.chroma_dir)
    bm25_store = BM25Store(db.list_chunks)
    retriever = HybridRetriever(
        embedder,
        vector_store,
        bm25_store,
        db.get_chunks,
        top_k_vector=settings.top_k_vector,
        top_k_bm25=settings.top_k_bm25,
        top_k_final=settings.top_k_final,
        rrf_k=settings.rrf_k,
        reranker=CrossEncoderReranker(
            settings.reranker_model, settings.reranker_revision, settings.model_device
        )
        if settings.use_reranker
        else None,
    )
    return Container(
        settings=settings,
        db=db,
        embedder=embedder,
        llm=llm,
        ingestion=IngestionService(settings, db, embedder, vector_store, bm25_store),
        rag=RagService(settings, db, retriever, llm),
    )
