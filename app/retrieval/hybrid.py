from collections.abc import Callable
from dataclasses import dataclass, replace

from app.models import Chunk, RetrievedChunk
from app.retrieval.bm25_store import BM25Store, tokenize
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    max_similarity: float
    lexical_coverage: float

    def is_relevant(self, min_similarity: float, min_coverage: float = 0.5) -> bool:
        return self.max_similarity >= min_similarity or self.lexical_coverage >= min_coverage


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        get_chunks: Callable[[list[str]], dict[str, Chunk]],
        *,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 6,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._bm25 = bm25_store
        self._get_chunks = get_chunks
        self._top_k_vector = top_k_vector
        self._top_k_bm25 = top_k_bm25
        self._top_k_final = top_k_final
        self._rrf_k = rrf_k
        self._reranker = reranker

    def rank(self, doc_id: str, query: str) -> dict[str, list[RetrievedChunk]]:
        dense = self._vectors.search(doc_id, self._embedder.embed_query(query), self._top_k_vector)
        lexical = self._bm25.get(doc_id).search(query, self._top_k_bm25)
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in dense], [cid for cid, _ in lexical]], k=self._rrf_k
        )
        similarity, bm25 = dict(dense), dict(lexical)
        chunks = self._get_chunks(list(fused))

        def materialize(ranking: list[tuple[str, float]]) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(chunks[cid], score, similarity.get(cid), bm25.get(cid))
                for cid, score in ranking
                if cid in chunks
            ]

        rankings = {
            "dense": materialize(dense),
            "bm25": materialize(lexical),
            "hybrid": materialize(sorted(fused.items(), key=lambda kv: kv[1], reverse=True)),
        }
        if self._reranker:
            rankings["rerank"] = self._reranker.rerank(
                query, [replace(rc) for rc in rankings["hybrid"]]
            )
        return rankings

    def retrieve(self, doc_id: str, query: str) -> RetrievalResult:
        rankings = self.rank(doc_id, query)
        selected = rankings["rerank" if self._reranker else "hybrid"][: self._top_k_final]
        return context_relevance(selected, query)


def context_relevance(selected: list[RetrievedChunk], query: str) -> RetrievalResult:
    terms = set(tokenize(query))
    coverage = (
        max(
            (len(terms & set(tokenize(rc.chunk.text))) / len(terms) for rc in selected),
            default=0.0,
        )
        if terms
        else 0.0
    )
    return RetrievalResult(
        chunks=selected,
        max_similarity=max(
            (rc.similarity for rc in selected if rc.similarity is not None), default=0.0
        ),
        lexical_coverage=coverage,
    )
