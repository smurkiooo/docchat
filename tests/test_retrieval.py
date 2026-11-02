from pathlib import Path

from app.models import Chunk
from app.retrieval.bm25_store import BM25Index, BM25Store, tokenize
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore
from tests.conftest import HashEmbedder

TEXTS = [
    (1, "Термостат питается от сети 230 вольт, максимальная нагрузка 16 ампер."),
    (2, "Гарантийный срок составляет 36 месяцев с даты покупки устройства."),
    (3, "Для сброса настроек удерживайте кнопку питания десять секунд."),
    (4, "Мобильное приложение позволяет задавать расписание обогрева по дням недели."),
]
CHUNKS = [Chunk(f"d:{i}", "d", i, text, page, page) for i, (page, text) in enumerate(TEXTS)]


def test_tokenize_stems_russian_and_drops_stopwords() -> None:
    question = tokenize("Какой гарантийный срок у термостата?")
    assert question == tokenize("гарантийного срока термостат")
    assert "и" not in tokenize("и в на")


def test_tokenize_splits_model_codes() -> None:
    assert {"ip21", "ip", "21"} <= set(tokenize("Степень защиты IP21"))


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused["a"] == fused["b"] > fused["c"] == fused["d"]


def test_bm25_finds_exact_terms_and_reports_coverage() -> None:
    index = BM25Index(CHUNKS)
    hits = index.search("сколько месяцев гарантия", k=3)
    assert hits[0][0] == "d:1"
    assert index.coverage("гарантийный срок") == 1.0
    assert index.coverage("столица Франции") == 0.0


def _retriever(tmp_path: Path) -> HybridRetriever:
    embedder = HashEmbedder()
    store = VectorStore(tmp_path / "chroma")
    store.add("d", CHUNKS, embedder.embed_passages([c.text for c in CHUNKS]))
    bm25 = BM25Store(lambda _doc_id: CHUNKS)
    by_id = {c.id: c for c in CHUNKS}
    return HybridRetriever(
        embedder, store, bm25, lambda ids: {i: by_id[i] for i in ids if i in by_id}, top_k_final=2
    )


def test_hybrid_retrieval_returns_relevant_page(tmp_path: Path) -> None:
    result = _retriever(tmp_path).retrieve("d", "Как сбросить настройки кнопкой питания?")
    assert result.chunks[0].chunk.page_start == 3
    assert len(result.chunks) == 2
    assert result.is_relevant(min_similarity=0.3)


def test_relevance_gate_rejects_unrelated_question(tmp_path: Path) -> None:
    result = _retriever(tmp_path).retrieve("d", "Какая столица Франции?")
    assert not result.is_relevant(min_similarity=0.3)
