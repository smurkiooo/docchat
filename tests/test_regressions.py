from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_pages
from app.models import Chunk, Page
from app.retrieval.bm25_store import BM25Index
from tests.test_api import _chat, _upload


@pytest.mark.parametrize("count", [0, 1, 2])
def test_bm25_small_corpus(count):
    chunks = [Chunk(f"d:{i}", "d", i, "Гарантия 36 месяцев", 1, 1) for i in range(count)]
    hits = BM25Index(chunks).search("гарантия", 10)
    assert len(hits) == count
    assert all(score > 0 for _, score in hits)


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_overlap": 800},
        {"top_k_final": 0},
        {"embedding_batch_size": 0},
        {"history_turns": -1},
        {"llm_timeout_s": 0},
        {"min_chunk_chars": 900},
    ],
)
def test_invalid_settings_fail_early(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_cleaner_preserves_body_matching_footer():
    pages = [
        Page(i, f"Заголовок\n\nТекст {i}\n\nЗаголовок\n\nЕщё текст {i}\n\nСтраница {i}")
        for i in range(1, 4)
    ]
    assert all("Заголовок" in p.text for p in clean_pages(pages))


def test_chunk_overlap_accounts_for_paragraph_separator():
    chunks = chunk_pages([Page(1, "a" * 90 + "\n\n" + "b" * 89)], "d", 100, 10, 20)
    assert all(len(c.text) <= 100 for c in chunks)


def test_whitespace_question_rejected(client, sample_pdf):
    doc_id = _upload(client, sample_pdf)["id"]
    assert (
        client.post(f"/api/documents/{doc_id}/chat", json={"question": " \n\t "}).status_code == 422
    )


def test_retrieval_failure_is_sse_error(client, sample_pdf, monkeypatch):
    doc_id = _upload(client, sample_pdf)["id"]
    retriever = client.app.state.container.rag._retriever
    monkeypatch.setattr(retriever, "retrieve", Mock(side_effect=RuntimeError("private detail")))
    events = _chat(client, doc_id, "гарантия")
    assert [kind for kind, _ in events] == ["error"]
    assert "private detail" not in str(events)
    assert not client.app.state.container.rag.active_documents


def test_empty_llm_answer_is_error(client, sample_pdf, fake_llm):
    doc_id = _upload(client, sample_pdf)["id"]
    fake_llm.answer = ""
    events = _chat(client, doc_id, "гарантийный срок")
    assert events[-1][0] == "error"
    assert not any(kind == "done" for kind, _ in events)


def test_active_chat_blocks_mutations(client, sample_pdf):
    doc_id = _upload(client, sample_pdf)["id"]
    active = client.app.state.container.rag.active_documents
    active.add(doc_id)
    try:
        assert client.delete(f"/api/documents/{doc_id}").status_code == 409
        assert client.delete(f"/api/documents/{doc_id}/messages").status_code == 409
        assert (
            client.post(f"/api/documents/{doc_id}/chat", json={"question": "test"}).status_code
            == 409
        )
    finally:
        active.discard(doc_id)


def test_failed_index_cleans_sqlite_and_can_retry(client, sample_pdf, monkeypatch):
    service = client.app.state.container.ingestion
    original = service._vectors.add
    monkeypatch.setattr(service._vectors, "add", Mock(side_effect=RuntimeError("index failure")))
    doc_id = _upload(client, sample_pdf)["id"]
    container = client.app.state.container
    assert container.db.get_document(doc_id).status == "error"
    assert container.db.list_chunks(doc_id) == []
    monkeypatch.setattr(service._vectors, "add", original)
    retry_id = _upload(client, sample_pdf)["id"]
    assert container.db.get_document(retry_id).status == "ready"
    assert container.db.get_document(doc_id) is None


def test_missing_pdf_returns_404(client, sample_pdf):
    doc_id = _upload(client, sample_pdf)["id"]
    path: Path = client.app.state.container.settings.uploads_dir / f"{doc_id}.pdf"
    path.unlink()
    assert client.get(f"/api/documents/{doc_id}/file").status_code == 404


def test_gate_checks_selected_context_only():
    from app.retrieval.bm25_store import BM25Store
    from app.retrieval.hybrid import HybridRetriever

    chunks = [Chunk("d:0", "d", 0, "гарантия", 1, 1), Chunk("d:1", "d", 1, "напряжение", 2, 2)]
    vectors = Mock()
    vectors.search.return_value = [("d:0", 0.99), ("d:1", 0.1)]
    reranker = Mock()
    reranker.rerank.side_effect = lambda q, candidates: list(reversed(candidates))
    retriever = HybridRetriever(
        Mock(),
        vectors,
        BM25Store(lambda _: chunks),
        lambda ids: {c.id: c for c in chunks},
        top_k_final=1,
        reranker=reranker,
    )
    result = retriever.retrieve("d", "гарантия")
    assert result.chunks[0].chunk.id == "d:1"
    assert not result.is_relevant(0.72, 0.5)


def test_vector_writes_are_batched():
    from app.retrieval.vector_store import VectorStore

    store = VectorStore.__new__(VectorStore)
    store._client = Mock()
    store._client.get_max_batch_size.return_value = 2
    collection = store._client.get_or_create_collection.return_value
    chunks = [Chunk(f"d:{i}", "d", i, "text", 1, 1) for i in range(5)]
    store.add("d", chunks, [[1.0] for _ in chunks])
    assert [len(call.kwargs["ids"]) for call in collection.add.call_args_list] == [2, 2, 1]


async def test_question_rewrite_rejects_invented_numbers(client, fake_llm, monkeypatch):
    from app.models import MessageOut

    history = [
        MessageOut(
            id=1,
            role="user",
            content="Как долго длится вводное занятие?",
            sources=[],
            created_at="2026-11-02",
        ),
        MessageOut(
            id=2, role="assistant", content="75 минут.", sources=[], created_at="2026-11-02"
        ),
    ]

    async def invented(*args):
        return "Какова численность курса 2023 года длительностью 75 минут?"

    monkeypatch.setattr(fake_llm, "complete", invented)
    result = await client.app.state.container.rag._condense(history, "А сколько человек?")
    assert "2023" not in result
    assert "75" not in result
    assert "вводное занятие" in result
    assert "сколько человек" in result


async def test_question_rewrite_preserves_user_model_code(client, fake_llm, monkeypatch):
    from app.models import MessageOut

    history = [
        MessageOut(
            id=1, role="user", content="Что означает R21?", sources=[], created_at="2026-11-02"
        )
    ]

    async def resolved(*args):
        return "Как устранить R21?"

    monkeypatch.setattr(fake_llm, "complete", resolved)
    assert (
        await client.app.state.container.rag._condense(history, "Как устранить?")
        == "Как устранить R21?"
    )
