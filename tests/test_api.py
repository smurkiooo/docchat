import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm.prompts import NO_ANSWER_EN, NO_ANSWER_RU
from tests.conftest import FakeLLM


def _upload(client: TestClient, path: Path) -> dict:
    with path.open("rb") as f:
        response = client.post("/api/documents", files={"file": (path.name, f, "application/pdf")})
    assert response.status_code in (200, 202), response.text
    return response.json()


def _chat(client: TestClient, doc_id: str, question: str) -> list[tuple[str, object]]:
    response = client.post(f"/api/documents/{doc_id}/chat", json={"question": question})
    assert response.status_code == 200, response.text
    events = []
    for block in response.text.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line[6:])))
    return events


def test_index_and_health(client: TestClient) -> None:
    assert "DocChat" in client.get("/").text
    health = client.get("/api/health").json()
    assert health["llm_available"] is True
    assert health["embedding_model"] == "test-hash-embedder"


def test_upload_indexes_document(client: TestClient, sample_pdf: Path) -> None:
    created = _upload(client, sample_pdf)
    document = client.get(f"/api/documents/{created['id']}").json()
    assert document["status"] == "ready"
    assert document["pages"] == 3
    assert document["chunks"] >= 1
    assert document["progress"] == 1.0
    assert client.get(f"/api/documents/{created['id']}/file").status_code == 200


def test_duplicate_upload_returns_existing(client: TestClient, sample_pdf: Path) -> None:
    first = _upload(client, sample_pdf)
    second = _upload(client, sample_pdf)
    assert first["id"] == second["id"]
    assert len(client.get("/api/documents").json()) == 1


@pytest.mark.parametrize(
    ("name", "content", "status"),
    [("notes.txt", b"hello", 415), ("fake.pdf", b"hello", 415)],
)
def test_upload_validation(client: TestClient, name: str, content: bytes, status: int) -> None:
    response = client.post("/api/documents", files={"file": (name, content)})
    assert response.status_code == status


def test_scanned_pdf_is_marked_as_error(client: TestClient, tmp_path: Path) -> None:
    from scripts.pdfgen import make_pdf

    created = _upload(client, make_pdf([["x"]], tmp_path / "scan.pdf", footer=None))
    document = client.get(f"/api/documents/{created['id']}").json()
    assert document["status"] == "error"
    assert "скан" in document["error"]


def test_chat_streams_sources_tokens_and_saves_history(
    client: TestClient, sample_pdf: Path, fake_llm: FakeLLM
) -> None:
    doc_id = _upload(client, sample_pdf)["id"]
    events = _chat(client, doc_id, "Какой гарантийный срок?")

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"}
    sources = events[0][1]
    assert sources[0]["page_start"] == 3

    system, messages = fake_llm.calls[-1]
    assert "<context>" in messages[-1]["content"]
    assert "Гарантийный срок составляет 36 месяцев" in messages[-1]["content"]
    assert "Используй только сведения из <context>" in system

    history = client.get(f"/api/documents/{doc_id}/messages").json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == fake_llm.answer
    assert history[1]["sources"][0]["page_start"] == 3


def test_follow_up_question_is_condensed_with_history(
    client: TestClient, sample_pdf: Path, fake_llm: FakeLLM
) -> None:
    doc_id = _upload(client, sample_pdf)["id"]
    _chat(client, doc_id, "Какой гарантийный срок?")
    _chat(client, doc_id, "А какое напряжение сети?")

    assert len(fake_llm.calls) == 3
    assert "История диалога" in fake_llm.calls[1][1][0]["content"]
    answer_messages = fake_llm.calls[2][1]
    assert [m["role"] for m in answer_messages] == ["user"]
    assert fake_llm.answer not in answer_messages[0]["content"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [("Какая столица Франции?", NO_ANSWER_RU), ("Who painted the Mona Lisa?", NO_ANSWER_EN)],
)
def test_irrelevant_question_is_refused_without_llm(
    client: TestClient, sample_pdf: Path, fake_llm: FakeLLM, question: str, expected: str
) -> None:
    doc_id = _upload(client, sample_pdf)["id"]
    events = _chat(client, doc_id, question)
    assert events[0] == ("sources", [])
    assert events[1] == ("token", {"t": expected})
    assert fake_llm.calls == []


def test_clear_history_and_delete(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf)["id"]
    _chat(client, doc_id, "Какой гарантийный срок?")

    assert client.delete(f"/api/documents/{doc_id}/messages").status_code == 204
    assert client.get(f"/api/documents/{doc_id}/messages").json() == []

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_chat_validation(client: TestClient, sample_pdf: Path) -> None:
    doc_id = _upload(client, sample_pdf)["id"]
    assert client.post(f"/api/documents/{doc_id}/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/documents/missing/chat", json={"question": "hi"}).status_code == 404
