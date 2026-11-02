import hashlib
import math
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.base import ChatMessage
from app.main import create_app
from app.retrieval.bm25_store import tokenize
from scripts.pdfgen import make_pdf

DIM = 256


class HashEmbedder:
    model_name = "test-hash-embedder"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for token in tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_passages(
        self, texts: Sequence[str], on_progress: Callable[[float], None] | None = None
    ) -> list[list[float]]:
        vectors = [self._embed(t) for t in texts]
        if on_progress:
            on_progress(1.0)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, answer: str = "Ответ по документу [стр. 2].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append((system, messages))
        for word in self.answer.split(" "):
            yield word + " "

    async def complete(self, system: str, messages: list[ChatMessage]) -> str:
        self.calls.append((system, messages))
        return messages[-1]["content"].rsplit("Последний вопрос: ", 1)[-1]

    async def is_available(self) -> bool:
        return True


SAMPLE_PAGES = [
    [
        "# Руководство по эксплуатации",
        "Это руководство описывает установку и настройку устройства. Прочитайте его полностью.",
    ],
    [
        "# Технические характеристики",
        "Термостат питается от сети переменного тока напряжением 230 вольт. "
        "Максимальная нагрузка на реле составляет 16 ампер.",
    ],
    [
        "# Гарантия",
        "Гарантийный срок составляет 36 месяцев с даты покупки. "
        "Гарантия не распространяется на механические повреждения корпуса.",
    ],
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        min_relevance_score=0.3,
        chunk_size=400,
        chunk_overlap=80,
    )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return make_pdf(SAMPLE_PAGES, tmp_path / "sample.pdf", header="ООО «Тест» — руководство")


@pytest.fixture
def client(settings: Settings, embedder: HashEmbedder, fake_llm: FakeLLM) -> Iterator[TestClient]:
    with TestClient(create_app(settings, embedder=embedder, llm=fake_llm)) as test_client:
        yield test_client
