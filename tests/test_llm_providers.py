import json

import httpx
import pytest

from app.llm.base import LLMError
from app.llm.openai_compat_provider import OpenAICompatibleProvider, ThinkTagFilter


@pytest.mark.parametrize(
    "deltas",
    [
        ["<think>hmm</think>Ответ"],
        ["<th", "ink>long ", "reasoning</thi", "nk>Отв", "ет"],
        ["Ответ"],
    ],
)
def test_think_filter_removes_reasoning(deltas: list[str]) -> None:
    f = ThinkTagFilter()
    assert "".join(f.feed(d) for d in deltas) + f.flush() == "Ответ"


def test_think_filter_keeps_text_that_only_looks_like_a_tag() -> None:
    f = ThinkTagFilter()
    assert f.feed("a <b> c <") + f.flush() == "a <b> c <"


def _sse(*deltas: str) -> str:
    events = ["data: " + json.dumps({"choices": [{"delta": {"content": d}}]}) for d in deltas]
    return "\n\n".join([*events, "data: [DONE]"]) + "\n\n"


@pytest.fixture
def mock_server(monkeypatch: pytest.MonkeyPatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "text-embedding-nomic"}, {"id": "gemma-3-4b-it"}]}
            )
        return httpx.Response(200, text=_sse("Гарантия ", "36 месяцев [стр. 9]."))

    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return requests


async def test_streams_and_autodetects_loaded_model(mock_server) -> None:
    llm = OpenAICompatibleProvider("http://lmstudio.test/v1")
    answer = await llm.complete("system", [{"role": "user", "content": "Гарантия?"}])

    assert answer == "Гарантия 36 месяцев [стр. 9]."
    body = json.loads(mock_server[-1].content)
    assert body["model"] == "gemma-3-4b-it"
    assert body["messages"][0] == {"role": "system", "content": "system"}
    assert body["stream"] is True
    assert await llm.is_available()


async def test_connection_error_has_helpful_message() -> None:
    llm = OpenAICompatibleProvider("http://127.0.0.1:9/v1", model="m", timeout_s=1)
    with pytest.raises(LLMError, match="LM Studio"):
        await llm.complete("s", [{"role": "user", "content": "q"}])
    assert not await llm.is_available()


@pytest.mark.parametrize(
    "body",
    [
        'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
        'data: {"choices":[{"finish_reason":"length","delta":{}}]}\n\n',
        'data: {"error":{"message":"upstream failure"}}\n\n',
        "data: invalid-json\n\n",
    ],
)
async def test_openai_stream_errors_are_not_success(monkeypatch, body):
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    provider = OpenAICompatibleProvider("http://model.test/v1", model="test")
    with pytest.raises(LLMError):
        await provider.complete("system", [{"role": "user", "content": "question"}])


@pytest.mark.parametrize(
    "done,reason,success", [(True, "stop", True), (True, "length", False), (False, None, False)]
)
async def test_ollama_requires_successful_terminal_event(monkeypatch, done, reason, success):
    from app.llm.ollama_provider import OllamaProvider

    original = httpx.AsyncClient.__init__
    body = (
        json.dumps({"message": {"content": "answer"}, "done": done, "done_reason": reason}) + "\n"
    )

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    provider = OllamaProvider("http://model.test", model="test")
    if success:
        assert await provider.complete("system", [{"role": "user", "content": "q"}]) == "answer"
    else:
        with pytest.raises(LLMError):
            await provider.complete("system", [{"role": "user", "content": "q"}])
