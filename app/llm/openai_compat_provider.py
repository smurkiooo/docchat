import json
from collections.abc import AsyncIterator

import httpx

from app.llm.base import ChatMessage, LLMError


class ThinkTagFilter:
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while (pos := self._buffer.find(self._tag)) != -1:
            if not self._inside:
                out.append(self._buffer[:pos])
            self._buffer = self._buffer[pos + len(self._tag) :]
            self._inside = not self._inside

        keep = next(
            (n for n in range(len(self._tag) - 1, 0, -1) if self._buffer.endswith(self._tag[:n])),
            0,
        )
        ready = self._buffer[: len(self._buffer) - keep]
        self._buffer = self._buffer[len(self._buffer) - keep :]
        if not self._inside:
            out.append(ready)
        return "".join(out)

    @property
    def _tag(self) -> str:
        return self.CLOSE if self._inside else self.OPEN

    def flush(self) -> str:
        rest, self._buffer = ("" if self._inside else self._buffer), ""
        return rest


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str = "",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._configured_model = model
        self.model = model or "auto"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        if self._configured_model:
            return self._configured_model
        models = await self._list_models(client)
        if not models:
            raise LLMError(
                f"На сервере {self._base_url} не загружена ни одна модель. "
                "Загрузите модель в LM Studio (вкладка Developer → Load model)."
            )
        self.model = models[0]
        return self.model

    async def _list_models(self, client: httpx.AsyncClient) -> list[str]:
        response = await client.get(f"{self._base_url}/models", headers=self._headers)
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", []) if "embed" not in m["id"]]

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        think = ThinkTagFilter()
        completed = False
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                payload = {
                    "model": await self._resolve_model(client),
                    "messages": [{"role": "system", "content": system}, *messages],
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "stream": True,
                }
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=self._headers,
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        raise LLMError(self._http_error(response))
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            completed = True
                            break
                        if not data:
                            continue
                        payload = json.loads(data)
                        if payload.get("error"):
                            raise LLMError("Сервер модели сообщил об ошибке генерации.")
                        choices = payload.get("choices") or [{}]
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason == "length":
                            raise LLMError(
                                "Ответ обрезан лимитом токенов. Увеличьте LLM_MAX_TOKENS."
                            )
                        if finish_reason == "content_filter":
                            raise LLMError("Сервер модели прервал ответ фильтром содержимого.")
                        if finish_reason == "stop":
                            completed = True
                        delta = choices[0].get("delta", {}).get("content")
                        if delta and (text := think.feed(delta)):
                            yield text
            if not completed:
                raise LLMError("Поток модели завершился без подтверждения окончания ответа.")
            if tail := think.flush():
                yield tail
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Не удалось подключиться к {self._base_url}. Запустите сервер в LM Studio "
                "(Developer → Start Server) или проверьте OPENAI_BASE_URL."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("Модель не ответила вовремя. Попробуйте ещё раз.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(self._http_error(exc.response)) from exc

        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LLMError("Соединение с моделью прервано или получен некорректный ответ.") from exc

    async def complete(self, system: str, messages: list[ChatMessage]) -> str:
        return "".join([part async for part in self.stream(system, messages)])

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                models = await self._list_models(client)
        except (httpx.HTTPError, ValueError, KeyError):
            return False
        if not self._configured_model and models:
            self.model = models[0]
        return bool(models) and (not self._configured_model or self._configured_model in models)

    def _http_error(self, response: httpx.Response) -> str:
        if response.status_code == 401:
            return "Сервер отклонил ключ: проверьте OPENAI_API_KEY."
        if response.status_code == 429:
            return "Превышен лимит запросов бесплатного тарифа, повторите позже."
        try:
            error = response.json().get("error", response.text)
            detail = error.get("message", error) if isinstance(error, dict) else error
        except ValueError:
            detail = response.text
        return f"Сервер модели вернул ошибку {response.status_code}: {detail}"
