import json
from collections.abc import AsyncIterator

import httpx

from app.llm.base import ChatMessage, LLMError


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._options = {"temperature": temperature, "num_predict": max_tokens}
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)

    def _payload(self, system: str, messages: list[ChatMessage], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "options": self._options,
        }

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        payload = self._payload(system, messages, stream=True)
        completed = False
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response,
            ):
                if response.status_code != 200:
                    await response.aread()
                    raise LLMError(self._http_error(response))
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if error := data.get("error"):
                        raise LLMError(f"Ollama: {error}")
                    if text := data.get("message", {}).get("content"):
                        yield text
                    if data.get("done"):
                        if data.get("done_reason") == "length":
                            raise LLMError(
                                "Ответ обрезан лимитом токенов. Увеличьте LLM_MAX_TOKENS."
                            )
                        completed = True
                        break
            if not completed:
                raise LLMError("Поток Ollama завершился без подтверждения окончания ответа.")
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Не удалось подключиться к Ollama ({self._base_url}). "
                "Запустите `ollama serve` или смените LLM_PROVIDER."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama не ответила вовремя. Попробуйте ещё раз.") from exc

        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LLMError("Соединение с моделью прервано или получен некорректный ответ.") from exc

    async def complete(self, system: str, messages: list[ChatMessage]) -> str:
        return "".join([part async for part in self.stream(system, messages)])

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
            names = {m["name"] for m in response.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, ValueError):
            return False

    def _http_error(self, response: httpx.Response) -> str:
        if response.status_code == 404:
            return (
                f"Модель «{self.model}» не найдена в Ollama. Выполните `ollama pull {self.model}`."
            )
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        return f"Ollama вернула ошибку {response.status_code}: {detail}"
