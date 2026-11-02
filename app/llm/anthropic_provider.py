from collections.abc import AsyncIterator

import anthropic

from app.llm.base import ChatMessage, LLMError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        effort: str = "low",
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s)

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Неверный ANTHROPIC_API_KEY.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Модель «{self.model}» недоступна для этого ключа.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Превышен лимит запросов к Anthropic API, повторите позже.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API вернул ошибку {exc.status_code}.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Нет соединения с Anthropic API.") from exc

        if final.stop_reason == "max_tokens":
            raise LLMError("Ответ обрезан лимитом токенов. Увеличьте LLM_MAX_TOKENS.")
        if final.stop_reason == "refusal":
            raise LLMError("Модель отказалась отвечать на этот запрос.")

    async def complete(self, system: str, messages: list[ChatMessage]) -> str:
        return "".join([part async for part in self.stream(system, messages)])

    async def is_available(self) -> bool:
        return bool(self._client.api_key)
