from collections.abc import AsyncIterator
from typing import Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]: ...

    async def complete(self, system: str, messages: list[ChatMessage]) -> str: ...

    async def is_available(self) -> bool: ...
