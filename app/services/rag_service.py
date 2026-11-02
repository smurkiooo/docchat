import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.db import Database
from app.llm import prompts
from app.llm.base import ChatMessage, LLMError, LLMProvider
from app.models import MessageOut, RetrievedChunk, SourceOut
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    sources: list[SourceOut]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class DoneEvent:
    message_id: int


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    detail: str


ChatEvent = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent


class RagService:
    def __init__(
        self, settings: Settings, db: Database, retriever: HybridRetriever, llm: LLMProvider
    ) -> None:
        self.active_documents: set[str] = set()
        self._settings = settings
        self._db = db
        self._retriever = retriever
        self._llm = llm

    async def answer(self, doc_id: str, question: str) -> AsyncIterator[ChatEvent]:
        try:
            async for event in self._answer(doc_id, question):
                yield event
        except Exception:
            logger.exception("Chat failed for %s", doc_id)
            yield ErrorEvent("Не удалось подготовить ответ. Попробуйте ещё раз.")

    async def _answer(self, doc_id: str, question: str) -> AsyncIterator[ChatEvent]:
        question = question.strip()
        history = self._db.list_messages(doc_id, limit=self._settings.history_turns * 2)
        self._db.add_message(doc_id, "user", question)

        started = time.perf_counter()
        search_query = await self._condense(history, question)
        result = await run_in_threadpool(self._retriever.retrieve, doc_id, search_query)
        logger.info(
            "Retrieval %.2fs | query=%r | max_sim=%.3f | coverage=%.2f | pages=%s",
            time.perf_counter() - started,
            search_query,
            result.max_similarity,
            result.lexical_coverage,
            [rc.chunk.pages_label for rc in result.chunks],
        )

        s = self._settings
        if not result.chunks or not result.is_relevant(
            s.min_relevance_score, s.min_lexical_coverage
        ):
            text = prompts.no_answer_text(question)
            yield SourcesEvent([])
            yield TokenEvent(text)
            yield DoneEvent(self._db.add_message(doc_id, "assistant", text))
            return

        sources = [_source(rc) for rc in result.chunks]
        yield SourcesEvent(sources)

        resolved_question = question
        if search_query != question:
            resolved_question += f"\nУточнённый вопрос для поиска: {search_query}"
        messages = [
            ChatMessage(
                role="user", content=prompts.build_user_message(resolved_question, result.chunks)
            ),
        ]
        parts: list[str] = []
        finished = False
        try:
            async for text in self._llm.stream(prompts.SYSTEM_PROMPT, messages):
                if not parts:
                    logger.info("First token after %.2fs", time.perf_counter() - started)
                parts.append(text)
                yield TokenEvent(text)
            if not "".join(parts).strip():
                raise LLMError("Модель вернула пустой ответ. Попробуйте ещё раз.")
            finished = True
        except LLMError as exc:
            yield ErrorEvent(str(exc))
        finally:
            answer = "".join(parts).strip()
            message_id = self._db.add_message(doc_id, "assistant", answer, sources) if answer else 0
        if finished:
            yield DoneEvent(message_id)

    async def _condense(self, history: list[MessageOut], question: str) -> str:
        if not history:
            return question
        turns = [(m.role, m.content) for m in history]
        previous = next((m.content for m in reversed(history) if m.role == "user"), "")
        fallback = (
            f"{question}\nКонтекст предыдущего вопроса: {previous[:400]}" if previous else question
        )
        try:
            rewritten = await self._llm.complete(
                prompts.CONDENSE_SYSTEM_PROMPT,
                [ChatMessage(role="user", content=prompts.build_condense_message(turns, question))],
            )
        except LLMError:
            logger.warning("Question condensing failed; searching with user question context")
            return fallback
        rewritten = rewritten.strip().strip('"«»')
        user_text = " ".join([question, *(m.content for m in history if m.role == "user")])
        known_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", user_text))
        rewritten_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", rewritten))
        if not rewritten_numbers <= known_numbers:
            logger.warning("Question rewrite added numbers absent from user turns; using fallback")
            return fallback
        return rewritten if 0 < len(rewritten) <= 3 * len(question) + 200 else fallback


def _source(rc: RetrievedChunk) -> SourceOut:
    return SourceOut(
        page_start=rc.chunk.page_start,
        page_end=rc.chunk.page_end,
        text=rc.chunk.text,
        score=round(rc.score, 4),
        similarity=round(rc.similarity, 4) if rc.similarity is not None else None,
    )
