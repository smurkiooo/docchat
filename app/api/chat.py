import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import ContainerDep, DocumentDep
from app.models import ChatRequest, DocumentStatus
from app.services.rag_service import ChatEvent, DoneEvent, ErrorEvent, SourcesEvent, TokenEvent

router = APIRouter(prefix="/api/documents", tags=["chat"])


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _encode(event: ChatEvent) -> str:
    match event:
        case SourcesEvent(sources):
            return _sse("sources", [s.model_dump() for s in sources])
        case TokenEvent(text):
            return _sse("token", {"t": text})
        case DoneEvent(message_id):
            return _sse("done", {"message_id": message_id})
        case ErrorEvent(detail):
            return _sse("error", {"detail": detail})
    raise TypeError(f"Unknown event {event!r}")


@router.post(
    "/{doc_id}/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat(document: DocumentDep, body: ChatRequest, container: ContainerDep):
    if document.status != DocumentStatus.READY:
        raise HTTPException(409, "Документ ещё не готов к вопросам.")

    if document.id in container.rag.active_documents:
        raise HTTPException(409, "Дождитесь завершения текущего ответа.")
    container.rag.active_documents.add(document.id)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in container.rag.answer(document.id, body.question):
                yield _encode(event)
        finally:
            container.rag.active_documents.discard(document.id)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
