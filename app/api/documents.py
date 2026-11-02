import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import ContainerDep, DocumentDep
from app.models import DocumentOut, DocumentStatus, MessageOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

PDF_MAGIC = b"%PDF-"
READ_CHUNK = 1024 * 1024


def _pdf_path(container: ContainerDep, doc_id: str) -> Path:
    return container.settings.uploads_dir / f"{doc_id}.pdf"


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile, background: BackgroundTasks, container: ContainerDep, response: Response
) -> DocumentOut:
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Поддерживаются только PDF-файлы.")

    max_bytes = container.settings.max_upload_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(READ_CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(413, f"Файл больше {container.settings.max_upload_mb} МБ.")
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(415, "Файл не является PDF.")

    db = container.db
    sha256 = hashlib.sha256(data).hexdigest()
    if existing := db.find_by_hash(sha256):
        if existing.status != DocumentStatus.ERROR:
            response.status_code = status.HTTP_200_OK
            return existing
        container.ingestion.delete(existing.id, _pdf_path(container, existing.id))

    doc_id = uuid.uuid4().hex
    path = _pdf_path(container, doc_id)
    path.write_bytes(data)
    try:
        db.create_document(doc_id, filename, sha256)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    background.add_task(container.ingestion.ingest, doc_id, path)
    document = db.get_document(doc_id)
    assert document is not None
    return document


@router.get("")
def list_documents(container: ContainerDep) -> list[DocumentOut]:
    return container.db.list_documents()


@router.get("/{doc_id}")
def get_document(document: DocumentDep) -> DocumentOut:
    return document


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document: DocumentDep, container: ContainerDep) -> None:
    if document.id in container.rag.active_documents:
        raise HTTPException(409, "Дождитесь завершения текущего ответа.")
    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(409, "Документ ещё индексируется, удалите его после завершения.")
    container.ingestion.delete(document.id, _pdf_path(container, document.id))


@router.get("/{doc_id}/file")
def download_document(document: DocumentDep, container: ContainerDep) -> FileResponse:
    path = _pdf_path(container, document.id)
    if not path.is_file():
        raise HTTPException(404, "Исходный PDF не найден. Загрузите документ снова.")
    return FileResponse(
        path,
        media_type="application/pdf",
        content_disposition_type="inline",
        filename=document.filename,
    )


@router.get("/{doc_id}/messages")
def list_messages(document: DocumentDep, container: ContainerDep) -> list[MessageOut]:
    return container.db.list_messages(document.id)


@router.delete("/{doc_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
async def clear_messages(document: DocumentDep, container: ContainerDep) -> None:
    if document.id in container.rag.active_documents:
        raise HTTPException(409, "Дождитесь завершения текущего ответа.")
    container.db.clear_messages(document.id)
