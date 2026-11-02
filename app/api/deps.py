from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.container import Container
from app.models import DocumentOut


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_document_or_404(doc_id: str, container: ContainerDep) -> DocumentOut:
    document = container.db.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не найден.")
    return document


DocumentDep = Annotated[DocumentOut, Depends(get_document_or_404)]
