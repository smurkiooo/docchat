from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    index: int
    text: str
    page_start: int
    page_end: int

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    similarity: float | None = None
    bm25_score: float | None = None


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    progress: float
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None
    created_at: str


class SourceOut(BaseModel):
    page_start: int
    page_end: int
    text: str
    score: float
    similarity: float | None = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[SourceOut] = Field(default_factory=list)
    created_at: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)


class HealthOut(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    llm_available: bool
    embedding_model: str
