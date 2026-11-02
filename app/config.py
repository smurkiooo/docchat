from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai_compatible", "ollama", "anthropic"] = "openai_compatible"

    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str | None = None
    openai_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(2048, gt=0)
    llm_timeout_s: float = Field(120.0, gt=0)

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_revision: str | None = "d128750597153bb5987e10b1c3493a34e5a4502a"
    model_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = Field(32, gt=0)
    chunk_size: int = Field(800, gt=100)
    chunk_overlap: int = Field(150, ge=0)
    min_chunk_chars: int = Field(200, gt=0)
    top_k_vector: int = Field(20, gt=0)
    top_k_bm25: int = Field(20, gt=0)
    top_k_final: int = Field(6, gt=0)
    rrf_k: int = Field(60, gt=0)
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str | None = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    min_relevance_score: float = Field(0.8, ge=0.0, le=1.0)
    min_lexical_coverage: float = Field(0.5, ge=0.0, le=1.0)

    max_upload_mb: int = Field(50, gt=0)
    history_turns: int = Field(4, ge=0)
    data_dir: Path = Path("data")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_chars > self.chunk_size:
            raise ValueError("min_chunk_chars must not exceed chunk_size")
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "docchat.db"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
