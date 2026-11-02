import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, documents
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.llm.base import LLMProvider
from app.models import HealthOut
from app.retrieval.embedder import Embedder

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logger = logging.getLogger("app")


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = build_container(settings, embedder=embedder, llm=llm)
        if interrupted := container.db.fail_interrupted():
            logger.warning("Marked %d interrupted ingestion(s) as failed", interrupted)
        app.state.container = container

        if warmup := getattr(container.embedder, "warmup", None):

            def safe_warmup() -> None:
                try:
                    warmup()
                except Exception:
                    logger.exception("Embedding model warmup failed; ingestion will retry")

            asyncio.get_running_loop().run_in_executor(None, safe_warmup)
        logger.info("LLM: %s / %s", container.llm.name, container.llm.model)
        yield

    app = FastAPI(
        title="DocChat",
        description="Local RAG chat over PDF documents.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/api/health", tags=["system"])
    async def health() -> HealthOut:
        container: Container = app.state.container
        return HealthOut(
            status="ok",
            llm_provider=container.llm.name,
            llm_model=container.llm.model,
            llm_available=await container.llm.is_available(),
            embedding_model=container.embedder.model_name,
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
