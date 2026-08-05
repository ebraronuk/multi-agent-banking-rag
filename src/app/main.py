from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agents.graph import build_graph
from app.api.routes import chat, health
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from schemas.dto import ErrorCode, ErrorResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    # Building the graph at startup (not per-request) means the vector store
    # connection, embeddings, and MCP tool client are all set up once — a
    # cold Chroma/embedding init on every request would make p99 latency
    # unpredictable for no benefit, since none of this is per-user state.
    app.state.settings = settings
    app.state.graph = build_graph(settings)
    logger.info(
        "app_startup",
        app_env=settings.app_env,
        llm_provider=settings.resolved_llm_provider(),
        embedding_provider=settings.embedding_provider,
    )
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="multi-agent-banking-rag",
    description="Reference multi-agent RAG + tool-calling assistant for a retail-banking support domain.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(chat.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # A stack trace in an API response is an information leak, not a helpful
    # error message — log the real exception server-side, return a generic,
    # typed error to the caller.
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Beklenmeyen bir hata oluştu.",
        ).model_dump(mode="json"),
    )
