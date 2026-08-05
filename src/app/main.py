from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from agents.graph import build_graph
from app.api.routes import chat, health
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import limiter
from schemas.dto import ErrorCode, ErrorResponse

logger = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Stamp every request/response pair with a correlation id.

    Set on `request.state` (routes that need it for logging, e.g. chat.py's
    conversation binding, read it from there) and echoed back as `X-Request-Id`
    so a caller — or a support ticket quoting this header — can be matched
    directly to a line in the structured logs, without depending on any one
    route implementing its own id generation.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


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
app.add_middleware(RequestIdMiddleware)
app.state.limiter = limiter
app.include_router(health.router)
app.include_router(chat.router)

# /metrics is unauthenticated, matching this demo's no-auth posture everywhere
# else (see README "Sınırlar"). A real deployment would put this behind the
# cluster's internal network (Prometheus scrapes it there) rather than expose
# it on the same public listener as /chat.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    # Every LLM call behind /chat costs real money and has real latency — a
    # naive client (or an abusive one) retrying in a tight loop shouldn't be
    # able to run either up unbounded. See `app/api/routes/chat.py` for the
    # actual per-route limit.
    logger.warning("rate_limit_exceeded", path=request.url.path, detail=str(exc.detail))
    response = JSONResponse(
        status_code=429,
        content=ErrorResponse(
            code=ErrorCode.RATE_LIMITED,
            message="Çok fazla istek gönderildi, lütfen biraz sonra tekrar deneyin.",
        ).model_dump(mode="json"),
    )
    return limiter._inject_headers(response, request.state.view_rate_limit)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # FastAPI's default 422 body doesn't match our documented ErrorResponse
    # contract — a client handling error responses would need a special case
    # just for validation failures. Reshape it into the same envelope as
    # every other error this API returns.
    logger.info("request_validation_failed", path=request.url.path, errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code=ErrorCode.VALIDATION_ERROR,
            message="Gönderilen istek geçersiz.",
            details={"errors": jsonable_encoder(exc.errors())},
        ).model_dump(mode="json"),
    )


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
