from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
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
    """Her istek/yanıt çiftine bir korelasyon id'si damgalar.

    `request.state`'e set edilir (loglama için buna ihtiyaç duyan route'lar,
    ör. chat.py'nin konuşma bağlaması, oradan okur) ve `X-Request-Id` olarak
    geri yankılanır ki bir çağıran — ya da bu header'ı alıntılayan bir destek
    talebi — doğrudan yapısal loglardaki bir satırla eşleştirilebilsin, her
    route'un kendi id üretimini yapmasına bağlı kalmadan.
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
    # Grafik başlangıçta bir kere kurulur — her istekte soğuk Chroma/embedding
    # init'i p99 gecikmesini öngörülemez yapardı.
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
    description="Retail-banking destek alanı için referans çoklu-ajan RAG + araç-çağırma asistanı.",
    version="0.1.0",
    lifespan=lifespan,
)
# Gerçek bir dağıtım CORS_ALLOWED_ORIGINS'i gerçek frontend origin'ine set
# etmeli, asla "*" değil — bu API'de auth yok.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)
app.state.limiter = limiter
app.include_router(health.router)
app.include_router(chat.router)

# /metrics kimlik doğrulamasız (bkz. README "Sınırlar") — gerçek bir dağıtım
# bunu public dinleyici yerine cluster'ın iç ağının arkasına koyardı.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    # /chat'in arkasındaki her LLM çağrısı gerçek para/gecikme maliyeti taşıyor.
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
    # FastAPI'nin varsayılan 422 gövdesi ErrorResponse sözleşmemize uymuyor.
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
    # API yanıtında stack trace bilgi sızıntısıdır — sunucu tarafında logla, çağırana jenerik hata dön.
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Beklenmeyen bir hata oluştu.",
        ).model_dump(mode="json"),
    )
