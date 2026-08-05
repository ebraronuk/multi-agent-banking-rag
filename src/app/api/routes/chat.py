"""Ajan grafiğine giden tek HTTP giriş noktası. Bilinçli olarak ince: parse et
→ derlenmiş grafiği çağır → yanıtı şekillendir. Tüm karar verme grafiğin
düğümlerinde yaşıyor, burada asla değil."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from agents.state import new_state
from app.core.config import get_settings
from app.core.logging import bind_request_context, get_logger
from app.core.rate_limit import limiter
from schemas.dto import ChatRequest, ChatResponse, ErrorResponse, IntentLabel

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Bankacılık asistanına bir mesaj gönder",
    responses={
        422: {"model": ErrorResponse, "description": "İstek doğrulamadan geçemedi"},
        429: {"model": ErrorResponse, "description": "Çok fazla istek"},
        500: {"model": ErrorResponse, "description": "Beklenmeyen sunucu hatası"},
    },
)
@limiter.limit(get_settings().chat_rate_limit)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Bir konuşma turn'ünü tüm ajan grafiği üzerinden çalıştırır.

    NER → niyet sınıflandırma → supervisor → eşleşen worker (RAG / araç
    çağırma / sohbet / aktarım) → guardrail sırasıyla ilerler, son cevabı
    alıntıları, araç çağrıları, çıkarılan varlıkları ve düğüm-bazlı trace'iyle
    (bkz. `AgentTraceStep`) birlikte döner.
    """
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    request_id: str = request.state.request_id

    with bind_request_context(request_id, conversation_id):
        logger.info("chat_request_received", message_length=len(payload.message))
        graph = request.app.state.graph
        final_state = await graph.ainvoke(new_state(conversation_id, payload.message))
        logger.info(
            "chat_request_completed",
            intent=final_state.get("intent"),
            iterations=final_state.get("iteration_count", 0),
            guardrail_flags=final_state.get("guardrail_flags", []),
        )

    return ChatResponse(
        conversation_id=conversation_id,
        answer=final_state.get("final_answer") or "",
        intent=final_state.get("intent") or IntentLabel.OUT_OF_SCOPE,
        entities=final_state.get("entities", []),
        citations=final_state.get("retrieved_docs", []),
        tool_calls=final_state.get("tool_calls", []),
        trace=final_state.get("trace", []),
        guardrail_flags=final_state.get("guardrail_flags", []),
        iterations=final_state.get("iteration_count", 0),
    )
