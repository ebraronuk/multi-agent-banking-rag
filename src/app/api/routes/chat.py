"""The one HTTP entry point into the agent graph. Intentionally thin: parse →
invoke the compiled graph → shape the response. All decision-making lives in
the graph's nodes, never here."""

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
    summary="Send a message to the banking assistant",
    responses={
        422: {"model": ErrorResponse, "description": "Request failed validation"},
        429: {"model": ErrorResponse, "description": "Too many requests"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
@limiter.limit(get_settings().chat_rate_limit)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run one conversational turn through the full agent graph.

    Routes through NER → intent classification → the supervisor → the
    matching worker (RAG / tool-calling / small talk / escalate) → the
    guardrail, and returns the final answer alongside its citations, tool
    calls, extracted entities, and a per-node trace (see `AgentTraceStep`).
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
