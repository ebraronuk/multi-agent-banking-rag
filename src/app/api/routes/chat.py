"""The one HTTP entry point into the agent graph. Intentionally thin: parse →
invoke the compiled graph → shape the response. All decision-making lives in
the graph's nodes, never here."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from agents.state import new_state
from app.core.logging import bind_request_context, get_logger
from schemas.dto import ChatRequest, ChatResponse, IntentLabel

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

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
