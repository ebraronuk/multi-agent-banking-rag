"""Grafiğin geri kalanının etrafında, konuşma başına hafızayı yükler/kaydeder.

`memory_load` önce çalışır (`ner_agent`'tan bile önce) ki niyet/varlık
çıkarımı önceki turn'ün ne beklediğini görebilsin (bkz. ADR-008).
`memory_save` en son çalışır (`guardrail`'den sonra) ki ara bir taslağı değil,
gerçekten gönderilen cevabı kalıcı hale getirsin.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agents.memory import ConversationMemory
from agents.state import GraphState
from app.core.config import Settings
from schemas.dto import AgentTraceStep


def build_memory_load_node(
    memory: ConversationMemory,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def memory_load_node(state: GraphState) -> dict[str, object]:
        context = await memory.load(state["conversation_id"])
        summary = f"loaded {len(context.turns)} prior turn(s)"
        if context.pending_entity_request:
            summary += f", pending {context.pending_entity_request.entity_type} for {context.pending_entity_request.intent}"

        return {
            "history": context.turns,
            "carried_pending_request": context.pending_entity_request,
            "trace": [AgentTraceStep(node="memory_load", summary=summary)],
        }

    return memory_load_node


def build_memory_save_node(
    memory: ConversationMemory, settings: Settings
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def memory_save_node(state: GraphState) -> dict[str, object]:
        await memory.save_turn(
            state["conversation_id"],
            user_message=state["user_query"],
            assistant_message=state.get("final_answer") or "",
            pending_entity_request=state.get("pending_entity_request"),
            history_limit=settings.conversation_history_limit,
        )
        return {"trace": [AgentTraceStep(node="memory_save", summary="turn persisted")]}

    return memory_save_node
