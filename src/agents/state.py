"""Paylaşılan LangGraph state şeması.

Her worker düğümü bu state'in tamamını alır, kısmi bir güncelleme dict'i
döner; LangGraph bunları aşağıdaki reducer'larla birleştirir (`add_messages`
sohbet transcript'i için, `operator.add` append-only trace/tool-call logları
için). Geri kalanı last-write-wins — `intent`/`entities`/`draft_answer`'ın
her biri tam olarak bir düğüme ait olduğu için bu bilinçli bir tercih.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from schemas.dto import (
    AgentTraceStep,
    ChatMessage,
    Citation,
    Entity,
    GuardrailFlag,
    IntentLabel,
    PendingEntityRequest,
    ToolCallRecord,
)


class GraphState(TypedDict, total=False):
    conversation_id: str
    messages: Annotated[list[BaseMessage], add_messages]

    user_query: str
    intent: IntentLabel | None
    intent_confidence: float | None
    entities: list[Entity]

    # memory_agent tarafından ner_agent'tan önce yüklenir, bu turn için salt
    # okunur girdi. carried_pending_request önceki turn'ün beklediği şey
    # (ADR-008); pending_entity_request bu turn'ün kendi çıkışı — bilinçli
    # olarak ayrı iki alan, biri diğerine karışırsa eski bir istek turn'ü
    # geçtikten sonra da sessizce askıda kalabilir.
    history: list[ChatMessage]
    carried_pending_request: PendingEntityRequest | None
    pending_entity_request: PendingEntityRequest | None

    retrieved_docs: list[Citation]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    draft_answer: str | None
    final_answer: str | None
    guardrail_flags: list[GuardrailFlag]

    trace: Annotated[list[AgentTraceStep], operator.add]
    iteration_count: int
    next_node: str | None
    tool_agent_done: bool


def new_state(conversation_id: str, user_query: str) -> GraphState:
    """Bir konuşmanın tek bir turn'ü için taze bir GraphState kurar.

    "Taze" olan turn, konuşmanın kendisi değil — sürekliliği asıl taşıyan
    memory_agent, bundan hemen sonra history/carried_pending_request'i
    depodan yüklüyor.
    """
    return GraphState(
        conversation_id=conversation_id,
        messages=[],
        user_query=user_query,
        intent=None,
        intent_confidence=None,
        entities=[],
        history=[],
        carried_pending_request=None,
        pending_entity_request=None,
        retrieved_docs=[],
        tool_calls=[],
        draft_answer=None,
        final_answer=None,
        guardrail_flags=[],
        trace=[],
        iteration_count=0,
        next_node=None,
        tool_agent_done=False,
    )
