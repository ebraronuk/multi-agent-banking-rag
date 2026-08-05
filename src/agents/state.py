"""Shared LangGraph state schema.

Every worker node (`agents/workers/*.py`) receives this full state and returns
a partial update `dict`; LangGraph merges partial updates into the running
state using the reducers declared below (`add_messages` for the chat
transcript, `operator.add` for append-only trace/tool-call logs). Everything
else is last-write-wins, which is intentional: `intent`/`entities`/
`draft_answer` are each owned by exactly one node.
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

    # Loaded by memory_agent (agents/workers/memory_agent.py) before ner_agent
    # runs; read-only input for this turn. `carried_pending_request` is what
    # the *previous* turn was waiting to hear back (see ADR-008) —
    # `pending_entity_request` below is this turn's own outgoing value and is
    # a completely separate field so a stale request can't linger past the
    # turn that actually resolves (or abandons) it.
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
    """Seed a fresh GraphState for one turn of a conversation.

    "Fresh" per *turn*, not per conversation — `memory_agent` is what actually
    carries continuity forward by loading `history`/`carried_pending_request`
    from storage immediately after this runs.
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
