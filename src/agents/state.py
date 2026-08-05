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

from schemas.dto import AgentTraceStep, Citation, Entity, IntentLabel, ToolCallRecord


class GraphState(TypedDict, total=False):
    conversation_id: str
    messages: Annotated[list[BaseMessage], add_messages]

    user_query: str
    intent: IntentLabel | None
    intent_confidence: float | None
    entities: list[Entity]

    retrieved_docs: list[Citation]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    draft_answer: str | None
    final_answer: str | None
    guardrail_flags: list[str]

    trace: Annotated[list[AgentTraceStep], operator.add]
    iteration_count: int
    next_node: str | None
    tool_agent_done: bool


def new_state(conversation_id: str, user_query: str) -> GraphState:
    """Seed a fresh GraphState for the first turn of a conversation."""
    return GraphState(
        conversation_id=conversation_id,
        messages=[],
        user_query=user_query,
        intent=None,
        intent_confidence=None,
        entities=[],
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
