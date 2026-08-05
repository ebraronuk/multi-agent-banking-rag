"""Supervisor: the routing brain of the graph.

Deliberately NOT an LLM call. Routing here is a handful of `IntentLabel`
comparisons plus the iteration guard — an LLM-as-router adds latency, cost, and
a new failure mode (misroute) for a decision that's fully determined by
upstream classification. The workers it dispatches to (`rag_agent`,
`tool_agent`, ...) still use the LLM for the parts that actually need
reasoning. See docs/decisions/ADR-002-supervisor-routing.md.
"""

from __future__ import annotations

from collections.abc import Callable

from agents.state import GraphState
from app.core.config import Settings
from app.core.logging import get_logger
from schemas.dto import AgentTraceStep, IntentLabel

logger = get_logger(__name__)

TOOL_DRIVEN_INTENTS = frozenset(
    {IntentLabel.ACCOUNT_ACTION, IntentLabel.TRANSACTION_ACTION, IntentLabel.CARD_ACTION}
)

NODE_RAG_AGENT = "rag_agent"
NODE_TOOL_AGENT = "tool_agent"
NODE_SMALLTALK = "smalltalk"
NODE_ESCALATE = "escalate"
NODE_GUARDRAIL = "guardrail"


def supervisor_node(state: GraphState) -> dict[str, object]:
    """No-op pass-through node whose only job is to leave a trace entry.

    LangGraph conditional edges can branch directly from any node, but giving
    the branch point its own named node makes the graph diagram (and the
    per-request trace returned to the API caller) legible: "supervisor
    evaluated routing" is a real, inspectable step, not an invisible `if`.
    """
    intent = state.get("intent")
    return {
        "trace": [
            AgentTraceStep(
                node="supervisor",
                summary=f"routing decision for intent={intent}",
                metadata={"iteration_count": state.get("iteration_count", 0)},
            )
        ]
    }


def build_supervisor_router(settings: Settings) -> Callable[[GraphState], str]:
    """Return the conditional-edge function bound to this run's iteration cap."""

    def route(state: GraphState) -> str:
        intent = state.get("intent")
        iteration_count = state.get("iteration_count", 0)

        if intent in TOOL_DRIVEN_INTENTS:
            if iteration_count >= settings.max_agent_iterations:
                logger.warning(
                    "agent_iteration_limit_hit", intent=intent, iteration_count=iteration_count
                )
                return NODE_GUARDRAIL
            if state.get("tool_agent_done"):
                return NODE_GUARDRAIL
            return NODE_TOOL_AGENT

        if intent == IntentLabel.RAG_QUERY:
            return NODE_RAG_AGENT

        if intent == IntentLabel.SMALL_TALK:
            return NODE_SMALLTALK

        # ESCALATE, OUT_OF_SCOPE, or an unclassified/None intent all fail safe
        # to a human-handoff message rather than guessing.
        return NODE_ESCALATE

    return route
