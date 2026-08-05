"""LangGraph node wrapping intent classification.

Runs after `ner_agent` in the graph so classification can use entities
already extracted (e.g. a `CARD_LAST4` hit corroborates `CARD_ACTION`) — see
`nlp/intent_classifier.py` for the rule-based/LLM split and fallback logic.

Also the consumer of `state["carried_pending_request"]` (see ADR-008): if the
previous turn was waiting on a specific entity and `ner_agent` found exactly
that entity this turn, the pending intent is reused directly instead of
reclassifying a bare "1234" from scratch — the rule-based classifier has no
keyword signal for that and would score it `OUT_OF_SCOPE`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.state import GraphState
from nlp.intent_classifier import classify_intent
from schemas.dto import AgentTraceStep


def build_intent_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def intent_node(state: GraphState) -> dict[str, object]:
        pending = state.get("carried_pending_request")
        entities = state.get("entities", [])

        if pending and any(e.type == pending.entity_type for e in entities):
            return {
                "intent": pending.intent,
                "intent_confidence": 1.0,
                "trace": [
                    AgentTraceStep(
                        node="intent_agent",
                        summary=f"continued pending {pending.intent} (slot-fill answered)",
                    )
                ],
            }

        intent, confidence = await classify_intent(state["user_query"], entities, llm)
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "trace": [
                AgentTraceStep(
                    node="intent_agent",
                    summary=f"classified as {intent} (confidence={confidence:.2f})",
                )
            ],
        }

    return intent_node
