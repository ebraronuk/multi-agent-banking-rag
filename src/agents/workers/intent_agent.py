"""LangGraph node wrapping intent classification.

Runs after `ner_agent` in the graph so classification can use entities
already extracted (e.g. a `CARD_LAST4` hit corroborates `CARD_ACTION`) — see
`nlp/intent_classifier.py` for the rule-based/LLM split and fallback logic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.state import GraphState
from nlp.intent_classifier import classify_intent
from schemas.dto import AgentTraceStep


def build_intent_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def intent_node(state: GraphState) -> dict[str, object]:
        intent, confidence = await classify_intent(
            state["user_query"], state.get("entities", []), llm
        )
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
