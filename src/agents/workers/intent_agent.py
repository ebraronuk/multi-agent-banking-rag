"""Niyet sınıflandırmasını saran LangGraph düğümü.

Grafikte `ner_agent`'tan sonra çalışır ki sınıflandırma çıkarılmış varlıkları
kullanabilsin (bkz. `nlp/intent_classifier.py`). Ayrıca
`state["carried_pending_request"]`'in tüketicisi (ADR-008): bekleyen bir slot
bu turn dolduysa, niyet sıfırdan yeniden sınıflandırılmak yerine aynen kullanılır.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.state import GraphState
from agents.supervisor import TOOL_DRIVEN_INTENTS
from nlp.intent_classifier import classify_intent
from schemas.dto import AgentTraceStep, IntentLabel

# Zincirlenebilir niyetler (ADR-012). ESCALATE/OUT_OF_SCOPE dahil değil —
# bir insana aktarım isteği konuşmanın o an bittiği anlamına geliyor.
_CHAINABLE_INTENTS = frozenset(
    {*TOOL_DRIVEN_INTENTS, IntentLabel.RAG_QUERY, IntentLabel.SMALL_TALK}
)
_MAX_EXTRA_INTENTS = 2


def _clean_extra_intents(primary: IntentLabel, raw: list[IntentLabel]) -> list[IntentLabel]:
    cleaned: list[IntentLabel] = []
    for candidate in raw:
        if candidate == primary or candidate in cleaned or candidate not in _CHAINABLE_INTENTS:
            continue
        cleaned.append(candidate)
        if len(cleaned) >= _MAX_EXTRA_INTENTS:
            break
    return cleaned


def build_intent_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def intent_node(state: GraphState) -> dict[str, object]:
        pending = state.get("carried_pending_request")
        entities = state.get("entities", [])

        if pending and any(e.type == pending.entity_type for e in entities):
            return {
                "intent": pending.intent,
                "intent_confidence": 1.0,
                "extra_intents": [],
                "trace": [
                    AgentTraceStep(
                        node="intent_agent",
                        summary=f"continued pending {pending.intent} (slot-fill answered)",
                    )
                ],
            }

        intent, confidence, extra_intents_raw = await classify_intent(
            state["user_query"], entities, llm
        )
        extra_intents = _clean_extra_intents(intent, extra_intents_raw)
        summary = f"classified as {intent} (confidence={confidence:.2f})"
        if extra_intents:
            summary += f", extra: {[str(i) for i in extra_intents]}"

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "extra_intents": extra_intents,
            "trace": [AgentTraceStep(node="intent_agent", summary=summary)],
        }

    return intent_node
