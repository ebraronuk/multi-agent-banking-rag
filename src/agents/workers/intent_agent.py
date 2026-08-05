"""Niyet sınıflandırmasını saran LangGraph düğümü.

Grafikte `ner_agent`'tan sonra çalışır ki sınıflandırma zaten çıkarılmış
varlıkları kullanabilsin (ör. bir CARD_LAST4 eşleşmesi CARD_ACTION'ı
destekler) — kural tabanlı/LLM ayrımı ve fallback mantığı için bkz.
`nlp/intent_classifier.py`.

Ayrıca `state["carried_pending_request"]`'in tüketicisi (bkz. ADR-008): önceki
turn belirli bir varlığı bekliyorduysa ve `ner_agent` bu turn tam olarak o
varlığı bulduysa, bekleyen niyet sıfırdan yeniden sınıflandırılmak yerine
aynen kullanılıyor — kural tabanlı sınıflandırıcının çıplak bir "1234" için
hiçbir anahtar kelime sinyali yok, bunu OUT_OF_SCOPE olarak puanlardı.
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
