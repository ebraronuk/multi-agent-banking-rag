"""Varlık çıkarımı (NER) düğümü.

`memory_load`'dan sonra, `intent_agent`'tan önce çalışıyor — böylece önceki
turn'ün ne beklediğini görebiliyor (`state["carried_pending_request"]`, bkz.
ADR-008) ve normal çıkarımına ek olarak, tek başına hiçbir anahtar kelimeye
anchor'lanamayan çıplak bir takip cevabını ("1234") tanıyabiliyor.

Ayrıca IBAN/kart son 4 hane gibi "kimlik" entity'lerini, bu turda tekrar
verilmemişse geçmiş turlardan geri çağırıyor (`_recall_entities_from_history`)
— aynı konuşmada bir kez verilen bir IBAN sonraki turlarda tekrar sorulmasın.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from collections.abc import Set as AbstractSet

from langchain_core.language_models.chat_models import BaseChatModel

from agents.memory import synthesize_bare_answer_entity
from agents.state import GraphState
from nlp.ner_extractor import extract_entities, extract_entities_with_llm
from schemas.dto import AgentTraceStep, ChatMessage, Entity, EntityType

# Sadece "bu istek kimin hesabıyla ilgili" entity'leri geri çağrılıyor — AMOUNT/
# DATE gibi turn'e özgü değerleri geçmişten taşımak, eski bir tutarı/tarihi
# yeni, ilgisiz bir isteğe sessizce uygulamak riskiyle karşı karşıya kalırdı.
_RECALLABLE_FROM_HISTORY = frozenset({EntityType.IBAN, EntityType.CARD_LAST4})


def _recall_entities_from_history(
    history: list[ChatMessage], missing_types: AbstractSet[EntityType]
) -> list[Entity]:
    recalled: dict[EntityType, Entity] = {}
    for turn in reversed(history):
        if turn.role != "user":
            continue
        remaining = missing_types - recalled.keys()
        if not remaining:
            break
        for entity in extract_entities(turn.content):
            if entity.type in remaining:
                # start/end bu turn'ün metnine değil, geçmiş bir turn'ün
                # metnine ait — kafa karıştırmasın diye taşınmıyor.
                recalled[entity.type] = Entity(
                    type=entity.type,
                    value=entity.value,
                    normalized=entity.normalized,
                    confidence=entity.confidence,
                )
    return list(recalled.values())


def build_ner_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def ner_node(state: GraphState) -> dict[str, object]:
        entities = await extract_entities_with_llm(state["user_query"], llm)
        summary = f"extracted {len(entities)} entity(ies)"

        missing_recallable = _RECALLABLE_FROM_HISTORY - {e.type for e in entities}
        if missing_recallable:
            recalled = _recall_entities_from_history(state.get("history", []), missing_recallable)
            if recalled:
                entities = [*entities, *recalled]
                summary += f", {len(recalled)} recalled from history"

        pending = state.get("carried_pending_request")
        if pending and not any(e.type == pending.entity_type for e in entities):
            synthesized = synthesize_bare_answer_entity(state["user_query"], pending)
            if synthesized:
                _intent, entity_type, value = synthesized
                entities = [
                    *entities,
                    Entity(type=entity_type, value=value, normalized=value, confidence=0.9),
                ]
                summary += " (+1 from pending slot-fill)"

        return {
            "entities": entities,
            "trace": [AgentTraceStep(node="ner_agent", summary=summary)],
        }

    return ner_node
