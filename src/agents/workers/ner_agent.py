"""Varlık çıkarımı (NER) düğümü.

`memory_load`'dan sonra, `intent_agent`'tan önce çalışıyor — böylece önceki
turn'ün ne beklediğini görebiliyor (`state["carried_pending_request"]`, bkz.
ADR-008) ve normal çıkarımına ek olarak, tek başına hiçbir anahtar kelimeye
anchor'lanamayan çıplak bir takip cevabını ("1234") tanıyabiliyor.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.memory import synthesize_bare_answer_entity
from agents.state import GraphState
from nlp.ner_extractor import extract_entities_with_llm
from schemas.dto import AgentTraceStep, Entity


def build_ner_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def ner_node(state: GraphState) -> dict[str, object]:
        entities = await extract_entities_with_llm(state["user_query"], llm)
        summary = f"extracted {len(entities)} entity(ies)"

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
