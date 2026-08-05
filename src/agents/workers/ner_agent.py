"""LangGraph node wrapping the rule-based NER pass.

Runs after `memory_load` and before `intent_agent`, so it can see what the
*previous* turn was waiting to hear back (`state["carried_pending_request"]`,
set by `memory_agent.py` — see ADR-008) and, in addition to its normal regex
extraction, recognize a bare follow-up answer ("1234") that on its own has no
keyword for the regexes to anchor on.
"""

from __future__ import annotations

from collections.abc import Callable

from agents.memory import synthesize_bare_answer_entity
from agents.state import GraphState
from nlp.ner_extractor import extract_entities
from schemas.dto import AgentTraceStep, Entity


def build_ner_node() -> Callable[[GraphState], dict[str, object]]:
    def ner_node(state: GraphState) -> dict[str, object]:
        entities = extract_entities(state["user_query"])
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
