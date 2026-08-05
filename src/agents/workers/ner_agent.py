"""LangGraph node wrapping the rule-based NER pass.

Sync and dependency-free: extraction is a pure function of `user_query`, so
this node needs no LLM/tool wiring (unlike `build_intent_node`, `build_smalltalk_node`,
etc., which take a `BaseChatModel`). It runs first in the graph so
`intent_agent` can use the entities it finds (e.g. a `CARD_LAST4` hit) as a
classification signal.
"""

from __future__ import annotations

from collections.abc import Callable

from agents.state import GraphState
from nlp.ner_extractor import extract_entities
from schemas.dto import AgentTraceStep


def build_ner_node() -> Callable[[GraphState], dict[str, object]]:
    def ner_node(state: GraphState) -> dict[str, object]:
        entities = extract_entities(state["user_query"])
        return {
            "entities": entities,
            "trace": [
                AgentTraceStep(
                    node="ner_agent",
                    summary=f"extracted {len(entities)} entity(ies)",
                )
            ],
        }

    return ner_node
