from __future__ import annotations

from agents.state import new_state
from agents.supervisor import (
    NODE_ESCALATE,
    NODE_GUARDRAIL,
    NODE_RAG_AGENT,
    NODE_SMALLTALK,
    NODE_TOOL_AGENT,
    build_supervisor_router,
)
from app.core.config import Settings
from schemas.dto import IntentLabel


def _settings(max_iterations: int = 6) -> Settings:
    return Settings(max_agent_iterations=max_iterations)


def test_routes_rag_query_to_rag_agent() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limiti nedir?")
    state["intent"] = IntentLabel.RAG_QUERY

    assert route(state) == NODE_RAG_AGENT


def test_routes_small_talk_to_smalltalk() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "merhaba")
    state["intent"] = IntentLabel.SMALL_TALK

    assert route(state) == NODE_SMALLTALK


def test_routes_unclassified_and_out_of_scope_to_escalate() -> None:
    route = build_supervisor_router(_settings())

    state = new_state("c1", "?")
    state["intent"] = None
    assert route(state) == NODE_ESCALATE

    state["intent"] = IntentLabel.OUT_OF_SCOPE
    assert route(state) == NODE_ESCALATE

    state["intent"] = IntentLabel.ESCALATE
    assert route(state) == NODE_ESCALATE


def test_tool_driven_intent_loops_until_done() -> None:
    route = build_supervisor_router(_settings(max_iterations=6))
    state = new_state("c1", "bakiyem ne kadar")
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 1

    assert route(state) == NODE_TOOL_AGENT

    state["tool_agent_done"] = True
    assert route(state) == NODE_GUARDRAIL


def test_tool_driven_intent_stops_at_iteration_cap_even_if_not_done() -> None:
    route = build_supervisor_router(_settings(max_iterations=3))
    state = new_state("c1", "kartımı blokla")
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 3

    assert route(state) == NODE_GUARDRAIL
