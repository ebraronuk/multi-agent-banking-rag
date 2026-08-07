from __future__ import annotations

from agents.state import new_state
from agents.supervisor import (
    NODE_ADVANCE_INTENT,
    NODE_ESCALATE,
    NODE_GUARDRAIL,
    NODE_RAG_AGENT,
    NODE_SMALLTALK,
    NODE_SYNTHESIZER,
    NODE_TOOL_AGENT,
    advance_intent_node,
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


def test_rag_query_second_pass_goes_to_guardrail_when_no_extra_intent() -> None:
    # rag_agent kendi işini bitirip supervisor'a döndüğünde (worker_pass_done),
    # kuyrukta bekleyen bir niyet yoksa bugünküyle birebir aynı davranış.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitiniz ne kadar?")
    state["intent"] = IntentLabel.RAG_QUERY
    state["worker_pass_done"] = True

    assert route(state) == NODE_GUARDRAIL


def test_rag_query_second_pass_advances_when_extra_intent_queued() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitiniz ne kadar ve merhaba")
    state["intent"] = IntentLabel.RAG_QUERY
    state["worker_pass_done"] = True
    state["extra_intents"] = [IntentLabel.SMALL_TALK]

    assert route(state) == NODE_ADVANCE_INTENT


def test_tool_driven_intent_advances_to_extra_intent_once_done() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = True
    state["extra_intents"] = [IntentLabel.RAG_QUERY]

    assert route(state) == NODE_ADVANCE_INTENT


def test_routes_to_synthesizer_once_queue_empty_and_drafts_collected() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "merhaba")
    state["intent"] = IntentLabel.SMALL_TALK
    state["worker_pass_done"] = True
    state["extra_intents"] = []
    state["collected_drafts"] = ["ilk taslak"]

    assert route(state) == NODE_SYNTHESIZER


def test_iteration_cap_never_advances_to_extra_intent() -> None:
    # Limit dolduysa kalan kuyruk sessizce bırakılıyor — advance etmek
    # sınırın amacını boşa çıkarırdı (bkz. ADR-012'nin bilinen sınırları).
    route = build_supervisor_router(_settings(max_iterations=2))
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 2
    state["extra_intents"] = [IntentLabel.RAG_QUERY]

    assert route(state) == NODE_GUARDRAIL


def test_advance_intent_node_pops_queue_and_resets_pass_flags() -> None:
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["extra_intents"] = [IntentLabel.RAG_QUERY, IntentLabel.SMALL_TALK]
    state["draft_answer"] = "kartınız bloklandı"
    state["worker_pass_done"] = True
    state["tool_agent_done"] = True

    result = advance_intent_node(state)

    assert result["intent"] == IntentLabel.RAG_QUERY
    assert result["extra_intents"] == [IntentLabel.SMALL_TALK]
    assert result["collected_drafts"] == ["kartınız bloklandı"]
    assert result["draft_answer"] is None
    assert result["worker_pass_done"] is False
    assert result["tool_agent_done"] is False


def test_advance_intent_node_does_not_collect_an_empty_draft() -> None:
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["extra_intents"] = [IntentLabel.RAG_QUERY]
    state["draft_answer"] = None

    result = advance_intent_node(state)

    assert result["collected_drafts"] == []
