from __future__ import annotations

from agents.state import new_state
from agents.workers.guardrail_agent import build_guardrail_node
from app.core.config import Settings
from schemas.dto import IntentLabel


def _settings(max_iterations: int = 6) -> Settings:
    return Settings(max_agent_iterations=max_iterations)


def test_passes_through_clean_answer_untouched() -> None:
    node = build_guardrail_node(_settings())
    state = new_state("c1", "çalışma saatleriniz nedir?")
    state["draft_answer"] = "Şubelerimiz hafta içi 09:00-17:00 arası hizmet vermektedir."
    state["intent"] = IntentLabel.RAG_QUERY
    state["tool_agent_done"] = True

    result = node(state)

    assert result["final_answer"] == state["draft_answer"]
    assert result["guardrail_flags"] == []


def test_redacts_iban_and_long_digit_runs() -> None:
    node = build_guardrail_node(_settings())
    state = new_state("c1", "bakiyem ne kadar")
    state["draft_answer"] = "Hesabınız TR330006100519786457841326 bakiyesi 1500 TL'dir."
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["tool_agent_done"] = True

    result = node(state)

    assert "TR33" not in result["final_answer"]
    assert (
        result["final_answer"].endswith("bakiyesi 1500 TL'dir.") or "***" in result["final_answer"]
    )
    assert "PII_REDACTED" in result["guardrail_flags"]


def test_blocks_financial_advice_language() -> None:
    node = build_guardrail_node(_settings())
    state = new_state("c1", "ne alayım")
    state["draft_answer"] = "Kesinlikle şu hisse al, garanti getiri sağlar."
    state["intent"] = IntentLabel.RAG_QUERY
    state["tool_agent_done"] = True

    result = node(state)

    assert "FINANCIAL_ADVICE_BLOCKED" in result["guardrail_flags"]
    assert "hisse al" not in result["final_answer"].lower()


def test_no_draft_falls_back_to_generic_message() -> None:
    node = build_guardrail_node(_settings())
    state = new_state("c1", "???")
    state["draft_answer"] = None
    state["intent"] = IntentLabel.OUT_OF_SCOPE
    state["tool_agent_done"] = True

    result = node(state)

    assert "NO_DRAFT_PRODUCED" in result["guardrail_flags"]
    assert result["final_answer"]


def test_iteration_limit_overrides_pending_tool_agent() -> None:
    node = build_guardrail_node(_settings(max_iterations=2))
    state = new_state("c1", "kartımı blokla")
    state["draft_answer"] = "işleniyor..."
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 2

    result = node(state)

    assert "ESCALATED_ITERATION_LIMIT" in result["guardrail_flags"]
