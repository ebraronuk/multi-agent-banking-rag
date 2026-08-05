"""Unit tests for `agents/workers/tool_agent.py`.

The integration suite only ever exercises the *missing-entity* path (no IBAN
given) — the actual happy path, where a real entity is present and a tool
genuinely gets called, was never covered anywhere. That's the path a real
user hits most often, so it's the one most worth pinning down here.
"""

from __future__ import annotations

from agents.state import new_state
from agents.tools.mcp_client import InProcessToolClient
from agents.workers.tool_agent import build_tool_agent_node
from app.core.config import Settings
from app.core.llm import FakeChatModel
from mcp_server.tools.banking_tools import _ACCOUNTS
from schemas.dto import Entity, EntityType, IntentLabel


def _settings() -> Settings:
    return Settings(llm_provider="fake")


async def test_account_action_with_iban_calls_the_tool_and_drafts_an_answer() -> None:
    node = build_tool_agent_node(InProcessToolClient(), FakeChatModel(), _settings())
    account_id = next(iter(_ACCOUNTS))
    state = new_state("c1", "bakiyem ne kadar?")
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["entities"] = [Entity(type=EntityType.IBAN, value=account_id, normalized=account_id)]
    state["iteration_count"] = 0

    result = await node(state)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].tool_name == "get_balance"
    assert result["tool_calls"][0].ok is True
    assert result["tool_agent_done"] is True
    assert result["iteration_count"] == 1
    assert result["draft_answer"]


async def test_card_action_with_card_last4_blocks_the_right_card() -> None:
    node = build_tool_agent_node(InProcessToolClient(), FakeChatModel(), _settings())
    account_id = next(iter(_ACCOUNTS))
    last4 = _ACCOUNTS[account_id]["cards"][0]["last4"]  # type: ignore[index]
    state = new_state("c1", "kartımı blokla")
    state["intent"] = IntentLabel.CARD_ACTION
    state["entities"] = [Entity(type=EntityType.CARD_LAST4, value=last4, normalized=last4)]

    result = await node(state)

    assert result["tool_calls"][0].tool_name == "block_card"
    assert result["tool_calls"][0].ok is True

    # restore fixture state so this test doesn't leak into others
    _ACCOUNTS[account_id]["cards"][0]["status"] = "active"  # type: ignore[index]


async def test_unmapped_intent_short_circuits_without_calling_any_tool() -> None:
    node = build_tool_agent_node(InProcessToolClient(), FakeChatModel(), _settings())
    state = new_state("c1", "bir şey")
    state["intent"] = IntentLabel.RAG_QUERY  # supervisor would never route here for this intent

    result = await node(state)

    # No tool_calls key at all here (not an empty list) — GraphState's
    # `tool_calls` reducer only fires when a node's partial update includes
    # the key; this short-circuit path never had a tool call to report.
    assert "tool_calls" not in result
    assert result["tool_agent_done"] is True
    assert result["draft_answer"]
