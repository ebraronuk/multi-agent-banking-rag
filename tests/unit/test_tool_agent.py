"""Unit tests for `agents/workers/tool_agent.py`.

The integration suite only ever exercises the *missing-entity* path (no IBAN
given) — the actual happy path, where a real entity is present and a tool
genuinely gets called, was never covered anywhere. That's the path a real
user hits most often, so it's the one most worth pinning down here.
"""

from __future__ import annotations

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agents.state import new_state
from agents.tools.mcp_client import InProcessToolClient
from agents.workers.tool_agent import build_tool_agent_node
from app.core.config import Settings
from app.core.llm import FakeChatModel
from mcp_server.tools.banking_tools import _ACCOUNTS
from schemas.dto import Entity, EntityType, IntentLabel


def _settings(max_agent_iterations: int = 6) -> Settings:
    return Settings(llm_provider="fake", max_agent_iterations=max_agent_iterations)


class _ScriptedToolCallingModel(BaseChatModel):
    """A minimal, deliberately dumb `bind_tools`-capable model: it plays back
    a scripted list of `AIMessage`s (one per `.ainvoke()` call) instead of
    reasoning about anything. Tests the reasoning *loop's mechanics*
    (execute proposed calls, feed results back, stop on no-more-tool-calls or
    at the hop cap) independent of whether a real model reasons well — that
    part is Anthropic's/OpenAI's job, not this codebase's to verify.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-test-model"

    def bind_tools(self, tools: object, **kwargs: object) -> _ScriptedToolCallingModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        index = min(self.call_index, len(self.responses) - 1)
        self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])


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


async def test_missing_entity_short_circuits_and_sets_pending_request_for_next_turn() -> None:
    node = build_tool_agent_node(InProcessToolClient(), FakeChatModel(), _settings())
    state = new_state("c1", "kartımı blokla")
    state["intent"] = IntentLabel.CARD_ACTION
    state["entities"] = []  # no card_last4 given this turn

    result = await node(state)

    assert "tool_calls" not in result
    assert result["tool_agent_done"] is True
    pending = result["pending_entity_request"]
    assert pending.intent == IntentLabel.CARD_ACTION
    assert pending.entity_type == EntityType.CARD_LAST4


async def test_reasoning_loop_calls_a_single_grounded_tool() -> None:
    account_id = next(iter(_ACCOUNTS))
    final = AIMessage(content="Bakiyeniz görüntülendi.")
    proposal = AIMessage(
        content="",
        tool_calls=[{"name": "get_balance", "args": {"account_id": account_id}, "id": "call_1"}],
    )
    llm = _ScriptedToolCallingModel(responses=[proposal, final])
    node = build_tool_agent_node(InProcessToolClient(), llm, _settings())

    state = new_state("c1", "bakiyem ne kadar?")
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["entities"] = [Entity(type=EntityType.IBAN, value=account_id, normalized=account_id)]

    result = await node(state)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].tool_name == "get_balance"
    assert result["tool_calls"][0].ok is True
    assert result["draft_answer"] == "Bakiyeniz görüntülendi."
    assert result["tool_agent_done"] is True


async def test_reasoning_loop_handles_a_compound_multi_tool_request_in_one_hop() -> None:
    account_id = next(iter(_ACCOUNTS))
    last4 = _ACCOUNTS[account_id]["cards"][0]["last4"]  # type: ignore[index]
    final = AIMessage(content="Kartınızı blokladım ve bir destek talebi açtım.")
    proposal = AIMessage(
        content="",
        tool_calls=[
            {"name": "block_card", "args": {"card_last4": last4, "reason": "çalındı"}, "id": "call_1"},
            {
                "name": "open_support_ticket",
                "args": {"subject": "Çalınan kart", "description": "Kart çalındı, bloklandı."},
                "id": "call_2",
            },
        ],
    )
    llm = _ScriptedToolCallingModel(responses=[proposal, final])
    node = build_tool_agent_node(InProcessToolClient(), llm, _settings())

    state = new_state("c1", "kartımı blokla ve bir destek talebi aç")
    state["intent"] = IntentLabel.CARD_ACTION
    state["entities"] = [Entity(type=EntityType.CARD_LAST4, value=last4, normalized=last4)]

    result = await node(state)

    called_tools = {tc.tool_name for tc in result["tool_calls"]}
    assert called_tools == {"block_card", "open_support_ticket"}
    assert all(tc.ok for tc in result["tool_calls"])

    _ACCOUNTS[account_id]["cards"][0]["status"] = "active"  # type: ignore[index]  # restore fixture


async def test_reasoning_loop_refuses_an_ungrounded_argument() -> None:
    # The model proposes a card_last4 that was never actually extracted from
    # the conversation — must be refused, not executed, regardless of how
    # plausible-looking it is.
    proposal = AIMessage(
        content="",
        tool_calls=[{"name": "block_card", "args": {"card_last4": "9999", "reason": "test"}, "id": "call_1"}],
    )
    final = AIMessage(content="Bu bilgiyi doğrulayamadım.")
    llm = _ScriptedToolCallingModel(responses=[proposal, final])
    node = build_tool_agent_node(InProcessToolClient(), llm, _settings())

    state = new_state("c1", "kartımı blokla, son 4 hane 9999")
    state["intent"] = IntentLabel.CARD_ACTION
    state["entities"] = []  # nothing grounded — NER never actually found "9999" as a card entity

    result = await node(state)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].ok is False
    assert "NOT_GROUNDED" in (result["tool_calls"][0].error or "")


async def test_reasoning_loop_stops_at_max_hops_instead_of_looping_forever() -> None:
    # A model that always wants "just one more" tool call must not be allowed
    # to run indefinitely — the cap is `settings.max_agent_iterations`.
    account_id = next(iter(_ACCOUNTS))
    always_another_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_balance", "args": {"account_id": account_id}, "id": "call_x"}],
    )
    llm = _ScriptedToolCallingModel(responses=[always_another_call])
    node = build_tool_agent_node(InProcessToolClient(), llm, _settings(max_agent_iterations=2))

    state = new_state("c1", "bakiyem ne kadar?")
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["entities"] = [Entity(type=EntityType.IBAN, value=account_id, normalized=account_id)]

    result = await node(state)

    assert len(result["tool_calls"]) == 2  # exactly max_agent_iterations hops, not more
    assert result["tool_agent_done"] is True
    assert result["draft_answer"]
