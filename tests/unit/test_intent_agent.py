from __future__ import annotations

from agents.state import new_state
from agents.workers.intent_agent import build_intent_node
from app.core.llm import FakeChatModel
from schemas.dto import Entity, EntityType, IntentLabel, PendingEntityRequest


async def test_intent_node_continues_pending_request_when_entity_answered() -> None:
    node = build_intent_node(FakeChatModel())
    state = new_state("c1", "1234")
    state["carried_pending_request"] = PendingEntityRequest(
        intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla"
    )
    state["entities"] = [Entity(type=EntityType.CARD_LAST4, value="1234", normalized="1234")]

    result = await node(state)

    assert result["intent"] == IntentLabel.CARD_ACTION
    assert result["intent_confidence"] == 1.0
    assert "continued pending" in result["trace"][0].summary


async def test_intent_node_classifies_normally_when_pending_entity_not_found() -> None:
    node = build_intent_node(FakeChatModel())
    state = new_state("c1", "bugün hava çok güzel")
    state["carried_pending_request"] = PendingEntityRequest(
        intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla"
    )
    state["entities"] = []  # nothing answered the pending request

    result = await node(state)

    assert result["intent"] == IntentLabel.OUT_OF_SCOPE


async def test_intent_node_classifies_normally_without_any_pending_request() -> None:
    node = build_intent_node(FakeChatModel())
    state = new_state("c1", "merhaba")

    result = await node(state)

    assert result["intent"] == IntentLabel.SMALL_TALK
