from __future__ import annotations

from agents.state import new_state
from agents.workers.ner_agent import build_ner_node
from app.core.llm import FakeChatModel
from schemas.dto import EntityType, IntentLabel, PendingEntityRequest


async def test_ner_node_extracts_normally_without_a_pending_request() -> None:
    node = build_ner_node(FakeChatModel())
    state = new_state("c1", "EFT limitiniz ne kadar?")

    result = await node(state)

    assert result["entities"] == []


async def test_ner_node_synthesizes_entity_from_bare_answer_when_pending() -> None:
    node = build_ner_node(FakeChatModel())
    state = new_state("c1", "1234")
    state["carried_pending_request"] = PendingEntityRequest(
        intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla"
    )

    result = await node(state)

    assert len(result["entities"]) == 1
    assert result["entities"][0].type == EntityType.CARD_LAST4
    assert result["entities"][0].value == "1234"
    assert "slot-fill" in result["trace"][0].summary


async def test_ner_node_does_not_duplicate_when_regex_already_found_the_entity() -> None:
    node = build_ner_node(FakeChatModel())
    state = new_state("c1", "kartımın son 4 hanesi 1234")
    state["carried_pending_request"] = PendingEntityRequest(
        intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla"
    )

    result = await node(state)

    card_entities = [e for e in result["entities"] if e.type == EntityType.CARD_LAST4]
    assert len(card_entities) == 1
