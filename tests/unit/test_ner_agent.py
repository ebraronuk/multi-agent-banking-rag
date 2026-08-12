from __future__ import annotations

from agents.state import new_state
from agents.workers.ner_agent import build_ner_node
from app.core.llm import FakeChatModel
from schemas.dto import ChatMessage, EntityType, IntentLabel, PendingEntityRequest


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


async def test_ner_node_recalls_iban_from_earlier_turn_when_not_repeated() -> None:
    # Regresyon: canlıda "IBAN'ım X, bakiyemi göster" sonrası "peki son
    # işlemlerim neler?" sorulduğunda sistem IBAN'ı hiç verilmemiş gibi
    # tekrar soruyordu — oysa birkaç turn önce zaten verilmişti.
    node = build_ner_node(FakeChatModel())
    state = new_state("c1", "peki son işlemlerim neler?")
    state["history"] = [
        ChatMessage(role="user", content="IBAN'ım TR330006100519786457841326, bakiyemi öğrenebilir miyim?"),
        ChatMessage(role="assistant", content="***1326 IBAN numaralı hesabınızın bakiyesi ..."),
    ]

    result = await node(state)

    iban_entities = [e for e in result["entities"] if e.type == EntityType.IBAN]
    assert len(iban_entities) == 1
    assert iban_entities[0].normalized == "TR330006100519786457841326"
    assert "recalled from history" in result["trace"][0].summary


async def test_ner_node_does_not_recall_when_entity_already_present_this_turn() -> None:
    node = build_ner_node(FakeChatModel())
    state = new_state("c1", "IBAN'ım TR640001000000012345678901, bakiyemi göster")
    state["history"] = [
        ChatMessage(role="user", content="IBAN'ım TR330006100519786457841326, bakiyemi öğrenebilir miyim?"),
        ChatMessage(role="assistant", content="***1326 IBAN numaralı hesabınızın bakiyesi ..."),
    ]

    result = await node(state)

    iban_entities = [e for e in result["entities"] if e.type == EntityType.IBAN]
    # Bu turda gerçekten verilen IBAN kazanmalı, geçmişteki eski hesap değil.
    assert len(iban_entities) == 1
    assert iban_entities[0].normalized == "TR640001000000012345678901"


async def test_ner_node_does_not_recall_amount_or_date_from_history() -> None:
    # Sadece kimlik entity'leri (IBAN/CARD_LAST4) geri çağrılıyor — bir
    # tutarı/tarihi eski bir turn'den sessizce yeni bir isteğe taşımak riskli
    # olurdu.
    node = build_ner_node(FakeChatModel())
    state = new_state("c1", "kartımı blokla")
    state["history"] = [
        ChatMessage(role="user", content="1000 TL gönderir misin, 12 Ocak 2026'da"),
        ChatMessage(role="assistant", content="Bu talebi işleyemedim."),
    ]

    result = await node(state)

    assert all(e.type not in ("AMOUNT", "DATE") for e in result["entities"])
