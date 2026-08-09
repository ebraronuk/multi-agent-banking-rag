from __future__ import annotations

from agents.state import new_state
from agents.workers.intent_agent import _clean_extra_intents, build_intent_node
from app.core.llm import FakeChatModel
from nlp.intent_classifier import _IntentClassification
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
    assert result["extra_intents"] == []
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


async def test_intent_node_fake_model_returns_empty_extra_intents_for_single_intent_message() -> None:
    node = build_intent_node(FakeChatModel())
    state = new_state("c1", "merhaba, nasılsınız")

    result = await node(state)

    assert result["extra_intents"] == []


async def test_intent_node_fake_model_detects_extra_intent_for_compound_message() -> None:
    # Fake modda da çoklu-niyet dispatch'in (ADR-012) tetiklenebildiğinin
    # intent_agent katmanındaki regresyon testi — nlp/intent_classifier.py'nin
    # kendi testleri _rule_based_extra_intents'i izole test ediyor, bu da
    # onun _clean_extra_intents ile birlikte doğru çalıştığını doğruluyor.
    node = build_intent_node(FakeChatModel())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")

    result = await node(state)

    assert result["intent"] == IntentLabel.RAG_QUERY
    assert result["extra_intents"] == [IntentLabel.CARD_ACTION]


class _StubIntentModel:
    """`classify_intent`'in gerçek-LLM yolunu, `with_structured_output` +
    `ainvoke` çağrısını taklit ederek test eder."""

    def __init__(self, classification: _IntentClassification) -> None:
        self._classification = classification

    def with_structured_output(self, schema: object) -> _StubIntentModel:
        return self

    async def ainvoke(self, messages: object) -> _IntentClassification:
        return self._classification


async def test_intent_node_passes_through_cleaned_extra_intents_from_real_model() -> None:
    stub = _StubIntentModel(
        _IntentClassification(
            intent=IntentLabel.CARD_ACTION,
            confidence=0.9,
            extra_intents=[IntentLabel.RAG_QUERY],
        )
    )
    node = build_intent_node(stub)  # type: ignore[arg-type]
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")

    result = await node(state)

    assert result["intent"] == IntentLabel.CARD_ACTION
    assert result["extra_intents"] == [IntentLabel.RAG_QUERY]
    assert "extra:" in result["trace"][0].summary


def test_clean_extra_intents_drops_the_primary_intent_itself() -> None:
    cleaned = _clean_extra_intents(IntentLabel.CARD_ACTION, [IntentLabel.CARD_ACTION, IntentLabel.RAG_QUERY])
    assert cleaned == [IntentLabel.RAG_QUERY]


def test_clean_extra_intents_deduplicates() -> None:
    cleaned = _clean_extra_intents(
        IntentLabel.CARD_ACTION, [IntentLabel.RAG_QUERY, IntentLabel.RAG_QUERY]
    )
    assert cleaned == [IntentLabel.RAG_QUERY]


def test_clean_extra_intents_excludes_non_chainable_categories() -> None:
    # ESCALATE/OUT_OF_SCOPE zincire dahil değil (bkz. ADR-012) — SMALL_TALK
    # dahil, ayrı bir testte doğrulanıyor.
    cleaned = _clean_extra_intents(
        IntentLabel.CARD_ACTION, [IntentLabel.ESCALATE, IntentLabel.OUT_OF_SCOPE]
    )
    assert cleaned == []


def test_clean_extra_intents_includes_small_talk() -> None:
    cleaned = _clean_extra_intents(IntentLabel.RAG_QUERY, [IntentLabel.SMALL_TALK])
    assert cleaned == [IntentLabel.SMALL_TALK]


def test_clean_extra_intents_caps_at_two() -> None:
    cleaned = _clean_extra_intents(
        IntentLabel.CARD_ACTION,
        [IntentLabel.RAG_QUERY, IntentLabel.ACCOUNT_ACTION, IntentLabel.TRANSACTION_ACTION],
    )
    assert len(cleaned) == 2
