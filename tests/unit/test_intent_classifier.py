"""Unit tests for nlp/intent_classifier.py — rule-based path only, no network/LLM."""

from __future__ import annotations

from nlp.intent_classifier import classify_intent_rule_based
from schemas.dto import Entity, EntityType, IntentLabel


def test_rag_query_turkish() -> None:
    intent, confidence = classify_intent_rule_based(
        "Kredi kartı komisyon ücreti ne kadar?", []
    )
    assert intent == IntentLabel.RAG_QUERY
    assert 0.0 < confidence <= 0.95


def test_account_action_turkish() -> None:
    intent, confidence = classify_intent_rule_based(
        "Hesap özetimi ve bakiyemi görmek istiyorum.", []
    )
    assert intent == IntentLabel.ACCOUNT_ACTION
    assert 0.0 < confidence <= 0.95


def test_transaction_action_english() -> None:
    intent, confidence = classify_intent_rule_based(
        "I want to check my transaction history and transfer some money.", []
    )
    assert intent == IntentLabel.TRANSACTION_ACTION
    assert 0.0 < confidence <= 0.95


def test_card_action_turkish() -> None:
    intent, confidence = classify_intent_rule_based(
        "Kartımı blokla, kartım çalındı sanırım.", []
    )
    assert intent == IntentLabel.CARD_ACTION
    assert 0.0 < confidence <= 0.95


def test_small_talk_english() -> None:
    intent, confidence = classify_intent_rule_based(
        "Hello! Thanks so much for your help today.", []
    )
    assert intent == IntentLabel.SMALL_TALK
    assert 0.0 < confidence <= 0.95


def test_escalate_turkish() -> None:
    intent, confidence = classify_intent_rule_based(
        "Gerçek bir kişiyle, yani müşteri temsilcisiyle konuşmak istiyorum.", []
    )
    assert intent == IntentLabel.ESCALATE
    assert 0.0 < confidence <= 0.95


def test_out_of_scope_for_unrelated_sentence() -> None:
    intent, confidence = classify_intent_rule_based(
        "Bugün dışarıda kediler parkta koşuyordu.", []
    )
    assert intent == IntentLabel.OUT_OF_SCOPE
    assert confidence == 0.3


def test_confidence_never_exceeds_cap_even_with_many_keyword_hits() -> None:
    # Every SMALL_TALK keyword crammed into one message — score is high, but
    # confidence must still respect the 0.95 cap (never claim full certainty).
    text = "Merhaba selam teşekkür nasılsın günaydın iyi günler hello thanks thank you how are you"
    intent, confidence = classify_intent_rule_based(text, [])
    assert intent == IntentLabel.SMALL_TALK
    assert confidence == 0.95


def test_entity_boost_corroborates_weak_keyword_signal() -> None:
    # No CARD_ACTION keyword phrase present, but a CARD_LAST4 entity nudges
    # the score above zero and CARD_ACTION becomes the argmax.
    entities = [
        Entity(type=EntityType.CARD_LAST4, value="1234", normalized="1234", confidence=1.0)
    ]
    intent, confidence = classify_intent_rule_based("Kartla ilgili bir şey var 1234", entities)
    assert intent == IntentLabel.CARD_ACTION
    assert confidence == 0.55  # base 0.4 + one boost point * 0.15
