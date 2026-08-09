"""nlp/intent_classifier.py için birim testler — sadece kural tabanlı yol, ağ/LLM yok."""

from __future__ import annotations

from app.core.llm import FakeChatModel
from nlp.intent_classifier import (
    _rule_based_extra_intents,
    classify_intent,
    classify_intent_rule_based,
)
from schemas.dto import Entity, EntityType, IntentLabel


def test_rag_query_turkish() -> None:
    intent, confidence = classify_intent_rule_based("Kredi kartı komisyon ücreti ne kadar?", [])
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
    intent, confidence = classify_intent_rule_based("Kartımı blokla, kartım çalındı sanırım.", [])
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


def test_transfer_execution_request_escalates_because_no_tool_can_send_money() -> None:
    # tool_agent'ın TRANSACTION_ACTION için tek aracı list_transactions
    # (salt görüntüleme) — parayı gerçekten gönderen bir araç yok. Bu yüzden
    # "gönder"/"transfer yap" gibi bir yürütme isteği TRANSACTION_ACTION'a
    # değil ESCALATE'e düşmeli; aksi halde asistan sessizce işlem geçmişini
    # gösterip transfer yapılmış izlenimi verirdi.
    intent, confidence = classify_intent_rule_based(
        "Şu hesaba 1000 TL için havale yapar mısınız, para transfer edin lütfen.", []
    )
    assert intent == IntentLabel.ESCALATE
    assert 0.0 < confidence <= 0.95


def test_transaction_history_query_still_resolves_to_transaction_action() -> None:
    intent, confidence = classify_intent_rule_based("Son işlemlerimi görebilir miyim?", [])
    assert intent == IntentLabel.TRANSACTION_ACTION
    assert 0.0 < confidence <= 0.95


def test_out_of_scope_for_unrelated_sentence() -> None:
    intent, confidence = classify_intent_rule_based("Bugün dışarıda kediler parkta koşuyordu.", [])
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
    entities = [Entity(type=EntityType.CARD_LAST4, value="1234", normalized="1234", confidence=1.0)]
    intent, confidence = classify_intent_rule_based("Kartla ilgili bir şey var 1234", entities)
    assert intent == IntentLabel.CARD_ACTION
    assert confidence == 0.55  # base 0.4 + one boost point * 0.15


async def test_classify_intent_fake_model_returns_empty_extra_intents_for_single_intent_message() -> None:
    # Async sarmalayıcı (classify_intent) fake modelde kural tabanlı yola
    # düşüyor — üçlü dönüş şeklini (intent, confidence, extra_intents) burada
    # doğruluyoruz. Tek niyetli bir mesajda (başka hiçbir kategori kelimesi
    # geçmiyor) extra_intents boş kalmalı.
    intent, confidence, extra_intents = await classify_intent("merhaba", [], FakeChatModel())

    assert intent == IntentLabel.SMALL_TALK
    assert confidence > 0.0
    assert extra_intents == []


async def test_classify_intent_fake_model_detects_extra_intent_for_compound_message() -> None:
    # Bu, çoklu-niyet dispatch'in (ADR-012) fake/anahtarsız modda da
    # tetiklenebildiğinin regresyon testi — daha önce classify_intent fake
    # modda extra_intents için hep [] dönüyordu, bu özelliği anahtarsız
    # tamamen görünmez yapıyordu.
    intent, confidence, extra_intents = await classify_intent(
        "Kartımı blokla ve EFT limitiniz ne kadar?", [], FakeChatModel()
    )

    assert intent == IntentLabel.RAG_QUERY  # "limit" berabereliği kazanıyor (dict sırası)
    assert confidence > 0.0
    assert extra_intents == [IntentLabel.CARD_ACTION]


def test_rule_based_extra_intents_ignores_primary_and_zero_score_intents() -> None:
    extra = _rule_based_extra_intents(
        "Kartımı blokla ve EFT limitiniz ne kadar?", [], IntentLabel.RAG_QUERY
    )

    assert extra == [IntentLabel.CARD_ACTION]


def test_rule_based_extra_intents_empty_for_single_intent_message() -> None:
    extra = _rule_based_extra_intents("Kartımı blokla, kartım çalındı.", [], IntentLabel.CARD_ACTION)

    assert extra == []


def test_rule_based_extra_intents_requires_corroboration_for_action_intents() -> None:
    # Regresyon: "bloke" tek başına hem "kartımı hemen bloke et" (imperatif)
    # hem "ne zaman bloke edebilirim" (bir politika sorusu) içinde geçebiliyor.
    # Bu saf bir RAG_QUERY — CARD_ACTION'ı bir entity ya da çok kelimeli,
    # spesifik bir kalıp (ör. "kartımı blokla") desteklemiyorsa extra intent
    # sayılmamalı, yoksa tool_agent alakasız bir "kartının son 4 hanesi?"
    # sorusu sorup tek-niyetli RAG cevabını kirletir.
    intent, confidence = classify_intent_rule_based(
        "Kartımı ne zaman bloke edebilirim, politikanız nedir?", []
    )
    assert intent == IntentLabel.RAG_QUERY

    extra = _rule_based_extra_intents(
        "Kartımı ne zaman bloke edebilirim, politikanız nedir?", [], IntentLabel.RAG_QUERY
    )
    assert extra == []
