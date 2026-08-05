"""Unit tests for nlp/ner_extractor.py — pure, deterministic, no network/LLM."""

from __future__ import annotations

from nlp.ner_extractor import extract_entities
from schemas.dto import EntityType


def test_extract_iban() -> None:
    iban = "TR330006100519786457841326"
    text = f"Paranızı {iban} numaralı hesaba gönderdim."

    entities = extract_entities(text)
    iban_entities = [e for e in entities if e.type == EntityType.IBAN]

    assert len(iban_entities) == 1
    entity = iban_entities[0]
    assert entity.value == iban
    assert entity.normalized == iban  # already uppercase, no whitespace
    assert (entity.start, entity.end) == (text.index(iban), text.index(iban) + len(iban))
    assert entity.confidence == 1.0


def test_extract_iban_with_spaces_strips_whitespace_when_normalized() -> None:
    spaced_iban = "TR33 0006 1005 1978 6457 8413 26"
    text = f"IBAN'ım {spaced_iban}, oraya gönderebilirsiniz."

    entities = extract_entities(text)
    iban_entities = [e for e in entities if e.type == EntityType.IBAN]

    assert len(iban_entities) == 1
    entity = iban_entities[0]
    assert entity.value == spaced_iban
    assert entity.normalized == spaced_iban.replace(" ", "")
    assert (entity.start, entity.end) == (
        text.index(spaced_iban),
        text.index(spaced_iban) + len(spaced_iban),
    )


def test_extract_amount_and_currency_turkish_style() -> None:
    amount = "1.250,50"
    text = f"Hesabımdan {amount} TL çekmek istiyorum."

    entities = extract_entities(text)
    amounts = [e for e in entities if e.type == EntityType.AMOUNT]
    currencies = [e for e in entities if e.type == EntityType.CURRENCY]

    assert len(amounts) == 1
    assert amounts[0].value == amount
    assert amounts[0].normalized == "1250.50"
    assert (amounts[0].start, amounts[0].end) == (
        text.index(amount),
        text.index(amount) + len(amount),
    )

    assert len(currencies) == 1
    assert currencies[0].value == "TL"
    assert currencies[0].normalized == "TRY"
    assert (currencies[0].start, currencies[0].end) == (
        text.index("TL"),
        text.index("TL") + len("TL"),
    )


def test_extract_amount_with_leading_currency_symbol() -> None:
    text = "Faturanız için €200 ödemeniz gerekiyor."

    entities = extract_entities(text)
    amounts = [e for e in entities if e.type == EntityType.AMOUNT]
    currencies = [e for e in entities if e.type == EntityType.CURRENCY]

    assert len(amounts) == 1
    assert amounts[0].value == "200"
    assert amounts[0].normalized == "200"

    assert len(currencies) == 1
    assert currencies[0].value == "€"
    assert currencies[0].normalized == "EUR"


def test_extract_amount_with_currency_word_suffix() -> None:
    text = "Kartıma 200 euro yükleyebilir misiniz?"

    entities = extract_entities(text)
    amounts = [e for e in entities if e.type == EntityType.AMOUNT]
    currencies = [e for e in entities if e.type == EntityType.CURRENCY]

    assert len(amounts) == 1
    assert amounts[0].value == "200"

    assert len(currencies) == 1
    assert currencies[0].value == "euro"
    assert currencies[0].normalized == "EUR"


def test_extract_date_dd_mm_yyyy() -> None:
    date = "15.03.2026"
    text = f"Son ödeme tarihi {date} olarak görünüyor."

    entities = extract_entities(text)
    dates = [e for e in entities if e.type == EntityType.DATE]

    assert len(dates) == 1
    entity = dates[0]
    assert entity.value == date
    assert entity.normalized == "2026-03-15"
    assert (entity.start, entity.end) == (text.index(date), text.index(date) + len(date))


def test_extract_date_yyyy_mm_dd() -> None:
    date = "2026-03-15"
    text = f"Ödeme planı {date} tarihinde başlıyor."

    entities = extract_entities(text)
    dates = [e for e in entities if e.type == EntityType.DATE]

    assert len(dates) == 1
    assert dates[0].value == date
    assert dates[0].normalized == date


def test_extract_card_last4_from_keyword_phrase() -> None:
    text = "Kartımın son 4 hanesi 1234 ile ilgili bir sorun var."

    entities = extract_entities(text)
    card_entities = [e for e in entities if e.type == EntityType.CARD_LAST4]

    assert len(card_entities) == 1
    entity = card_entities[0]
    assert entity.value == "1234"
    assert entity.normalized == "1234"
    assert (entity.start, entity.end) == (text.index("1234"), text.index("1234") + 4)


def test_extract_card_last4_from_masked_pattern() -> None:
    text = "Kart numaram **** 1234 ile bitiyor."

    entities = extract_entities(text)
    card_entities = [e for e in entities if e.type == EntityType.CARD_LAST4]

    assert len(card_entities) == 1
    assert card_entities[0].value == "1234"


def test_extract_account_type() -> None:
    keyword = "vadeli hesap"
    text = f"Bir {keyword} açmak istiyorum, faiz oranı nedir?"

    entities = extract_entities(text)
    account_entities = [e for e in entities if e.type == EntityType.ACCOUNT_TYPE]

    assert len(account_entities) == 1
    entity = account_entities[0]
    assert entity.value == keyword
    assert entity.normalized == keyword
    assert (entity.start, entity.end) == (text.index(keyword), text.index(keyword) + len(keyword))


def test_extract_entities_returns_empty_list_for_irrelevant_text() -> None:
    assert extract_entities("Bugün dışarıda kediler parkta koşuyordu.") == []
