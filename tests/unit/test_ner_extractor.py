"""nlp/ner_extractor.py için birim testler.

Regex katmanı (`extract_entities`) saf ve deterministik, ağ/LLM gerektirmez.
`extract_entities_with_llm`'in fake-model ve mocklu-LLM yolları dosyanın
sonunda ayrıca test ediliyor.
"""

from __future__ import annotations

from app.core.llm import FakeChatModel
from nlp.ner_extractor import (
    _ExtractedEntity,
    _NERExtraction,
    extract_entities,
    extract_entities_with_llm,
)
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


class _StubStructuredModel:
    """`with_structured_output(...).ainvoke(...)` çağrısını taklit eden minimal
    bir test çifti — gerçek bir `BaseChatModel` olmasına gerek yok, sadece
    `is_fake_model()`'in False dönmesi (FakeChatModel örneği olmadığı için
    zaten öyle) ve bu iki metodun var olması yeterli."""

    def __init__(self, extraction: _NERExtraction | None = None, raises: Exception | None = None) -> None:
        self._extraction = extraction
        self._raises = raises

    def with_structured_output(self, schema: object) -> _StubStructuredModel:
        return self

    async def ainvoke(self, messages: object) -> _NERExtraction:
        if self._raises:
            raise self._raises
        assert self._extraction is not None
        return self._extraction


async def test_extract_entities_with_llm_uses_only_regex_for_fake_model() -> None:
    text = "Kartımın son 4 hanesi 1234 ile ilgili bir sorun var."

    result = await extract_entities_with_llm(text, FakeChatModel())

    assert result == extract_entities(text)


async def test_extract_entities_with_llm_merges_person_name_from_llm() -> None:
    text = "Ayşe Yılmaz adına bir işlem yapmak istiyorum."
    stub = _StubStructuredModel(
        _NERExtraction(entities=[_ExtractedEntity(type=EntityType.PERSON_NAME, value="Ayşe Yılmaz")])
    )

    result = await extract_entities_with_llm(text, stub)  # type: ignore[arg-type]

    person_entities = [e for e in result if e.type == EntityType.PERSON_NAME]
    assert len(person_entities) == 1
    assert person_entities[0].value == "Ayşe Yılmaz"
    assert person_entities[0].confidence == 0.75  # regex'in 1.0'ı kadar kesin değil


async def test_extract_entities_with_llm_does_not_duplicate_entity_already_found_by_regex() -> None:
    iban = "TR330006100519786457841326"
    text = f"IBAN'ım {iban}."
    # Model aynı IBAN'ı da rapor ediyor — regex zaten bulduğu için tekrar sayılmamalı.
    stub = _StubStructuredModel(
        _NERExtraction(entities=[_ExtractedEntity(type=EntityType.IBAN, value=iban, normalized=iban)])
    )

    result = await extract_entities_with_llm(text, stub)  # type: ignore[arg-type]

    iban_entities = [e for e in result if e.type == EntityType.IBAN]
    assert len(iban_entities) == 1


async def test_extract_entities_with_llm_falls_back_to_regex_on_llm_failure() -> None:
    text = "Kartımın son 4 hanesi 1234."
    stub = _StubStructuredModel(raises=RuntimeError("provider outage"))

    result = await extract_entities_with_llm(text, stub)  # type: ignore[arg-type]

    assert result == extract_entities(text)
