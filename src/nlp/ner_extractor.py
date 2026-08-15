"""Bankacılık sohbet mesajları için varlık çıkarımı (NER).

Regex katmanı (`extract_entities`) hızlı ve tamamen denetlenebilir bir ilk
geçiş — her eşleşme açık bir desene dayanıyor. Ama regex'in hiç yakalayamadığı
bir şey var: serbest metin kişi adları, ya da alışılmadık bir ifadeyle
söylenmiş tarih/tutar. `extract_entities_with_llm` bunun için var — regex
sonucunu her zaman baz alıp, gerçek bir model bağlıysa üstüne bir LLM geçişi
ekliyor (bkz. ADR-011).
"""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.prompts.ner_prompt import NER_SYSTEM_PROMPT
from app.core.llm import is_fake_model
from app.core.logging import get_logger
from nlp.text_utils import turkish_lower
from schemas.dto import Entity, EntityType

logger = get_logger(__name__)

# TR IBAN: "TR" + 2 kontrol hanesi + 5x4 grup + son 2 hane = 26 karakter,
# gruplar arası tekli boşluklar opsiyonel (insanların yazma şekli).
_IBAN_RE = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)

# Önde para birimi sembolü ("€200") ya da sonda kod/kelime ("500 TRY").
# İki ayrı named-group seti: Python'ın `re`'si alternation dallarında aynı
# grup adını tekrar kullanmaya izin vermiyor.
_AMOUNT_CURRENCY_RE = re.compile(
    r"(?P<symbol>[€$£])\s?(?P<amt1>\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?)"
    r"|"
    r"(?P<amt2>\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s?"
    r"(?P<word>TL|TRY|USD|EUR|GBP|₺|euro|dolar|dollar|lira|pound)(?![A-Za-z])",
    re.IGNORECASE,
)

_CURRENCY_ISO_MAP = {
    "TL": "TRY",
    "TRY": "TRY",
    "₺": "TRY",
    "LIRA": "TRY",
    "USD": "USD",
    "$": "USD",
    "DOLAR": "USD",
    "DOLLAR": "USD",
    "EUR": "EUR",
    "€": "EUR",
    "EURO": "EUR",
    "GBP": "GBP",
    "£": "GBP",
    "POUND": "GBP",
}

_DATE_DMY_RE = re.compile(r"\b(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{4})\b")
_DATE_YMD_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")

# "**** 1234" / "****1234" tarzı maskeleme.
_CARD_MASK_RE = re.compile(r"\*{2,}\s?(?P<last4>\d{4})(?!\d)")
# "kart" kelimesinden kısa bir pencere sonra 4 haneli bir sayı. Pencere
# `[^\d]` değil `.` kullanıyor çünkü "son 4 hanesi" gibi ifadeler anahtar
# kelimeyle hedef arasına alakasız tek bir rakam sıkıştırabiliyor.
_CARD_KEYWORD_RE = re.compile(r"kart\w*.{0,30}?(?<!\d)(?P<last4>\d{4})(?!\d)", re.IGNORECASE)

_ACCOUNT_TYPE_VOCAB = (
    "vadeli hesap",
    "vadesiz hesap",
    "kredi kartı",
    "döviz hesabı",
    "birikim hesabı",
)
_ACCOUNT_TYPE_PATTERNS = tuple(
    (canonical, re.compile(re.escape(canonical), re.IGNORECASE))
    for canonical in _ACCOUNT_TYPE_VOCAB
)


def _normalize_amount(raw: str) -> str:
    """Türkçe (1.250,50) ve düz (500 / 50,00) formatları, sistemin beklediği
    '.' ondalık ayraçlı düz sayı string'ine indirger. İki ayraç da varsa
    sonuncusu ondalık sayılır; tek ayraç varsa ardından tam 2 hane geliyorsa
    ondalık, aksi halde binlik ayracı sayılıp atılır.
    """
    has_dot = "." in raw
    has_comma = "," in raw

    if has_dot and has_comma:
        if raw.rfind(",") > raw.rfind("."):
            integer_part, _, decimal_part = raw.replace(".", "").rpartition(",")
        else:
            integer_part, _, decimal_part = raw.replace(",", "").rpartition(".")
        return f"{integer_part}.{decimal_part}"

    if has_comma:
        integer_part, _, decimal_part = raw.rpartition(",")
        if len(decimal_part) == 2:
            return f"{integer_part}.{decimal_part}"
        return raw.replace(",", "")

    if has_dot:
        integer_part, _, decimal_part = raw.rpartition(".")
        if len(decimal_part) == 2:
            return f"{integer_part}.{decimal_part}"
        return raw.replace(".", "")

    return raw


def _normalize_currency(raw: str) -> str:
    return _CURRENCY_ISO_MAP.get(raw.upper(), raw.upper())


def _extract_ibans(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for match in _IBAN_RE.finditer(text):
        raw = match.group(0)
        entities.append(
            Entity(
                type=EntityType.IBAN,
                value=raw,
                normalized=re.sub(r"\s+", "", raw).upper(),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            )
        )
    return entities


def _extract_amounts_and_currency(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for match in _AMOUNT_CURRENCY_RE.finditer(text):
        if match.group("amt1") is not None:
            amount_raw = match.group("amt1")
            amount_start, amount_end = match.span("amt1")
            currency_raw = match.group("symbol")
            currency_start, currency_end = match.span("symbol")
        else:
            amount_raw = match.group("amt2")
            amount_start, amount_end = match.span("amt2")
            currency_raw = match.group("word")
            currency_start, currency_end = match.span("word")

        entities.append(
            Entity(
                type=EntityType.AMOUNT,
                value=amount_raw,
                normalized=_normalize_amount(amount_raw),
                start=amount_start,
                end=amount_end,
                confidence=1.0,
            )
        )
        if currency_raw:
            entities.append(
                Entity(
                    type=EntityType.CURRENCY,
                    value=currency_raw,
                    normalized=_normalize_currency(currency_raw),
                    start=currency_start,
                    end=currency_end,
                    confidence=1.0,
                )
            )
    return entities


def _extract_dates(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for match in _DATE_DMY_RE.finditer(text):
        day, month, year = match.group("day"), match.group("month"), match.group("year")
        entities.append(
            Entity(
                type=EntityType.DATE,
                value=match.group(0),
                normalized=f"{year}-{int(month):02d}-{int(day):02d}",
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            )
        )
    for match in _DATE_YMD_RE.finditer(text):
        entities.append(
            Entity(
                type=EntityType.DATE,
                value=match.group(0),
                normalized=match.group(0),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            )
        )
    return entities


def _extract_card_last4(text: str) -> list[Entity]:
    entities: list[Entity] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in (_CARD_MASK_RE, _CARD_KEYWORD_RE):
        for match in pattern.finditer(text):
            span = match.span("last4")
            if span in seen_spans:
                continue
            seen_spans.add(span)
            last4 = match.group("last4")
            entities.append(
                Entity(
                    type=EntityType.CARD_LAST4,
                    value=last4,
                    normalized=last4,
                    start=span[0],
                    end=span[1],
                    confidence=1.0,
                )
            )
    return entities


def _extract_account_types(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for canonical, pattern in _ACCOUNT_TYPE_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(
                Entity(
                    type=EntityType.ACCOUNT_TYPE,
                    value=match.group(0),
                    normalized=canonical,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                )
            )
    return entities


def extract_entities(text: str) -> list[Entity]:
    entities: list[Entity] = [
        *_extract_ibans(text),
        *_extract_amounts_and_currency(text),
        *_extract_dates(text),
        *_extract_card_last4(text),
        *_extract_account_types(text),
    ]
    # Pozisyona göre sırala ki trace/API çıktısı, hangi alt-extractor bulmuş
    # olursa olsun kaynak mesajla aynı soldan-sağa sırayı taşısın.
    entities.sort(key=lambda entity: entity.start if entity.start is not None else 0)
    return entities


class _ExtractedEntity(BaseModel):
    type: EntityType
    value: str
    normalized: str | None = None


class _NERExtraction(BaseModel):
    entities: list[_ExtractedEntity] = Field(default_factory=list)


def _dedup_key(entity_type: EntityType, value: str) -> tuple[EntityType, str]:
    return entity_type, turkish_lower(value.strip())


async def extract_entities_with_llm(text: str, llm: BaseChatModel) -> list[Entity]:
    """Regex geçişini her zaman çalıştırır; gerçek bir model bağlıysa üstüne
    bir LLM geçişi ekleyip regex'in kaçırdıklarını (özellikle PERSON_NAME)
    birleştirir.

    Fake modelde (`is_fake_model`) ya da LLM çağrısı/parse başarısız olursa
    sadece regex sonucuna düşer — `intent_classifier.classify_intent`'teki
    aynı fail-open mantık.
    """
    rule_based = extract_entities(text)

    if is_fake_model(llm):
        return rule_based

    try:
        structured_llm = llm.with_structured_output(_NERExtraction)
        result = await structured_llm.ainvoke(
            [SystemMessage(content=NER_SYSTEM_PROMPT), HumanMessage(content=text)]
        )
        if not isinstance(result, _NERExtraction):
            raise TypeError(f"beklenmeyen structured output tipi: {type(result)!r}")
    except Exception:
        logger.warning("ner_llm_extraction_failed", text_preview=text[:120], exc_info=True)
        return rule_based

    seen = {_dedup_key(e.type, e.normalized or e.value) for e in rule_based}
    merged = list(rule_based)
    for extracted in result.entities:
        key = _dedup_key(extracted.type, extracted.normalized or extracted.value)
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            Entity(
                type=extracted.type,
                value=extracted.value,
                normalized=extracted.normalized,
                # LLM karakter offset'i vermiyor; confidence de regex'in tam
                # eşleşmesi kadar kesin değil — bunu 1.0 göstermek yanıltıcı olurdu.
                confidence=0.75,
            )
        )

    # Stabil sıralama: regex'in bulduğu (start bilinen, zaten pozisyona göre
    # sıralı) varlıklar önce, LLM'in eklediği (start'ı olmayan) varlıklar sonra.
    merged.sort(key=lambda entity: entity.start is None)
    return merged
